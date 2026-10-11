"""Post-confirmation explicit-MF robustness experiments.

This module never edits the original development outputs or selection lock. It
provides three resumable stages: nested retention construction, the expanded
explicit-MF grid on the original retained artifacts, and the same expanded grid
over independently seeded nested retention families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file
from .explicit_mf import (
    BiasedMatrixFactorization,
    FittedMF,
    MFRunConfig,
    _id_sort_key,
    evaluate_validation,
    set_deterministic_seed,
)


DEFAULT_PROTOCOL_PATH = Path(
    "configs/latent_factor_beyond_rmse/robustness.yaml"
)
DATASETS = ("movielens_1m", "ciaodvd", "yelp_tampa_restaurants")
RETENTIONS = ("r100", "r075", "r050", "r025")
NON_FULL_RETENTIONS = ("r075", "r050", "r025")
RETENTION_VALUES = {"r100": 1.0, "r075": 0.75, "r050": 0.50, "r025": 0.25}

_FRAME_CACHE: dict[str, pd.DataFrame] = {}


def _load_frame(path: Path) -> pd.DataFrame:
    key = str(path.resolve())
    if key not in _FRAME_CACHE:
        _FRAME_CACHE[key] = pd.read_parquet(path)
    return _FRAME_CACHE[key]


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _priority(seed: int, interaction_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{interaction_id}".encode("utf-8")).hexdigest()


def nested_retention_family(
    full_train: pd.DataFrame,
    *,
    seed: int,
    minimum_per_user: int = 3,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build a fully nested 100/75/50/25 family with catalog repair.

    The priority ordering is independent of retention. The 25% subset is
    repaired to the full-training catalog and those repair rows are propagated
    to every larger subset, guaranteeing exact set inclusion.
    """
    required = {"interaction_id", "user_id", "item_id", "timestamp", "source_row_id"}
    missing = sorted(required - set(full_train.columns))
    if missing:
        raise ValueError(f"full_train is missing columns: {missing}")
    if full_train["interaction_id"].duplicated().any():
        raise ValueError("full_train contains duplicate interaction identifiers")

    train = full_train.copy()
    train["_priority"] = train["interaction_id"].astype(str).map(
        lambda value: _priority(seed, value)
    )
    selected: dict[str, set[int]] = {label: set() for label in RETENTIONS}
    requested: dict[str, int] = {label: 0 for label in RETENTIONS}
    for user_id, rows in train.groupby("user_id", sort=False):
        if len(rows) < minimum_per_user:
            raise ValueError(
                f"User {user_id!r} has {len(rows)} rows, below {minimum_per_user}"
            )
        ordered = rows.sort_values(["_priority", "interaction_id"], kind="stable")
        for label in RETENTIONS:
            q = RETENTION_VALUES[label]
            target = min(
                len(rows), max(minimum_per_user, int(q * len(rows) + 0.5))
            )
            requested[label] += target
            selected[label].update(int(index) for index in ordered.head(target).index)

    base_selected = {label: set(indices) for label, indices in selected.items()}
    full_catalog = set(train["item_id"].astype(str))
    smallest = selected["r025"]
    present = set(train.loc[sorted(smallest), "item_id"].astype(str))
    missing_items = sorted(full_catalog - present)
    repaired_ids: list[str] = []
    for item_id in missing_items:
        candidate = (
            train.loc[train["item_id"].astype(str) == item_id]
            .sort_values(["_priority", "interaction_id"], kind="stable")
            .iloc[0]
        )
        smallest.add(int(candidate.name))
        repaired_ids.append(str(candidate["interaction_id"]))

    selected["r050"].update(selected["r025"])
    selected["r075"].update(selected["r050"])
    selected["r100"] = set(train.index)

    artifacts: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}
    previous_ids: set[str] = set()
    for label in reversed(RETENTIONS):
        frame = (
            train.loc[sorted(selected[label])]
            .drop(columns="_priority")
            .sort_values(
                ["user_id", "timestamp", "source_row_id", "item_id"],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        ids = set(frame["interaction_id"].astype(str))
        if previous_ids and not previous_ids.issubset(ids):
            raise AssertionError("Nested retention inclusion failed")
        previous_ids = ids
        if set(frame["item_id"].astype(str)) != full_catalog:
            raise AssertionError("Nested retention catalog repair failed")
        if int(frame.groupby("user_id").size().min()) < minimum_per_user:
            raise AssertionError("Nested retention minimum user degree failed")
        artifacts[label] = frame
        audit[label] = {
            "retention_nominal": RETENTION_VALUES[label],
            "seed": int(seed),
            "initial_requested_rows": int(requested[label]),
            "realized_rows": int(len(frame)),
            "realized_retention": float(len(frame) / len(full_train)),
            "users": int(frame["user_id"].nunique()),
            "items": int(frame["item_id"].nunique()),
            "minimum_realized_user_degree": int(frame.groupby("user_id").size().min()),
            "repair_rows_propagated": int(
                len(selected[label] - base_selected[label])
            ),
        }

    audit["family"] = {
        "seed": int(seed),
        "nested": True,
        "catalog_repair_origin": "r025_then_propagated_upward",
        "repaired_items": len(missing_items),
        "repaired_interaction_ids": repaired_ids,
        "priority_rule": "sha256(seed, interaction_id)",
    }
    return artifacts, audit


def build_nested_retention_artifacts(
    config_path: Path = DEFAULT_CONFIG_PATH,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> list[Path]:
    dataset_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    output_root = _resolve(root, protocol["nested_retention_root"])
    settings = protocol["repeated_retention"]
    paths: list[Path] = []
    for seed in [int(value) for value in settings["retention_seeds"]]:
        for dataset_name in protocol["common"]["datasets"]:
            processed_dir = _resolve(
                root, dataset_config["datasets"][dataset_name]["output_dir"]
            )
            full_train = pd.read_parquet(processed_dir / "train.parquet")
            artifacts, audit = nested_retention_family(
                full_train,
                seed=seed,
                minimum_per_user=int(settings["minimum_per_user"]),
            )
            destination = output_root / f"seed_{seed}" / dataset_name
            condition_payload: dict[str, Any] = {}
            for label, frame in artifacts.items():
                path = destination / f"train_{label}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                condition_payload[label] = {
                    **audit[label],
                    "artifact": {
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    },
                }
                paths.append(path)
            manifest = {
                "schema_version": 1,
                "stage": "post_confirmation_nested_retention",
                "dataset": dataset_name,
                "protocol_sha256": sha256_file(protocol_path),
                "family": audit["family"],
                "conditions": condition_payload,
            }
            manifest_path = destination / "nested_retention_manifest.json"
            _json_write(manifest_path, manifest)
            paths.append(manifest_path)
    return paths


def _train_explicit_checkpoints(
    *,
    sparse_train: pd.DataFrame,
    full_train: pd.DataFrame,
    validation: pd.DataFrame,
    dimension: int,
    regularization: float,
    learning_rate: float,
    batch_size: int,
    checkpoints: list[int],
    convergence_patience: int,
    convergence_relative_tolerance: float,
    seed: int,
    ranking_batch_size: int,
    device_name: str,
    torch_num_threads: int,
) -> dict[str, Any]:
    set_deterministic_seed(seed)
    torch.set_num_threads(torch_num_threads)
    device = torch.device(device_name)
    user_ids = sorted(set(sparse_train["user_id"].astype(str)), key=_id_sort_key)
    item_ids = sorted(set(sparse_train["item_id"].astype(str)), key=_id_sort_key)
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}
    user_index = sparse_train["user_id"].astype(str).map(user_to_index).to_numpy(np.int64)
    item_index = sparse_train["item_id"].astype(str).map(item_to_index).to_numpy(np.int64)
    ratings = sparse_train["rating"].to_numpy(np.float32)
    model = BiasedMatrixFactorization(
        num_users=len(user_ids),
        num_items=len(item_ids),
        dimension=dimension,
        global_mean=float(ratings.mean()),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    checkpoint_set = set(checkpoints)
    max_epoch = max(checkpoints)
    history: list[dict[str, Any]] = []
    validation_results: dict[str, Any] = {}
    training_seconds = 0.0
    best_objective = math.inf
    stale_epochs = 0
    convergence_epoch: int | None = None

    run_config = MFRunConfig(
        dimension=dimension,
        regularization=regularization,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epoch,
        convergence_patience=convergence_patience,
        convergence_relative_tolerance=convergence_relative_tolerance,
        seed=seed,
        device=device_name,
        torch_num_threads=torch_num_threads,
    )

    for epoch in range(1, max_epoch + 1):
        started = time.perf_counter()
        permutation = rng.permutation(len(sparse_train))
        objective_sum = 0.0
        squared_error_sum = 0.0
        model.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            users = torch.as_tensor(user_index[indices], dtype=torch.long, device=device)
            items = torch.as_tensor(item_index[indices], dtype=torch.long, device=device)
            target = torch.as_tensor(ratings[indices], dtype=torch.float32, device=device)
            prediction = model(users, items)
            squared_errors = (prediction - target).square()
            objective = squared_errors.mean()
            if regularization:
                objective = objective + regularization * model.regularization_per_observation(
                    users, items
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            objective_sum += float(objective.detach()) * len(indices)
            squared_error_sum += float(squared_errors.detach().sum())
        training_seconds += time.perf_counter() - started
        epoch_objective = objective_sum / len(sparse_train)
        train_rmse = math.sqrt(squared_error_sum / len(sparse_train))
        relative_improvement = (
            math.inf
            if not math.isfinite(best_objective)
            else (best_objective - epoch_objective) / max(abs(best_objective), 1.0)
        )
        if epoch_objective < best_objective:
            best_objective = epoch_objective
        if relative_improvement > convergence_relative_tolerance:
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= convergence_patience and convergence_epoch is None:
            convergence_epoch = epoch
        history.append(
            {
                "epoch": epoch,
                "training_objective": epoch_objective,
                "training_rmse": train_rmse,
                "relative_objective_improvement": relative_improvement,
                "stale_epochs": stale_epochs,
            }
        )
        if epoch in checkpoint_set:
            fitted = FittedMF(
                model=model,
                user_ids=user_ids,
                item_ids=item_ids,
                user_to_index=user_to_index,
                item_to_index=item_to_index,
                epoch_history=history.copy(),
                training_seconds=training_seconds,
                config=run_config,
            )
            summary, _ = evaluate_validation(
                fitted,
                sparse_train=sparse_train,
                full_train=full_train,
                validation=validation,
                ranking_batch_size=ranking_batch_size,
            )
            validation_results[str(epoch)] = summary

    return {
        "run_config": asdict(run_config),
        "training": {
            "rows": int(len(sparse_train)),
            "users": len(user_ids),
            "items": len(item_ids),
            "epochs_completed": max_epoch,
            "training_seconds": training_seconds,
            "convergence_epoch_by_training_tolerance": convergence_epoch,
            "converged_by_max_epoch": convergence_epoch is not None,
            "epoch_history": history,
        },
        "validation_checkpoints": validation_results,
    }


def _summary_matches(path: Path, signature: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("signature") == signature


def _explicit_task(task: dict[str, Any]) -> str:
    output_path = Path(task["output_path"])
    signature = task["signature"]
    if _summary_matches(output_path, signature):
        return str(output_path)
    sparse_train = _load_frame(Path(task["sparse_train_path"]))
    full_train = _load_frame(Path(task["full_train_path"]))
    validation = _load_frame(Path(task["validation_path"]))
    result = _train_explicit_checkpoints(
        sparse_train=sparse_train,
        full_train=full_train,
        validation=validation,
        **task["train_arguments"],
    )
    payload = {
        "schema_version": 1,
        "stage": task["stage"],
        "dataset": task["dataset"],
        "retention_label": task["retention_label"],
        "retention_seed": task.get("retention_seed"),
        "signature": signature,
        **result,
    }
    _json_write(output_path, payload)
    return str(output_path)


def _configuration_id(dimension: int, regularization: float, seed: int) -> str:
    return f"d{dimension}_lambda{regularization:g}_seed{seed}"


def _execute_tasks(tasks: list[dict[str, Any]], workers: int, label: str) -> None:
    pending = [
        task
        for task in tasks
        if not _summary_matches(Path(task["output_path"]), task["signature"])
    ]
    complete = len(tasks) - len(pending)
    print(f"{label}: {complete}/{len(tasks)} already complete", flush=True)
    if not pending:
        return
    if workers == 1:
        for task in pending:
            _explicit_task(task)
            complete += 1
            if complete % 10 == 0 or complete == len(tasks):
                print(f"{label}: {complete}/{len(tasks)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_explicit_task, task) for task in pending]
        for future in as_completed(futures):
            future.result()
            complete += 1
            if complete % 10 == 0 or complete == len(tasks):
                print(f"{label}: {complete}/{len(tasks)}", flush=True)


def _aggregate_explicit(
    summary_paths: Iterable[Path], output_dir: Path, checkpoints: list[int]
) -> tuple[Path, Path]:
    records: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload["run_config"]
        for epoch in checkpoints:
            validation = payload["validation_checkpoints"][str(epoch)]
            records.append(
                {
                    "dataset": payload["dataset"],
                    "retention_label": payload["retention_label"],
                    "retention_seed": payload.get("retention_seed"),
                    "dimension": int(run["dimension"]),
                    "regularization": float(run["regularization"]),
                    "seed": int(run["seed"]),
                    "epoch": epoch,
                    "training_seconds": payload["training"]["training_seconds"],
                    "convergence_epoch": payload["training"][
                        "convergence_epoch_by_training_tolerance"
                    ],
                    "validation_rmse": validation["prediction"]["clipped"]["rmse"],
                    "validation_ndcg@10": validation["ranking"]["ndcg@10"],
                    "validation_recall@10": validation["ranking"]["recall@10"],
                }
            )
    frame = pd.DataFrame.from_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.csv"
    frame.to_csv(runs_path, index=False)
    group_columns = ["dataset", "retention_label", "epoch"]
    if frame["retention_seed"].notna().any():
        group_columns.insert(2, "retention_seed")
    selections: list[dict[str, Any]] = []
    for keys, condition in frame.groupby(group_columns, dropna=False, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, key_values))
        grouped = (
            condition.groupby(["dimension", "regularization"], as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                mean_validation_rmse=("validation_rmse", "mean"),
                std_validation_rmse=("validation_rmse", "std"),
                mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
                std_validation_ndcg_at_10=("validation_ndcg@10", "std"),
            )
        )
        rmse = grouped.sort_values(
            ["mean_validation_rmse", "dimension", "regularization"],
            ascending=[True, True, False],
            kind="stable",
        ).iloc[0]
        ndcg = grouped.sort_values(
            ["mean_validation_ndcg_at_10", "dimension", "regularization"],
            ascending=[False, True, False],
            kind="stable",
        ).iloc[0]
        selections.append(
            {
                **identity,
                "rmse_dimension": int(rmse.dimension),
                "rmse_regularization": float(rmse.regularization),
                "mean_validation_rmse": float(rmse.mean_validation_rmse),
                "ndcg_dimension": int(ndcg.dimension),
                "ndcg_regularization": float(ndcg.regularization),
                "mean_validation_ndcg@10": float(ndcg.mean_validation_ndcg_at_10),
                "same_configuration": bool(
                    rmse.dimension == ndcg.dimension
                    and rmse.regularization == ndcg.regularization
                ),
            }
        )
    selection_frame = pd.DataFrame.from_records(selections)
    selection_path = output_dir / "selections.csv"
    selection_frame.to_csv(selection_path, index=False)
    return runs_path, selection_path


def execute_expanded_condition(
    *,
    config_path: Path,
    protocol_path: Path,
    dataset_name: str,
    retention_label: str,
    workers: int,
) -> tuple[Path, Path]:
    dataset_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    common = protocol["common"]
    settings = protocol["explicit_expanded"]
    processed = _resolve(root, dataset_config["datasets"][dataset_name]["output_dir"])
    sparse = (
        _resolve(root, dataset_config["controlled_sparsity"]["output_root"])
        / dataset_name
        / f"train_{retention_label}.parquet"
    )
    output_dir = (
        _resolve(root, protocol["output_root"])
        / "explicit_expanded"
        / dataset_name
        / retention_label
    )
    checkpoints = [int(value) for value in settings["checkpoint_epochs"]]
    protocol_hash = sha256_file(protocol_path)
    sparse_hash = sha256_file(sparse)
    tasks: list[dict[str, Any]] = []
    paths: list[Path] = []
    for dimension in settings["dimensions"]:
        for regularization in settings["regularizations"]:
            for seed in common["selection_seeds"]:
                run_id = _configuration_id(int(dimension), float(regularization), int(seed))
                output_path = output_dir / "runs" / run_id / "summary.json"
                signature = {
                    "protocol_sha256": protocol_hash,
                    "sparse_train_sha256": sparse_hash,
                    "dimension": int(dimension),
                    "regularization": float(regularization),
                    "seed": int(seed),
                    "checkpoints": checkpoints,
                }
                tasks.append(
                    {
                        "output_path": str(output_path),
                        "signature": signature,
                        "stage": "post_confirmation_explicit_expanded",
                        "dataset": dataset_name,
                        "retention_label": retention_label,
                        "sparse_train_path": str(sparse),
                        "full_train_path": str(processed / "train.parquet"),
                        "validation_path": str(processed / "validation.parquet"),
                        "train_arguments": {
                            "dimension": int(dimension),
                            "regularization": float(regularization),
                            "learning_rate": float(settings["learning_rate"]),
                            "batch_size": int(common["batch_size"]),
                            "checkpoints": checkpoints,
                            "convergence_patience": int(settings["convergence_patience"]),
                            "convergence_relative_tolerance": float(
                                settings["convergence_relative_tolerance"]
                            ),
                            "seed": int(seed),
                            "ranking_batch_size": int(common["ranking_batch_size"]),
                            "device_name": str(common["device"]),
                            "torch_num_threads": int(common["torch_num_threads"]),
                        },
                    }
                )
                paths.append(output_path)
    _execute_tasks(tasks, workers, f"expanded/{dataset_name}/{retention_label}")
    return _aggregate_explicit(paths, output_dir, checkpoints)


def execute_repeated_condition(
    *,
    config_path: Path,
    protocol_path: Path,
    retention_seed: int,
    dataset_name: str,
    retention_label: str,
    workers: int,
) -> tuple[Path, Path]:
    dataset_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    common = protocol["common"]
    settings = protocol["repeated_retention"]
    processed = _resolve(root, dataset_config["datasets"][dataset_name]["output_dir"])
    sparse = (
        _resolve(root, protocol["nested_retention_root"])
        / f"seed_{retention_seed}"
        / dataset_name
        / f"train_{retention_label}.parquet"
    )
    if not sparse.exists():
        raise FileNotFoundError(f"Build nested retention artifacts first: {sparse}")
    output_dir = (
        _resolve(root, protocol["output_root"])
        / "explicit_repeated_retention"
        / f"seed_{retention_seed}"
        / dataset_name
        / retention_label
    )
    checkpoints = [int(settings["max_epochs"])]
    protocol_hash = sha256_file(protocol_path)
    sparse_hash = sha256_file(sparse)
    tasks: list[dict[str, Any]] = []
    paths: list[Path] = []
    for dimension in settings["explicit_dimensions"]:
        for regularization in settings["explicit_regularizations"]:
            for seed in settings["selection_seeds"]:
                run_id = _configuration_id(int(dimension), float(regularization), int(seed))
                output_path = output_dir / "runs" / run_id / "summary.json"
                signature = {
                    "protocol_sha256": protocol_hash,
                    "sparse_train_sha256": sparse_hash,
                    "retention_seed": retention_seed,
                    "dimension": int(dimension),
                    "regularization": float(regularization),
                    "seed": int(seed),
                    "checkpoints": checkpoints,
                }
                tasks.append(
                    {
                        "output_path": str(output_path),
                        "signature": signature,
                        "stage": "post_confirmation_explicit_repeated_retention",
                        "dataset": dataset_name,
                        "retention_label": retention_label,
                        "retention_seed": retention_seed,
                        "sparse_train_path": str(sparse),
                        "full_train_path": str(processed / "train.parquet"),
                        "validation_path": str(processed / "validation.parquet"),
                        "train_arguments": {
                            "dimension": int(dimension),
                            "regularization": float(regularization),
                            "learning_rate": float(settings["learning_rate"]),
                            "batch_size": int(common["batch_size"]),
                            "checkpoints": checkpoints,
                            "convergence_patience": int(
                                protocol["explicit_expanded"]["convergence_patience"]
                            ),
                            "convergence_relative_tolerance": float(
                                protocol["explicit_expanded"][
                                    "convergence_relative_tolerance"
                                ]
                            ),
                            "seed": int(seed),
                            "ranking_batch_size": int(common["ranking_batch_size"]),
                            "device_name": str(common["device"]),
                            "torch_num_threads": int(common["torch_num_threads"]),
                        },
                    }
                )
                paths.append(output_path)
    _execute_tasks(
        tasks,
        workers,
        f"repeated/{retention_seed}/{dataset_name}/{retention_label}",
    )
    return _aggregate_explicit(paths, output_dir, checkpoints)


def _choices(value: str, allowed: tuple[str, ...]) -> list[str]:
    return list(allowed) if value == "all" else [value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    subparsers.add_parser("nested-retention")

    expanded = subparsers.add_parser("expanded")
    expanded.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    expanded.add_argument("--retention", choices=["all", *RETENTIONS], default="all")
    expanded.add_argument("--workers", type=int, default=1)

    repeated = subparsers.add_parser("repeated")
    repeated.add_argument("--retention-seed", default="all")
    repeated.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    repeated.add_argument(
        "--retention", choices=["all", *NON_FULL_RETENTIONS], default="all"
    )
    repeated.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()
    if args.stage == "nested-retention":
        for path in build_nested_retention_artifacts(args.config, args.protocol):
            print(path)
        return
    if args.stage == "expanded":
        for dataset in _choices(args.dataset, DATASETS):
            for retention in _choices(args.retention, RETENTIONS):
                execute_expanded_condition(
                    config_path=args.config,
                    protocol_path=args.protocol,
                    dataset_name=dataset,
                    retention_label=retention,
                    workers=args.workers,
                )
        return
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8")) or {}
    seeds = [int(value) for value in protocol["repeated_retention"]["retention_seeds"]]
    if args.retention_seed != "all":
        requested_seed = int(args.retention_seed)
        if requested_seed not in seeds:
            raise ValueError(f"Unknown retention seed: {requested_seed}")
        seeds = [requested_seed]
    for retention_seed in seeds:
        for dataset in _choices(args.dataset, DATASETS):
            for retention in _choices(args.retention, NON_FULL_RETENTIONS):
                execute_repeated_condition(
                    config_path=args.config,
                    protocol_path=args.protocol,
                    retention_seed=retention_seed,
                    dataset_name=dataset,
                    retention_label=retention,
                    workers=args.workers,
                )


if __name__ == "__main__":
    main()
