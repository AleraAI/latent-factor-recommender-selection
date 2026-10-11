"""Post-confirmation test robustness for the expanded explicit-MF searches.

This runner consumes only completed validation selections.  It uses fresh seeds,
stores results below the isolated robustness tree, and never edits the original
selection lock or confirmation outputs.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file
from .explicit_mf import MFRunConfig, evaluate_stage, fit_explicit_mf
from .robustness_explicit import (
    DATASETS,
    DEFAULT_PROTOCOL_PATH,
    NON_FULL_RETENTIONS,
    RETENTIONS,
)


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


def _summary_matches(path: Path, signature: dict[str, Any], per_user: Path) -> bool:
    if not path.exists() or not per_user.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("signature") == signature


def _fit_and_evaluate(task: dict[str, Any]) -> str:
    summary_path = Path(task["summary_path"])
    per_user_path = Path(task["per_user_path"])
    if _summary_matches(summary_path, task["signature"], per_user_path):
        return str(summary_path)
    sparse_train = _load_frame(Path(task["sparse_train_path"]))
    full_train = _load_frame(Path(task["full_train_path"]))
    validation = _load_frame(Path(task["validation_path"]))
    test = _load_frame(Path(task["test_path"]))
    run_config = MFRunConfig(**task["run_config"])
    fitted = fit_explicit_mf(sparse_train, run_config)
    if len(fitted.epoch_history) != run_config.max_epochs:
        raise AssertionError("The post-confirmation fit stopped before its selected epoch")
    test_summary, per_user = evaluate_stage(
        fitted,
        full_train=full_train,
        evaluation=test,
        prior_evaluation=validation,
        ranking_batch_size=int(task["ranking_batch_size"]),
        relevance_threshold=float(task["relevance_threshold"]),
    )
    payload = {
        "schema_version": 1,
        "stage": "post_confirmation_explicit_test_robustness",
        "test_status": "post_confirmation_robustness_not_a_new_locked_test",
        "dataset": task["dataset"],
        "retention_label": task["retention_label"],
        "retention_seed": task.get("retention_seed"),
        "selection_source": task["selection_source"],
        "selection_roles": task["selection_roles"],
        "signature": task["signature"],
        "run_config": asdict(run_config),
        "training": {
            "rows": int(len(sparse_train)),
            "users": int(sparse_train["user_id"].astype(str).nunique()),
            "items": int(sparse_train["item_id"].astype(str).nunique()),
            "epochs_completed": len(fitted.epoch_history),
            "training_seconds": fitted.training_seconds,
            "epoch_history": fitted.epoch_history,
        },
        "test": test_summary,
    }
    _json_write(summary_path, payload)
    per_user_path.parent.mkdir(parents=True, exist_ok=True)
    per_user.to_parquet(per_user_path, index=False)
    return str(summary_path)


def _execute(tasks: list[dict[str, Any]], workers: int, label: str) -> None:
    pending = [
        task
        for task in tasks
        if not _summary_matches(
            Path(task["summary_path"]),
            task["signature"],
            Path(task["per_user_path"]),
        )
    ]
    complete = len(tasks) - len(pending)
    print(f"{label}: {complete}/{len(tasks)} already complete", flush=True)
    if workers == 1:
        for task in pending:
            _fit_and_evaluate(task)
            complete += 1
            if complete % 10 == 0 or complete == len(tasks):
                print(f"{label}: {complete}/{len(tasks)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fit_and_evaluate, task) for task in pending]
        for future in as_completed(futures):
            future.result()
            complete += 1
            if complete % 10 == 0 or complete == len(tasks):
                print(f"{label}: {complete}/{len(tasks)}", flush=True)


def _selected_expanded(selection_path: Path) -> list[dict[str, Any]]:
    selection = pd.read_csv(selection_path)
    rmse = selection.sort_values(
        ["mean_validation_rmse", "epoch", "rmse_dimension", "rmse_regularization"],
        ascending=[True, True, True, False],
        kind="stable",
    ).iloc[0]
    ndcg = selection.sort_values(
        [
            "mean_validation_ndcg@10",
            "epoch",
            "ndcg_dimension",
            "ndcg_regularization",
        ],
        ascending=[False, True, True, False],
        kind="stable",
    ).iloc[0]
    return [
        {
            "role": "rmse_selected",
            "dimension": int(rmse.rmse_dimension),
            "regularization": float(rmse.rmse_regularization),
            "epoch": int(rmse.epoch),
        },
        {
            "role": "ndcg_selected",
            "dimension": int(ndcg.ndcg_dimension),
            "regularization": float(ndcg.ndcg_regularization),
            "epoch": int(ndcg.epoch),
        },
    ]


def _selected_repeated(selection_path: Path) -> list[dict[str, Any]]:
    selected = pd.read_csv(selection_path).iloc[0]
    return [
        {
            "role": "rmse_selected",
            "dimension": int(selected.rmse_dimension),
            "regularization": float(selected.rmse_regularization),
            "epoch": int(selected.epoch),
        },
        {
            "role": "ndcg_selected",
            "dimension": int(selected.ndcg_dimension),
            "regularization": float(selected.ndcg_regularization),
            "epoch": int(selected.epoch),
        },
    ]


def _collapse_roles(selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: dict[tuple[int, float, int], dict[str, Any]] = {}
    for selected in selections:
        key = (
            int(selected["dimension"]),
            float(selected["regularization"]),
            int(selected["epoch"]),
        )
        if key not in collapsed:
            collapsed[key] = {
                "dimension": key[0],
                "regularization": key[1],
                "epoch": key[2],
                "roles": [],
            }
        collapsed[key]["roles"].append(str(selected["role"]))
    return list(collapsed.values())


def _condition_tasks(
    *,
    config: dict[str, Any],
    protocol: dict[str, Any],
    root: Path,
    protocol_path: Path,
    dataset: str,
    retention: str,
    retention_seed: int | None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    processed = _resolve(root, config["datasets"][dataset]["output_dir"])
    robustness_root = _resolve(root, protocol["output_root"])
    if retention_seed is None:
        selection_source = "explicit_expanded_validation"
        selection_path = (
            robustness_root
            / "explicit_expanded"
            / dataset
            / retention
            / "selections.csv"
        )
        sparse = (
            _resolve(root, config["controlled_sparsity"]["output_root"])
            / dataset
            / f"train_{retention}.parquet"
        )
        output_dir = robustness_root / "explicit_confirmation" / dataset / retention
        selected = _selected_expanded(selection_path)
    else:
        selection_source = "explicit_repeated_retention_validation"
        selection_path = (
            robustness_root
            / "explicit_repeated_retention"
            / f"seed_{retention_seed}"
            / dataset
            / retention
            / "selections.csv"
        )
        sparse = (
            _resolve(root, protocol["nested_retention_root"])
            / f"seed_{retention_seed}"
            / dataset
            / f"train_{retention}.parquet"
        )
        output_dir = (
            robustness_root
            / "explicit_repeated_confirmation"
            / f"seed_{retention_seed}"
            / dataset
            / retention
        )
        selected = _selected_repeated(selection_path)
    if not selection_path.exists():
        raise FileNotFoundError(f"Run the validation search first: {selection_path}")
    common = protocol["common"]
    protocol_hash = sha256_file(protocol_path)
    selection_hash = sha256_file(selection_path)
    sparse_hash = sha256_file(sparse)
    tasks: list[dict[str, Any]] = []
    paths: list[Path] = []
    for choice in _collapse_roles(selected):
        for seed in common["robustness_confirmation_seeds"]:
            run_id = (
                f"d{choice['dimension']}_lambda{choice['regularization']:g}_"
                f"epoch{choice['epoch']}_seed{seed}"
            )
            run_dir = output_dir / "runs" / run_id
            summary_path = run_dir / "summary.json"
            per_user_path = run_dir / "test_per_user.parquet"
            signature = {
                "protocol_sha256": protocol_hash,
                "selection_sha256": selection_hash,
                "sparse_train_sha256": sparse_hash,
                "dimension": int(choice["dimension"]),
                "regularization": float(choice["regularization"]),
                "epoch": int(choice["epoch"]),
                "seed": int(seed),
                "selection_roles": sorted(choice["roles"]),
                "test_is_post_confirmation": True,
            }
            run_config = {
                "dimension": int(choice["dimension"]),
                "regularization": float(choice["regularization"]),
                "learning_rate": float(protocol["explicit_expanded"]["learning_rate"]),
                "batch_size": int(common["batch_size"]),
                "max_epochs": int(choice["epoch"]),
                # A patience longer than the fixed budget makes the fit exact.
                "convergence_patience": int(choice["epoch"]) + 1,
                "convergence_relative_tolerance": float(
                    protocol["explicit_expanded"]["convergence_relative_tolerance"]
                ),
                "seed": int(seed),
                "device": str(common["device"]),
                "torch_num_threads": int(common["torch_num_threads"]),
            }
            tasks.append(
                {
                    "summary_path": str(summary_path),
                    "per_user_path": str(per_user_path),
                    "signature": signature,
                    "dataset": dataset,
                    "retention_label": retention,
                    "retention_seed": retention_seed,
                    "selection_source": selection_source,
                    "selection_roles": sorted(choice["roles"]),
                    "sparse_train_path": str(sparse),
                    "full_train_path": str(processed / "train.parquet"),
                    "validation_path": str(processed / "validation.parquet"),
                    "test_path": str(processed / "test.parquet"),
                    "ranking_batch_size": int(common["ranking_batch_size"]),
                    "relevance_threshold": float(common["relevance_threshold"]),
                    "run_config": run_config,
                }
            )
            paths.append(summary_path)
    return tasks, paths


def _aggregate(paths: Iterable[Path], output_dir: Path) -> Path:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload["run_config"]
        for role in payload["selection_roles"]:
            records.append(
                {
                    "dataset": payload["dataset"],
                    "retention_label": payload["retention_label"],
                    "retention_seed": payload.get("retention_seed"),
                    "selection_source": payload["selection_source"],
                    "selection_role": role,
                    "dimension": int(run["dimension"]),
                    "regularization": float(run["regularization"]),
                    "epoch": int(run["max_epochs"]),
                    "seed": int(run["seed"]),
                    "training_seconds": payload["training"]["training_seconds"],
                    "test_rmse": payload["test"]["prediction"]["clipped"]["rmse"],
                    "test_mae": payload["test"]["prediction"]["clipped"]["mae"],
                    "test_ndcg@10": payload["test"]["ranking"]["ndcg@10"],
                    "test_recall@10": payload["test"]["ranking"]["recall@10"],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runs.csv"
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
    return output_path


def execute_expanded(
    config_path: Path, protocol_path: Path, dataset: str, retention: str, workers: int
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    tasks, paths = _condition_tasks(
        config=config,
        protocol=protocol,
        root=root,
        protocol_path=protocol_path,
        dataset=dataset,
        retention=retention,
        retention_seed=None,
    )
    _execute(tasks, workers, f"explicit-confirmation/{dataset}/{retention}")
    output_dir = _resolve(root, protocol["output_root"]) / "explicit_confirmation" / dataset / retention
    return _aggregate(paths, output_dir)


def execute_repeated(
    config_path: Path,
    protocol_path: Path,
    retention_seed: int,
    dataset: str,
    retention: str,
    workers: int,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    tasks, paths = _condition_tasks(
        config=config,
        protocol=protocol,
        root=root,
        protocol_path=protocol_path,
        dataset=dataset,
        retention=retention,
        retention_seed=retention_seed,
    )
    _execute(
        tasks,
        workers,
        f"explicit-repeated-confirmation/{retention_seed}/{dataset}/{retention}",
    )
    output_dir = (
        _resolve(root, protocol["output_root"])
        / "explicit_repeated_confirmation"
        / f"seed_{retention_seed}"
        / dataset
        / retention
    )
    return _aggregate(paths, output_dir)


def _choices(value: str, allowed: tuple[str, ...]) -> list[str]:
    return list(allowed) if value == "all" else [value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["expanded", "repeated"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--retention", default="all")
    parser.add_argument("--retention-seed", default="all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "expanded":
        if args.retention != "all" and args.retention not in RETENTIONS:
            parser.error(f"unknown retention: {args.retention}")
        for dataset in _choices(args.dataset, DATASETS):
            for retention in _choices(args.retention, RETENTIONS):
                execute_expanded(
                    args.config, args.protocol, dataset, retention, args.workers
                )
        return
    if args.retention != "all" and args.retention not in NON_FULL_RETENTIONS:
        parser.error(f"unknown repeated-retention condition: {args.retention}")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8")) or {}
    seeds = [int(value) for value in protocol["repeated_retention"]["retention_seeds"]]
    if args.retention_seed != "all":
        requested = int(args.retention_seed)
        if requested not in seeds:
            parser.error(f"unknown retention seed: {requested}")
        seeds = [requested]
    for retention_seed in seeds:
        for dataset in _choices(args.dataset, DATASETS):
            for retention in _choices(args.retention, NON_FULL_RETENTIONS):
                execute_repeated(
                    args.config,
                    args.protocol,
                    retention_seed,
                    dataset,
                    retention,
                    args.workers,
                )


if __name__ == "__main__":
    main()
