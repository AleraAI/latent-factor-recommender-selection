"""Independent BPR tuning and negative-sampling robustness experiments.

All development searches are validation-only. Test confirmation is a separate
explicitly post-confirmation command and never modifies the original BPR lock.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from .bpr_mf import (
    BPRMatrixFactorization,
    BPRRunConfig,
    FittedBPR,
    evaluate_bpr_stage,
    set_deterministic_seed,
)
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file
from .explicit_mf import _id_sort_key
from .robustness_explicit import DATASETS, RETENTIONS


DEFAULT_PROTOCOL_PATH = Path("configs/latent_factor_beyond_rmse/robustness.yaml")
PRIMARY_MODE = "retained_unobserved"
ORACLE_MODE = "full_training_positive_oracle"
EXPLICIT_MODE = "retained_explicit_negative"
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


def sample_uniform_unblocked(
    *,
    rng: np.random.Generator,
    users: np.ndarray,
    blocked: np.ndarray,
    count: int,
) -> np.ndarray:
    """Sample ``count`` independent unblocked items per user."""
    if count < 1:
        raise ValueError("count must be positive")
    if np.any(blocked[users].all(axis=1)):
        raise ValueError("At least one user has no eligible negative")
    columns: list[np.ndarray] = []
    for _ in range(count):
        negatives = rng.integers(0, blocked.shape[1], size=len(users), dtype=np.int64)
        invalid = blocked[users, negatives]
        while bool(invalid.any()):
            negatives[invalid] = rng.integers(
                0, blocked.shape[1], size=int(invalid.sum()), dtype=np.int64
            )
            invalid = blocked[users, negatives]
        columns.append(negatives)
    return np.column_stack(columns)


def _build_boolean_matrix(
    frame: pd.DataFrame,
    *,
    user_to_index: dict[str, int],
    item_to_index: dict[str, int],
    rating_threshold: float | None = None,
) -> np.ndarray:
    matrix = np.zeros((len(user_to_index), len(item_to_index)), dtype=np.bool_)
    selected = frame
    if rating_threshold is not None:
        selected = selected[selected["rating"] >= rating_threshold]
    users = selected["user_id"].astype(str).map(user_to_index)
    items = selected["item_id"].astype(str).map(item_to_index)
    known = users.notna() & items.notna()
    if known.any():
        matrix[
            users[known].to_numpy(dtype=np.int64),
            items[known].to_numpy(dtype=np.int64),
        ] = True
    return matrix


def _explicit_negative_lists(
    sparse_train: pd.DataFrame,
    *,
    user_to_index: dict[str, int],
    item_to_index: dict[str, int],
    threshold: float,
) -> dict[int, np.ndarray]:
    negative = sparse_train[sparse_train["rating"] < threshold]
    output: dict[int, np.ndarray] = {}
    for user_id, rows in negative.groupby("user_id", sort=False):
        user_index = user_to_index[str(user_id)]
        values = rows["item_id"].astype(str).map(item_to_index).dropna().to_numpy(np.int64)
        if len(values):
            output[user_index] = np.unique(values)
    return output


def _sample_bpr_negatives(
    *,
    rng: np.random.Generator,
    users: np.ndarray,
    retained_observed: np.ndarray,
    oracle_blocked: np.ndarray,
    explicit_lists: dict[int, np.ndarray],
    mode: str,
    count: int,
) -> tuple[np.ndarray, int]:
    if mode == PRIMARY_MODE:
        return (
            sample_uniform_unblocked(
                rng=rng, users=users, blocked=retained_observed, count=count
            ),
            0,
        )
    if mode == ORACLE_MODE:
        return (
            sample_uniform_unblocked(
                rng=rng, users=users, blocked=oracle_blocked, count=count
            ),
            0,
        )
    if mode != EXPLICIT_MODE:
        raise ValueError(f"Unknown negative mode: {mode}")
    sampled = np.empty((len(users), count), dtype=np.int64)
    fallback_rows = 0
    for row_index, user in enumerate(users):
        values = explicit_lists.get(int(user))
        if values is not None and len(values):
            sampled[row_index] = rng.choice(values, size=count, replace=True)
        else:
            sampled[row_index] = sample_uniform_unblocked(
                rng=rng,
                users=np.asarray([user], dtype=np.int64),
                blocked=retained_observed,
                count=count,
            )[0]
            fallback_rows += 1
    return sampled, fallback_rows


def train_bpr_checkpoints(
    *,
    sparse_train: pd.DataFrame,
    full_train: pd.DataFrame,
    evaluation: pd.DataFrame,
    prior_evaluation: pd.DataFrame | None,
    dimension: int,
    regularization: float,
    learning_rate: float,
    batch_size: int,
    negatives_per_positive: int,
    negative_mode: str,
    checkpoints: list[int],
    relevance_threshold: float,
    seed: int,
    ranking_batch_size: int,
    device_name: str,
    torch_num_threads: int,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    required = {"user_id", "item_id", "rating"}
    if missing := sorted(required - set(sparse_train.columns)):
        raise ValueError(f"Training data are missing columns: {missing}")
    set_deterministic_seed(seed)
    torch.set_num_threads(torch_num_threads)
    device = torch.device(device_name)
    user_ids = sorted(set(sparse_train["user_id"].astype(str)), key=_id_sort_key)
    item_ids = sorted(set(sparse_train["item_id"].astype(str)), key=_id_sort_key)
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}
    all_users = sparse_train["user_id"].astype(str).map(user_to_index).to_numpy(np.int64)
    all_items = sparse_train["item_id"].astype(str).map(item_to_index).to_numpy(np.int64)
    retained_observed = _build_boolean_matrix(
        sparse_train, user_to_index=user_to_index, item_to_index=item_to_index
    )
    full_positive = _build_boolean_matrix(
        full_train,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        rating_threshold=relevance_threshold,
    )
    oracle_blocked = retained_observed | full_positive
    explicit_lists = _explicit_negative_lists(
        sparse_train,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        threshold=relevance_threshold,
    )

    positive_mask = sparse_train["rating"].to_numpy(float) >= relevance_threshold
    positive_users = all_users[positive_mask]
    positive_items = all_items[positive_mask]
    sampling_blocked = oracle_blocked if negative_mode == ORACLE_MODE else retained_observed
    if negative_mode == EXPLICIT_MODE:
        eligible_users = np.asarray(
            [
                bool(len(explicit_lists.get(index, []))) or not retained_observed[index].all()
                for index in range(len(user_ids))
            ],
            dtype=np.bool_,
        )
    else:
        eligible_users = ~sampling_blocked.all(axis=1)
    eligible_rows = eligible_users[positive_users]
    positive_users = positive_users[eligible_rows]
    positive_items = positive_items[eligible_rows]
    if len(positive_users) == 0:
        raise ValueError("No eligible BPR positive interactions")

    model = BPRMatrixFactorization(
        num_users=len(user_ids), num_items=len(item_ids), dimension=dimension
    ).to(device)
    trained_user_mask = np.zeros(len(user_ids), dtype=np.bool_)
    trained_user_mask[np.unique(positive_users)] = True
    with torch.no_grad():
        untrained = torch.as_tensor(
            np.flatnonzero(~trained_user_mask), dtype=torch.long, device=device
        )
        if len(untrained):
            model.user_factors.weight[untrained] = 0.0
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )
    rng = np.random.default_rng(seed)
    checkpoints_set = set(checkpoints)
    max_epoch = max(checkpoints)
    history: list[dict[str, Any]] = []
    evaluation_results: dict[str, Any] = {}
    final_per_user: pd.DataFrame | None = None
    training_seconds = 0.0

    run_config = BPRRunConfig(
        dimension=dimension,
        regularization=regularization,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epoch,
        negatives_per_positive=negatives_per_positive,
        relevance_threshold=relevance_threshold,
        seed=seed,
        device=device_name,
        torch_num_threads=torch_num_threads,
    )

    for epoch in range(1, max_epoch + 1):
        started = time.perf_counter()
        permutation = rng.permutation(len(positive_users))
        pairwise_sum = 0.0
        objective_sum = 0.0
        sampled_count = 0
        sampled_full_positive = 0
        fallback_rows = 0
        model.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            base_users = positive_users[indices]
            base_positives = positive_items[indices]
            sampled, fallback = _sample_bpr_negatives(
                rng=rng,
                users=base_users,
                retained_observed=retained_observed,
                oracle_blocked=oracle_blocked,
                explicit_lists=explicit_lists,
                mode=negative_mode,
                count=negatives_per_positive,
            )
            fallback_rows += fallback
            users_np = np.repeat(base_users, negatives_per_positive)
            positives_np = np.repeat(base_positives, negatives_per_positive)
            negatives_np = sampled.reshape(-1)
            sampled_count += len(negatives_np)
            sampled_full_positive += int(full_positive[users_np, negatives_np].sum())
            users = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
            positive_scores = model.score_pairs(users, positives)
            negative_scores = model.score_pairs(users, negatives)
            pairwise = torch.nn.functional.softplus(negative_scores - positive_scores)
            objective = pairwise.mean()
            if regularization:
                objective = objective + regularization * model.regularization_per_triplet(
                    users, positives, negatives
                ).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            pairwise_sum += float(pairwise.detach().sum())
            objective_sum += float(objective.detach()) * len(negatives_np)
        training_seconds += time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "training_pairwise_loss": pairwise_sum / sampled_count,
                "training_objective": objective_sum / sampled_count,
                "sampled_negatives": sampled_count,
                "sampled_full_training_positive": sampled_full_positive,
                "sampled_full_training_positive_rate": sampled_full_positive
                / sampled_count,
                "explicit_negative_fallback_positive_rows": fallback_rows,
            }
        )
        if epoch in checkpoints_set:
            fitted = FittedBPR(
                model=model,
                user_ids=user_ids,
                item_ids=item_ids,
                user_to_index=user_to_index,
                item_to_index=item_to_index,
                epoch_history=history.copy(),
                training_seconds=training_seconds,
                positive_rows=int(len(positive_users)),
                users_without_positive_training=int((~trained_user_mask).sum()),
                config=run_config,
            )
            summary, per_user = evaluate_bpr_stage(
                fitted,
                full_train=full_train,
                evaluation=evaluation,
                prior_evaluation=prior_evaluation,
                ranking_batch_size=ranking_batch_size,
            )
            evaluation_results[str(epoch)] = summary
            if epoch == max_epoch:
                final_per_user = per_user

    return (
        {
            "run_config": {
                **run_config.__dict__,
                "negative_mode": negative_mode,
            },
            "training": {
                "rows": int(len(sparse_train)),
                "positive_rows": int(len(positive_users)),
                "users": len(user_ids),
                "items": len(item_ids),
                "users_without_positive_training": int((~trained_user_mask).sum()),
                "epochs_completed": max_epoch,
                "training_seconds": training_seconds,
                "epoch_history": history,
            },
            "evaluation_checkpoints": evaluation_results,
        },
        final_per_user,
    )


def _summary_matches(path: Path, signature: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("signature") == signature


def _bpr_task(task: dict[str, Any]) -> str:
    output_path = Path(task["output_path"])
    signature = task["signature"]
    per_user_path = task.get("per_user_path")
    if _summary_matches(output_path, signature) and (
        per_user_path is None or Path(per_user_path).exists()
    ):
        return str(output_path)
    sparse_train = _load_frame(Path(task["sparse_train_path"]))
    full_train = _load_frame(Path(task["full_train_path"]))
    evaluation = _load_frame(Path(task["evaluation_path"]))
    prior = (
        _load_frame(Path(task["prior_evaluation_path"]))
        if task.get("prior_evaluation_path")
        else None
    )
    result, per_user = train_bpr_checkpoints(
        sparse_train=sparse_train,
        full_train=full_train,
        evaluation=evaluation,
        prior_evaluation=prior,
        **task["train_arguments"],
    )
    payload = {
        "schema_version": 1,
        "stage": task["stage"],
        "dataset": task["dataset"],
        "retention_label": task["retention_label"],
        "evaluation_stage": task["evaluation_stage"],
        "signature": signature,
        **result,
    }
    _json_write(output_path, payload)
    if per_user_path is not None:
        if per_user is None:
            raise AssertionError("Final per-user evaluation was not produced")
        path = Path(per_user_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        per_user.to_parquet(path, index=False)
    return str(output_path)


def _execute_tasks(tasks: list[dict[str, Any]], workers: int, label: str) -> None:
    pending = [
        task
        for task in tasks
        if not _summary_matches(Path(task["output_path"]), task["signature"])
        or (
            task.get("per_user_path") is not None
            and not Path(task["per_user_path"]).exists()
        )
    ]
    complete = len(tasks) - len(pending)
    print(f"{label}: {complete}/{len(tasks)} already complete", flush=True)
    if not pending:
        return
    if workers == 1:
        for task in pending:
            _bpr_task(task)
            complete += 1
            if complete % 10 == 0 or complete == len(tasks):
                print(f"{label}: {complete}/{len(tasks)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_bpr_task, task) for task in pending]
        for future in as_completed(futures):
            future.result()
            complete += 1
            if complete % 10 == 0 or complete == len(tasks):
                print(f"{label}: {complete}/{len(tasks)}", flush=True)


def _run_id(
    dimension: int,
    regularization: float,
    learning_rate: float,
    negatives: int,
    seed: int,
    mode: str,
) -> str:
    return (
        f"d{dimension}_lambda{regularization:g}_lr{learning_rate:g}_"
        f"neg{negatives}_{mode}_seed{seed}"
    )


def _aggregate(
    summary_paths: Iterable[Path], output_dir: Path, checkpoints: list[int]
) -> tuple[Path, Path, Path]:
    records: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload["run_config"]
        for epoch in checkpoints:
            evaluation = payload["evaluation_checkpoints"][str(epoch)]
            history = payload["training"]["epoch_history"]
            epoch_history = history[epoch - 1]
            records.append(
                {
                    "dataset": payload["dataset"],
                    "retention_label": payload["retention_label"],
                    "dimension": int(run["dimension"]),
                    "regularization": float(run["regularization"]),
                    "learning_rate": float(run["learning_rate"]),
                    "negatives_per_positive": int(run["negatives_per_positive"]),
                    "negative_mode": run["negative_mode"],
                    "seed": int(run["seed"]),
                    "epoch": epoch,
                    "training_seconds": payload["training"]["training_seconds"],
                    "training_objective": epoch_history["training_objective"],
                    "sampled_full_training_positive_rate": epoch_history[
                        "sampled_full_training_positive_rate"
                    ],
                    "evaluation_ndcg@10": evaluation["ndcg@10"],
                    "evaluation_recall@10": evaluation["recall@10"],
                    "evaluation_ndcg@20": evaluation["ndcg@20"],
                }
            )
    frame = pd.DataFrame.from_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "runs.csv"
    frame.to_csv(runs_path, index=False)
    grouped = (
        frame.groupby(
            [
                "dataset",
                "retention_label",
                "dimension",
                "regularization",
                "learning_rate",
                "negatives_per_positive",
                "negative_mode",
                "epoch",
            ],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            mean_evaluation_ndcg_at_10=("evaluation_ndcg@10", "mean"),
            std_evaluation_ndcg_at_10=("evaluation_ndcg@10", "std"),
            mean_evaluation_recall_at_10=("evaluation_recall@10", "mean"),
            mean_false_negative_rate=("sampled_full_training_positive_rate", "mean"),
        )
    )
    ranked = grouped.sort_values(
        [
            "dataset",
            "retention_label",
            "mean_evaluation_ndcg_at_10",
            "epoch",
            "dimension",
            "regularization",
            "negatives_per_positive",
            "learning_rate",
        ],
        ascending=[True, True, False, True, True, False, True, True],
        kind="stable",
    )
    ranked_path = output_dir / "ranked_configurations.csv"
    ranked.to_csv(ranked_path, index=False)
    selection = ranked.groupby(["dataset", "retention_label"], sort=False).head(1)
    selection_path = output_dir / "selections.csv"
    selection.to_csv(selection_path, index=False)
    return runs_path, ranked_path, selection_path


def _base_paths(
    config_path: Path, protocol_path: Path, dataset: str, retention: str
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, Path, Path]:
    dataset_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    processed = _resolve(root, dataset_config["datasets"][dataset]["output_dir"])
    sparse = (
        _resolve(root, dataset_config["controlled_sparsity"]["output_root"])
        / dataset
        / f"train_{retention}.parquet"
    )
    return (
        dataset_config,
        protocol,
        root,
        processed,
        sparse,
        processed / "validation.parquet",
        processed / "test.parquet",
    )


def execute_coarse(
    config_path: Path,
    protocol_path: Path,
    dataset: str,
    retention: str,
    workers: int,
) -> tuple[Path, Path, Path]:
    _, protocol, root, processed, sparse, validation, _ = _base_paths(
        config_path, protocol_path, dataset, retention
    )
    common = protocol["common"]
    settings = protocol["bpr_independent"]["coarse"]
    mode = protocol["bpr_independent"]["negative_pool_primary"]
    output_dir = _resolve(root, protocol["output_root"]) / "bpr_coarse" / dataset / retention
    checkpoints = [int(value) for value in settings["checkpoint_epochs"]]
    protocol_hash = sha256_file(protocol_path)
    sparse_hash = sha256_file(sparse)
    tasks: list[dict[str, Any]] = []
    paths: list[Path] = []
    for dimension in settings["dimensions"]:
        for regularization in settings["regularizations"]:
            for learning_rate in settings["learning_rates"]:
                for negatives in settings["negatives_per_positive"]:
                    for seed in settings["selection_seeds"]:
                        run_id = _run_id(
                            int(dimension),
                            float(regularization),
                            float(learning_rate),
                            int(negatives),
                            int(seed),
                            mode,
                        )
                        output_path = output_dir / "runs" / run_id / "summary.json"
                        signature = {
                            "protocol_sha256": protocol_hash,
                            "sparse_train_sha256": sparse_hash,
                            "dimension": int(dimension),
                            "regularization": float(regularization),
                            "learning_rate": float(learning_rate),
                            "negatives_per_positive": int(negatives),
                            "negative_mode": mode,
                            "seed": int(seed),
                            "checkpoints": checkpoints,
                        }
                        tasks.append(
                            {
                                "output_path": str(output_path),
                                "signature": signature,
                                "stage": "post_confirmation_bpr_independent_coarse",
                                "evaluation_stage": "validation",
                                "dataset": dataset,
                                "retention_label": retention,
                                "sparse_train_path": str(sparse),
                                "full_train_path": str(processed / "train.parquet"),
                                "evaluation_path": str(validation),
                                "train_arguments": {
                                    "dimension": int(dimension),
                                    "regularization": float(regularization),
                                    "learning_rate": float(learning_rate),
                                    "batch_size": int(common["batch_size"]),
                                    "negatives_per_positive": int(negatives),
                                    "negative_mode": mode,
                                    "checkpoints": checkpoints,
                                    "relevance_threshold": float(
                                        common["relevance_threshold"]
                                    ),
                                    "seed": int(seed),
                                    "ranking_batch_size": int(
                                        common["ranking_batch_size"]
                                    ),
                                    "device_name": str(common["device"]),
                                    "torch_num_threads": int(
                                        common["torch_num_threads"]
                                    ),
                                },
                            }
                        )
                        paths.append(output_path)
    _execute_tasks(tasks, workers, f"bpr-coarse/{dataset}/{retention}")
    return _aggregate(paths, output_dir, checkpoints)


def _top_coarse_hyperparameters(
    coarse_ranked_path: Path, count: int
) -> list[dict[str, Any]]:
    ranked = pd.read_csv(coarse_ranked_path)
    unique = ranked.drop_duplicates(
        ["dimension", "regularization", "learning_rate"], keep="first"
    ).head(count)
    return unique[["dimension", "regularization", "learning_rate"]].to_dict("records")


def execute_refine(
    config_path: Path,
    protocol_path: Path,
    dataset: str,
    retention: str,
    workers: int,
) -> tuple[Path, Path, Path]:
    _, protocol, root, processed, sparse, validation, _ = _base_paths(
        config_path, protocol_path, dataset, retention
    )
    common = protocol["common"]
    bpr = protocol["bpr_independent"]
    settings = bpr["refine"]
    coarse_ranked = (
        _resolve(root, protocol["output_root"])
        / "bpr_coarse"
        / dataset
        / retention
        / "ranked_configurations.csv"
    )
    if not coarse_ranked.exists():
        raise FileNotFoundError(f"Run BPR coarse search first: {coarse_ranked}")
    hyperparameters = _top_coarse_hyperparameters(
        coarse_ranked, int(settings["top_coarse_configurations"])
    )
    mode = bpr["negative_pool_primary"]
    checkpoints = [int(value) for value in settings["checkpoint_epochs"]]
    output_dir = _resolve(root, protocol["output_root"]) / "bpr_refine" / dataset / retention
    protocol_hash = sha256_file(protocol_path)
    sparse_hash = sha256_file(sparse)
    tasks: list[dict[str, Any]] = []
    paths: list[Path] = []
    for values in hyperparameters:
        for negatives in settings["negatives_per_positive"]:
            for seed in settings["selection_seeds"]:
                dimension = int(values["dimension"])
                regularization = float(values["regularization"])
                learning_rate = float(values["learning_rate"])
                run_id = _run_id(
                    dimension,
                    regularization,
                    learning_rate,
                    int(negatives),
                    int(seed),
                    mode,
                )
                output_path = output_dir / "runs" / run_id / "summary.json"
                signature = {
                    "protocol_sha256": protocol_hash,
                    "sparse_train_sha256": sparse_hash,
                    "dimension": dimension,
                    "regularization": regularization,
                    "learning_rate": learning_rate,
                    "negatives_per_positive": int(negatives),
                    "negative_mode": mode,
                    "seed": int(seed),
                    "checkpoints": checkpoints,
                }
                tasks.append(
                    {
                        "output_path": str(output_path),
                        "signature": signature,
                        "stage": "post_confirmation_bpr_independent_refine",
                        "evaluation_stage": "validation",
                        "dataset": dataset,
                        "retention_label": retention,
                        "sparse_train_path": str(sparse),
                        "full_train_path": str(processed / "train.parquet"),
                        "evaluation_path": str(validation),
                        "train_arguments": {
                            "dimension": dimension,
                            "regularization": regularization,
                            "learning_rate": learning_rate,
                            "batch_size": int(common["batch_size"]),
                            "negatives_per_positive": int(negatives),
                            "negative_mode": mode,
                            "checkpoints": checkpoints,
                            "relevance_threshold": float(common["relevance_threshold"]),
                            "seed": int(seed),
                            "ranking_batch_size": int(common["ranking_batch_size"]),
                            "device_name": str(common["device"]),
                            "torch_num_threads": int(common["torch_num_threads"]),
                        },
                    }
                )
                paths.append(output_path)
    _execute_tasks(tasks, workers, f"bpr-refine/{dataset}/{retention}")
    return _aggregate(paths, output_dir, checkpoints)


def _selected_refine(root: Path, protocol: dict[str, Any], dataset: str, retention: str) -> pd.Series:
    path = (
        _resolve(root, protocol["output_root"])
        / "bpr_refine"
        / dataset
        / retention
        / "selections.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Run BPR refinement first: {path}")
    return pd.read_csv(path).iloc[0]


def execute_sensitivity(
    config_path: Path,
    protocol_path: Path,
    dataset: str,
    retention: str,
    workers: int,
) -> tuple[Path, Path, Path]:
    _, protocol, root, processed, sparse, validation, _ = _base_paths(
        config_path, protocol_path, dataset, retention
    )
    common = protocol["common"]
    bpr = protocol["bpr_independent"]
    selected = _selected_refine(root, protocol, dataset, retention)
    modes = list(bpr["sensitivity_modes"])
    output_dir = _resolve(root, protocol["output_root"]) / "bpr_sensitivity" / dataset / retention
    checkpoints = [int(selected.epoch)]
    protocol_hash = sha256_file(protocol_path)
    sparse_hash = sha256_file(sparse)
    tasks: list[dict[str, Any]] = []
    paths: list[Path] = []
    for mode in modes:
        for seed in bpr["refine"]["selection_seeds"]:
            run_id = _run_id(
                int(selected.dimension),
                float(selected.regularization),
                float(selected.learning_rate),
                int(selected.negatives_per_positive),
                int(seed),
                mode,
            )
            output_path = output_dir / "runs" / run_id / "summary.json"
            signature = {
                "protocol_sha256": protocol_hash,
                "sparse_train_sha256": sparse_hash,
                "selected_from": "bpr_refine",
                "dimension": int(selected.dimension),
                "regularization": float(selected.regularization),
                "learning_rate": float(selected.learning_rate),
                "negatives_per_positive": int(selected.negatives_per_positive),
                "negative_mode": mode,
                "seed": int(seed),
                "checkpoints": checkpoints,
            }
            tasks.append(
                {
                    "output_path": str(output_path),
                    "signature": signature,
                    "stage": "post_confirmation_bpr_negative_sensitivity",
                    "evaluation_stage": "validation",
                    "dataset": dataset,
                    "retention_label": retention,
                    "sparse_train_path": str(sparse),
                    "full_train_path": str(processed / "train.parquet"),
                    "evaluation_path": str(validation),
                    "train_arguments": {
                        "dimension": int(selected.dimension),
                        "regularization": float(selected.regularization),
                        "learning_rate": float(selected.learning_rate),
                        "batch_size": int(common["batch_size"]),
                        "negatives_per_positive": int(selected.negatives_per_positive),
                        "negative_mode": mode,
                        "checkpoints": checkpoints,
                        "relevance_threshold": float(common["relevance_threshold"]),
                        "seed": int(seed),
                        "ranking_batch_size": int(common["ranking_batch_size"]),
                        "device_name": str(common["device"]),
                        "torch_num_threads": int(common["torch_num_threads"]),
                    },
                }
            )
            paths.append(output_path)
    _execute_tasks(tasks, workers, f"bpr-sensitivity/{dataset}/{retention}")
    return _aggregate(paths, output_dir, checkpoints)


def execute_confirmation(
    config_path: Path,
    protocol_path: Path,
    dataset: str,
    retention: str,
    workers: int,
) -> tuple[Path, Path, Path]:
    _, protocol, root, processed, sparse, _, test = _base_paths(
        config_path, protocol_path, dataset, retention
    )
    common = protocol["common"]
    bpr = protocol["bpr_independent"]
    selected = _selected_refine(root, protocol, dataset, retention)
    modes = [bpr["negative_pool_primary"], *bpr["sensitivity_modes"]]
    output_dir = _resolve(root, protocol["output_root"]) / "bpr_confirmation" / dataset / retention
    checkpoints = [int(selected.epoch)]
    protocol_hash = sha256_file(protocol_path)
    sparse_hash = sha256_file(sparse)
    tasks: list[dict[str, Any]] = []
    paths: list[Path] = []
    for mode in modes:
        for seed in common["robustness_confirmation_seeds"]:
            run_id = _run_id(
                int(selected.dimension),
                float(selected.regularization),
                float(selected.learning_rate),
                int(selected.negatives_per_positive),
                int(seed),
                mode,
            )
            run_dir = output_dir / "runs" / run_id
            output_path = run_dir / "summary.json"
            per_user_path = run_dir / "test_per_user.parquet"
            signature = {
                "protocol_sha256": protocol_hash,
                "sparse_train_sha256": sparse_hash,
                "selected_from": "bpr_refine",
                "dimension": int(selected.dimension),
                "regularization": float(selected.regularization),
                "learning_rate": float(selected.learning_rate),
                "negatives_per_positive": int(selected.negatives_per_positive),
                "negative_mode": mode,
                "seed": int(seed),
                "checkpoints": checkpoints,
                "test_is_post_confirmation": True,
            }
            tasks.append(
                {
                    "output_path": str(output_path),
                    "per_user_path": str(per_user_path),
                    "signature": signature,
                    "stage": "post_confirmation_bpr_independent_test_robustness",
                    "evaluation_stage": "test",
                    "dataset": dataset,
                    "retention_label": retention,
                    "sparse_train_path": str(sparse),
                    "full_train_path": str(processed / "train.parquet"),
                    "evaluation_path": str(test),
                    "prior_evaluation_path": str(processed / "validation.parquet"),
                    "train_arguments": {
                        "dimension": int(selected.dimension),
                        "regularization": float(selected.regularization),
                        "learning_rate": float(selected.learning_rate),
                        "batch_size": int(common["batch_size"]),
                        "negatives_per_positive": int(selected.negatives_per_positive),
                        "negative_mode": mode,
                        "checkpoints": checkpoints,
                        "relevance_threshold": float(common["relevance_threshold"]),
                        "seed": int(seed),
                        "ranking_batch_size": int(common["ranking_batch_size"]),
                        "device_name": str(common["device"]),
                        "torch_num_threads": int(common["torch_num_threads"]),
                    },
                }
            )
            paths.append(output_path)
    _execute_tasks(tasks, workers, f"bpr-confirmation/{dataset}/{retention}")
    return _aggregate(paths, output_dir, checkpoints)


def _choices(value: str, allowed: tuple[str, ...]) -> list[str]:
    return list(allowed) if value == "all" else [value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["coarse", "refine", "sensitivity", "confirmation"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--retention", choices=["all", *RETENTIONS], default="all")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    functions = {
        "coarse": execute_coarse,
        "refine": execute_refine,
        "sensitivity": execute_sensitivity,
        "confirmation": execute_confirmation,
    }
    function = functions[args.stage]
    for dataset in _choices(args.dataset, DATASETS):
        for retention in _choices(args.retention, RETENTIONS):
            function(args.config, args.protocol, dataset, retention, args.workers)


if __name__ == "__main__":
    main()
