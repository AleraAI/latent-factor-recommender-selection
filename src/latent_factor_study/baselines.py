"""Bias-only and deterministic popularity baselines for the journal study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve
from .evaluation import ranking_metrics, rating_gain, top_k_items
from .explicit_mf import (
    MFRunConfig,
    _id_sort_key,
    evaluate_validation,
    fit_explicit_mf,
)


DEFAULT_BASELINE_MODELING_PATH = Path(
    "configs/latent_factor_beyond_rmse/baseline_modeling.yaml"
)


def run_bias_validation(
    *,
    config_path: Path,
    modeling_path: Path,
    dataset_name: str,
    retention_label: str,
    regularization: float,
    seed: int,
    smoke: bool,
) -> Path:
    dataset_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    dataset = dataset_config["datasets"][dataset_name]
    processed_dir = _resolve(root, dataset["output_dir"])
    sparsity_dir = _resolve(root, dataset_config["controlled_sparsity"]["output_root"]) / dataset_name
    sparse_train = pd.read_parquet(sparsity_dir / f"train_{retention_label}.parquet")
    full_train = pd.read_parquet(processed_dir / "train.parquet")
    validation = pd.read_parquet(processed_dir / "validation.parquet")
    # Deliberately do not resolve or read test.parquet in a development run.

    settings = modeling["bias_only"]
    run_config = MFRunConfig(
        dimension=0,
        regularization=regularization,
        learning_rate=float(settings["learning_rate"]),
        batch_size=int(settings["batch_size"]),
        max_epochs=int(settings["smoke_epochs"] if smoke else settings["max_epochs"]),
        convergence_patience=int(settings["convergence_patience"]),
        convergence_relative_tolerance=float(
            settings["convergence_relative_tolerance"]
        ),
        seed=seed,
        device=str(settings["device"]),
        torch_num_threads=int(settings["torch_num_threads"]),
    )
    fitted = fit_explicit_mf(sparse_train, run_config)
    validation_summary, per_user = evaluate_validation(
        fitted,
        sparse_train=sparse_train,
        full_train=full_train,
        validation=validation,
        ranking_batch_size=int(settings["ranking_batch_size"]),
    )
    run_id = f"lambda{regularization:g}_seed{seed}" + ("_smoke" if smoke else "")
    output_dir = (
        _resolve(root, modeling["development_output_root"])
        / dataset_name
        / retention_label
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user.to_parquet(output_dir / "validation_per_user.parquet", index=False)
    summary = {
        "schema_version": 1,
        "stage": "baseline_development_validation_only",
        "model": "bias_only",
        "dataset": dataset_name,
        "retention_label": retention_label,
        "run_id": run_id,
        "smoke": smoke,
        "run_config": asdict(run_config),
        "training": {
            "rows": int(len(sparse_train)),
            "users": len(fitted.user_ids),
            "items": len(fitted.item_ids),
            "epochs_completed": len(fitted.epoch_history),
            "seconds": fitted.training_seconds,
            "epoch_history": fitted.epoch_history,
        },
        "validation": validation_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary_path


def popularity_scores(
    sparse_train: pd.DataFrame, *, relevance_threshold: float = 4.0
) -> tuple[list[str], np.ndarray]:
    """Return catalog IDs and training-only relevant-rating counts."""
    item_ids = sorted(set(sparse_train["item_id"].astype(str)), key=_id_sort_key)
    relevant = sparse_train[sparse_train["rating"] >= relevance_threshold]
    counts = relevant["item_id"].astype(str).value_counts()
    scores = np.asarray([float(counts.get(item_id, 0)) for item_id in item_ids])
    return item_ids, scores


@torch.no_grad()
def evaluate_popularity_stage(
    *,
    sparse_train: pd.DataFrame,
    full_train: pd.DataFrame,
    evaluation: pd.DataFrame,
    prior_evaluation: pd.DataFrame | None = None,
    relevance_threshold: float = 4.0,
    tie_salt: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    item_ids, base_scores = popularity_scores(
        sparse_train, relevance_threshold=relevance_threshold
    )
    item_array = np.asarray(item_ids, dtype=str)
    tie_keys = None
    if tie_salt is not None:
        tie_keys = np.asarray(
            [
                hashlib.sha256(f"{tie_salt}\0{item_id}".encode("utf-8")).hexdigest()
                + "\0"
                + item_id
                for item_id in item_ids
            ],
            dtype=str,
        )
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    catalog_lookup = set(item_ids)
    history = (
        pd.concat([full_train, prior_evaluation], ignore_index=True)
        if prior_evaluation is not None
        else full_train
    )
    historical_by_user = (
        history.groupby(history["user_id"].astype(str), sort=False)["item_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    evaluation_by_user = {
        str(user): rows
        for user, rows in evaluation.groupby(evaluation["user_id"].astype(str), sort=False)
    }
    rows_out: list[dict[str, Any]] = []
    for user_id in sorted(evaluation_by_user, key=_id_sort_key):
        scores = base_scores.copy()
        historical = historical_by_user.get(user_id, set())
        historical_indices = [
            item_to_index[item] for item in historical if item in item_to_index
        ]
        scores[historical_indices] = -np.inf
        ranked = top_k_items(item_array, scores, 20, tie_keys=tie_keys)
        relevant = evaluation_by_user[user_id]
        relevant = relevant[relevant["rating"] >= relevance_threshold]
        gains = {
            str(row.item_id): rating_gain(
                float(row.rating), threshold=relevance_threshold
            )
            for row in relevant.itertuples(index=False)
            if str(row.item_id) in catalog_lookup and str(row.item_id) not in historical
        }
        cold = {
            str(item)
            for item in relevant["item_id"].astype(str)
            if str(item) not in catalog_lookup
        }
        rows_out.append(
            {
                "user_id": user_id,
                "num_candidates": len(item_ids) - len(historical_indices),
                "num_relevant": len(gains),
                "num_cold_relevant": len(cold),
                **ranking_metrics(ranked, gains),
            }
        )
    per_user = pd.DataFrame.from_records(rows_out)
    evaluable = per_user[per_user["num_relevant"] > 0]
    summary = {
        "users_total": int(len(per_user)),
        "users_evaluable": int(len(evaluable)),
        "users_without_eligible_relevant_items": int(len(per_user) - len(evaluable)),
        "cold_relevant_events": int(per_user["num_cold_relevant"].sum()),
        **{
            metric: float(evaluable[metric].mean())
            for metric in ("ndcg@10", "recall@10", "ndcg@20", "recall@20")
        },
    }
    return summary, per_user
