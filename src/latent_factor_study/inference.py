"""Paired user-level inference for locked MF confirmation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .confirmation import DEFAULT_CONFIRMATION_PATH
from .data import _repo_root, _resolve, sha256_file
from .selection_lock import DATASETS, RETENTIONS


def bootstrap_mean_difference(
    differences: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
    batch_size: int = 500,
) -> tuple[float, float, np.ndarray]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Bootstrap requires finite paired differences.")
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch_size):
        size = min(batch_size, resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        sampled_means[start : start + size] = values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(sampled_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper), sampled_means


def paired_sign_randomization_p_value(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
    batch_size: int = 500,
) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Randomization test requires finite paired differences.")
    observed = abs(float(values.mean()))
    if np.all(values == 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, resamples, batch_size):
        size = min(batch_size, resamples - start)
        signs = rng.integers(0, 2, size=(size, len(values)), dtype=np.int8)
        signs = signs * 2 - 1
        null_means = np.abs((signs * values).mean(axis=1))
        exceedances += int((null_means >= observed).sum())
    return float((exceedances + 1) / (resamples + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _role_frames(root: Path, manifest: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {
        "rmse_selected": [],
        "ndcg_selected": [],
    }
    for run in manifest["runs"]:
        per_user_path = root / run["per_user_path"]
        if sha256_file(per_user_path) != run["per_user_sha256"]:
            raise ValueError(f"Per-user checksum mismatch: {per_user_path}")
        frame = pd.read_parquet(per_user_path)
        frame["seed"] = int(run["seed"])
        for role in run["selection_roles"]:
            frames[role].append(frame.copy())
    output: dict[str, pd.DataFrame] = {}
    for role, parts in frames.items():
        if not parts:
            raise ValueError(f"Confirmation manifest has no runs for {role}.")
        combined = pd.concat(parts, ignore_index=True)
        seed_counts = combined.groupby("user_id")["seed"].nunique()
        if seed_counts.min() != 5 or seed_counts.max() != 5:
            raise ValueError(f"{role} does not have five fresh seeds for every user.")
        numeric = [
            "ndcg@10",
            "recall@10",
            "ndcg@20",
            "recall@20",
            "prediction_rmse",
            "prediction_mae",
            "prediction_raw_rmse",
            "prediction_raw_mae",
        ]
        output[role] = combined.groupby("user_id", as_index=False)[numeric].mean()
    return output


def _role_summary_metrics(
    root: Path, manifest: dict[str, Any]
) -> dict[str, dict[str, list[float]]]:
    metrics = {
        "rmse_selected": {"global_rmse": [], "ndcg@10": []},
        "ndcg_selected": {"global_rmse": [], "ndcg@10": []},
    }
    for run in manifest["runs"]:
        summary_path = root / run["summary_path"]
        if sha256_file(summary_path) != run["summary_sha256"]:
            raise ValueError(f"Summary checksum mismatch: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for role in run["selection_roles"]:
            metrics[role]["global_rmse"].append(
                float(summary["test"]["prediction"]["clipped"]["rmse"])
            )
            metrics[role]["ndcg@10"].append(
                float(summary["test"]["ranking"]["ndcg@10"])
            )
    return metrics


def analyze_confirmation(confirmation_path: Path) -> Path:
    config = yaml.safe_load(confirmation_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(confirmation_path)
    output_root = _resolve(root, config["output_root"])
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    inference = config["inference"]
    bootstrap_resamples = int(inference["user_cluster_bootstrap_resamples"])
    randomization_resamples = int(inference["paired_randomization_resamples"])
    confidence_level = float(inference["confidence_level"])
    base_seed = int(inference["seed"])
    lock = yaml.safe_load(
        _resolve(root, config["selection_lock"]).read_text(encoding="utf-8")
    ) or {}
    practical_threshold = float(
        lock["minimum_practically_important_ndcg_at_10_difference"]
    )

    records: list[dict[str, Any]] = []
    condition_index = 0
    for dataset in DATASETS:
        for retention in RETENTIONS:
            manifest_path = output_root / dataset / retention / "confirmation_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["observed_runs"] != manifest["expected_runs"]:
                raise ValueError(f"Incomplete confirmation manifest: {dataset}/{retention}")
            role_frames = _role_frames(root, manifest)
            summary_metrics = _role_summary_metrics(root, manifest)
            paired = role_frames["rmse_selected"].merge(
                role_frames["ndcg_selected"],
                on="user_id",
                suffixes=("_rmse_selected", "_ndcg_selected"),
                validate="one_to_one",
            )
            paired["delta_ndcg@10"] = (
                paired["ndcg@10_ndcg_selected"] - paired["ndcg@10_rmse_selected"]
            )
            paired["delta_prediction_rmse"] = (
                paired["prediction_rmse_ndcg_selected"]
                - paired["prediction_rmse_rmse_selected"]
            )
            paired_path = analysis_dir / f"{dataset}_{retention}_paired_users.parquet"
            paired.to_parquet(paired_path, index=False)

            ranking_delta = paired["delta_ndcg@10"].dropna().to_numpy(dtype=float)
            prediction_delta = paired["delta_prediction_rmse"].dropna().to_numpy(dtype=float)
            rank_seed = base_seed + condition_index * 10
            rank_lower, rank_upper, rank_bootstrap = bootstrap_mean_difference(
                ranking_delta,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=rank_seed,
            )
            pred_lower, pred_upper, pred_bootstrap = bootstrap_mean_difference(
                prediction_delta,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=rank_seed + 1,
            )
            distribution_path = analysis_dir / f"{dataset}_{retention}_resamples.npz"
            np.savez_compressed(
                distribution_path,
                ranking_mean_differences=rank_bootstrap,
                prediction_rmse_mean_differences=pred_bootstrap,
            )
            rank_mean = float(ranking_delta.mean())
            pred_mean = float(prediction_delta.mean())
            baseline_ndcg = float(paired["ndcg@10_rmse_selected"].mean())
            practical = abs(rank_mean) >= practical_threshold
            if rank_mean > 0 and practical:
                interpretation = "ranking_selection_gain_practically_meaningful"
            elif rank_mean > 0:
                interpretation = "ranking_selection_gain_below_practical_threshold"
            elif rank_mean < 0 and practical:
                interpretation = "rmse_selection_gain_practically_meaningful"
            elif rank_mean < 0:
                interpretation = "rmse_selection_gain_below_practical_threshold"
            else:
                interpretation = "identical_test_ranking"
            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "users_paired_ranking": int(len(ranking_delta)),
                    "users_paired_prediction": int(len(prediction_delta)),
                    "rmse_selected_mean_ndcg@10": baseline_ndcg,
                    "ndcg_selected_mean_ndcg@10": float(
                        paired["ndcg@10_ndcg_selected"].mean()
                    ),
                    "delta_rank_ndcg@10": rank_mean,
                    "delta_rank_relative": (
                        rank_mean / baseline_ndcg if baseline_ndcg != 0.0 else float("nan")
                    ),
                    "delta_rank_ci_lower": rank_lower,
                    "delta_rank_ci_upper": rank_upper,
                    "delta_rank_randomization_p": paired_sign_randomization_p_value(
                        ranking_delta,
                        resamples=randomization_resamples,
                        seed=rank_seed + 2,
                    ),
                    "practical_threshold": practical_threshold,
                    "practically_meaningful": practical,
                    "interpretation": interpretation,
                    "rmse_selected_global_rmse": float(
                        np.mean(summary_metrics["rmse_selected"]["global_rmse"])
                    ),
                    "ndcg_selected_global_rmse": float(
                        np.mean(summary_metrics["ndcg_selected"]["global_rmse"])
                    ),
                    "delta_prediction_global_rmse": float(
                        np.mean(summary_metrics["ndcg_selected"]["global_rmse"])
                        - np.mean(summary_metrics["rmse_selected"]["global_rmse"])
                    ),
                    "rmse_selected_mean_user_rmse": float(
                        paired["prediction_rmse_rmse_selected"].mean()
                    ),
                    "ndcg_selected_mean_user_rmse": float(
                        paired["prediction_rmse_ndcg_selected"].mean()
                    ),
                    "delta_mean_user_rmse": pred_mean,
                    "delta_mean_user_rmse_ci_lower": pred_lower,
                    "delta_mean_user_rmse_ci_upper": pred_upper,
                    "delta_mean_user_rmse_randomization_p": paired_sign_randomization_p_value(
                        prediction_delta,
                        resamples=randomization_resamples,
                        seed=rank_seed + 3,
                    ),
                    "paired_users_sha256": sha256_file(paired_path),
                    "resamples_sha256": sha256_file(distribution_path),
                }
            )
            condition_index += 1

    adjusted = holm_adjust([row["delta_rank_randomization_p"] for row in records])
    for row, value in zip(records, adjusted):
        row["delta_rank_holm_p"] = value
    frame = pd.DataFrame.from_records(records)
    csv_path = analysis_dir / "mf_selection_regret.csv"
    json_path = analysis_dir / "mf_selection_regret.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_CONFIRMATION_PATH
    )
    args = parser.parse_args()
    print(analyze_confirmation(args.confirmation_config))


if __name__ == "__main__":
    main()
