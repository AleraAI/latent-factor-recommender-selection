"""Paired inference for latent-factor models against non-latent baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .baseline_lock import DEFAULT_BASELINE_CONFIRMATION_PATH
from .bpr_grid import DATASETS, RETENTIONS
from .data import _repo_root, _resolve, sha256_file
from .inference import (
    _role_frames,
    bootstrap_mean_difference,
    holm_adjust,
    paired_sign_randomization_p_value,
)


def _bias_role_frames(root: Path, manifest: dict[str, Any]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {
        "rmse_selected": [],
        "ndcg_selected": [],
    }
    for run in manifest["bias_runs"]:
        path = root / run["per_user_path"]
        if sha256_file(path) != run["per_user_sha256"]:
            raise ValueError(f"Bias per-user checksum mismatch: {path}")
        frame = pd.read_parquet(path)
        frame["user_id"] = frame["user_id"].astype(str)
        frame["seed"] = int(run["seed"])
        for role in run["selection_roles"]:
            parts[role].append(frame.copy())
    output: dict[str, pd.DataFrame] = {}
    for role, frames in parts.items():
        if not frames:
            raise ValueError(f"Baseline manifest contains no {role} bias runs.")
        combined = pd.concat(frames, ignore_index=True)
        seed_counts = combined.groupby("user_id")["seed"].nunique()
        if int(seed_counts.min()) != 5 or int(seed_counts.max()) != 5:
            raise ValueError(f"Bias {role} lacks five fresh seeds for every user.")
        output[role] = combined.groupby("user_id", as_index=False)[
            ["ndcg@10", "recall@10", "prediction_rmse", "prediction_mae"]
        ].mean()
    return output


def _popularity_users(root: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    record = manifest["popularity"]
    path = root / record["per_user_path"]
    if sha256_file(path) != record["per_user_sha256"]:
        raise ValueError(f"Popularity per-user checksum mismatch: {path}")
    frame = pd.read_parquet(path)
    frame["user_id"] = frame["user_id"].astype(str)
    return frame


def analyze_baseline_confirmation(confirmation_path: Path) -> Path:
    config = yaml.safe_load(confirmation_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(confirmation_path)
    baseline_output = _resolve(root, config["output_root"])
    explicit_config_path = _resolve(root, config["explicit_confirmation_config"])
    explicit_config = yaml.safe_load(explicit_config_path.read_text(encoding="utf-8")) or {}
    explicit_output = _resolve(root, explicit_config["output_root"])
    analysis_dir = baseline_output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    settings = config["inference"]
    bootstrap_resamples = int(settings["user_cluster_bootstrap_resamples"])
    randomization_resamples = int(settings["paired_randomization_resamples"])
    confidence_level = float(settings["confidence_level"])
    threshold = float(settings["minimum_practically_important_ndcg_at_10_difference"])
    base_seed = int(settings["seed"])
    records: list[dict[str, Any]] = []

    condition_index = 0
    for dataset in DATASETS:
        for retention in RETENTIONS:
            baseline_manifest_path = (
                baseline_output / dataset / retention / "confirmation_manifest.json"
            )
            explicit_manifest_path = (
                explicit_output / dataset / retention / "confirmation_manifest.json"
            )
            baseline_manifest = json.loads(
                baseline_manifest_path.read_text(encoding="utf-8")
            )
            explicit_manifest = json.loads(
                explicit_manifest_path.read_text(encoding="utf-8")
            )
            if int(baseline_manifest["observed_bias_runs"]) != int(
                baseline_manifest["expected_bias_runs"]
            ) or int(explicit_manifest["observed_runs"]) != int(
                explicit_manifest["expected_runs"]
            ):
                raise ValueError(f"Incomplete baseline comparison: {dataset}/{retention}")
            explicit = _role_frames(root, explicit_manifest)
            bias = _bias_role_frames(root, baseline_manifest)
            popularity = _popularity_users(root, baseline_manifest)

            prediction = explicit["rmse_selected"][
                ["user_id", "prediction_rmse"]
            ].merge(
                bias["rmse_selected"][["user_id", "prediction_rmse"]],
                on="user_id",
                suffixes=("_explicit", "_bias"),
                validate="one_to_one",
            )
            prediction["delta_bias_minus_explicit_rmse"] = (
                prediction["prediction_rmse_bias"]
                - prediction["prediction_rmse_explicit"]
            )
            ranking_bias = explicit["ndcg_selected"][["user_id", "ndcg@10"]].merge(
                bias["ndcg_selected"][["user_id", "ndcg@10"]],
                on="user_id",
                suffixes=("_explicit", "_bias"),
                validate="one_to_one",
            )
            ranking_bias["delta_explicit_minus_bias_ndcg@10"] = (
                ranking_bias["ndcg@10_explicit"] - ranking_bias["ndcg@10_bias"]
            )
            ranking_popularity = explicit["ndcg_selected"][
                ["user_id", "ndcg@10"]
            ].merge(
                popularity[["user_id", "ndcg@10"]],
                on="user_id",
                suffixes=("_explicit", "_popularity"),
                validate="one_to_one",
            )
            ranking_popularity["delta_explicit_minus_popularity_ndcg@10"] = (
                ranking_popularity["ndcg@10_explicit"]
                - ranking_popularity["ndcg@10_popularity"]
            )
            prediction_path = analysis_dir / f"{dataset}_{retention}_prediction_users.parquet"
            bias_path = analysis_dir / f"{dataset}_{retention}_bias_ranking_users.parquet"
            popularity_path = (
                analysis_dir / f"{dataset}_{retention}_popularity_ranking_users.parquet"
            )
            prediction.to_parquet(prediction_path, index=False)
            ranking_bias.to_parquet(bias_path, index=False)
            ranking_popularity.to_parquet(popularity_path, index=False)

            pred_delta = prediction["delta_bias_minus_explicit_rmse"].dropna().to_numpy()
            bias_delta = ranking_bias[
                "delta_explicit_minus_bias_ndcg@10"
            ].dropna().to_numpy()
            popularity_delta = ranking_popularity[
                "delta_explicit_minus_popularity_ndcg@10"
            ].dropna().to_numpy()
            condition_seed = base_seed + condition_index * 20
            pred_lower, pred_upper, pred_bootstrap = bootstrap_mean_difference(
                pred_delta,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=condition_seed,
            )
            bias_lower, bias_upper, bias_bootstrap = bootstrap_mean_difference(
                bias_delta,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=condition_seed + 1,
            )
            pop_lower, pop_upper, pop_bootstrap = bootstrap_mean_difference(
                popularity_delta,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=condition_seed + 2,
            )
            resamples_path = analysis_dir / f"{dataset}_{retention}_resamples.npz"
            np.savez_compressed(
                resamples_path,
                prediction_rmse_mean_differences=pred_bootstrap,
                explicit_minus_bias_ndcg_mean_differences=bias_bootstrap,
                explicit_minus_popularity_ndcg_mean_differences=pop_bootstrap,
            )
            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "users_paired_prediction": int(len(pred_delta)),
                    "users_paired_ranking": int(len(bias_delta)),
                    "explicit_rmse_selected_mean_user_rmse": float(
                        prediction["prediction_rmse_explicit"].mean()
                    ),
                    "bias_rmse_selected_mean_user_rmse": float(
                        prediction["prediction_rmse_bias"].mean()
                    ),
                    "delta_bias_minus_explicit_mean_user_rmse": float(pred_delta.mean()),
                    "delta_prediction_ci_lower": pred_lower,
                    "delta_prediction_ci_upper": pred_upper,
                    "delta_prediction_p": paired_sign_randomization_p_value(
                        pred_delta,
                        resamples=randomization_resamples,
                        seed=condition_seed + 3,
                    ),
                    "explicit_ndcg_selected_mean_ndcg@10": float(
                        ranking_bias["ndcg@10_explicit"].mean()
                    ),
                    "bias_ndcg_selected_mean_ndcg@10": float(
                        ranking_bias["ndcg@10_bias"].mean()
                    ),
                    "delta_explicit_minus_bias_ndcg@10": float(bias_delta.mean()),
                    "delta_bias_ranking_ci_lower": bias_lower,
                    "delta_bias_ranking_ci_upper": bias_upper,
                    "delta_bias_ranking_p": paired_sign_randomization_p_value(
                        bias_delta,
                        resamples=randomization_resamples,
                        seed=condition_seed + 4,
                    ),
                    "popularity_mean_ndcg@10": float(
                        ranking_popularity["ndcg@10_popularity"].mean()
                    ),
                    "delta_explicit_minus_popularity_ndcg@10": float(
                        popularity_delta.mean()
                    ),
                    "delta_popularity_ci_lower": pop_lower,
                    "delta_popularity_ci_upper": pop_upper,
                    "delta_popularity_p": paired_sign_randomization_p_value(
                        popularity_delta,
                        resamples=randomization_resamples,
                        seed=condition_seed + 5,
                    ),
                    "practical_threshold": threshold,
                    "explicit_vs_bias_practical": abs(float(bias_delta.mean()))
                    >= threshold,
                    "explicit_vs_popularity_practical": abs(
                        float(popularity_delta.mean())
                    )
                    >= threshold,
                    "resamples_sha256": sha256_file(resamples_path),
                }
            )
            condition_index += 1

    for source, target in (
        ("delta_prediction_p", "delta_prediction_holm_p"),
        ("delta_bias_ranking_p", "delta_bias_ranking_holm_p"),
        ("delta_popularity_p", "delta_popularity_holm_p"),
    ):
        adjusted = holm_adjust([row[source] for row in records])
        for row, value in zip(records, adjusted):
            row[target] = value
    frame = pd.DataFrame.from_records(records)
    csv_path = analysis_dir / "baseline_comparison.csv"
    frame.to_csv(csv_path, index=False)
    (analysis_dir / "baseline_comparison.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_BASELINE_CONFIRMATION_PATH
    )
    args = parser.parse_args()
    print(analyze_baseline_confirmation(args.confirmation_config))


if __name__ == "__main__":
    main()
