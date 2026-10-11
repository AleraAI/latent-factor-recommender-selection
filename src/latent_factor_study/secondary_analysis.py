"""Post-confirmation cutoff, prediction, and activity-stratified analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import _repo_root, _resolve, sha256_file
from .inference import (
    _role_frames,
    bootstrap_mean_difference,
    holm_adjust,
    paired_sign_randomization_p_value,
)
from .selection_lock import DATASETS, RETENTIONS


DEFAULT_SECONDARY_PATH = Path(
    "configs/latent_factor_beyond_rmse/secondary_analysis.yaml"
)


def assign_activity_bin(degrees: pd.Series, bins: list[dict[str, Any]]) -> pd.Series:
    values = degrees.to_numpy(dtype=int)
    labels = np.full(len(values), "unassigned", dtype=object)
    for spec in bins:
        lower = int(spec["lower_inclusive"])
        upper = spec.get("upper_inclusive")
        mask = values >= lower
        if upper is not None:
            mask &= values <= int(upper)
        labels[mask] = str(spec["label"])
    if bool((labels == "unassigned").any()):
        missing = sorted(set(values[labels == "unassigned"].tolist()))
        raise ValueError(f"Activity bins do not cover degrees: {missing[:10]}")
    return pd.Series(labels, index=degrees.index, dtype="string")


def analyze_secondary(config_path: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    confirmation_path = _resolve(root, config["confirmation_config"])
    confirmation = yaml.safe_load(confirmation_path.read_text(encoding="utf-8")) or {}
    confirmation_root = _resolve(root, confirmation["output_root"])
    dataset_path = _resolve(root, config["dataset_config"])
    datasets_cfg = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
    sparsity_root = _resolve(
        root, datasets_cfg["controlled_sparsity"]["output_root"]
    )
    output_root = _resolve(root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    inference = config["inference"]
    bootstrap_resamples = int(inference["user_cluster_bootstrap_resamples"])
    randomization_resamples = int(inference["paired_randomization_resamples"])
    confidence_level = float(inference["confidence_level"])
    minimum_users = int(inference["minimum_users_per_activity_interval"])
    base_seed = int(inference["seed"])
    metrics = [str(metric) for metric in config["metrics"]]
    bins = list(config["training_activity_bins"])
    metric_records: list[dict[str, Any]] = []
    activity_records: list[dict[str, Any]] = []

    condition_index = 0
    for dataset in DATASETS:
        for retention in RETENTIONS:
            manifest_path = (
                confirmation_root / dataset / retention / "confirmation_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest["observed_runs"]) != int(manifest["expected_runs"]):
                raise ValueError(f"Incomplete confirmation: {dataset}/{retention}")
            roles = _role_frames(root, manifest)
            paired = roles["rmse_selected"].merge(
                roles["ndcg_selected"],
                on="user_id",
                suffixes=("_rmse_selected", "_ndcg_selected"),
                validate="one_to_one",
            )
            condition_seed = base_seed + condition_index * 100
            for metric_index, metric in enumerate(metrics):
                delta_name = f"delta_{metric}"
                paired[delta_name] = (
                    paired[f"{metric}_ndcg_selected"]
                    - paired[f"{metric}_rmse_selected"]
                )
                differences = paired[delta_name].dropna().to_numpy(dtype=float)
                lower, upper, bootstrap = bootstrap_mean_difference(
                    differences,
                    resamples=bootstrap_resamples,
                    confidence_level=confidence_level,
                    seed=condition_seed + metric_index * 5,
                )
                resamples_path = output_root / f"{dataset}_{retention}_{metric.replace('@', '_at_')}.npz"
                np.savez_compressed(resamples_path, mean_differences=bootstrap)
                metric_records.append(
                    {
                        "dataset": dataset,
                        "retention_label": retention,
                        "metric": metric,
                        "direction": (
                            "ndcg_selected_minus_rmse_selected; positive_is_ranking_gain"
                            if not metric.startswith("prediction")
                            else "ndcg_selected_minus_rmse_selected; positive_is_prediction_cost"
                        ),
                        "users_paired": int(len(differences)),
                        "rmse_selected_mean": float(
                            paired[f"{metric}_rmse_selected"].mean()
                        ),
                        "ndcg_selected_mean": float(
                            paired[f"{metric}_ndcg_selected"].mean()
                        ),
                        "delta": float(differences.mean()),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "randomization_p": paired_sign_randomization_p_value(
                            differences,
                            resamples=randomization_resamples,
                            seed=condition_seed + metric_index * 5 + 1,
                        ),
                        "resamples_sha256": sha256_file(resamples_path),
                    }
                )

            sparse_train = pd.read_parquet(
                sparsity_root / dataset / f"train_{retention}.parquet",
                columns=["user_id"],
            )
            degrees = (
                sparse_train["user_id"].astype(str).value_counts().rename("training_degree")
            )
            degree_frame = degrees.rename_axis("user_id").reset_index()
            degree_frame["activity_bin"] = assign_activity_bin(
                degree_frame["training_degree"], bins
            )
            paired["delta_ndcg@10"] = (
                paired["ndcg@10_ndcg_selected"] - paired["ndcg@10_rmse_selected"]
            )
            stratified = paired.merge(
                degree_frame, on="user_id", how="left", validate="one_to_one"
            )
            if stratified["activity_bin"].isna().any():
                raise ValueError(f"Missing activity degree for {dataset}/{retention}.")
            for bin_index, spec in enumerate(bins):
                label = str(spec["label"])
                differences = stratified.loc[
                    stratified["activity_bin"] == label, "delta_ndcg@10"
                ].dropna().to_numpy(dtype=float)
                record: dict[str, Any] = {
                    "dataset": dataset,
                    "retention_label": retention,
                    "activity_bin": label,
                    "users_paired": int(len(differences)),
                    "minimum_users_for_interval": minimum_users,
                    "delta_ndcg@10": (
                        float(differences.mean()) if len(differences) else float("nan")
                    ),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                }
                if len(differences) >= minimum_users:
                    lower, upper, _ = bootstrap_mean_difference(
                        differences,
                        resamples=bootstrap_resamples,
                        confidence_level=confidence_level,
                        seed=condition_seed + 50 + bin_index,
                    )
                    record["ci_lower"] = lower
                    record["ci_upper"] = upper
                activity_records.append(record)
            paired.to_parquet(
                output_root / f"{dataset}_{retention}_paired_users.parquet", index=False
            )
            condition_index += 1

    for metric in metrics:
        indices = [
            index for index, row in enumerate(metric_records) if row["metric"] == metric
        ]
        adjusted = holm_adjust(
            [metric_records[index]["randomization_p"] for index in indices]
        )
        for index, value in zip(indices, adjusted):
            metric_records[index]["holm_p_within_metric"] = value
    metric_path = output_root / "secondary_metric_differences.csv"
    activity_path = output_root / "activity_stratified_ndcg_regret.csv"
    pd.DataFrame.from_records(metric_records).to_csv(metric_path, index=False)
    pd.DataFrame.from_records(activity_records).to_csv(activity_path, index=False)
    (output_root / "secondary_metric_differences.json").write_text(
        json.dumps(metric_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "activity_stratified_ndcg_regret.json").write_text(
        json.dumps(activity_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metric_path, activity_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SECONDARY_PATH)
    args = parser.parse_args()
    for path in analyze_secondary(args.config):
        print(path)


if __name__ == "__main__":
    main()
