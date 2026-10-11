"""Predeclared external-validation inference for the Epinions framework test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import _repo_root, _resolve, sha256_file
from .inference import bootstrap_mean_difference
from .prospective_confirmation import DEFAULT_CONFIRMATION_CONFIG, ROLE_ORDER
from .prospective_lock import verify_prospective_selection_lock
from .uncertainty_selector import RETENTIONS


NUMERIC_METRICS = (
    "ndcg@10",
    "recall@10",
    "ndcg@20",
    "recall@20",
    "prediction_rmse",
    "prediction_mae",
    "prediction_raw_rmse",
    "prediction_raw_mae",
)


def _role_frames(root: Path, manifest: dict[str, Any]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {role: [] for role in ROLE_ORDER}
    for run in manifest["runs"]:
        path = root / run["per_user_path"]
        if sha256_file(path) != run["per_user_sha256"]:
            raise ValueError(f"Prospective per-user checksum mismatch: {path}")
        frame = pd.read_parquet(path)
        frame["seed"] = int(run["seed"])
        for role in run["selection_roles"]:
            parts[role].append(frame.copy())
    output: dict[str, pd.DataFrame] = {}
    for role, role_parts in parts.items():
        if not role_parts:
            raise ValueError(f"No prospective confirmation runs for {role}.")
        combined = pd.concat(role_parts, ignore_index=True)
        counts = combined.groupby("user_id")["seed"].nunique()
        if counts.min() != 5 or counts.max() != 5:
            raise ValueError(f"{role} does not contain five seeds per user.")
        output[role] = combined.groupby("user_id", as_index=False)[
            list(NUMERIC_METRICS)
        ].mean()
    return output


def _role_run_metrics(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    output = {
        role: {
            "global_rmse": [],
            "global_mae": [],
            "raw_global_rmse": [],
            "raw_global_mae": [],
            "ndcg@10": [],
            "recall@10": [],
            "ndcg@20": [],
            "recall@20": [],
            "training_seconds": [],
            "epochs": [],
            "item_cold_rows": [],
            "cold_relevant_events": [],
            "users_evaluable": [],
            "users_without_eligible_relevant_items": [],
        }
        for role in ROLE_ORDER
    }
    for run in manifest["runs"]:
        path = root / run["summary_path"]
        if sha256_file(path) != run["summary_sha256"]:
            raise ValueError(f"Prospective summary checksum mismatch: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        for role in run["selection_roles"]:
            output[role]["global_rmse"].append(
                float(summary["test"]["prediction"]["clipped"]["rmse"])
            )
            output[role]["global_mae"].append(
                float(summary["test"]["prediction"]["clipped"]["mae"])
            )
            output[role]["raw_global_rmse"].append(
                float(summary["test"]["prediction"]["raw"]["rmse"])
            )
            output[role]["raw_global_mae"].append(
                float(summary["test"]["prediction"]["raw"]["mae"])
            )
            output[role]["ndcg@10"].append(
                float(summary["test"]["ranking"]["ndcg@10"])
            )
            for metric in ("recall@10", "ndcg@20", "recall@20"):
                output[role][metric].append(
                    float(summary["test"]["ranking"][metric])
                )
            output[role]["training_seconds"].append(float(summary["training"]["seconds"]))
            output[role]["epochs"].append(float(summary["training"]["epochs_completed"]))
            output[role]["item_cold_rows"].append(
                float(summary["test"]["prediction"]["item_cold_rows"])
            )
            for metric in (
                "cold_relevant_events",
                "users_evaluable",
                "users_without_eligible_relevant_items",
            ):
                output[role][metric].append(
                    float(summary["test"]["ranking"][metric])
                )
    return output


def _comparison(
    differences: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    lower, upper, distribution = bootstrap_mean_difference(
        values,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "users": int(len(values)),
        "mean": float(values.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_means": distribution,
    }


def analyze_prospective_confirmation(
    confirmation_config_path: Path = DEFAULT_CONFIRMATION_CONFIG,
) -> Path:
    config = yaml.safe_load(confirmation_config_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(confirmation_config_path)
    dataset_config_path = _resolve(root, config["dataset_config"])
    modeling_path = _resolve(root, config["modeling_config"])
    protocol_lock_path = _resolve(root, config["protocol_lock"])
    analysis_plan_path = _resolve(root, config["analysis_plan"])
    selection_lock_path = _resolve(root, config["selection_lock"])
    selection_lock = verify_prospective_selection_lock(
        selection_lock_path=selection_lock_path,
        dataset_config_path=dataset_config_path,
        modeling_path=modeling_path,
        protocol_lock_path=protocol_lock_path,
        analysis_plan_path=analysis_plan_path,
    )
    plan = yaml.safe_load(analysis_plan_path.read_text(encoding="utf-8")) or {}
    primary = plan["primary_external_validation"]
    resamples = int(primary["bootstrap_resamples"])
    confidence_level = 0.95
    base_seed = int(primary["bootstrap_seed"])
    margin = float(primary["noninferiority_margin"])
    output_root = _resolve(root, config["output_root"])
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    decisions: list[bool] = []
    for index, retention in enumerate(RETENTIONS):
        manifest_path = output_root / "epinions" / retention / "confirmation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["observed_runs"] != manifest["expected_runs"]:
            raise ValueError(f"Incomplete prospective confirmation: {retention}")
        if manifest["selection_lock_sha256"] != sha256_file(selection_lock_path):
            raise ValueError(f"Prospective manifest lock mismatch: {retention}")
        role_frames = _role_frames(root, manifest)
        run_metrics = _role_run_metrics(root, manifest)

        paired = role_frames["framework_selected"].merge(
            role_frames["ndcg_selected"],
            on="user_id",
            suffixes=("_framework", "_ndcg"),
            validate="one_to_one",
        ).merge(
            role_frames["rmse_selected"].add_suffix("_rmse").rename(
                columns={"user_id_rmse": "user_id"}
            ),
            on="user_id",
            validate="one_to_one",
        )
        paired["ndcg_regret_framework_vs_ndcg"] = (
            paired["ndcg@10_ndcg"] - paired["ndcg@10_framework"]
        )
        paired["rmse_framework_minus_ndcg"] = (
            paired["prediction_rmse_framework"] - paired["prediction_rmse_ndcg"]
        )
        paired["ndcg_framework_minus_rmse"] = (
            paired["ndcg@10_framework"] - paired["ndcg@10_rmse"]
        )
        paired["rmse_framework_minus_rmse"] = (
            paired["prediction_rmse_framework"] - paired["prediction_rmse_rmse"]
        )
        generic_secondary_columns: list[str] = []
        for metric in NUMERIC_METRICS:
            for comparator in ("ndcg", "rmse"):
                column = f"{metric}_framework_minus_{comparator}"
                paired[column] = (
                    paired[f"{metric}_framework"] - paired[f"{metric}_{comparator}"]
                )
                generic_secondary_columns.append(column)
        paired.to_parquet(analysis_dir / f"epinions_{retention}_paired_users.parquet", index=False)

        seed = base_seed + index * 10
        primary_result = _comparison(
            paired["ndcg_regret_framework_vs_ndcg"].to_numpy(),
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        primary_distribution = primary_result.pop("bootstrap_means")
        primary_upper = float(
            np.quantile(primary_distribution, confidence_level)
        )
        passes = bool(primary_upper <= margin)
        decisions.append(passes)

        secondary_results: dict[str, dict[str, Any]] = {}
        secondary_columns = (
            "rmse_framework_minus_ndcg",
            "ndcg_framework_minus_rmse",
            "rmse_framework_minus_rmse",
            *generic_secondary_columns,
        )
        distributions: dict[str, np.ndarray] = {}
        for offset, column in enumerate(secondary_columns, start=1):
            result = _comparison(
                paired[column].to_numpy(),
                resamples=resamples,
                confidence_level=confidence_level,
                seed=seed + offset,
            )
            distributions[column] = result.pop("bootstrap_means")
            secondary_results[column] = result
        np.savez_compressed(
            analysis_dir / f"epinions_{retention}_bootstrap_resamples.npz",
            ndcg_regret_framework_vs_ndcg=primary_distribution,
            **distributions,
        )

        condition = selection_lock["selections"][retention]
        rows.append(
            {
                "dataset": "epinions",
                "retention_label": retention,
                "admissible_configurations": condition["admissible_configurations"],
                "rmse_selected_dimension": condition["rmse_selected"]["dimension"],
                "rmse_selected_regularization": condition["rmse_selected"]["regularization"],
                "ndcg_selected_dimension": condition["ndcg_selected"]["dimension"],
                "ndcg_selected_regularization": condition["ndcg_selected"]["regularization"],
                "framework_selected_dimension": condition["framework_selected"]["dimension"],
                "framework_selected_regularization": condition["framework_selected"]["regularization"],
                "framework_matches_rmse": condition["framework_matches_rmse"],
                "framework_matches_ndcg": condition["framework_matches_ndcg"],
                "users_primary": primary_result["users"],
                "ndcg_regret_framework_vs_ndcg": primary_result["mean"],
                "ndcg_regret_two_sided_ci_lower": primary_result["ci_lower"],
                "ndcg_regret_two_sided_ci_upper": primary_result["ci_upper"],
                "ndcg_regret_one_sided_95_upper": primary_upper,
                "noninferiority_margin": margin,
                "condition_passes_noninferiority": passes,
                "framework_minus_ndcg_user_rmse": secondary_results[
                    "rmse_framework_minus_ndcg"
                ]["mean"],
                "framework_minus_ndcg_user_rmse_ci_lower": secondary_results[
                    "rmse_framework_minus_ndcg"
                ]["ci_lower"],
                "framework_minus_ndcg_user_rmse_ci_upper": secondary_results[
                    "rmse_framework_minus_ndcg"
                ]["ci_upper"],
                "framework_minus_rmse_ndcg@10": secondary_results[
                    "ndcg_framework_minus_rmse"
                ]["mean"],
                "framework_minus_rmse_ndcg@10_ci_lower": secondary_results[
                    "ndcg_framework_minus_rmse"
                ]["ci_lower"],
                "framework_minus_rmse_ndcg@10_ci_upper": secondary_results[
                    "ndcg_framework_minus_rmse"
                ]["ci_upper"],
                "framework_minus_rmse_user_rmse": secondary_results[
                    "rmse_framework_minus_rmse"
                ]["mean"],
                "framework_minus_rmse_user_rmse_ci_lower": secondary_results[
                    "rmse_framework_minus_rmse"
                ]["ci_lower"],
                "framework_minus_rmse_user_rmse_ci_upper": secondary_results[
                    "rmse_framework_minus_rmse"
                ]["ci_upper"],
                "rmse_selected_mean_global_rmse": float(
                    np.mean(run_metrics["rmse_selected"]["global_rmse"])
                ),
                "ndcg_selected_mean_global_rmse": float(
                    np.mean(run_metrics["ndcg_selected"]["global_rmse"])
                ),
                "framework_selected_mean_global_rmse": float(
                    np.mean(run_metrics["framework_selected"]["global_rmse"])
                ),
                "framework_median_training_seconds": float(
                    np.median(run_metrics["framework_selected"]["training_seconds"])
                ),
                "framework_median_epochs": float(
                    np.median(run_metrics["framework_selected"]["epochs"])
                ),
                "item_cold_test_rows": int(
                    run_metrics["framework_selected"]["item_cold_rows"][0]
                ),
                "cold_relevant_test_events": int(
                    run_metrics["framework_selected"]["cold_relevant_events"][0]
                ),
                "ranking_users_evaluable": int(
                    run_metrics["framework_selected"]["users_evaluable"][0]
                ),
                "users_without_eligible_relevant_items": int(
                    run_metrics["framework_selected"][
                        "users_without_eligible_relevant_items"
                    ][0]
                ),
                "unique_confirmation_configurations": manifest[
                    "unique_selected_configurations"
                ],
                **{
                    f"{column}_{field}": value
                    for column, values in secondary_results.items()
                    if column in generic_secondary_columns
                    for field, value in values.items()
                },
                **{
                    f"{role}_{metric}_mean": float(np.mean(values))
                    for role, metrics in run_metrics.items()
                    for metric, values in metrics.items()
                    if metric
                    in {
                        "global_mae",
                        "raw_global_rmse",
                        "raw_global_mae",
                        "ndcg@10",
                        "recall@10",
                        "ndcg@20",
                        "recall@20",
                    }
                },
            }
        )

    frame = pd.DataFrame.from_records(rows)
    csv_path = analysis_dir / "prospective_external_validation.csv"
    frame.to_csv(csv_path, index=False)
    report = {
        "schema_version": 1,
        "stage": "prospective_external_validation",
        "selection_lock_sha256": sha256_file(selection_lock_path),
        "analysis_plan_sha256": sha256_file(analysis_plan_path),
        "inference_code_sha256": sha256_file(Path(__file__)),
        "primary_margin": margin,
        "condition_decisions": dict(zip(RETENTIONS, decisions)),
        "family_decision": "supports_external_ranking_noninferiority"
        if all(decisions)
        else "does_not_support_external_ranking_noninferiority",
        "decision_rule": primary["decision_rule"],
        "result_csv": str(csv_path.relative_to(root)),
        "result_csv_sha256": sha256_file(csv_path),
    }
    json_path = analysis_dir / "prospective_external_validation.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG)
    args = parser.parse_args()
    print(analyze_prospective_confirmation(args.config))


if __name__ == "__main__":
    main()
