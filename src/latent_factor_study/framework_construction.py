"""Retrospective construction-dataset evaluation of the frozen selector.

This module deliberately labels every output as post-confirmation. It applies
the uncertainty-aware validation rule to the three datasets used to understand
the prediction--ranking problem, reuses an original confirmation run only when
the selected configuration and seed are identical, and otherwise performs a
fresh-seed fit under the original evaluation contract. These results describe
framework construction; Epinions supplies the independent prospective test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .data import _repo_root, _resolve, sha256_file
from .explicit_mf import MFRunConfig, evaluate_stage, fit_explicit_mf
from .inference import bootstrap_mean_difference
from .selection_lock import DATASETS, RETENTIONS, verify_selection_lock
from .uncertainty_selector import select_uncertainty_aware_configuration


DEFAULT_CONFIG = Path(
    "configs/latent_factor_beyond_rmse/framework_construction.yaml"
)
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


def _same_config(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return int(left["dimension"]) == int(right["dimension"]) and float(
        left["regularization"]
    ) == float(right["regularization"])


def _load_and_verify(config_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if config.get("claim_status") != "exploratory_not_prospective":
        raise ValueError("Framework-construction outputs must remain exploratory.")
    root = _repo_root(config_path)
    dataset_path = _resolve(root, config["dataset_config"])
    modeling_path = _resolve(root, config["modeling_config"])
    confirmation_path = _resolve(root, config["confirmation_config"])
    lock_path = _resolve(root, config["historical_selection_lock"])
    verified = verify_selection_lock(
        lock_path=lock_path,
        dataset_config_path=dataset_path,
        modeling_path=modeling_path,
        confirmation_path=confirmation_path,
    )
    if verified["status"] != "verified":
        raise RuntimeError("Historical selection lock did not verify.")
    return root, config, verified


def write_construction_selections(config_path: Path = DEFAULT_CONFIG) -> list[Path]:
    root, config, _ = _load_and_verify(config_path)
    modeling_path = _resolve(root, config["modeling_config"])
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    historical_lock_path = _resolve(root, config["historical_selection_lock"])
    historical_lock = yaml.safe_load(
        historical_lock_path.read_text(encoding="utf-8")
    ) or {}
    output_root = _resolve(root, config["output_root"]) / "selections"
    settings = config["selection"]
    expected = (
        len(modeling["explicit_mf"]["dimensions"])
        * len(modeling["explicit_mf"]["regularizations"])
        * len(modeling["explicit_mf"]["selection_seeds"])
    )
    paths: list[Path] = []
    for dataset in settings["construction_datasets"]:
        for retention in settings["retentions"]:
            grid_path = (
                _resolve(root, modeling["development_output_root"])
                / dataset
                / retention
                / "grid_runs.csv"
            )
            rows = pd.read_csv(grid_path)
            if len(rows) != expected:
                raise ValueError(
                    f"Incomplete construction grid for {dataset}/{retention}: "
                    f"expected {expected}, observed {len(rows)}."
                )
            selection = select_uncertainty_aware_configuration(
                rows,
                practical_regret=float(settings["practical_ndcg_at_10_regret"]),
                confidence_level=float(settings["confidence_level"]),
            )
            historical = historical_lock["selections"][dataset][retention]
            if not _same_config(selection["rmse_selected"], historical["rmse_selected"]):
                raise ValueError(f"RMSE selection changed for {dataset}/{retention}.")
            if not _same_config(selection["ndcg_selected"], historical["ndcg_selected"]):
                raise ValueError(f"NDCG selection changed for {dataset}/{retention}.")
            artifact = {
                "schema_version": 1,
                "stage": "post_confirmation_framework_construction",
                "claim_status": "exploratory_not_prospective",
                "test_metrics_used_for_this_selection": False,
                "historical_test_results_predated_framework_construction": True,
                "dataset": dataset,
                "retention_label": retention,
                "historical_selection_lock_sha256": sha256_file(historical_lock_path),
                "grid_runs_sha256": sha256_file(grid_path),
                "expected_runs": expected,
                "observed_runs": len(rows),
                "selection": selection,
            }
            path = output_root / dataset / retention / "framework_selection.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths.append(path)
    return paths


def _reusable_record(
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    seed: int,
) -> dict[str, Any] | None:
    for record in manifest["runs"]:
        if (
            int(record["seed"]) == int(seed)
            and int(record["dimension"]) == int(selected["dimension"])
            and float(record["regularization"])
            == float(selected["regularization"])
        ):
            return dict(record)
    return None


def _run_matches(path: Path, run_config: MFRunConfig) -> bool:
    if not path.exists() or not (path.parent / "test_per_user.parquet").exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("stage") == "post_confirmation_framework_construction"
        and value.get("selection_roles") == ["framework_selected"]
        and value.get("run_config") == asdict(run_config)
    )


def execute_construction_confirmation(
    config_path: Path = DEFAULT_CONFIG,
    *,
    datasets: list[str] | None = None,
    retentions: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    root, config, _ = _load_and_verify(config_path)
    dataset_config_path = _resolve(root, config["dataset_config"])
    modeling_path = _resolve(root, config["modeling_config"])
    dataset_cfg = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    settings = modeling["explicit_mf"]
    confirmation_seeds = [int(seed) for seed in config["confirmation"]["seeds"]]
    output_root = _resolve(root, config["output_root"])
    selection_root = output_root / "selections"
    confirmation_root = output_root / "confirmation"
    historical_confirmation_root = _resolve(
        root,
        (yaml.safe_load(
            _resolve(root, config["confirmation_config"]).read_text(encoding="utf-8")
        ) or {})["output_root"],
    )
    selected_datasets = datasets or list(config["selection"]["construction_datasets"])
    selected_retentions = retentions or list(config["selection"]["retentions"])
    manifests: list[Path] = []

    for dataset in selected_datasets:
        dataset_entry = dataset_cfg["datasets"][dataset]
        processed_dir = _resolve(root, dataset_entry["output_dir"])
        sparsity_dir = (
            _resolve(root, dataset_cfg["controlled_sparsity"]["output_root"])
            / dataset
        )
        full_train = pd.read_parquet(processed_dir / "train.parquet")
        validation = pd.read_parquet(processed_dir / "validation.parquet")
        test = pd.read_parquet(processed_dir / "test.parquet")
        for retention in selected_retentions:
            selection_path = (
                selection_root / dataset / retention / "framework_selection.json"
            )
            artifact = json.loads(selection_path.read_text(encoding="utf-8"))
            if artifact.get("claim_status") != "exploratory_not_prospective":
                raise ValueError("Construction selection lost exploratory status.")
            selected = artifact["selection"]["framework_selected"]
            historical_manifest_path = (
                historical_confirmation_root
                / dataset
                / retention
                / "confirmation_manifest.json"
            )
            historical_manifest = json.loads(
                historical_manifest_path.read_text(encoding="utf-8")
            )
            sparse_train = pd.read_parquet(
                sparsity_dir / f"train_{retention}.parquet"
            )
            records: list[dict[str, Any]] = []
            for index, seed in enumerate(confirmation_seeds, start=1):
                reused = _reusable_record(historical_manifest, selected, seed)
                if reused is not None:
                    if sha256_file(root / reused["summary_path"]) != reused["summary_sha256"]:
                        raise ValueError("Reusable summary checksum mismatch.")
                    if sha256_file(root / reused["per_user_path"]) != reused["per_user_sha256"]:
                        raise ValueError("Reusable per-user checksum mismatch.")
                    record = {
                        **reused,
                        "selection_roles": ["framework_selected"],
                        "evidence_source": "reused_identical_historical_confirmation",
                    }
                else:
                    run_config = MFRunConfig(
                        dimension=int(selected["dimension"]),
                        regularization=float(selected["regularization"]),
                        learning_rate=float(settings["learning_rate"]),
                        batch_size=int(settings["batch_size"]),
                        max_epochs=int(settings["max_epochs"]),
                        convergence_patience=int(settings["convergence_patience"]),
                        convergence_relative_tolerance=float(
                            settings["convergence_relative_tolerance"]
                        ),
                        seed=seed,
                        device=str(settings["device"]),
                        torch_num_threads=int(settings["torch_num_threads"]),
                    )
                    run_id = (
                        f"d{run_config.dimension}_lambda{run_config.regularization:g}"
                        f"_seed{run_config.seed}"
                    )
                    run_dir = confirmation_root / dataset / retention / run_id
                    summary_path = run_dir / "summary.json"
                    if force or not _run_matches(summary_path, run_config):
                        fitted = fit_explicit_mf(sparse_train, run_config)
                        test_summary, per_user = evaluate_stage(
                            fitted,
                            full_train=full_train,
                            evaluation=test,
                            prior_evaluation=validation,
                            ranking_batch_size=int(settings["ranking_batch_size"]),
                        )
                        run_dir.mkdir(parents=True, exist_ok=True)
                        per_user_path = run_dir / "test_per_user.parquet"
                        per_user.to_parquet(per_user_path, index=False)
                        summary = {
                            "schema_version": 1,
                            "stage": "post_confirmation_framework_construction",
                            "claim_status": "exploratory_not_prospective",
                            "selection_roles": ["framework_selected"],
                            "dataset": dataset,
                            "retention_label": retention,
                            "run_id": run_id,
                            "run_config": asdict(run_config),
                            "training": {
                                "rows": int(len(sparse_train)),
                                "users": len(fitted.user_ids),
                                "items": len(fitted.item_ids),
                                "epochs_completed": len(fitted.epoch_history),
                                "seconds": fitted.training_seconds,
                                "epoch_history": fitted.epoch_history,
                            },
                            "test": test_summary,
                        }
                        summary_path.write_text(
                            json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    per_user_path = run_dir / "test_per_user.parquet"
                    record = {
                        "selection_roles": ["framework_selected"],
                        "dimension": run_config.dimension,
                        "regularization": run_config.regularization,
                        "seed": run_config.seed,
                        "summary_path": str(summary_path.relative_to(root)),
                        "summary_sha256": sha256_file(summary_path),
                        "per_user_path": str(per_user_path.relative_to(root)),
                        "per_user_sha256": sha256_file(per_user_path),
                        "evidence_source": "new_post_confirmation_fit",
                    }
                records.append(record)
                print(
                    f"{dataset}/{retention}: {index}/{len(confirmation_seeds)} "
                    "framework confirmation runs",
                    flush=True,
                )
            manifest = {
                "schema_version": 1,
                "stage": "post_confirmation_framework_construction",
                "claim_status": "exploratory_not_prospective",
                "dataset": dataset,
                "retention_label": retention,
                "framework_selection_sha256": sha256_file(selection_path),
                "historical_confirmation_manifest_sha256": sha256_file(
                    historical_manifest_path
                ),
                "expected_runs": len(confirmation_seeds),
                "observed_runs": len(records),
                "runs": records,
            }
            manifest_path = (
                confirmation_root / dataset / retention / "confirmation_manifest.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifests.append(manifest_path)
    return manifests


def _averaged_role_frame(
    root: Path,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for record in records:
        path = root / record["per_user_path"]
        if sha256_file(path) != record["per_user_sha256"]:
            raise ValueError(f"Per-user checksum mismatch: {path}")
        frame = pd.read_parquet(path)
        frame["seed"] = int(record["seed"])
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    counts = combined.groupby("user_id")["seed"].nunique()
    if counts.min() != 5 or counts.max() != 5:
        raise ValueError("Every construction comparison requires five seeds per user.")
    return combined.groupby("user_id", as_index=False)[list(NUMERIC_METRICS)].mean()


def _original_role_records(
    manifest: Mapping[str, Any], role: str
) -> list[dict[str, Any]]:
    records = [record for record in manifest["runs"] if role in record["selection_roles"]]
    if len(records) != 5:
        raise ValueError(f"Original manifest does not contain five {role} runs.")
    return records


def _interval(values: pd.Series, *, resamples: int, confidence: float, seed: int) -> dict[str, float]:
    lower, upper, _ = bootstrap_mean_difference(
        values.to_numpy(),
        resamples=resamples,
        confidence_level=confidence,
        seed=seed,
    )
    return {"mean": float(values.mean()), "ci_lower": lower, "ci_upper": upper}


def analyze_construction(config_path: Path = DEFAULT_CONFIG) -> Path:
    root, config, _ = _load_and_verify(config_path)
    output_root = _resolve(root, config["output_root"])
    historical_confirmation_root = _resolve(
        root,
        (yaml.safe_load(
            _resolve(root, config["confirmation_config"]).read_text(encoding="utf-8")
        ) or {})["output_root"],
    )
    inference = config["descriptive_inference"]
    resamples = int(inference["user_cluster_bootstrap_resamples"])
    confidence = float(inference["confidence_level"])
    base_seed = int(inference["seed"])
    rows: list[dict[str, Any]] = []
    condition_index = 0
    for dataset in config["selection"]["construction_datasets"]:
        for retention in config["selection"]["retentions"]:
            selection_path = (
                output_root / "selections" / dataset / retention / "framework_selection.json"
            )
            selection = json.loads(selection_path.read_text(encoding="utf-8"))["selection"]
            framework_manifest = json.loads(
                (
                    output_root
                    / "confirmation"
                    / dataset
                    / retention
                    / "confirmation_manifest.json"
                ).read_text(encoding="utf-8")
            )
            original_manifest = json.loads(
                (
                    historical_confirmation_root
                    / dataset
                    / retention
                    / "confirmation_manifest.json"
                ).read_text(encoding="utf-8")
            )
            framework = _averaged_role_frame(root, framework_manifest["runs"])
            rmse = _averaged_role_frame(
                root, _original_role_records(original_manifest, "rmse_selected")
            )
            ndcg = _averaged_role_frame(
                root, _original_role_records(original_manifest, "ndcg_selected")
            )
            paired = framework.merge(
                ndcg,
                on="user_id",
                suffixes=("_framework", "_ndcg"),
                validate="one_to_one",
            ).merge(
                rmse.add_suffix("_rmse").rename(columns={"user_id_rmse": "user_id"}),
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
            paired_dir = output_root / "analysis" / "paired_users"
            paired_dir.mkdir(parents=True, exist_ok=True)
            paired.to_parquet(
                paired_dir / f"{dataset}_{retention}.parquet", index=False
            )
            metrics = {}
            for offset, column in enumerate(
                (
                    "ndcg_regret_framework_vs_ndcg",
                    "rmse_framework_minus_ndcg",
                    "ndcg_framework_minus_rmse",
                    "rmse_framework_minus_rmse",
                )
            ):
                metrics[column] = _interval(
                    paired[column],
                    resamples=resamples,
                    confidence=confidence,
                    seed=base_seed + condition_index * 10 + offset,
                )
            rows.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "admissible_configurations": selection["admissible_configurations"],
                    "framework_dimension": selection["framework_selected"]["dimension"],
                    "framework_regularization": selection["framework_selected"]["regularization"],
                    "framework_matches_rmse": selection["framework_matches_rmse"],
                    "framework_matches_ndcg": selection["framework_matches_ndcg"],
                    **{
                        f"{metric}_{field}": value
                        for metric, values in metrics.items()
                        for field, value in values.items()
                    },
                }
            )
            condition_index += 1
    analysis_dir = output_root / "analysis"
    csv_path = analysis_dir / "construction_framework_results.csv"
    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)
    report = {
        "schema_version": 1,
        "stage": "post_confirmation_framework_construction",
        "claim_status": "exploratory_not_prospective",
        "conditions": len(rows),
        "result_csv": str(csv_path.relative_to(root)),
        "result_csv_sha256": sha256_file(csv_path),
        "interpretation": inference["interpretation"],
    }
    json_path = analysis_dir / "construction_framework_results.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "confirm", "analyze", "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", choices=["all", *DATASETS])
    parser.add_argument("--retention", action="append", choices=["all", *RETENTIONS])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = None if not args.dataset or args.dataset == ["all"] else args.dataset
    retentions = None if not args.retention or args.retention == ["all"] else args.retention
    if args.action in {"select", "all"}:
        for path in write_construction_selections(args.config):
            print(path)
    if args.action in {"confirm", "all"}:
        for path in execute_construction_confirmation(
            args.config,
            datasets=datasets,
            retentions=retentions,
            force=args.force,
        ):
            print(path)
    if args.action in {"analyze", "all"}:
        print(analyze_construction(args.config))


if __name__ == "__main__":
    main()
