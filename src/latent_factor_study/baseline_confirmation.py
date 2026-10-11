"""Locked-test confirmation for bias-only and popularity baselines."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .baseline_lock import (
    DEFAULT_BASELINE_CONFIRMATION_PATH,
    verify_baseline_selection_lock,
)
from .baselines import (
    DEFAULT_BASELINE_MODELING_PATH,
    evaluate_popularity_stage,
)
from .bpr_grid import DATASETS, RETENTIONS
from .confirmation import selected_config_groups
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file
from .explicit_mf import MFRunConfig, evaluate_stage, fit_explicit_mf


def _bias_run_matches(path: Path, config: MFRunConfig, roles: list[str]) -> bool:
    if not path.exists() or not (path.parent / "test_per_user.parquet").exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("stage") == "locked_baseline_test_confirmation"
        and value.get("model") == "bias_only"
        and sorted(value.get("selection_roles", [])) == sorted(roles)
        and value.get("run_config") == asdict(config)
    )


def execute_baseline_confirmation(
    *,
    dataset_config_path: Path,
    modeling_path: Path,
    confirmation_path: Path,
    datasets: list[str],
    retentions: list[str],
    force: bool = False,
) -> list[Path]:
    root = _repo_root(dataset_config_path)
    confirmation_cfg = yaml.safe_load(confirmation_path.read_text(encoding="utf-8")) or {}
    lock_path = _resolve(root, confirmation_cfg["selection_lock"])
    gate = verify_baseline_selection_lock(
        lock_path=lock_path,
        dataset_config_path=dataset_config_path,
        modeling_path=modeling_path,
        confirmation_path=confirmation_path,
    )
    if gate["status"] != "verified":
        raise RuntimeError("Baseline selection lock did not verify; test access prohibited.")

    dataset_cfg = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    settings = modeling["bias_only"]
    output_root = _resolve(root, confirmation_cfg["output_root"])
    manifests: list[Path] = []

    for dataset_name in datasets:
        dataset = dataset_cfg["datasets"][dataset_name]
        processed_dir = _resolve(root, dataset["output_dir"])
        sparsity_dir = _resolve(
            root, dataset_cfg["controlled_sparsity"]["output_root"]
        ) / dataset_name
        # Test data are resolved only after the baseline lock verifies above.
        full_train = pd.read_parquet(processed_dir / "train.parquet")
        validation = pd.read_parquet(processed_dir / "validation.parquet")
        test = pd.read_parquet(processed_dir / "test.parquet")

        for retention in retentions:
            sparse_train = pd.read_parquet(sparsity_dir / f"train_{retention}.parquet")
            condition = lock["selections"][dataset_name][retention]
            condition_dir = output_root / dataset_name / retention
            condition_dir.mkdir(parents=True, exist_ok=True)
            bias_records: list[dict[str, Any]] = []
            groups = selected_config_groups(condition)
            total = len(groups) * len(gate["confirmation_seeds"])
            completed = 0
            for group in groups:
                role_label = "shared" if len(group["roles"]) > 1 else group["roles"][0]
                for seed in gate["confirmation_seeds"]:
                    run_config = MFRunConfig(
                        dimension=0,
                        regularization=float(group["regularization"]),
                        learning_rate=float(settings["learning_rate"]),
                        batch_size=int(settings["batch_size"]),
                        max_epochs=int(settings["max_epochs"]),
                        convergence_patience=int(settings["convergence_patience"]),
                        convergence_relative_tolerance=float(
                            settings["convergence_relative_tolerance"]
                        ),
                        seed=int(seed),
                        device=str(settings["device"]),
                        torch_num_threads=int(settings["torch_num_threads"]),
                    )
                    run_id = f"lambda{run_config.regularization:g}_seed{run_config.seed}"
                    run_dir = condition_dir / "bias_only" / role_label / run_id
                    summary_path = run_dir / "summary.json"
                    if force or not _bias_run_matches(
                        summary_path, run_config, group["roles"]
                    ):
                        fitted = fit_explicit_mf(sparse_train, run_config)
                        evaluation_started = time.perf_counter()
                        test_summary, per_user = evaluate_stage(
                            fitted,
                            full_train=full_train,
                            evaluation=test,
                            prior_evaluation=validation,
                            ranking_batch_size=int(settings["ranking_batch_size"]),
                        )
                        evaluation_seconds = time.perf_counter() - evaluation_started
                        run_dir.mkdir(parents=True, exist_ok=True)
                        per_user_path = run_dir / "test_per_user.parquet"
                        per_user.to_parquet(per_user_path, index=False)
                        summary = {
                            "schema_version": 1,
                            "stage": "locked_baseline_test_confirmation",
                            "model": "bias_only",
                            "selection_roles": group["roles"],
                            "dataset": dataset_name,
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
                                "test_evaluation_seconds": evaluation_seconds,
                            },
                            "test": test_summary,
                        }
                        summary_path.write_text(
                            json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    completed += 1
                    print(
                        f"{dataset_name}/{retention}: {completed}/{total} bias runs",
                        flush=True,
                    )
                    per_user_path = run_dir / "test_per_user.parquet"
                    bias_records.append(
                        {
                            "model": "bias_only",
                            "selection_roles": group["roles"],
                            "regularization": run_config.regularization,
                            "seed": run_config.seed,
                            "summary_path": str(summary_path.relative_to(root)),
                            "summary_sha256": sha256_file(summary_path),
                            "per_user_path": str(per_user_path.relative_to(root)),
                            "per_user_sha256": sha256_file(per_user_path),
                        }
                    )

            popularity_dir = condition_dir / "popularity"
            popularity_dir.mkdir(parents=True, exist_ok=True)
            popularity_started = time.perf_counter()
            popularity_summary, popularity_users = evaluate_popularity_stage(
                sparse_train=sparse_train,
                full_train=full_train,
                evaluation=test,
                prior_evaluation=validation,
                relevance_threshold=float(modeling["ranking"]["relevance_threshold"]),
            )
            popularity_seconds = time.perf_counter() - popularity_started
            popularity_users_path = popularity_dir / "test_per_user.parquet"
            popularity_summary_path = popularity_dir / "summary.json"
            popularity_users.to_parquet(popularity_users_path, index=False)
            popularity_summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": "locked_baseline_test_confirmation",
                        "model": "popularity",
                        "dataset": dataset_name,
                        "retention_label": retention,
                        "test_evaluation_seconds": popularity_seconds,
                        "test": {"ranking": popularity_summary},
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            popularity_record = {
                "model": "popularity",
                "summary_path": str(popularity_summary_path.relative_to(root)),
                "summary_sha256": sha256_file(popularity_summary_path),
                "per_user_path": str(popularity_users_path.relative_to(root)),
                "per_user_sha256": sha256_file(popularity_users_path),
            }
            manifest = {
                "schema_version": 1,
                "stage": "locked_baseline_test_confirmation",
                "selection_lock_sha256": sha256_file(lock_path),
                "dataset": dataset_name,
                "retention_label": retention,
                "expected_bias_runs": total,
                "observed_bias_runs": len(bias_records),
                "bias_runs": bias_records,
                "popularity": popularity_record,
            }
            manifest_path = condition_dir / "confirmation_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifests.append(manifest_path)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--modeling-config", type=Path, default=DEFAULT_BASELINE_MODELING_PATH
    )
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_BASELINE_CONFIRMATION_PATH
    )
    parser.add_argument(
        "--dataset", action="append", choices=["all", *DATASETS], dest="datasets"
    )
    parser.add_argument(
        "--retention", action="append", choices=["all", *RETENTIONS], dest="retentions"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = list(DATASETS) if not args.datasets or args.datasets == ["all"] else args.datasets
    retentions = (
        list(RETENTIONS)
        if not args.retentions or args.retentions == ["all"]
        else args.retentions
    )
    for path in execute_baseline_confirmation(
        dataset_config_path=args.config,
        modeling_path=args.modeling_config,
        confirmation_path=args.confirmation_config,
        datasets=datasets,
        retentions=retentions,
        force=args.force,
    ):
        print(path)


if __name__ == "__main__":
    main()
