"""Retrain primary selections and evaluate a five-star relevance sensitivity."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .confirmation import (
    DEFAULT_CONFIRMATION_PATH,
    DEFAULT_MODELING_PATH,
    selected_config_groups,
)
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file
from .explicit_mf import MFRunConfig, evaluate_stage, fit_explicit_mf
from .selection_lock import DATASETS, RETENTIONS, verify_selection_lock


DEFAULT_FIVE_STAR_PATH = Path(
    "configs/latent_factor_beyond_rmse/five_star_sensitivity.yaml"
)


def _run_matches(path: Path, config: MFRunConfig, roles: list[str]) -> bool:
    if not path.exists() or not (path.parent / "test_per_user.parquet").exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("stage") == "post_confirmation_five_star_relevance_sensitivity"
        and float(value.get("relevance_threshold", -1.0)) == 5.0
        and sorted(value.get("selection_roles", [])) == sorted(roles)
        and value.get("run_config") == asdict(config)
    )


def execute_five_star_sensitivity(
    *,
    dataset_config_path: Path,
    modeling_path: Path,
    sensitivity_path: Path,
    datasets: list[str],
    retentions: list[str],
    force: bool = False,
) -> list[Path]:
    root = _repo_root(dataset_config_path)
    sensitivity = yaml.safe_load(sensitivity_path.read_text(encoding="utf-8")) or {}
    primary_confirmation_path = _resolve(
        root, sensitivity["primary_confirmation_config"]
    )
    lock_path = _resolve(root, sensitivity["selection_lock"])
    gate = verify_selection_lock(
        lock_path=lock_path,
        dataset_config_path=dataset_config_path,
        modeling_path=modeling_path,
        confirmation_path=primary_confirmation_path,
    )
    if gate["status"] != "verified":
        raise RuntimeError("Primary selection lock does not verify.")
    dataset_cfg = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    settings = modeling["explicit_mf"]
    output_root = _resolve(root, sensitivity["output_root"])
    relevance_threshold = float(sensitivity["relevance_threshold"])
    manifests: list[Path] = []

    for dataset_name in datasets:
        dataset = dataset_cfg["datasets"][dataset_name]
        processed_dir = _resolve(root, dataset["output_dir"])
        sparsity_dir = _resolve(
            root, dataset_cfg["controlled_sparsity"]["output_root"]
        ) / dataset_name
        full_train = pd.read_parquet(processed_dir / "train.parquet")
        validation = pd.read_parquet(processed_dir / "validation.parquet")
        test = pd.read_parquet(processed_dir / "test.parquet")
        for retention in retentions:
            sparse_train = pd.read_parquet(sparsity_dir / f"train_{retention}.parquet")
            condition = lock["selections"][dataset_name][retention]
            condition_dir = output_root / dataset_name / retention
            run_records: list[dict[str, Any]] = []
            groups = selected_config_groups(condition)
            total = len(groups) * len(gate["confirmation_seeds"])
            completed = 0
            for group in groups:
                role_label = "shared" if len(group["roles"]) > 1 else group["roles"][0]
                for seed in gate["confirmation_seeds"]:
                    run_config = MFRunConfig(
                        dimension=int(group["dimension"]),
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
                    run_id = (
                        f"d{run_config.dimension}_lambda{run_config.regularization:g}"
                        f"_seed{run_config.seed}"
                    )
                    run_dir = condition_dir / role_label / run_id
                    summary_path = run_dir / "summary.json"
                    if force or not _run_matches(summary_path, run_config, group["roles"]):
                        fitted = fit_explicit_mf(sparse_train, run_config)
                        test_summary, per_user = evaluate_stage(
                            fitted,
                            full_train=full_train,
                            evaluation=test,
                            prior_evaluation=validation,
                            ranking_batch_size=int(settings["ranking_batch_size"]),
                            relevance_threshold=relevance_threshold,
                        )
                        run_dir.mkdir(parents=True, exist_ok=True)
                        per_user_path = run_dir / "test_per_user.parquet"
                        per_user.to_parquet(per_user_path, index=False)
                        summary = {
                            "schema_version": 1,
                            "stage": "post_confirmation_five_star_relevance_sensitivity",
                            "relevance_threshold": relevance_threshold,
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
                            },
                            "test": test_summary,
                        }
                        summary_path.write_text(
                            json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    completed += 1
                    print(
                        f"{dataset_name}/{retention}: {completed}/{total} five-star runs",
                        flush=True,
                    )
                    per_user_path = run_dir / "test_per_user.parquet"
                    run_records.append(
                        {
                            "selection_roles": group["roles"],
                            "dimension": run_config.dimension,
                            "regularization": run_config.regularization,
                            "seed": run_config.seed,
                            "summary_path": str(summary_path.relative_to(root)),
                            "summary_sha256": sha256_file(summary_path),
                            "per_user_path": str(per_user_path.relative_to(root)),
                            "per_user_sha256": sha256_file(per_user_path),
                        }
                    )
            manifest = {
                "schema_version": 1,
                "stage": "post_confirmation_five_star_relevance_sensitivity",
                "selection_lock_sha256": sha256_file(lock_path),
                "dataset": dataset_name,
                "retention_label": retention,
                "relevance_threshold": relevance_threshold,
                "expected_runs": total,
                "observed_runs": len(run_records),
                "runs": run_records,
            }
            manifest_path = condition_dir / "confirmation_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifests.append(manifest_path)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--modeling-config", type=Path, default=DEFAULT_MODELING_PATH)
    parser.add_argument("--sensitivity-config", type=Path, default=DEFAULT_FIVE_STAR_PATH)
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
    for path in execute_five_star_sensitivity(
        dataset_config_path=args.config,
        modeling_path=args.modeling_config,
        sensitivity_path=args.sensitivity_config,
        datasets=datasets,
        retentions=retentions,
        force=args.force,
    ):
        print(path)


if __name__ == "__main__":
    main()
