"""Locked-test confirmation for validation-selected BPR-MF configurations."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .bpr_grid import DATASETS, RETENTIONS
from .bpr_lock import (
    DEFAULT_BPR_CONFIRMATION_PATH,
    DEFAULT_BPR_LOCK_PATH,
    verify_bpr_selection_lock,
)
from .bpr_mf import (
    DEFAULT_BPR_MODELING_PATH,
    BPRRunConfig,
    evaluate_bpr_stage,
    fit_bpr_mf,
)
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file


def _run_matches(path: Path, config: BPRRunConfig) -> bool:
    if not path.exists() or not (path.parent / "test_per_user.parquet").exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("stage") == "locked_bpr_test_confirmation"
        and value.get("selection_role") == "bpr_selected"
        and value.get("run_config") == asdict(config)
    )


def execute_bpr_confirmation(
    *,
    dataset_config_path: Path,
    modeling_path: Path,
    confirmation_path: Path,
    datasets: list[str],
    retentions: list[str],
    force: bool = False,
) -> list[Path]:
    """Verify the BPR lock, then execute only the selected BPR configurations."""
    root = _repo_root(dataset_config_path)
    confirmation_cfg = yaml.safe_load(confirmation_path.read_text(encoding="utf-8")) or {}
    lock_path = _resolve(root, confirmation_cfg["selection_lock"])
    gate = verify_bpr_selection_lock(
        lock_path=lock_path,
        dataset_config_path=dataset_config_path,
        modeling_path=modeling_path,
        confirmation_path=confirmation_path,
    )
    if gate["status"] != "verified":
        raise RuntimeError("BPR selection lock did not verify; test access is prohibited.")

    dataset_cfg = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    settings = modeling["bpr_mf"]
    output_root = _resolve(root, confirmation_cfg["output_root"])
    manifests: list[Path] = []

    for dataset_name in datasets:
        dataset = dataset_cfg["datasets"][dataset_name]
        processed_dir = _resolve(root, dataset["output_dir"])
        sparsity_dir = _resolve(
            root, dataset_cfg["controlled_sparsity"]["output_root"]
        ) / dataset_name
        # Test data are resolved only after the BPR selection lock passes above.
        full_train = pd.read_parquet(processed_dir / "train.parquet")
        validation = pd.read_parquet(processed_dir / "validation.parquet")
        test = pd.read_parquet(processed_dir / "test.parquet")

        for retention in retentions:
            sparse_train = pd.read_parquet(sparsity_dir / f"train_{retention}.parquet")
            selected = lock["selections"][dataset_name][retention]["bpr_selected"]
            condition_dir = output_root / dataset_name / retention
            condition_dir.mkdir(parents=True, exist_ok=True)
            run_records: list[dict[str, Any]] = []
            total = len(gate["confirmation_seeds"])

            for completed, seed in enumerate(gate["confirmation_seeds"], start=1):
                run_config = BPRRunConfig(
                    dimension=int(selected["dimension"]),
                    regularization=float(selected["regularization"]),
                    learning_rate=float(settings["learning_rate"]),
                    batch_size=int(settings["batch_size"]),
                    max_epochs=int(settings["max_epochs"]),
                    negatives_per_positive=int(settings["negatives_per_positive"]),
                    relevance_threshold=float(settings["relevance_threshold"]),
                    seed=int(seed),
                    device=str(settings["device"]),
                    torch_num_threads=int(settings["torch_num_threads"]),
                )
                run_id = (
                    f"d{run_config.dimension}_lambda{run_config.regularization:g}"
                    f"_seed{run_config.seed}"
                )
                run_dir = condition_dir / "bpr_selected" / run_id
                summary_path = run_dir / "summary.json"
                if force or not _run_matches(summary_path, run_config):
                    fitted = fit_bpr_mf(sparse_train, run_config)
                    evaluation_started = time.perf_counter()
                    test_summary, per_user = evaluate_bpr_stage(
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
                        "stage": "locked_bpr_test_confirmation",
                        "selection_role": "bpr_selected",
                        "dataset": dataset_name,
                        "retention_label": retention,
                        "run_id": run_id,
                        "run_config": asdict(run_config),
                        "training": {
                            "rows": int(len(sparse_train)),
                            "positive_rows": fitted.positive_rows,
                            "users": len(fitted.user_ids),
                            "items": len(fitted.item_ids),
                            "users_without_positive_training": (
                                fitted.users_without_positive_training
                            ),
                            "epochs_completed": len(fitted.epoch_history),
                            "seconds": fitted.training_seconds,
                            "epoch_history": fitted.epoch_history,
                            "test_evaluation_seconds": evaluation_seconds,
                        },
                        "test": {"ranking": test_summary},
                    }
                    summary_path.write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                print(
                    f"{dataset_name}/{retention}: {completed}/{total} BPR confirmation runs",
                    flush=True,
                )
                run_records.append(
                    {
                        "selection_role": "bpr_selected",
                        "dimension": run_config.dimension,
                        "regularization": run_config.regularization,
                        "seed": run_config.seed,
                        "summary_path": str(summary_path.relative_to(root)),
                        "summary_sha256": sha256_file(summary_path),
                        "per_user_path": str(
                            (run_dir / "test_per_user.parquet").relative_to(root)
                        ),
                        "per_user_sha256": sha256_file(
                            run_dir / "test_per_user.parquet"
                        ),
                    }
                )
            manifest = {
                "schema_version": 1,
                "stage": "locked_bpr_test_confirmation",
                "selection_lock_sha256": sha256_file(lock_path),
                "dataset": dataset_name,
                "retention_label": retention,
                "expected_runs": total,
                "observed_runs": len(run_records),
                "runs": run_records,
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
    parser.add_argument("--modeling-config", type=Path, default=DEFAULT_BPR_MODELING_PATH)
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_BPR_CONFIRMATION_PATH
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
    for path in execute_bpr_confirmation(
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
