"""Fresh-seed locked-test confirmation for the prospective Epinions study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .data import _repo_root, _resolve, sha256_file
from .explicit_mf import MFRunConfig, evaluate_stage, fit_explicit_mf
from .prospective_lock import verify_prospective_selection_lock
from .uncertainty_selector import RETENTIONS


DEFAULT_CONFIRMATION_CONFIG = Path(
    "configs/latent_factor_beyond_rmse/epinions_prospective_confirmation.yaml"
)
ROLE_ORDER = ("rmse_selected", "ndcg_selected", "framework_selected")


def selected_config_groups(condition: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collapse any identical selections into one fresh-seed fit."""
    groups: dict[tuple[int, float], list[str]] = {}
    for role in ROLE_ORDER:
        selected = condition[role]
        key = (int(selected["dimension"]), float(selected["regularization"]))
        groups.setdefault(key, []).append(role)
    return [
        {"roles": roles, "dimension": key[0], "regularization": key[1]}
        for key, roles in groups.items()
    ]


def _run_matches(path: Path, run_config: MFRunConfig, roles: list[str]) -> bool:
    if not path.exists() or not (path.parent / "test_per_user.parquet").exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("stage") == "prospective_locked_test_confirmation"
        and sorted(value.get("selection_roles", [])) == sorted(roles)
        and value.get("run_config") == asdict(run_config)
    )


def execute_prospective_confirmation(
    *,
    confirmation_config_path: Path,
    retentions: list[str],
    force: bool = False,
) -> list[Path]:
    config = yaml.safe_load(confirmation_config_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(confirmation_config_path)
    dataset_config_path = _resolve(root, config["dataset_config"])
    modeling_path = _resolve(root, config["modeling_config"])
    protocol_lock_path = _resolve(root, config["protocol_lock"])
    analysis_plan_path = _resolve(root, config["analysis_plan"])
    selection_lock_path = _resolve(root, config["selection_lock"])
    lock = verify_prospective_selection_lock(
        selection_lock_path=selection_lock_path,
        dataset_config_path=dataset_config_path,
        modeling_path=modeling_path,
        protocol_lock_path=protocol_lock_path,
        analysis_plan_path=analysis_plan_path,
    )

    dataset_cfg = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    settings = modeling["explicit_mf"]
    dataset = dataset_cfg["datasets"]["epinions"]
    processed_dir = _resolve(root, dataset["output_dir"])
    sparsity_dir = _resolve(root, dataset_cfg["controlled_sparsity"]["output_root"]) / "epinions"
    output_root = _resolve(root, config["output_root"])

    # Test data are resolved only after the complete prospective gate above.
    full_train = pd.read_parquet(processed_dir / "train.parquet")
    validation = pd.read_parquet(processed_dir / "validation.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    manifests: list[Path] = []

    for retention in retentions:
        sparse_train = pd.read_parquet(sparsity_dir / f"train_{retention}.parquet")
        condition = lock["selections"][retention]
        groups = selected_config_groups(condition)
        total = len(groups) * len(lock["confirmation_seeds"])
        completed = 0
        run_records: list[dict[str, Any]] = []
        condition_dir = output_root / "epinions" / retention
        condition_dir.mkdir(parents=True, exist_ok=True)

        for group_index, group in enumerate(groups, start=1):
            role_label = (
                group["roles"][0]
                if len(group["roles"]) == 1
                else f"shared_{group_index}"
            )
            for seed in lock["confirmation_seeds"]:
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
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)
                    per_user.to_parquet(run_dir / "test_per_user.parquet", index=False)
                    summary = {
                        "schema_version": 1,
                        "stage": "prospective_locked_test_confirmation",
                        "selection_roles": group["roles"],
                        "dataset": "epinions",
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
                print(f"epinions/{retention}: {completed}/{total} confirmation runs", flush=True)
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
            "stage": "prospective_locked_test_confirmation",
            "selection_lock_sha256": sha256_file(selection_lock_path),
            "dataset": "epinions",
            "retention_label": retention,
            "expected_runs": total,
            "observed_runs": len(run_records),
            "unique_selected_configurations": len(groups),
            "runs": run_records,
        }
        manifest_path = condition_dir / "confirmation_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifests.append(manifest_path)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG)
    parser.add_argument("--retention", choices=["all", *RETENTIONS], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    retentions = list(RETENTIONS) if args.retention == "all" else [args.retention]
    for path in execute_prospective_confirmation(
        confirmation_config_path=args.config,
        retentions=retentions,
        force=args.force,
    ):
        print(path)


if __name__ == "__main__":
    main()

