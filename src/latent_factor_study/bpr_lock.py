"""Build and verify the BPR validation-selection lock before BPR test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .bpr_grid import DATASETS, RETENTIONS
from .bpr_mf import DEFAULT_BPR_MODELING_PATH
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file


DEFAULT_BPR_CONFIRMATION_PATH = Path(
    "configs/latent_factor_beyond_rmse/bpr_confirmation.yaml"
)
DEFAULT_BPR_LOCK_PATH = Path(
    "configs/latent_factor_beyond_rmse/bpr_selection_lock.yaml"
)


def build_bpr_selection_lock(
    *,
    dataset_config_path: Path,
    modeling_path: Path,
    confirmation_path: Path,
    output_path: Path,
) -> Path:
    datasets_cfg = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(dataset_config_path)
    development_root = _resolve(root, modeling["development_output_root"])
    dimensions = [int(value) for value in modeling["bpr_mf"]["dimensions"]]
    regularizations = [float(value) for value in modeling["bpr_mf"]["regularizations"]]
    selection_seeds = [int(value) for value in modeling["bpr_mf"]["selection_seeds"]]
    confirmation_seeds = [
        int(value) for value in modeling["bpr_mf"]["confirmation_seeds"]
    ]
    if set(selection_seeds).intersection(confirmation_seeds):
        raise ValueError("BPR selection and confirmation seeds must be disjoint.")
    expected_per_condition = len(dimensions) * len(regularizations) * len(selection_seeds)
    selections: dict[str, dict[str, Any]] = {}

    for dataset in DATASETS:
        if dataset not in datasets_cfg["datasets"]:
            raise ValueError(f"BPR dataset is absent from dataset config: {dataset}")
        selections[dataset] = {}
        for retention in RETENTIONS:
            selection_path = development_root / dataset / retention / "selection.json"
            raw = selection_path.read_bytes()
            current = json.loads(raw)
            if current.get("stage") != "bpr_development_validation_only":
                raise ValueError(f"Unexpected BPR stage for {dataset}/{retention}.")
            if int(current["observed_runs"]) != expected_per_condition or int(
                current["expected_runs"]
            ) != expected_per_condition:
                raise ValueError(f"Incomplete BPR grid for {dataset}/{retention}.")
            selected = current["selection"]["bpr_selected"]
            selections[dataset][retention] = {
                "selection_sha256": hashlib.sha256(raw).hexdigest(),
                "bpr_selected": {
                    "dimension": int(selected["dimension"]),
                    "regularization": float(selected["regularization"]),
                    "selection_seed_count": int(selected["selection_seed_count"]),
                    "mean_validation_ndcg@10": float(
                        selected["mean_validation_ndcg@10"]
                    ),
                },
            }

    payload = {
        "schema_version": 1,
        "status": "locked_before_bpr_test_access",
        "scientific_role": "secondary_ranking_native_boundary",
        "development_conditions": len(DATASETS) * len(RETENTIONS),
        "development_runs": expected_per_condition * len(DATASETS) * len(RETENTIONS),
        "configurations_per_condition": len(dimensions) * len(regularizations),
        "selection_seeds": selection_seeds,
        "confirmation_seeds": confirmation_seeds,
        "locked_input_sha256": {
            "datasets_yaml": sha256_file(dataset_config_path),
            "bpr_modeling_yaml": sha256_file(modeling_path),
            "bpr_confirmation_yaml": sha256_file(confirmation_path),
        },
        "selections": selections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def verify_bpr_selection_lock(
    *,
    lock_path: Path,
    dataset_config_path: Path,
    modeling_path: Path,
    confirmation_path: Path,
) -> dict[str, Any]:
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(dataset_config_path)
    development_root = _resolve(root, modeling["development_output_root"])
    expected_per_condition = (
        len(modeling["bpr_mf"]["dimensions"])
        * len(modeling["bpr_mf"]["regularizations"])
        * len(modeling["bpr_mf"]["selection_seeds"])
    )
    if lock.get("status") != "locked_before_bpr_test_access":
        raise ValueError("BPR selection lock status does not permit test access.")
    if int(lock.get("development_conditions", 0)) != 12:
        raise ValueError("BPR selection lock must contain all 12 conditions.")
    if int(lock.get("development_runs", 0)) != expected_per_condition * 12:
        raise ValueError("BPR selection lock contains the wrong development-run count.")
    if [int(value) for value in lock["selection_seeds"]] != [
        int(value) for value in modeling["bpr_mf"]["selection_seeds"]
    ]:
        raise ValueError("Locked BPR selection seeds differ from modeling config.")
    if [int(value) for value in lock["confirmation_seeds"]] != [
        int(value) for value in modeling["bpr_mf"]["confirmation_seeds"]
    ]:
        raise ValueError("Locked BPR confirmation seeds differ from modeling config.")
    hashes = lock["locked_input_sha256"]
    for label, path in {
        "datasets_yaml": dataset_config_path,
        "bpr_modeling_yaml": modeling_path,
        "bpr_confirmation_yaml": confirmation_path,
    }.items():
        if sha256_file(path) != str(hashes[label]):
            raise ValueError(f"Locked BPR input hash mismatch for {label}.")

    verified = 0
    for dataset in DATASETS:
        for retention in RETENTIONS:
            selection_path = development_root / dataset / retention / "selection.json"
            raw = selection_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != str(
                lock["selections"][dataset][retention]["selection_sha256"]
            ):
                raise ValueError(f"BPR selection hash mismatch for {dataset}/{retention}.")
            current = json.loads(raw)
            if int(current["observed_runs"]) != expected_per_condition:
                raise ValueError(f"Incomplete BPR selection for {dataset}/{retention}.")
            chosen = current["selection"]["bpr_selected"]
            locked = lock["selections"][dataset][retention]["bpr_selected"]
            if int(chosen["dimension"]) != int(locked["dimension"]) or float(
                chosen["regularization"]
            ) != float(locked["regularization"]):
                raise ValueError(f"Locked BPR choice differs for {dataset}/{retention}.")
            verified += 1
    return {
        "status": "verified",
        "conditions_verified": verified,
        "development_runs_verified": expected_per_condition * 12,
        "confirmation_seeds": [int(value) for value in lock["confirmation_seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--modeling-config", type=Path, default=DEFAULT_BPR_MODELING_PATH)
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_BPR_CONFIRMATION_PATH
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_BPR_LOCK_PATH)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(
            verify_bpr_selection_lock(
                lock_path=args.lock,
                dataset_config_path=args.config,
                modeling_path=args.modeling_config,
                confirmation_path=args.confirmation_config,
            )
        )
    else:
        print(
            build_bpr_selection_lock(
                dataset_config_path=args.config,
                modeling_path=args.modeling_config,
                confirmation_path=args.confirmation_config,
                output_path=args.lock,
            )
        )


if __name__ == "__main__":
    main()
