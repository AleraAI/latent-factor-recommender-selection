"""Build and verify the bias/popularity baseline lock before test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .baselines import DEFAULT_BASELINE_MODELING_PATH
from .bpr_grid import DATASETS, RETENTIONS
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file


DEFAULT_BASELINE_CONFIRMATION_PATH = Path(
    "configs/latent_factor_beyond_rmse/baseline_confirmation.yaml"
)
DEFAULT_BASELINE_LOCK_PATH = Path(
    "configs/latent_factor_beyond_rmse/baseline_selection_lock.yaml"
)


def build_baseline_selection_lock(
    *,
    dataset_config_path: Path,
    modeling_path: Path,
    confirmation_path: Path,
    output_path: Path,
) -> Path:
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(dataset_config_path)
    development_root = _resolve(root, modeling["development_output_root"])
    settings = modeling["bias_only"]
    expected = len(settings["regularizations"]) * len(settings["selection_seeds"])
    selection_seeds = [int(value) for value in settings["selection_seeds"]]
    confirmation_seeds = [int(value) for value in settings["confirmation_seeds"]]
    if set(selection_seeds).intersection(confirmation_seeds):
        raise ValueError("Baseline selection and confirmation seeds must be disjoint.")
    selections: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        selections[dataset] = {}
        for retention in RETENTIONS:
            selection_path = development_root / dataset / retention / "selection.json"
            raw = selection_path.read_bytes()
            current = json.loads(raw)
            if current.get("stage") != "baseline_development_validation_only":
                raise ValueError(f"Unexpected baseline stage for {dataset}/{retention}.")
            if int(current["observed_bias_runs"]) != expected or int(
                current["expected_bias_runs"]
            ) != expected:
                raise ValueError(f"Incomplete baseline grid for {dataset}/{retention}.")
            popularity_path = root / current["popularity_summary"]
            roles = {
                role: {
                    "dimension": int(current["selection"][role]["dimension"]),
                    "regularization": float(
                        current["selection"][role]["regularization"]
                    ),
                    "selection_seed_count": int(
                        current["selection"][role]["selection_seed_count"]
                    ),
                }
                for role in ("rmse_selected", "ndcg_selected")
            }
            selections[dataset][retention] = {
                "selection_sha256": hashlib.sha256(raw).hexdigest(),
                "popularity_validation_summary_sha256": sha256_file(popularity_path),
                "same_configuration": bool(
                    current["selection"]["same_configuration"]
                ),
                **roles,
            }
    payload = {
        "schema_version": 1,
        "status": "locked_before_baseline_test_access",
        "scientific_role": "non_latent_reference_baselines",
        "development_conditions": 12,
        "development_bias_runs": expected * 12,
        "selection_seeds": selection_seeds,
        "confirmation_seeds": confirmation_seeds,
        "popularity_rule": modeling["popularity"],
        "locked_input_sha256": {
            "datasets_yaml": sha256_file(dataset_config_path),
            "baseline_modeling_yaml": sha256_file(modeling_path),
            "baseline_confirmation_yaml": sha256_file(confirmation_path),
        },
        "selections": selections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def verify_baseline_selection_lock(
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
    settings = modeling["bias_only"]
    expected = len(settings["regularizations"]) * len(settings["selection_seeds"])
    if lock.get("status") != "locked_before_baseline_test_access":
        raise ValueError("Baseline lock status does not permit test access.")
    if int(lock.get("development_conditions", 0)) != 12 or int(
        lock.get("development_bias_runs", 0)
    ) != expected * 12:
        raise ValueError("Baseline lock has an incomplete development count.")
    for label, path in {
        "datasets_yaml": dataset_config_path,
        "baseline_modeling_yaml": modeling_path,
        "baseline_confirmation_yaml": confirmation_path,
    }.items():
        if sha256_file(path) != str(lock["locked_input_sha256"][label]):
            raise ValueError(f"Locked baseline input hash mismatch for {label}.")
    if [int(value) for value in lock["selection_seeds"]] != [
        int(value) for value in settings["selection_seeds"]
    ] or [int(value) for value in lock["confirmation_seeds"]] != [
        int(value) for value in settings["confirmation_seeds"]
    ]:
        raise ValueError("Locked baseline seeds differ from modeling config.")

    verified = 0
    for dataset in DATASETS:
        for retention in RETENTIONS:
            selection_path = development_root / dataset / retention / "selection.json"
            raw = selection_path.read_bytes()
            frozen = lock["selections"][dataset][retention]
            if hashlib.sha256(raw).hexdigest() != frozen["selection_sha256"]:
                raise ValueError(f"Baseline selection hash mismatch for {dataset}/{retention}.")
            current = json.loads(raw)
            if int(current["observed_bias_runs"]) != expected:
                raise ValueError(f"Incomplete baseline selection for {dataset}/{retention}.")
            popularity_path = root / current["popularity_summary"]
            if sha256_file(popularity_path) != frozen[
                "popularity_validation_summary_sha256"
            ]:
                raise ValueError(f"Popularity hash mismatch for {dataset}/{retention}.")
            for role in ("rmse_selected", "ndcg_selected"):
                chosen = current["selection"][role]
                if int(chosen["dimension"]) != 0 or float(
                    chosen["regularization"]
                ) != float(frozen[role]["regularization"]):
                    raise ValueError(f"Locked baseline choice differs for {dataset}/{retention}.")
            verified += 1
    return {
        "status": "verified",
        "conditions_verified": verified,
        "development_bias_runs_verified": expected * 12,
        "confirmation_seeds": [int(value) for value in lock["confirmation_seeds"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--modeling-config", type=Path, default=DEFAULT_BASELINE_MODELING_PATH
    )
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_BASELINE_CONFIRMATION_PATH
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_BASELINE_LOCK_PATH)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(
            verify_baseline_selection_lock(
                lock_path=args.lock,
                dataset_config_path=args.config,
                modeling_path=args.modeling_config,
                confirmation_path=args.confirmation_config,
            )
        )
    else:
        print(
            build_baseline_selection_lock(
                dataset_config_path=args.config,
                modeling_path=args.modeling_config,
                confirmation_path=args.confirmation_config,
                output_path=args.lock,
            )
        )


if __name__ == "__main__":
    main()
