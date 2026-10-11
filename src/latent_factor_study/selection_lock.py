"""Verification gate for the frozen validation selections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .data import _repo_root, _resolve, sha256_file


DATASETS = ("movielens_1m", "ciaodvd", "yelp_tampa_restaurants")
RETENTIONS = ("r100", "r075", "r050", "r025")


def _same_config(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return int(left["dimension"]) == int(right["dimension"]) and float(
        left["regularization"]
    ) == float(right["regularization"])


def verify_selection_lock(
    *,
    lock_path: Path,
    dataset_config_path: Path,
    modeling_path: Path,
    confirmation_path: Path | None = None,
) -> dict[str, Any]:
    """Verify every development artifact before allowing test access."""
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    datasets_cfg = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(dataset_config_path)
    if lock.get("status") != "locked_before_test_access":
        raise ValueError("Selection lock status does not permit test access.")
    if int(lock.get("development_conditions", 0)) != 12:
        raise ValueError("Selection lock must contain all 12 conditions.")
    if int(lock.get("development_runs", 0)) != 2100:
        raise ValueError("Selection lock must account for all 2,100 development runs.")
    if float(lock.get("minimum_practically_important_ndcg_at_10_difference", 0.0)) <= 0:
        raise ValueError("A positive practical NDCG@10 threshold must be locked.")

    selection_seeds = [int(value) for value in lock["selection_seeds"]]
    confirmation_seeds = [int(value) for value in lock["confirmation_seeds"]]
    if set(selection_seeds).intersection(confirmation_seeds):
        raise ValueError("Selection and confirmation seeds must be disjoint.")
    if selection_seeds != [int(value) for value in modeling["explicit_mf"]["selection_seeds"]]:
        raise ValueError("Locked selection seeds differ from modeling.yaml.")
    if confirmation_seeds != [
        int(value) for value in modeling["explicit_mf"]["confirmation_seeds"]
    ]:
        raise ValueError("Locked confirmation seeds differ from modeling.yaml.")

    hashes = lock["locked_input_sha256"]
    expected_inputs = {
        "datasets_yaml": dataset_config_path,
        "modeling_yaml": modeling_path,
        "development_landscape_summary_csv": _resolve(
            root, modeling["development_output_root"]
        )
        / "development_landscape_summary.csv",
    }
    if confirmation_path is not None:
        expected_inputs["confirmation_yaml"] = confirmation_path
    for label, path in expected_inputs.items():
        actual = sha256_file(path)
        if actual != str(hashes[label]):
            raise ValueError(f"Locked input hash mismatch for {label}: {actual}")

    development_root = _resolve(root, modeling["development_output_root"])
    verified = 0
    for dataset in DATASETS:
        if dataset not in datasets_cfg["datasets"]:
            raise ValueError(f"Locked dataset is absent from dataset config: {dataset}")
        for retention in RETENTIONS:
            locked_condition = lock["selections"][dataset][retention]
            selection_path = development_root / dataset / retention / "selection.json"
            raw = selection_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != str(locked_condition["selection_sha256"]):
                raise ValueError(f"Selection hash mismatch for {dataset}/{retention}.")
            current = json.loads(raw)
            if current["observed_runs"] != 175 or current["expected_runs"] != 175:
                raise ValueError(f"Incomplete selection grid for {dataset}/{retention}.")
            for role in ("rmse_selected", "ndcg_selected"):
                if not _same_config(
                    locked_condition[role], current["selection"][role]
                ):
                    raise ValueError(
                        f"Locked {role} differs from current selection for {dataset}/{retention}."
                    )
            if bool(locked_condition["same_configuration"]) != bool(
                current["selection"]["same_configuration"]
            ):
                raise ValueError(f"Same-configuration flag mismatch for {dataset}/{retention}.")
            verified += 1
    return {
        "status": "verified",
        "conditions_verified": verified,
        "development_runs_verified": 2100,
        "confirmation_seeds": confirmation_seeds,
        "minimum_practically_important_ndcg_at_10_difference": float(
            lock["minimum_practically_important_ndcg_at_10_difference"]
        ),
    }
