"""Create and verify the Epinions prospective selection lock."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from .data import _repo_root, _resolve, sha256_file
from .uncertainty_selector import RETENTIONS, verify_prospective_protocol_lock


DEFAULT_DATASET_CONFIG = Path(
    "configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml"
)
DEFAULT_MODELING_CONFIG = Path(
    "configs/latent_factor_beyond_rmse/epinions_prospective_modeling.yaml"
)
DEFAULT_PROTOCOL_LOCK = Path(
    "configs/latent_factor_beyond_rmse/epinions_prospective_protocol_lock.yaml"
)
DEFAULT_ANALYSIS_PLAN = Path(
    "configs/latent_factor_beyond_rmse/epinions_prospective_analysis_plan.yaml"
)
DEFAULT_SELECTION_LOCK = Path(
    "configs/latent_factor_beyond_rmse/epinions_prospective_selection_lock.yaml"
)


def _config(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dimension": int(value["dimension"]),
        "regularization": float(value["regularization"]),
    }


def create_prospective_selection_lock(
    *,
    dataset_config_path: Path,
    modeling_path: Path,
    protocol_lock_path: Path,
    analysis_plan_path: Path,
    output_path: Path,
    locked_on: str,
) -> Path:
    """Freeze all validation selections before any test evaluation."""
    if output_path.exists():
        raise FileExistsError(
            f"Prospective selection lock already exists and is immutable: {output_path}"
        )
    protocol_lock = verify_prospective_protocol_lock(
        protocol_lock_path=protocol_lock_path,
        dataset_config_path=dataset_config_path,
        modeling_path=modeling_path,
    )
    analysis_plan = yaml.safe_load(analysis_plan_path.read_text(encoding="utf-8")) or {}
    if analysis_plan.get("status") != "frozen_before_epinions_test_access":
        raise ValueError("Prospective analysis plan is not frozen before test access.")
    if list(analysis_plan.get("conditions", [])) != list(RETENTIONS):
        raise ValueError("Analysis-plan retention conditions differ from the protocol.")

    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(dataset_config_path)
    development_root = _resolve(root, modeling["development_output_root"])
    selections: dict[str, Any] = {}
    observed_runs = 0
    for retention in RETENTIONS:
        path = development_root / "epinions" / retention / "framework_selection.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("stage") != "prospective_development_validation_only":
            raise ValueError(f"Unexpected framework-selection stage: {retention}")
        if artifact.get("test_metrics_accessed") is not False:
            raise ValueError(f"Framework selection does not certify test isolation: {retention}")
        if artifact.get("protocol_lock_sha256") != sha256_file(protocol_lock_path):
            raise ValueError(f"Protocol-lock hash mismatch: {retention}")
        if int(artifact.get("observed_runs", 0)) != 175:
            raise ValueError(f"Incomplete framework selection: {retention}")
        selection = artifact["selection"]
        selections[retention] = {
            "framework_selection_sha256": sha256_file(path),
            "grid_runs_sha256": artifact["grid_runs_sha256"],
            "rmse_selected": _config(selection["rmse_selected"]),
            "ndcg_selected": _config(selection["ndcg_selected"]),
            "framework_selected": _config(selection["framework_selected"]),
            "admissible_configurations": int(selection["admissible_configurations"]),
            "framework_matches_rmse": bool(selection["framework_matches_rmse"]),
            "framework_matches_ndcg": bool(selection["framework_matches_ndcg"]),
        }
        observed_runs += int(artifact["observed_runs"])
    if observed_runs != int(protocol_lock["prospective_development_runs"]):
        raise ValueError(f"Prospective development total is {observed_runs}, expected 700.")

    lock = {
        "schema_version": 1,
        "status": "locked_before_epinions_test_access",
        "locked_on": locked_on,
        "dataset": "epinions",
        "development_conditions": len(RETENTIONS),
        "development_runs": observed_runs,
        "test_metrics_accessed_at_lock": False,
        "confirmation_seeds": [
            int(value) for value in modeling["explicit_mf"]["confirmation_seeds"]
        ],
        "practical_ndcg_at_10_regret": float(
            modeling["uncertainty_aware_selection"]["practical_ndcg_at_10_regret"]
        ),
        "locked_input_sha256": {
            "datasets_yaml": sha256_file(dataset_config_path),
            "modeling_yaml": sha256_file(modeling_path),
            "protocol_lock_yaml": sha256_file(protocol_lock_path),
            "analysis_plan_yaml": sha256_file(analysis_plan_path),
        },
        "selections": selections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(lock, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return output_path


def verify_prospective_selection_lock(
    *,
    selection_lock_path: Path,
    dataset_config_path: Path,
    modeling_path: Path,
    protocol_lock_path: Path,
    analysis_plan_path: Path,
) -> dict[str, Any]:
    """Verify the selection lock before allowing Epinions test access."""
    lock = yaml.safe_load(selection_lock_path.read_text(encoding="utf-8")) or {}
    if lock.get("status") != "locked_before_epinions_test_access":
        raise ValueError("Prospective selection lock does not permit test access.")
    if lock.get("test_metrics_accessed_at_lock") is not False:
        raise ValueError("Prospective lock does not certify pre-test creation.")
    if int(lock.get("development_conditions", 0)) != 4:
        raise ValueError("Prospective lock must contain four conditions.")
    if int(lock.get("development_runs", 0)) != 700:
        raise ValueError("Prospective lock must account for 700 development runs.")
    expected = {
        "datasets_yaml": dataset_config_path,
        "modeling_yaml": modeling_path,
        "protocol_lock_yaml": protocol_lock_path,
        "analysis_plan_yaml": analysis_plan_path,
    }
    for label, path in expected.items():
        actual = sha256_file(path)
        if actual != str(lock["locked_input_sha256"].get(label, "")):
            raise ValueError(f"Prospective selection-lock input mismatch: {label}")

    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(dataset_config_path)
    development_root = _resolve(root, modeling["development_output_root"])
    for retention in RETENTIONS:
        condition = lock["selections"][retention]
        path = development_root / "epinions" / retention / "framework_selection.json"
        if sha256_file(path) != condition["framework_selection_sha256"]:
            raise ValueError(f"Framework-selection hash mismatch: {retention}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("grid_runs_sha256") != condition.get("grid_runs_sha256"):
            raise ValueError(f"Recorded grid hash mismatch: {retention}")
        grid_path = path.with_name("grid_runs.csv")
        if sha256_file(grid_path) != condition.get("grid_runs_sha256"):
            raise ValueError(f"Development-grid content mismatch: {retention}")
        for role in ("rmse_selected", "ndcg_selected", "framework_selected"):
            if _config(artifact["selection"][role]) != _config(condition[role]):
                raise ValueError(f"Locked {role} mismatch: {retention}")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--modeling-config", type=Path, default=DEFAULT_MODELING_CONFIG)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--analysis-plan", type=Path, default=DEFAULT_ANALYSIS_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_SELECTION_LOCK)
    parser.add_argument("--locked-on", default=date.today().isoformat())
    args = parser.parse_args()
    print(
        create_prospective_selection_lock(
            dataset_config_path=args.config,
            modeling_path=args.modeling_config,
            protocol_lock_path=args.protocol_lock,
            analysis_plan_path=args.analysis_plan,
            output_path=args.output,
            locked_on=args.locked_on,
        )
    )


if __name__ == "__main__":
    main()
