#!/usr/bin/env python3
"""Verify every historical and prospective selection lock in the release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_factor_study.baseline_lock import verify_baseline_selection_lock
from src.latent_factor_study.bpr_lock import verify_bpr_selection_lock
from src.latent_factor_study.prospective_lock import verify_prospective_selection_lock
from src.latent_factor_study.selection_lock import verify_selection_lock


def verify_all(root: Path) -> dict[str, dict[str, object]]:
    config = root / "configs/latent_factor_beyond_rmse"
    results = {
        "primary_explicit_mf": verify_selection_lock(
            lock_path=config / "selection_lock.yaml",
            dataset_config_path=config / "datasets.yaml",
            modeling_path=config / "modeling.yaml",
            confirmation_path=config / "confirmation.yaml",
        ),
        "bpr_mf": verify_bpr_selection_lock(
            lock_path=config / "bpr_selection_lock.yaml",
            dataset_config_path=config / "datasets.yaml",
            modeling_path=config / "bpr_modeling.yaml",
            confirmation_path=config / "bpr_confirmation.yaml",
        ),
        "bias_popularity": verify_baseline_selection_lock(
            lock_path=config / "baseline_selection_lock.yaml",
            dataset_config_path=config / "datasets.yaml",
            modeling_path=config / "baseline_modeling.yaml",
            confirmation_path=config / "baseline_confirmation.yaml",
        ),
    }
    prospective_lock = config / "epinions_prospective_selection_lock.yaml"
    if prospective_lock.exists():
        results["prospective_epinions_framework"] = verify_prospective_selection_lock(
            selection_lock_path=prospective_lock,
            dataset_config_path=config / "epinions_prospective_datasets.yaml",
            modeling_path=config / "epinions_prospective_modeling.yaml",
            protocol_lock_path=config / "epinions_prospective_protocol_lock.yaml",
            analysis_plan_path=config / "epinions_prospective_analysis_plan.yaml",
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    args = parser.parse_args()
    result = verify_all(args.project_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
