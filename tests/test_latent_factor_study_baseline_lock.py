from pathlib import Path

import pytest

from src.latent_factor_study.baseline_lock import verify_baseline_selection_lock


def test_missing_baseline_lock_prohibits_verification(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_baseline_selection_lock(
            lock_path=tmp_path / "missing.yaml",
            dataset_config_path=tmp_path / "datasets.yaml",
            modeling_path=tmp_path / "modeling.yaml",
            confirmation_path=tmp_path / "confirmation.yaml",
        )
