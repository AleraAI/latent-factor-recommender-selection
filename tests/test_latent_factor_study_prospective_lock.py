from __future__ import annotations

from src.latent_factor_study.prospective_lock import _config


def test_prospective_lock_normalizes_configurations() -> None:
    assert _config({"dimension": "16", "regularization": "0.01", "extra": 1}) == {
        "dimension": 16,
        "regularization": 0.01,
    }

