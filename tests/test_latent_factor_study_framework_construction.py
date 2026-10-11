from __future__ import annotations

from src.latent_factor_study.framework_construction import _reusable_record


def test_reusable_record_requires_identical_configuration_and_seed() -> None:
    manifest = {
        "runs": [
            {
                "dimension": 16,
                "regularization": 0.1,
                "seed": 601,
                "summary_path": "summary.json",
            }
        ]
    }

    assert _reusable_record(
        manifest, {"dimension": 16, "regularization": 0.1}, 601
    ) == manifest["runs"][0]
    assert _reusable_record(
        manifest, {"dimension": 32, "regularization": 0.1}, 601
    ) is None
    assert _reusable_record(
        manifest, {"dimension": 16, "regularization": 0.1}, 701
    ) is None
