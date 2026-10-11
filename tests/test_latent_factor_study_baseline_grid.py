from __future__ import annotations

import json
from pathlib import Path

from src.latent_factor_study.baseline_grid import _summary_matches


def test_bias_resume_requires_non_smoke_matching_run(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "stage": "baseline_development_validation_only",
                "model": "bias_only",
                "smoke": False,
                "run_config": {
                    "dimension": 0,
                    "regularization": 0.1,
                    "seed": 7,
                    "max_epochs": 20,
                },
            }
        ),
        encoding="utf-8",
    )
    assert _summary_matches(path, regularization=0.1, seed=7, max_epochs=20)
    assert not _summary_matches(path, regularization=0.01, seed=7, max_epochs=20)
