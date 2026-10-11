from __future__ import annotations

import json
from pathlib import Path

from src.latent_factor_study.bpr_confirmation import _run_matches
from src.latent_factor_study.bpr_mf import BPRRunConfig


def test_bpr_confirmation_resume_requires_matching_config_and_per_user_file(
    tmp_path: Path,
) -> None:
    config = BPRRunConfig(
        dimension=8,
        regularization=0.01,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=2,
        negatives_per_positive=1,
        relevance_threshold=4.0,
        seed=17,
    )
    summary = tmp_path / "run" / "summary.json"
    summary.parent.mkdir()
    summary.write_text(
        json.dumps(
            {
                "stage": "locked_bpr_test_confirmation",
                "selection_role": "bpr_selected",
                "run_config": {
                    "dimension": 8,
                    "regularization": 0.01,
                    "learning_rate": 0.01,
                    "batch_size": 16,
                    "max_epochs": 2,
                    "negatives_per_positive": 1,
                    "relevance_threshold": 4.0,
                    "seed": 17,
                    "device": "cpu",
                    "torch_num_threads": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    assert not _run_matches(summary, config)
    (summary.parent / "test_per_user.parquet").touch()
    assert _run_matches(summary, config)
