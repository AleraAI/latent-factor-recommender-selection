from __future__ import annotations

import json
from pathlib import Path

from src.latent_factor_study.baseline_confirmation import _bias_run_matches
from src.latent_factor_study.explicit_mf import MFRunConfig


def test_bias_confirmation_resume_requires_matching_roles(tmp_path: Path) -> None:
    config = MFRunConfig(
        dimension=0,
        regularization=0.1,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=2,
        convergence_patience=5,
        convergence_relative_tolerance=1e-5,
        seed=17,
    )
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "stage": "locked_baseline_test_confirmation",
                "model": "bias_only",
                "selection_roles": ["ndcg_selected"],
                "run_config": {
                    "dimension": 0,
                    "regularization": 0.1,
                    "learning_rate": 0.01,
                    "batch_size": 16,
                    "max_epochs": 2,
                    "convergence_patience": 5,
                    "convergence_relative_tolerance": 1e-5,
                    "seed": 17,
                    "device": "cpu",
                    "torch_num_threads": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_per_user.parquet").touch()
    assert _bias_run_matches(path, config, ["ndcg_selected"])
    assert not _bias_run_matches(path, config, ["rmse_selected"])
