from pathlib import Path

from src.latent_factor_study.explicit_mf import MFRunConfig
from src.latent_factor_study.five_star_sensitivity import _run_matches


def test_missing_five_star_run_is_not_resumable(tmp_path: Path) -> None:
    config = MFRunConfig(
        dimension=8,
        regularization=0.1,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=2,
        convergence_patience=5,
        convergence_relative_tolerance=1e-5,
        seed=17,
    )
    assert not _run_matches(tmp_path / "summary.json", config, ["ndcg_selected"])
