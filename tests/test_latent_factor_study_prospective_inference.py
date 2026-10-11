from __future__ import annotations

import numpy as np

from src.latent_factor_study.prospective_inference import _comparison


def test_prospective_comparison_returns_paired_bootstrap_distribution() -> None:
    result = _comparison(
        np.array([0.001, 0.002, 0.003, 0.004]),
        resamples=200,
        confidence_level=0.95,
        seed=7,
    )

    assert result["users"] == 4
    assert result["mean"] == 0.0025
    assert result["bootstrap_means"].shape == (200,)
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


def test_one_sided_upper_bound_uses_predeclared_quantile() -> None:
    result = _comparison(
        np.array([-0.001, 0.0, 0.001, 0.002]),
        resamples=500,
        confidence_level=0.95,
        seed=17,
    )

    upper = float(np.quantile(result["bootstrap_means"], 0.95))
    assert upper >= result["mean"]
    assert upper < 0.005
