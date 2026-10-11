import numpy as np

from src.latent_factor_study.inference import (
    bootstrap_mean_difference,
    holm_adjust,
    paired_sign_randomization_p_value,
)


def test_bootstrap_constant_difference_has_degenerate_interval() -> None:
    lower, upper, samples = bootstrap_mean_difference(
        np.ones(20), resamples=100, confidence_level=0.95, seed=1
    )
    assert lower == 1.0
    assert upper == 1.0
    assert np.all(samples == 1.0)


def test_zero_differences_have_unit_randomization_p_value() -> None:
    assert paired_sign_randomization_p_value(
        np.zeros(10), resamples=100, seed=1
    ) == 1.0


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
