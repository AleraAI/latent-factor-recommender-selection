"""Regression checks for the new, validation-only user-resampling diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts/latent_factor/revision_user_resampling.py"
spec = importlib.util.spec_from_file_location("revision_user_resampling", MODULE_PATH)
revision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revision)


def test_global_rmse_uses_events_before_root_and_seeds_after_root():
    weights = np.array([[1., 1.], [2., 0.]])
    counts = np.array([1., 9.])
    rmse_u = np.array([[[1., 3.], [2., 4.]]])
    sse = counts * rmse_u ** 2
    ndcg = np.array([[[.5, 0.], [.25, 0.]]])
    ranking, rmse, nr, npred = revision.aggregate_users(weights, ndcg, sse, counts, np.array([1, 0]))
    np.testing.assert_allclose(rmse[0, 0], np.sqrt([8.2, 14.8]))
    np.testing.assert_allclose(rmse[1, 0], [1., 2.])
    np.testing.assert_allclose(ranking[:, 0], [[.5, .25], [.5, .25]])
    np.testing.assert_array_equal(nr, [1, 2])
    np.testing.assert_array_equal(npred, [10, 2])


def test_no_rank_eligible_draw_fails():
    with pytest.raises(ValueError, match="denominators"):
        revision.aggregate_users(np.array([[0., 2.]]), np.zeros((1, 2, 2)), np.ones((1, 2, 2)), np.ones(2), np.array([1, 0]))


def test_mean_ablation_preserves_fit_time_tiebreak_and_ua_changes_only_screen():
    configs = ((1, 0.), (2, .1), (4, 0.))
    ndcg = np.array([[.1, .1, .1, .1, .1], [.09, .11, .09, .09, .11], [.096]*5])
    rmse = np.array([[1.2]*5, [1.]*5, [1.]*5])
    result = revision.select_arrays(ndcg, rmse, np.array([1., .5, 2.]), margins=[.005], configs=configs)
    mean = next(r for r in result if r["selector"] == "mean")
    ua = next(r for r in result if r["selector"] == "UA")
    assert mean["selected_index"] == 1  # lower fit time, not smaller dimension alone
    assert ua["selected_index"] == 2  # seed uncertainty excludes config 1
    assert ua["reference_index"] == mean["reference_index"] == 0


def test_finite_and_shape_checks():
    with pytest.raises(ValueError, match="finite"):
        revision.select_arrays(np.array([[np.nan, .1]]), np.ones((1, 2)), np.ones(1), configs=((1, 0.),))
    with pytest.raises(ValueError, match="margins|Margins"):
        revision.select_arrays(np.array([[.1, .1]]), np.ones((1, 2)), np.ones(1), configs=((1, 0.),), margins=[-.1])


def test_baseline_ua_matches_frozen_implementation_on_synthetic_grid():
    from src.latent_factor_study.uncertainty_selector import select_uncertainty_aware_configuration
    rng = np.random.default_rng(41)
    ndcg = rng.uniform(.05, .07, (35, 5))
    rmse = rng.uniform(.9, 1.1, (35, 5))
    times = rng.uniform(1, 20, 35)
    records = []
    for i, (d, reg) in enumerate(revision.CONFIGS):
        for j, seed in enumerate(revision.SEEDS):
            records.append(dict(dimension=d, regularization=reg, seed=seed, training_seconds=times[i],
                                validation_rmse=rmse[i, j], **{"validation_ndcg@10": ndcg[i, j]}))
    historical = select_uncertainty_aware_configuration(pd.DataFrame(records))
    new = next(r for r in revision.select_arrays(ndcg, rmse, times) if r["selector"] == "UA" and r["delta"] == .005)
    choice = historical["framework_selected"]
    assert revision.CONFIGS[new["selected_index"]] == (choice["dimension"], choice["regularization"])
    assert new["admissible_count"] == historical["admissible_configurations"]
    assert abs(new["selected_upper_bound"] - choice["paired_regret_upper_bound"]) < 1e-14
