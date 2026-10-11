import numpy as np
import pandas as pd
import pytest

from src.latent_factor_study.major_revision_selection import (
    bootstrap_metrics, constrained_choice, pareto_flags, validate_grid,
)


def test_mean_ablation_preserves_fit_time_tie():
    table = pd.DataFrame([
        dict(dimension=1, regularization=.1, mean_validation_rmse=1., mean_training_seconds=20.,
             paired_mean_ndcg_at_10_regret=.002, paired_regret_upper_bound=.004),
        dict(dimension=8, regularization=.01, mean_validation_rmse=1., mean_training_seconds=10.,
             paired_mean_ndcg_at_10_regret=.002, paired_regret_upper_bound=.006),
    ])
    assert constrained_choice(table, .005, False).dimension == 8
    assert constrained_choice(table, .005, True).dimension == 1
    assert constrained_choice(table, .010, True).dimension == 8


def test_missing_nonfinite_and_duplicate_grid_rejected():
    rows = pd.DataFrame([dict(dimension=d, regularization=l, seed=s, training_seconds=1.,
                             validation_rmse=1., **{"validation_ndcg@10": .1})
                         for d in [1, 2, 4, 8, 16, 32, 64]
                         for l in [0., .0001, .001, .01, .1]
                         for s in [101, 211, 307, 401, 503]])
    validate_grid(rows)
    with pytest.raises(ValueError):
        validate_grid(rows.iloc[:-1])
    duplicate = rows.copy()
    duplicate.iloc[-1] = duplicate.iloc[0]
    with pytest.raises(ValueError):
        validate_grid(duplicate)
    rows.loc[0, "validation_rmse"] = np.nan
    with pytest.raises(ValueError):
        validate_grid(rows)


def test_pareto_retains_equal_points_and_rejects_dominated():
    table = pd.DataFrame({"mean_validation_rmse": [1., 1., 2., .5],
                          "mean_validation_ndcg@10": [.1, .1, .05, .01]})
    assert pareto_flags(table).tolist() == [True, True, False, True]


def test_global_rmse_not_mean_user_and_not_ensemble():
    frame = pd.DataFrame({"user_id": ["a", "b"], "num_prediction_rows": [1, 9],
                          "num_candidates": [10, 10], "num_relevant": [1, 1],
                          "num_cold_relevant": [0, 0], "prediction_rmse": [1., 3.],
                          "ndcg@10": [.2, .4]})
    frames = [[frame.copy() for _ in range(5)], [frame.copy() for _ in range(5)]]
    frames[0][0]["prediction_rmse"] = [3., 1.]
    point, samples = bootstrap_metrics(frames, resamples=20, seed=3)
    expected = (np.sqrt(1 * 9 / 10 + 9 * 1 / 10) + 4 * np.sqrt(1 * 1 / 10 + 9 * 9 / 10)) / 5
    assert point["global_rmse"][0] == pytest.approx(expected)
    assert point["mean_user_rmse"][0] == pytest.approx(2.)
    assert point["global_rmse"][0] != point["mean_user_rmse"][0]
    assert samples["global_rmse"].shape == (20, 2)


def test_unsupported_partial_seed_group_rejected():
    frame = pd.DataFrame({"user_id": ["a"], "num_prediction_rows": [1],
                          "num_candidates": [10], "num_relevant": [1],
                          "num_cold_relevant": [0], "prediction_rmse": [1.], "ndcg@10": [.2]})
    with pytest.raises(ValueError, match="five seed"):
        bootstrap_metrics([[frame] * 4], resamples=2, seed=1)


def test_identical_selected_configuration_has_exact_zero_contrasts():
    frame = pd.DataFrame({"user_id": ["a", "b"], "num_prediction_rows": [2, 5],
                          "num_candidates": [10, 10], "num_relevant": [1, 1],
                          "num_cold_relevant": [0, 0], "prediction_rmse": [.5, 1.2], "ndcg@10": [.2, .5]})
    point, draws = bootstrap_metrics([[frame] * 5, [frame] * 5], resamples=10, seed=1)
    for key in point:
        assert point[key][0] == point[key][1]
        assert np.array_equal(draws[key][:, 0], draws[key][:, 1])
