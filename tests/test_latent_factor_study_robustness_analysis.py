from __future__ import annotations

import pandas as pd

from src.latent_factor_study.robustness_analysis import (
    analyze_bpr_refine,
    analyze_bpr_sampler_confirmation,
    analyze_explicit_confirmation,
    analyze_expanded_frame,
    analyze_repeated_variance,
    selected_repeated_observations,
)


def test_expanded_diagnostics_report_budget_change_boundary_and_convergence() -> None:
    rows = []
    for epoch in (20, 100):
        for dimension, regularization in ((1, 0.0), (4, 0.1), (256, 1.0)):
            for seed in (101, 307):
                rmse = 1.0 + dimension / 1000 + seed / 1e7
                ndcg = 0.1 - dimension / 10000 + seed / 1e8
                if epoch == 100 and dimension == 256:
                    rmse = 0.8 + seed / 1e7
                rows.append(
                    {
                        "dataset": "toy",
                        "retention_label": "r050",
                        "dimension": dimension,
                        "regularization": regularization,
                        "seed": seed,
                        "epoch": epoch,
                        "validation_rmse": rmse,
                        "validation_ndcg@10": ndcg,
                        "convergence_epoch": 50 if seed == 101 else None,
                        "training_seconds": 2.0,
                    }
                )
    selections, budgets, convergence = analyze_expanded_frame(pd.DataFrame(rows))
    rmse_100 = selections[
        (selections["selection_role"] == "rmse_selected")
        & (selections["epoch"] == 100)
    ].iloc[0]
    assert rmse_100.dimension == 256
    assert bool(rmse_100.any_grid_boundary)
    rmse_budget = budgets[budgets["selection_role"] == "rmse_selected"].iloc[0]
    assert bool(rmse_budget.selection_changed_across_budgets)
    assert convergence.iloc[0].convergence_rate == 0.5


def test_repeated_variance_separates_retention_and_initialization_levels() -> None:
    runs = []
    selections = []
    for retention_seed, dimension in ((11, 1), (13, 4), (17, 4)):
        selections.append(
            {
                "dataset": "toy",
                "retention_label": "r025",
                "retention_seed": retention_seed,
                "epoch": 100,
                "rmse_dimension": dimension,
                "rmse_regularization": 0.1,
                "ndcg_dimension": dimension,
                "ndcg_regularization": 0.1,
            }
        )
        for seed, offset in ((101, -0.01), (307, 0.01)):
            runs.append(
                {
                    "dataset": "toy",
                    "retention_label": "r025",
                    "retention_seed": retention_seed,
                    "epoch": 100,
                    "dimension": dimension,
                    "regularization": 0.1,
                    "seed": seed,
                    "validation_rmse": 1.0 + retention_seed / 100 + offset,
                    "validation_ndcg@10": 0.2 + retention_seed / 1000 + offset,
                }
            )
    observations = selected_repeated_observations(
        pd.DataFrame(runs), pd.DataFrame(selections)
    )
    variance = analyze_repeated_variance(observations)
    assert len(observations) == 12
    assert set(variance["selection_role"]) == {"rmse_selected", "ndcg_selected"}
    assert (variance["retention_seed_count"] == 3).all()
    assert (variance["distinct_selected_configurations"] == 2).all()


def test_bpr_refine_diagnostics_select_mean_ndcg_winner() -> None:
    rows = []
    for dimension, value in ((4, 0.20), (16, 0.22), (64, 0.21)):
        for seed, perturbation in ((101, -0.001), (307, 0.001)):
            rows.append(
                {
                    "dataset": "toy",
                    "retention_label": "r100",
                    "dimension": dimension,
                    "regularization": 0.01,
                    "learning_rate": 0.003,
                    "negatives_per_positive": 3,
                    "negative_mode": "retained_unobserved",
                    "epoch": 50,
                    "seed": seed,
                    "evaluation_ndcg@10": value + perturbation,
                    "sampled_full_training_positive_rate": 0.02,
                }
            )
    result = analyze_bpr_refine(pd.DataFrame(rows)).iloc[0]
    assert result.dimension == 16
    assert result.runner_up_gap > 0


def test_post_confirmation_summaries_preserve_directional_estimands() -> None:
    explicit_rows = []
    for role, rmse, ndcg in (
        ("rmse_selected", 0.90, 0.20),
        ("ndcg_selected", 0.92, 0.23),
    ):
        for seed in (1201, 1301):
            explicit_rows.append(
                {
                    "dataset": "toy",
                    "retention_label": "r050",
                    "retention_seed": None,
                    "selection_role": role,
                    "seed": seed,
                    "test_rmse": rmse,
                    "test_ndcg@10": ndcg,
                }
            )
    explicit = analyze_explicit_confirmation(pd.DataFrame(explicit_rows)).iloc[0]
    assert abs(explicit["delta_rank_ndcg@10"] - 0.03) < 1e-12
    assert abs(explicit["delta_prediction_rmse"] - 0.02) < 1e-12

    bpr = analyze_bpr_sampler_confirmation(
        pd.DataFrame(
            [
                {
                    "dataset": "toy",
                    "retention_label": "r050",
                    "negative_mode": mode,
                    "seed": seed,
                    "evaluation_ndcg@10": 0.2,
                    "evaluation_recall@10": 0.3,
                    "sampled_full_training_positive_rate": false_rate,
                }
                for mode, false_rate in (
                    ("retained_unobserved", 0.02),
                    ("full_training_positive_oracle", 0.0),
                )
                for seed in (1201, 1301)
            ]
        )
    )
    assert len(bpr) == 2
    assert bpr.loc[
        bpr["negative_mode"] == "full_training_positive_oracle",
        "mean_sampled_full_positive_rate",
    ].iloc[0] == 0.0
