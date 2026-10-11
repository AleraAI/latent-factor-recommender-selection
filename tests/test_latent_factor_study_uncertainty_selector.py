from __future__ import annotations

import pandas as pd

from src.latent_factor_study.uncertainty_selector import (
    select_uncertainty_aware_configuration,
)


def _rows() -> pd.DataFrame:
    records = []
    # The RMSE winner has stable ranking regret below 0.005 and should be used.
    for seed, reference, proxy in zip(
        [1, 2, 3, 4, 5],
        [0.100, 0.101, 0.099, 0.100, 0.100],
        [0.097, 0.098, 0.096, 0.097, 0.097],
    ):
        records.extend(
            [
                {
                    "dimension": 16,
                    "regularization": 0.1,
                    "seed": seed,
                    "training_seconds": 1.0,
                    "validation_rmse": 0.90,
                    "validation_ndcg@10": proxy,
                },
                {
                    "dimension": 32,
                    "regularization": 0.1,
                    "seed": seed,
                    "training_seconds": 2.0,
                    "validation_rmse": 0.95,
                    "validation_ndcg@10": reference,
                },
                {
                    "dimension": 64,
                    "regularization": 0.01,
                    "seed": seed,
                    "training_seconds": 3.0,
                    "validation_rmse": 0.89,
                    "validation_ndcg@10": reference - 0.02,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def test_framework_chooses_lowest_rmse_configuration_inside_bound() -> None:
    result = select_uncertainty_aware_configuration(_rows())

    assert result["rmse_selected"]["dimension"] == 64
    assert result["ndcg_selected"]["dimension"] == 32
    assert result["framework_selected"]["dimension"] == 16
    assert not result["framework_matches_rmse"]
    assert not result["framework_matches_ndcg"]
    assert result["framework_selected"]["paired_regret_upper_bound"] < 0.005


def test_ranking_reference_is_always_admissible() -> None:
    result = select_uncertainty_aware_configuration(_rows())
    reference = next(
        row
        for row in result["configuration_table"]
        if row["dimension"] == 32 and row["regularization"] == 0.1
    )

    assert reference["admissible"]
    assert reference["paired_regret_upper_bound"] == 0.0

