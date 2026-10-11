import math

import pandas as pd

from src.latent_factor_study.development_analysis import configuration_agreement


def test_configuration_agreement_reports_perfect_alignment() -> None:
    rows = pd.DataFrame(
        [
            {"dimension": 1, "regularization": 0.0, "seed": 1, "validation_rmse": 1.0, "validation_ndcg@10": 0.3},
            {"dimension": 1, "regularization": 0.0, "seed": 2, "validation_rmse": 1.0, "validation_ndcg@10": 0.3},
            {"dimension": 2, "regularization": 0.0, "seed": 1, "validation_rmse": 1.1, "validation_ndcg@10": 0.2},
            {"dimension": 2, "regularization": 0.0, "seed": 2, "validation_rmse": 1.1, "validation_ndcg@10": 0.2},
            {"dimension": 4, "regularization": 0.0, "seed": 1, "validation_rmse": 1.2, "validation_ndcg@10": 0.1},
            {"dimension": 4, "regularization": 0.0, "seed": 2, "validation_rmse": 1.2, "validation_ndcg@10": 0.1},
        ]
    )
    result = configuration_agreement(rows)
    assert math.isclose(result["spearman_rho"], 1.0)
    assert math.isclose(result["kendall_tau_b"], 1.0)
    assert result["top3_overlap"] == 3
    assert result["same_top_configuration"]
    assert result["minimum_seed_count"] == 2
