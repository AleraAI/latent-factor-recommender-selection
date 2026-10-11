import pandas as pd

from src.latent_factor_study.grid import select_configurations


def test_selection_uses_seed_means_and_declared_tie_breaks() -> None:
    rows = pd.DataFrame(
        [
            {"dimension": 1, "regularization": 0.0, "seed": 1, "validation_rmse": 1.0, "validation_ndcg@10": 0.10},
            {"dimension": 1, "regularization": 0.0, "seed": 2, "validation_rmse": 1.2, "validation_ndcg@10": 0.20},
            {"dimension": 2, "regularization": 0.1, "seed": 1, "validation_rmse": 1.1, "validation_ndcg@10": 0.30},
            {"dimension": 2, "regularization": 0.1, "seed": 2, "validation_rmse": 1.1, "validation_ndcg@10": 0.40},
        ]
    )

    selected = select_configurations(rows)

    # RMSE means tie at 1.1, so smaller dimension wins.
    assert selected["rmse_selected"]["dimension"] == 1
    assert selected["ndcg_selected"]["dimension"] == 2
    assert not selected["same_configuration"]
