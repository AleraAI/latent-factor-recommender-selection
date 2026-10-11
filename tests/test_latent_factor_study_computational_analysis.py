import pandas as pd

from src.latent_factor_study.computational_analysis import aggregate_confirmation_costs


def test_aggregate_confirmation_costs_uses_fresh_seed_distribution() -> None:
    runs = pd.DataFrame(
        {
            "dataset": ["example"] * 5,
            "retention_label": ["r100"] * 5,
            "model": ["explicit_mf"] * 5,
            "selection_role": ["ndcg_selected"] * 5,
            "dimension": [8] * 5,
            "regularization": [0.01] * 5,
            "seed": [1, 2, 3, 4, 5],
            "epochs_completed": [20] * 5,
            "training_seconds": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = aggregate_confirmation_costs(runs).iloc[0]
    assert result["fresh_seed_fits"] == 5
    assert result["median_training_seconds"] == 3.0
    assert result["q25_training_seconds"] == 2.0
    assert result["q75_training_seconds"] == 4.0
