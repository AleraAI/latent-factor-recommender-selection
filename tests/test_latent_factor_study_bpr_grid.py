from __future__ import annotations

import pandas as pd

from src.latent_factor_study.bpr_grid import select_bpr_configuration


def test_bpr_selection_uses_seed_mean_and_declared_tie_breakers() -> None:
    rows = pd.DataFrame(
        [
            {
                "dimension": dimension,
                "regularization": regularization,
                "seed": seed,
                "validation_ndcg@10": ndcg,
                "validation_recall@10": ndcg + 0.1,
                "training_seconds": 1.0,
            }
            for dimension, regularization, seed, ndcg in [
                (2, 0.01, 1, 0.3),
                (2, 0.01, 2, 0.3),
                (1, 0.01, 1, 0.3),
                (1, 0.01, 2, 0.3),
                (1, 0.1, 1, 0.3),
                (1, 0.1, 2, 0.3),
            ]
        ]
    )
    selected = select_bpr_configuration(rows)["bpr_selected"]
    assert selected["dimension"] == 1
    assert selected["regularization"] == 0.1
    assert selected["selection_seed_count"] == 2
