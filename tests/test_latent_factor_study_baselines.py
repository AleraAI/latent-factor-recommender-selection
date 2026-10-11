from __future__ import annotations

import pandas as pd

from src.latent_factor_study.baselines import (
    evaluate_popularity_stage,
    popularity_scores,
)


def test_popularity_uses_retained_positive_counts_and_exact_history() -> None:
    train = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rating": 5.0},
            {"user_id": "u1", "item_id": "i2", "rating": 2.0},
            {"user_id": "u2", "item_id": "i1", "rating": 4.0},
            {"user_id": "u2", "item_id": "i3", "rating": 5.0},
            {"user_id": "u3", "item_id": "i3", "rating": 3.0},
        ]
    )
    validation = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i3", "rating": 5.0},
            {"user_id": "u2", "item_id": "i2", "rating": 5.0},
            {"user_id": "u3", "item_id": "i2", "rating": 5.0},
        ]
    )
    items, scores = popularity_scores(train)
    assert dict(zip(items, scores)) == {"i1": 2.0, "i2": 0.0, "i3": 1.0}
    summary, per_user = evaluate_popularity_stage(
        sparse_train=train,
        full_train=train,
        evaluation=validation,
    )
    assert summary["users_total"] == 3
    assert summary["users_evaluable"] == 3
    assert per_user.set_index("user_id")["num_candidates"].to_dict() == {
        "u1": 1,
        "u2": 1,
        "u3": 2,
    }
