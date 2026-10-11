from __future__ import annotations

import numpy as np
import pandas as pd

from src.latent_factor_study.bpr_mf import (
    BPRRunConfig,
    evaluate_bpr_validation,
    fit_bpr_mf,
    sample_unobserved_items,
)


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rating": 5.0},
            {"user_id": "u1", "item_id": "i2", "rating": 2.0},
            {"user_id": "u2", "item_id": "i2", "rating": 5.0},
            {"user_id": "u2", "item_id": "i3", "rating": 3.0},
            {"user_id": "u3", "item_id": "i3", "rating": 4.0},
            {"user_id": "u3", "item_id": "i4", "rating": 1.0},
            {"user_id": "u4", "item_id": "i4", "rating": 5.0},
            {"user_id": "u4", "item_id": "i1", "rating": 2.0},
        ]
    )
    validation = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i3", "rating": 5.0},
            {"user_id": "u2", "item_id": "i4", "rating": 5.0},
            {"user_id": "u3", "item_id": "i1", "rating": 5.0},
            {"user_id": "u4", "item_id": "i2", "rating": 5.0},
        ]
    )
    return train, validation


def test_negative_sampler_never_returns_observed_items() -> None:
    observed = np.asarray(
        [[True, True, False, False], [False, True, True, False]], dtype=np.bool_
    )
    users = np.asarray([0, 1] * 1000, dtype=np.int64)
    negatives = sample_unobserved_items(
        rng=np.random.default_rng(17), user_indices=users, observed=observed
    )
    assert not observed[users, negatives].any()


def test_bpr_is_reproducible_and_uses_exact_validation_contract() -> None:
    train, validation = _toy()
    config = BPRRunConfig(
        dimension=3,
        regularization=0.01,
        learning_rate=0.02,
        batch_size=4,
        max_epochs=3,
        negatives_per_positive=1,
        relevance_threshold=4.0,
        seed=19,
        torch_num_threads=1,
    )
    first = fit_bpr_mf(train, config)
    second = fit_bpr_mf(train, config)
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        np.testing.assert_allclose(
            left.detach().cpu().numpy(), right.detach().cpu().numpy()
        )

    summary, per_user = evaluate_bpr_validation(
        first, full_train=train, validation=validation, ranking_batch_size=2
    )
    assert summary["users_total"] == 4
    assert summary["users_evaluable"] == 4
    assert summary["cold_relevant_events"] == 0
    assert set(per_user.columns) == {
        "user_id",
        "num_candidates",
        "num_relevant",
        "num_cold_relevant",
        "ndcg@10",
        "recall@10",
        "ndcg@20",
        "recall@20",
    }
