from __future__ import annotations

import numpy as np
import pandas as pd

from src.latent_factor_study.robustness_bpr import (
    EXPLICIT_MODE,
    ORACLE_MODE,
    PRIMARY_MODE,
    _sample_bpr_negatives,
    sample_uniform_unblocked,
    train_bpr_checkpoints,
)


def _toy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_train = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1", "rating": 5.0},
            {"user_id": "u1", "item_id": "i2", "rating": 2.0},
            {"user_id": "u2", "item_id": "i2", "rating": 5.0},
            {"user_id": "u2", "item_id": "i3", "rating": 2.0},
            {"user_id": "u3", "item_id": "i3", "rating": 5.0},
            {"user_id": "u3", "item_id": "i4", "rating": 1.0},
            # These positives are absent from the retained sample and therefore
            # distinguish the primary and oracle negative pools.
            {"user_id": "u1", "item_id": "i4", "rating": 4.0},
            {"user_id": "u2", "item_id": "i1", "rating": 4.0},
            {"user_id": "u3", "item_id": "i2", "rating": 4.0},
        ]
    )
    sparse_train = full_train.iloc[:6].copy()
    validation = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i3", "rating": 5.0},
            {"user_id": "u2", "item_id": "i4", "rating": 5.0},
            {"user_id": "u3", "item_id": "i1", "rating": 5.0},
        ]
    )
    return sparse_train, full_train, validation


def test_uniform_multi_negative_sampler_never_returns_blocked_items() -> None:
    blocked = np.asarray(
        [[True, True, False, False], [False, True, True, False]], dtype=np.bool_
    )
    users = np.asarray([0, 1] * 1000, dtype=np.int64)
    sampled = sample_uniform_unblocked(
        rng=np.random.default_rng(17), users=users, blocked=blocked, count=5
    )
    assert sampled.shape == (len(users), 5)
    assert not blocked[np.repeat(users, 5), sampled.reshape(-1)].any()


def test_oracle_and_explicit_negative_pools_obey_their_contracts() -> None:
    retained = np.asarray(
        [[True, True, False, False], [False, True, True, False]], dtype=np.bool_
    )
    oracle = retained.copy()
    oracle[0, 3] = True
    oracle[1, 0] = True
    users = np.asarray([0, 1] * 100, dtype=np.int64)

    sampled, fallback = _sample_bpr_negatives(
        rng=np.random.default_rng(19),
        users=users,
        retained_observed=retained,
        oracle_blocked=oracle,
        explicit_lists={},
        mode=ORACLE_MODE,
        count=3,
    )
    assert fallback == 0
    assert not oracle[np.repeat(users, 3), sampled.reshape(-1)].any()

    explicit, fallback = _sample_bpr_negatives(
        rng=np.random.default_rng(23),
        users=np.asarray([0, 1], dtype=np.int64),
        retained_observed=retained,
        oracle_blocked=oracle,
        explicit_lists={0: np.asarray([1]), 1: np.asarray([2])},
        mode=EXPLICIT_MODE,
        count=4,
    )
    np.testing.assert_array_equal(explicit[0], np.asarray([1, 1, 1, 1]))
    np.testing.assert_array_equal(explicit[1], np.asarray([2, 2, 2, 2]))
    assert fallback == 0


def test_checkpoint_trainer_supports_all_negative_modes_and_three_negatives() -> None:
    sparse_train, full_train, validation = _toy()
    for mode in (PRIMARY_MODE, ORACLE_MODE, EXPLICIT_MODE):
        result, per_user = train_bpr_checkpoints(
            sparse_train=sparse_train,
            full_train=full_train,
            evaluation=validation,
            prior_evaluation=None,
            dimension=3,
            regularization=0.01,
            learning_rate=0.02,
            batch_size=4,
            negatives_per_positive=3,
            negative_mode=mode,
            checkpoints=[1, 2],
            relevance_threshold=4.0,
            seed=29,
            ranking_batch_size=2,
            device_name="cpu",
            torch_num_threads=1,
        )
        assert set(result["evaluation_checkpoints"]) == {"1", "2"}
        assert result["training"]["epochs_completed"] == 2
        assert per_user is not None
        if mode == ORACLE_MODE:
            assert all(
                row["sampled_full_training_positive_rate"] == 0
                for row in result["training"]["epoch_history"]
            )

