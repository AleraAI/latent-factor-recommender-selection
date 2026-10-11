from __future__ import annotations

import math

import pandas as pd

from src.latent_factor_study.evaluation import (
    build_candidate_set,
    graded_ndcg_at_k,
    prediction_metrics,
    rank_items,
    recall_at_k,
    top_k_items,
)
from src.latent_factor_study.sparsity import thin_training_interactions


def _frame(rows: list[tuple[str, str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "interaction_id": f"x{index}",
                "user_id": user,
                "item_id": item,
                "rating": rating,
                "timestamp": timestamp,
                "source_row_id": index,
            }
            for index, (user, item, rating, timestamp) in enumerate(rows)
        ]
    )


def test_sparsity_is_deterministic_and_preserves_full_training_catalog() -> None:
    train = _frame(
        [
            ("u1", "a", 5, 1), ("u1", "b", 4, 2), ("u1", "c", 3, 3), ("u1", "d", 2, 4),
            ("u2", "a", 5, 1), ("u2", "b", 4, 2), ("u2", "c", 3, 3), ("u2", "e", 2, 4),
        ]
    )
    first, first_audit = thin_training_interactions(
        train, retention=0.25, seed=19, minimum_per_user=1
    )
    second, second_audit = thin_training_interactions(
        train, retention=0.25, seed=19, minimum_per_user=1
    )

    assert first["interaction_id"].tolist() == second["interaction_id"].tolist()
    assert set(first["item_id"]) == set(train["item_id"])
    assert first_audit == second_audit
    assert first_audit["full_training_catalog_items"] == 5


def test_candidate_contract_excludes_removed_history_and_prior_validation() -> None:
    full_train = _frame(
        [("u", "a", 5, 1), ("u", "b", 4, 2), ("v", "c", 5, 1), ("v", "d", 5, 2)]
    )
    sparse_train = full_train[full_train["item_id"] != "b"].copy()
    validation = _frame([("u", "c", 4, 3)])
    test = _frame([("u", "d", 5, 4), ("u", "z", 5, 5)])

    candidates = build_candidate_set(
        user_id="u",
        sparse_train=sparse_train,
        full_train=full_train,
        evaluation=test,
        prior_evaluation=validation,
    )

    assert candidates.item_ids == ("d",)
    assert candidates.relevant_gains == {"d": 3.0}
    assert candidates.cold_relevant_items == frozenset({"z"})
    assert {"a", "b", "c"}.issubset(candidates.excluded_historical_items)


def test_graded_metrics_and_deterministic_ties_match_hand_calculation() -> None:
    ranked = rank_items(["b", "a", "c"], [0.5, 0.5, 0.1])
    assert ranked == ["a", "b", "c"]
    gains = {"a": 1.0, "c": 3.0}

    actual = graded_ndcg_at_k(ranked, gains, 3)
    expected_dcg = 1.0 + 3.0 / math.log2(4)
    ideal_dcg = 3.0 + 1.0 / math.log2(3)
    assert math.isclose(actual, expected_dcg / ideal_dcg)
    assert recall_at_k(ranked, gains, 2) == 0.5


def test_exact_top_k_resolves_cutoff_ties_by_item_id() -> None:
    assert top_k_items(["d", "c", "b", "a"], [1.0, 1.0, 1.0, 2.0], 2) == ["a", "b"]
    assert top_k_items(["d", "c"], [float("-inf"), 1.0], 2) == ["c"]


def test_exact_top_k_accepts_an_explicit_deterministic_tie_order() -> None:
    assert top_k_items(
        ["a", "b", "c"],
        [1.0, 1.0, 1.0],
        2,
        tie_keys=["3", "1", "2"],
    ) == ["b", "c"]


def test_prediction_metrics_clip_only_when_requested() -> None:
    clipped = prediction_metrics([1.0, 5.0], [-1.0, 7.0])
    raw = prediction_metrics([1.0, 5.0], [-1.0, 7.0], clip_min=None, clip_max=None)
    assert clipped == {"rmse": 0.0, "mae": 0.0}
    assert raw == {"rmse": 2.0, "mae": 2.0}
