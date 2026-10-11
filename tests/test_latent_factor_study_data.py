from __future__ import annotations

import pandas as pd
import pytest

from src.latent_factor_study.data import (
    canonicalize_interactions,
    chronological_user_split,
    iterative_k_core,
    validate_split,
)


def _rows_for_user(user_id: str, item_offset: int, count: int) -> list[dict[str, object]]:
    return [
        {
            "user_id": user_id,
            "item_id": str(item_offset + index),
            "rating": float(index % 5 + 1),
            "timestamp": 100 + index,
            "source_row_id": item_offset + index,
        }
        for index in range(count)
    ]


def test_canonicalization_keeps_latest_duplicate_and_stable_tie() -> None:
    interactions = pd.DataFrame(
        [
            {"user_id": "u", "item_id": "i", "rating": 1, "timestamp": 10, "source_row_id": 1},
            {"user_id": "u", "item_id": "i", "rating": 4, "timestamp": 20, "source_row_id": 2},
            {"user_id": "u", "item_id": "i", "rating": 5, "timestamp": 20, "source_row_id": 3},
            {"user_id": "u", "item_id": "j", "rating": 3, "timestamp": 15, "source_row_id": 4},
        ]
    )

    canonical, audit = canonicalize_interactions(interactions, dataset_name="toy")

    assert canonical[["item_id", "rating", "source_row_id"]].to_dict("records") == [
        {"item_id": "j", "rating": 3.0, "source_row_id": 4},
        {"item_id": "i", "rating": 5.0, "source_row_id": 3},
    ]
    assert audit == {"input_rows": 4, "deduplicated_rows": 2, "duplicate_rows_removed": 2}
    assert canonical["interaction_id"].is_unique


def test_iterative_core_repeats_until_both_sides_satisfy_threshold() -> None:
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "i1"},
            {"user_id": "u1", "item_id": "i2"},
            {"user_id": "u2", "item_id": "i1"},
            {"user_id": "u2", "item_id": "i2"},
            {"user_id": "u2", "item_id": "i3"},
            {"user_id": "u3", "item_id": "i2"},
            {"user_id": "u3", "item_id": "i3"},
        ]
    )

    core, audit = iterative_k_core(
        interactions, minimum_user_degree=2, minimum_item_degree=2
    )

    assert set(core["user_id"]) == {"u1", "u2", "u3"}
    assert set(core["item_id"]) == {"i1", "i2", "i3"}
    assert audit[-1]["rows_before"] == audit[-1]["rows_after"]


def test_chronological_split_uses_published_floor_boundaries() -> None:
    raw = pd.DataFrame(_rows_for_user("u10", 0, 10) + _rows_for_user("u11", 100, 11))
    canonical, _ = canonicalize_interactions(raw, dataset_name="toy")

    split = chronological_user_split(canonical)

    counts = split.groupby(["user_id", "split"]).size().to_dict()
    assert counts == {
        ("u10", "test"): 2,
        ("u10", "train"): 7,
        ("u10", "validation"): 1,
        ("u11", "test"): 2,
        ("u11", "train"): 7,
        ("u11", "validation"): 2,
    }
    validate_split(split)


def test_split_validator_detects_temporal_leakage() -> None:
    raw = pd.DataFrame(_rows_for_user("u", 0, 10))
    canonical, _ = canonicalize_interactions(raw, dataset_name="toy")
    split = chronological_user_split(canonical)
    train_index = split.index[split["split"] == "train"][-1]
    split.loc[train_index, "timestamp"] = 10_000

    with pytest.raises(ValueError, match="temporal leakage"):
        validate_split(split)
