"""Exact-catalog evaluation contracts and metrics for the journal study."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateSet:
    user_id: str
    item_ids: tuple[str, ...]
    relevant_gains: Mapping[str, float]
    excluded_historical_items: frozenset[str]
    cold_relevant_items: frozenset[str]


def rating_gain(rating: float, *, threshold: float = 4.0) -> float:
    if rating < threshold:
        return 0.0
    if rating >= 5.0:
        return 3.0
    return 1.0


def build_candidate_set(
    *,
    user_id: str,
    sparse_train: pd.DataFrame,
    full_train: pd.DataFrame,
    evaluation: pd.DataFrame,
    prior_evaluation: pd.DataFrame | None = None,
    relevance_threshold: float = 4.0,
) -> CandidateSet:
    """Build a train-known exact candidate catalog for one user.

    All full-training interactions are excluded even when an interaction was
    removed to create a sparse condition.  At test time, pass validation rows
    as ``prior_evaluation`` so they are excluded as observed history too.
    Relevant events whose items never occur in the sparse training catalog are
    reported as cold rather than inserted with untrained model parameters.
    """
    uid = str(user_id)
    catalog = set(sparse_train["item_id"].astype(str))
    historical = set(
        full_train.loc[full_train["user_id"].astype(str) == uid, "item_id"].astype(str)
    )
    if prior_evaluation is not None:
        historical.update(
            prior_evaluation.loc[
                prior_evaluation["user_id"].astype(str) == uid, "item_id"
            ].astype(str)
        )
    candidates = tuple(sorted(catalog - historical))
    candidate_lookup = set(candidates)

    user_eval = evaluation[evaluation["user_id"].astype(str) == uid]
    relevant = user_eval[user_eval["rating"] >= relevance_threshold]
    gains = {
        str(row.item_id): rating_gain(float(row.rating), threshold=relevance_threshold)
        for row in relevant.itertuples(index=False)
        if str(row.item_id) in candidate_lookup
    }
    cold = frozenset(
        str(item_id)
        for item_id in relevant["item_id"].astype(str)
        if str(item_id) not in catalog
    )
    return CandidateSet(
        user_id=uid,
        item_ids=candidates,
        relevant_gains=gains,
        excluded_historical_items=frozenset(historical),
        cold_relevant_items=cold,
    )


def rank_items(
    item_ids: Sequence[str], scores: Sequence[float]
) -> list[str]:
    """Rank descending by score and ascending by item ID on exact ties."""
    if len(item_ids) != len(scores):
        raise ValueError("item_ids and scores must have equal length.")
    if not np.isfinite(np.asarray(scores, dtype=float)).all():
        raise ValueError("Ranking scores must be finite.")
    return [
        item_id
        for item_id, _ in sorted(
            zip((str(item) for item in item_ids), (float(score) for score in scores)),
            key=lambda pair: (-pair[1], pair[0]),
        )
    ]


def top_k_items(
    item_ids: Sequence[str],
    scores: Sequence[float],
    k: int,
    *,
    tie_keys: Sequence[str] | None = None,
) -> list[str]:
    """Return the exact deterministic top-k without sorting the full catalog."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if isinstance(item_ids, np.ndarray) and item_ids.dtype.kind in {"U", "S"}:
        items = item_ids
    else:
        items = np.asarray([str(item) for item in item_ids], dtype=str)
    values = np.asarray(scores)
    if items.shape != values.shape or items.ndim != 1:
        raise ValueError("item_ids and scores must be equal-length vectors.")
    if tie_keys is None:
        keys = items
    else:
        keys = np.asarray([str(value) for value in tie_keys], dtype=str)
        if keys.shape != items.shape:
            raise ValueError("tie_keys must have the same shape as item_ids.")
    eligible = np.flatnonzero(np.isfinite(values))
    if len(eligible) <= k:
        selected = eligible
    else:
        eligible_scores = values[eligible]
        threshold = np.partition(eligible_scores, len(eligible_scores) - k)[
            len(eligible_scores) - k
        ]
        strict = eligible[eligible_scores > threshold]
        tied = eligible[eligible_scores == threshold]
        needed = k - len(strict)
        tied_order = np.argsort(keys[tied], kind="stable")[:needed]
        selected = np.concatenate([strict, tied[tied_order]])
    order = np.lexsort((keys[selected], -values[selected]))
    return items[selected][order].tolist()


def graded_dcg_at_k(
    ranked_items: Sequence[str], gains: Mapping[str, float], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    return sum(
        float(gains.get(str(item_id), 0.0)) / math.log2(rank + 1.0)
        for rank, item_id in enumerate(ranked_items[:k], start=1)
    )


def graded_ndcg_at_k(
    ranked_items: Sequence[str], gains: Mapping[str, float], k: int
) -> float:
    positive = sorted((float(gain) for gain in gains.values() if gain > 0.0), reverse=True)
    if not positive:
        return float("nan")
    ideal = sum(gain / math.log2(rank + 1.0) for rank, gain in enumerate(positive[:k], start=1))
    return graded_dcg_at_k(ranked_items, gains, k) / ideal


def recall_at_k(
    ranked_items: Sequence[str], gains: Mapping[str, float], k: int
) -> float:
    relevant = {str(item_id) for item_id, gain in gains.items() if gain > 0.0}
    if not relevant:
        return float("nan")
    return len(relevant.intersection(str(item) for item in ranked_items[:k])) / len(relevant)


def ranking_metrics(
    ranked_items: Sequence[str], gains: Mapping[str, float], ks: Sequence[int] = (10, 20)
) -> dict[str, float]:
    values: dict[str, float] = {}
    for k in ks:
        values[f"ndcg@{k}"] = graded_ndcg_at_k(ranked_items, gains, k)
        values[f"recall@{k}"] = recall_at_k(ranked_items, gains, k)
    return values


def prediction_metrics(
    ratings: Sequence[float],
    predictions: Sequence[float],
    *,
    clip_min: float | None = 1.0,
    clip_max: float | None = 5.0,
) -> dict[str, float]:
    truth = np.asarray(ratings, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("ratings and predictions must be equal-length vectors.")
    if truth.size == 0:
        raise ValueError("Prediction metrics require at least one observation.")
    if clip_min is not None or clip_max is not None:
        predicted = np.clip(predicted, clip_min, clip_max)
    errors = predicted - truth
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }
