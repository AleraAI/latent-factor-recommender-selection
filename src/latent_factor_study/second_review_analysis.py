"""Submission diagnostics requested by the second manuscript review.

The analyses are descriptive post-confirmation checks.  They do not alter any
selection lock or replace the primary paired inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .baselines import evaluate_popularity_stage, popularity_scores
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve
from .robustness_explicit import DATASETS, RETENTIONS


DISPLAY = {
    "movielens_1m": "MovieLens 1M",
    "ciaodvd": "CiaoDVD",
    "yelp_tampa_restaurants": "Yelp--Tampa",
}
RETENTION_DISPLAY = {"r100": "100\\%", "r075": "75\\%", "r050": "50\\%", "r025": "25\\%"}
TIE_SALTS = tuple(f"popularity-tie-{seed}" for seed in range(20260821, 20260826))


def _condition_frames(
    root: Path, config: dict[str, Any], dataset: str, retention: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed = _resolve(root, config["datasets"][dataset]["output_dir"])
    sparse_root = _resolve(root, config["controlled_sparsity"]["output_root"])
    return (
        pd.read_parquet(sparse_root / dataset / f"train_{retention}.parquet"),
        pd.read_parquet(processed / "train.parquet"),
        pd.read_parquet(processed / "validation.parquet"),
        pd.read_parquet(processed / "test.parquet"),
    )


def _cutoff_tie_crossing_rate(
    *,
    sparse_train: pd.DataFrame,
    full_train: pd.DataFrame,
    validation: pd.DataFrame,
    test_users: pd.DataFrame,
    relevance_threshold: float,
    cutoff: int = 10,
) -> float:
    item_ids, scores = popularity_scores(
        sparse_train, relevance_threshold=relevance_threshold
    )
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    history = pd.concat([full_train, validation], ignore_index=True)
    historical_by_user = (
        history.groupby(history["user_id"].astype(str), sort=False)["item_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    evaluable = set(
        test_users.loc[test_users["num_relevant"] > 0, "user_id"].astype(str)
    )
    crossing = 0
    for user_id in evaluable:
        eligible = scores.copy()
        excluded = historical_by_user.get(user_id, set())
        excluded_indices = [
            item_to_index[item] for item in excluded if item in item_to_index
        ]
        eligible[excluded_indices] = -np.inf
        finite = eligible[np.isfinite(eligible)]
        if len(finite) <= cutoff:
            continue
        threshold = np.partition(finite, len(finite) - cutoff)[len(finite) - cutoff]
        strictly_above = int(np.sum(finite > threshold))
        tied_at_cutoff = int(np.sum(finite == threshold))
        if strictly_above < cutoff and tied_at_cutoff > cutoff - strictly_above:
            crossing += 1
    return crossing / len(evaluable) if evaluable else float("nan")


def analyze_popularity_ties(
    root: Path, config: dict[str, Any], relevance_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            sparse, full_train, validation, test = _condition_frames(
                root, config, dataset, retention
            )
            id_summary, id_users = evaluate_popularity_stage(
                sparse_train=sparse,
                full_train=full_train,
                evaluation=test,
                prior_evaluation=validation,
                relevance_threshold=relevance_threshold,
            )
            paired_path = (
                root
                / "outputs/latent_factor_beyond_rmse/bpr_confirmation/analysis"
                / f"{dataset}_{retention}_paired_users.parquet"
            )
            bpr_users = pd.read_parquet(
                paired_path, columns=["user_id", "ndcg@10_bpr"]
            )
            bpr_users["user_id"] = bpr_users["user_id"].astype(str)
            hash_ndcg: list[float] = []
            hash_bpr_delta: list[float] = []
            for salt in TIE_SALTS:
                summary, users = evaluate_popularity_stage(
                    sparse_train=sparse,
                    full_train=full_train,
                    evaluation=test,
                    prior_evaluation=validation,
                    relevance_threshold=relevance_threshold,
                    tie_salt=salt,
                )
                users["user_id"] = users["user_id"].astype(str)
                paired = bpr_users.merge(
                    users[["user_id", "ndcg@10"]],
                    on="user_id",
                    how="inner",
                    validate="one_to_one",
                )
                mean_delta = float((paired["ndcg@10_bpr"] - paired["ndcg@10"]).mean())
                mean_ndcg = float(summary["ndcg@10"])
                hash_ndcg.append(mean_ndcg)
                hash_bpr_delta.append(mean_delta)
                detail.append(
                    {
                        "dataset": dataset,
                        "retention_label": retention,
                        "tie_salt": salt,
                        "users_evaluable": int(summary["users_evaluable"]),
                        "popularity_mean_ndcg@10": mean_ndcg,
                        "bpr_minus_popularity_mean_ndcg@10": mean_delta,
                    }
                )
            summaries.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "users_evaluable": int(id_summary["users_evaluable"]),
                    "cutoff_tie_crossing_rate": _cutoff_tie_crossing_rate(
                        sparse_train=sparse,
                        full_train=full_train,
                        validation=validation,
                        test_users=id_users,
                        relevance_threshold=relevance_threshold,
                    ),
                    "ascending_id_popularity_ndcg@10": float(id_summary["ndcg@10"]),
                    "hash_popularity_ndcg@10_min": min(hash_ndcg),
                    "hash_popularity_ndcg@10_max": max(hash_ndcg),
                    "hash_popularity_ndcg@10_range": max(hash_ndcg) - min(hash_ndcg),
                    "hash_bpr_minus_popularity_min": min(hash_bpr_delta),
                    "hash_bpr_minus_popularity_max": max(hash_bpr_delta),
                }
            )
    return pd.DataFrame.from_records(summaries), pd.DataFrame.from_records(detail)


def _expected_unique(pool_size: int, draws: int) -> float:
    if pool_size <= 0 or draws <= 0:
        return 0.0
    if pool_size == 1:
        return 1.0
    return pool_size * (1.0 - math.exp(draws * math.log1p(-1.0 / pool_size)))


def analyze_negative_pools(
    root: Path, config: dict[str, Any], relevance_threshold: float
) -> pd.DataFrame:
    selections = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/robustness/analysis"
        / "bpr_independent_selection_diagnostics.csv"
    ).set_index(["dataset", "retention_label"])
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            sparse, _, _, _ = _condition_frames(root, config, dataset, retention)
            sparse = sparse.assign(
                user_id=sparse["user_id"].astype(str),
                item_id=sparse["item_id"].astype(str),
            )
            positive = sparse[sparse["rating"] >= relevance_threshold]
            explicit = sparse[sparse["rating"] < relevance_threshold]
            positive_counts = positive.groupby("user_id").size()
            pool_sizes = explicit.groupby("user_id")["item_id"].nunique()
            positive_users = positive_counts.index
            sizes_all = pool_sizes.reindex(positive_users, fill_value=0).astype(int)
            fallback = sizes_all == 0
            selection = selections.loc[(dataset, retention)]
            negatives = int(selection["negatives_per_positive"])
            epochs = int(selection["epoch"])
            explicit_users = positive_users[~fallback]
            expected_unique = [
                _expected_unique(
                    int(sizes_all.loc[user]),
                    int(positive_counts.loc[user]) * negatives * epochs,
                )
                for user in explicit_users
            ]
            fallback_triplets = int(positive_counts.reindex(positive_users[fallback]).sum())
            total_positive = int(positive_counts.sum())
            eligible_sizes = sizes_all[~fallback]
            rows.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "positive_users": int(len(positive_users)),
                    "explicit_negative_users": int((~fallback).sum()),
                    "fallback_users": int(fallback.sum()),
                    "fallback_user_rate": float(fallback.mean()),
                    "fallback_triplet_rate": fallback_triplets / total_positive,
                    "explicit_pool_median": float(eligible_sizes.median()),
                    "explicit_pool_q05": float(eligible_sizes.quantile(0.05)),
                    "explicit_pool_q95": float(eligible_sizes.quantile(0.95)),
                    "expected_unique_sampled_median": float(np.median(expected_unique)),
                    "negatives_per_positive": negatives,
                    "epochs": epochs,
                    "sampling_with_replacement": True,
                }
            )
    return pd.DataFrame.from_records(rows)


def analyze_robustness_search_cost(root: Path) -> pd.DataFrame:
    candidate_work = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/revision_analysis/selection_cost.csv"
    )
    candidate_work = candidate_work[candidate_work["family"] == "explicit_mf"].set_index(
        ["dataset", "retention_label"]
    )
    records: list[dict[str, Any]] = []
    for stage, label in (
        ("explicit_expanded", "Expanded explicit MF"),
        ("bpr_coarse", "BPR coarse"),
        ("bpr_refine", "BPR refinement"),
    ):
        checkpoint_rows = 0
        fits = 0
        fit_seconds = 0.0
        candidate_scores = 0
        for path in sorted(
            (root / "outputs/latent_factor_beyond_rmse/robustness" / stage).glob(
                "*/*/runs.csv"
            )
        ):
            frame = pd.read_csv(path)
            checkpoint_rows += len(frame)
            max_epoch = frame.groupby(
                [
                    "dataset",
                    "retention_label",
                    "dimension",
                    "regularization",
                    "seed",
                    *(
                        ["learning_rate", "negatives_per_positive", "negative_mode"]
                        if stage.startswith("bpr_")
                        else []
                    ),
                ],
                dropna=False,
            )["epoch"].transform("max")
            completed = frame[frame["epoch"] == max_epoch]
            fits += len(completed)
            fit_seconds += float(completed["training_seconds"].sum())
            dataset = str(frame.iloc[0]["dataset"])
            retention = str(frame.iloc[0]["retention_label"])
            per_evaluation = int(
                candidate_work.loc[(dataset, retention), "validation_candidate_scores_per_run"]
            )
            candidate_scores += len(frame) * per_evaluation
        records.append(
            {
                "stage": stage,
                "display": label,
                "fits": fits,
                "checkpoint_evaluations": checkpoint_rows,
                "aggregate_fit_seconds": fit_seconds,
                "aggregate_fit_hours": fit_seconds / 3600.0,
                "validation_candidate_scores": candidate_scores,
            }
        )
    return pd.DataFrame.from_records(records)


def _tex_popularity(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        subset = frame[frame["dataset"] == dataset]
        for index, retention in enumerate(RETENTIONS):
            row = subset[subset["retention_label"] == retention].iloc[0]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention],
                        f"{100*row.cutoff_tie_crossing_rate:.1f}\\%",
                        f"{row['ascending_id_popularity_ndcg@10']:.4f}",
                        f"[{row['hash_popularity_ndcg@10_min']:.4f},{row['hash_popularity_ndcg@10_max']:.4f}]",
                        f"{row['hash_popularity_ndcg@10_range']:.5f}",
                        f"[{row.hash_bpr_minus_popularity_min:+.4f},{row.hash_bpr_minus_popularity_max:+.4f}]",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post-confirmation popularity tie-order sensitivity. Tie@10 is the percentage of ranking-evaluable users for whom an equal-popularity group straddles rank 10. ID gives the original ascending-item-ID result; Hash gives the range across five predeclared SHA-256 item permutations. The final column gives the corresponding range of mean BPR-minus-popularity NDCG@10. This descriptive check preserves the locked primary result and tests whether its deployment boundary depends on source item numbering.}",
            r"\label{tab:popularity-tie-sensitivity}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Tie@10 & ID NDCG & Hash NDCG range & Width & BPR--pop. range \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _tex_negative_pool(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        subset = frame[frame["dataset"] == dataset]
        for index, retention in enumerate(RETENTIONS):
            row = subset[subset["retention_label"] == retention].iloc[0]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention],
                        f"{row.explicit_pool_median:.0f} [{row.explicit_pool_q05:.0f},{row.explicit_pool_q95:.0f}]",
                        f"{100*row.fallback_user_rate:.1f}\\%",
                        f"{100*row.fallback_triplet_rate:.1f}\\%",
                        f"{row.expected_unique_sampled_median:.1f}",
                        str(int(row.negatives_per_positive)),
                    ]
                )
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post-confirmation diagnostics for the fixed-configuration retained-explicit-negative sensitivity. Pool size is the median [5th,95th percentile] number of distinct retained ratings below four among positive-training users with at least one such item. Users without an explicit-negative pool fall back to retained-unobserved sampling. Fallback triplets weights those users by their number of positive training events. Sampling is with replacement; expected unique is the analytic expectation over the selected 100-epoch sampling schedule, not a separately logged realized count.}",
            r"\label{tab:bpr-negative-pool-diagnostics}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Pool med. [$q_{.05},q_{.95}$] & Fallback users & Fallback triplets & Expected unique & $m$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _tex_cost(frame: pd.DataFrame) -> str:
    rows = [
        " & ".join(
            [
                str(row.display),
                f"{int(row.fits):,}",
                f"{int(row.checkpoint_evaluations):,}",
                f"{row.aggregate_fit_hours:.2f}",
                f"{int(row.validation_candidate_scores):,}",
            ]
        )
        + r" \\"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Measured post-confirmation validation-search work. Fit hours sum recorded single-threaded training time and are not elapsed parallel wall time. Checkpoint evaluations count complete validation rankings; candidate scores are exact train-known-catalog score work units. BPR sensitivity and fresh-seed test confirmations are excluded because they are not search opportunities.}",
            r"\label{tab:robustness-search-cost}",
            r"\small",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Stage & Fits & Checkpoint evaluations & Fit h & Candidate scores \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def _tex_secondary_metric(frame: pd.DataFrame, metric: str, title: str, label: str) -> str:
    selected = frame[frame["metric"] == metric]
    rows: list[str] = []
    for dataset in DATASETS:
        subset = selected[selected["dataset"] == dataset]
        for index, retention in enumerate(RETENTIONS):
            row = subset[subset["retention_label"] == retention].iloc[0]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention],
                        f"{int(row.users_paired):,}",
                        f"{row.rmse_selected_mean:.5f}",
                        f"{row.ndcg_selected_mean:.5f}",
                        f"{row.delta:+.5f}",
                        f"[{row.ci_lower:+.5f},{row.ci_upper:+.5f}]",
                        f"{row.holm_p_within_metric:.4f}",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{{title}. Effects are NDCG-selected minus RMSE-selected after averaging five fresh seeds per user. Intervals use 10,000 user-cluster bootstrap resamples; Holm adjustment covers the twelve conditions for this metric.}}",
            rf"\label{{{label}}}",
            r"\scriptsize",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Users & RMSE-selected & NDCG-selected & $\Delta$ & 95\% CI & $p_{\rm Holm}$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _tex_clipping(frame: pd.DataFrame, primary: pd.DataFrame) -> str:
    pivot = frame[frame["metric"].isin(["prediction_rmse", "prediction_raw_rmse"])].set_index(
        ["dataset", "retention_label", "metric"]
    )
    primary = primary.set_index(["dataset", "retention_label"])
    rows: list[str] = []
    for dataset in DATASETS:
        for index, retention in enumerate(RETENTIONS):
            clipped = primary.loc[(dataset, retention)]
            raw = pivot.loc[(dataset, retention, "prediction_raw_rmse")]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention],
                        f"{int(clipped.users_paired_prediction):,}",
                        f"{clipped.delta_mean_user_rmse:+.5f}",
                        f"[{clipped.delta_mean_user_rmse_ci_lower:+.5f},{clipped.delta_mean_user_rmse_ci_upper:+.5f}]",
                        f"{raw.delta:+.5f}",
                        f"[{raw.ci_lower:+.5f},{raw.ci_upper:+.5f}]",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Clipped and raw mean-user prediction-RMSE sensitivity. Positive effects indicate a prediction cost for the NDCG-selected explicit-MF configuration. The clipped columns reproduce the canonical primary paired estimate and interval reported in the main manuscript; the raw columns are the separately versioned secondary sensitivity. Each interval uses 10,000 user-cluster bootstrap resamples after averaging five fresh seeds per user.}",
            r"\label{tab:supp-clipping}",
            r"\scriptsize",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Users & Clipped $\Delta$ & Clipped 95\% CI & Raw $\Delta$ & Raw 95\% CI \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _tex_mae(frame: pd.DataFrame) -> str:
    pivot = frame[frame["metric"].isin(["prediction_mae", "prediction_raw_mae"])].set_index(
        ["dataset", "retention_label", "metric"]
    )
    rows: list[str] = []
    for dataset in DATASETS:
        for index, retention in enumerate(RETENTIONS):
            clipped = pivot.loc[(dataset, retention, "prediction_mae")]
            raw = pivot.loc[(dataset, retention, "prediction_raw_mae")]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention],
                        f"{int(clipped.users_paired):,}",
                        f"{clipped.rmse_selected_mean:.5f}",
                        f"{clipped.ndcg_selected_mean:.5f}",
                        f"{clipped.delta:+.5f}",
                        f"[{clipped.ci_lower:+.5f},{clipped.ci_upper:+.5f}]",
                        f"{raw.delta:+.5f}",
                        f"[{raw.ci_lower:+.5f},{raw.ci_upper:+.5f}]",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Mean-user prediction-MAE sensitivity. Positive effects indicate greater prediction error for the NDCG-selected explicit-MF configuration. Clipped selector means and differences are shown with the raw-prediction difference; all intervals use 10,000 user-cluster bootstrap resamples after averaging five fresh seeds per user.}",
            r"\label{tab:supp-mae}",
            r"\scriptsize",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Users & RMSE-selected & NDCG-selected & Clipped $\Delta$ & Clipped 95\% CI & Raw $\Delta$ & Raw 95\% CI \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def analyze_primary_epochs(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    stages = (
        ("Development", root / "outputs/latent_factor_beyond_rmse/development"),
        ("Confirmation", root / "outputs/latent_factor_beyond_rmse/confirmation"),
    )
    for stage, stage_root in stages:
        for path in stage_root.glob("**/summary.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if stage == "Development" and bool(payload.get("smoke", False)):
                continue
            training = payload["training"]
            configured_max = int(payload["run_config"]["max_epochs"])
            completed = int(training["epochs_completed"])
            records.append(
                {
                    "stage": stage,
                    "dataset": payload["dataset"],
                    "retention_label": payload["retention_label"],
                    "run_id": payload["run_id"],
                    "configured_max_epochs": configured_max,
                    "epochs_completed": completed,
                    "stopped_before_maximum": completed < configured_max,
                    "summary_path": str(path.relative_to(root)),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["stage", "dataset", "retention_label", "run_id"], kind="stable"
    )


def _tex_epoch_audit(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for stage in ("Development", "Confirmation"):
        subset = frame[frame["stage"] == stage]
        counts = subset["epochs_completed"].value_counts().sort_index()
        for index, (epochs, count) in enumerate(counts.items()):
            rows.append(
                " & ".join(
                    [
                        stage if index == 0 else "",
                        f"{int(epochs)}",
                        f"{int(count):,}",
                        f"{100.0 * float(count) / len(subset):.2f}\\%",
                    ]
                )
                + r" \\"
            )
        if stage == "Development":
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Audit of completed epochs in the primary explicit-MF artifacts. The development audit excludes the two validation-only smoke runs. Early stopping used five stale epochs at relative training-objective tolerance $10^{-5}$ under a maximum of 20 epochs.}",
            r"\label{tab:supp-primary-epochs}",
            r"\small",
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            r"Stage & Epochs completed & Runs & Share within stage \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def _tex_bias_only(frame: pd.DataFrame) -> str:
    ranking_rows: list[str] = []
    prediction_rows: list[str] = []
    for dataset in DATASETS:
        subset = frame[frame["dataset"] == dataset].set_index("retention_label")
        for index, retention in enumerate(RETENTIONS):
            row = subset.loc[retention]
            label = DISPLAY[dataset] if index == 0 else ""
            ranking_rows.append(
                " & ".join(
                    [
                        label,
                        RETENTION_DISPLAY[retention],
                        f"{int(row.users_paired_ranking):,}",
                        f"{row['bias_ndcg_selected_mean_ndcg@10']:.5f}",
                        f"{row['explicit_ndcg_selected_mean_ndcg@10']:.5f}",
                        f"{row['delta_explicit_minus_bias_ndcg@10']:+.5f}",
                        f"[{row.delta_bias_ranking_ci_lower:+.5f},{row.delta_bias_ranking_ci_upper:+.5f}]",
                        f"{row.delta_bias_ranking_holm_p:.4f}",
                    ]
                )
                + r" \\"
            )
            prediction_rows.append(
                " & ".join(
                    [
                        label,
                        RETENTION_DISPLAY[retention],
                        f"{int(row.users_paired_prediction):,}",
                        f"{row.bias_rmse_selected_mean_user_rmse:.5f}",
                        f"{row.explicit_rmse_selected_mean_user_rmse:.5f}",
                        f"{row.delta_bias_minus_explicit_mean_user_rmse:+.5f}",
                        f"[{row.delta_prediction_ci_lower:+.5f},{row.delta_prediction_ci_upper:+.5f}]",
                        f"{row.delta_prediction_holm_p:.4f}",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            ranking_rows.append(r"\addlinespace")
            prediction_rows.append(r"\addlinespace")
    common_header = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
    ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Complete bias-only boundary. Panel (a) reports ranking-selected explicit MF minus ranking-selected bias-only NDCG@10, so positive $\Delta$ favors explicit MF. Panel (b) reports bias-only minus explicit-MF mean-user RMSE under RMSE selection, so positive $\Delta$ favors explicit MF. Intervals are paired user-cluster bootstrap intervals and $p_{\rm Holm}$ adjusts the twelve conditions within each comparison family.}",
            r"\label{tab:supp-bias-only}",
            r"\scriptsize",
            r"\textbf{(a) Ranking boundary}\par\smallskip",
            r"\resizebox{\textwidth}{!}{%",
            *common_header,
            r"Dataset & Ret. & Users & Bias NDCG & Explicit NDCG & $\Delta$ & 95\% CI & $p_{\rm Holm}$ \\",
            r"\midrule",
            *ranking_rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\par\medskip\textbf{(b) Prediction boundary}\par\smallskip",
            r"\resizebox{\textwidth}{!}{%",
            *common_header,
            r"Dataset & Ret. & Users & Bias RMSE & Explicit RMSE & $\Delta$ & 95\% CI & $p_{\rm Holm}$ \\",
            r"\midrule",
            *prediction_rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _tex_five_star(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        subset = frame[frame["dataset"] == dataset]
        for index, retention in enumerate(RETENTIONS):
            row = subset[subset["retention_label"] == retention].iloc[0]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention],
                        f"{int(row['users_paired_ranking']):,}",
                        f"{row['rmse_selected_mean_ndcg@10']:.5f}",
                        f"{row['ndcg_selected_mean_ndcg@10']:.5f}",
                        f"{row['delta_rank_ndcg@10']:+.5f}",
                        f"[{row.delta_rank_ci_lower:+.5f},{row.delta_rank_ci_upper:+.5f}]",
                        f"{row.delta_rank_holm_p:.4f}",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Five-star-only post-confirmation relevance sensitivity using the original four-star-selected configurations. Effects, intervals, and Holm-adjusted randomization tests are reported for auditability; this table is not a second confirmatory family.}",
            r"\label{tab:supp-five-star}",
            r"\scriptsize",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Users & RMSE-selected & NDCG-selected & $\Delta$ & 95\% CI & $p_{\rm Holm}$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _tex_activity(frame: pd.DataFrame, dataset: str) -> str:
    subset = frame[frame["dataset"] == dataset]
    rows: list[str] = []
    for retention in RETENTIONS:
        for index, activity in enumerate(("3--9", "10--19", "20--49", "50+")):
            row = subset[
                (subset["retention_label"] == retention)
                & (subset["activity_bin"] == activity)
            ].iloc[0]
            if pd.isna(row["delta_ndcg@10"]):
                effect = ci = "--"
            else:
                effect = f"{row['delta_ndcg@10']:+.5f}"
                ci = (
                    "--"
                    if pd.isna(row.ci_lower)
                    else f"[{row.ci_lower:+.5f},{row.ci_upper:+.5f}]"
                )
            rows.append(
                " & ".join(
                    [
                        RETENTION_DISPLAY[retention] if index == 0 else "",
                        activity.replace("+", r"$+$"),
                        f"{int(row.users_paired):,}",
                        effect,
                        ci,
                    ]
                )
                + r" \\"
            )
        if retention != RETENTIONS[-1]:
            rows.append(r"\addlinespace")
    label = dataset.replace("_", "-")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            rf"\caption{{Exploratory retained-training activity strata for {DISPLAY[dataset]}. Intervals are shown only when at least 30 paired ranking-evaluable users enter the stratum; sample sizes are reported for every cell.}}",
            rf"\label{{tab:supp-activity-{label}}}",
            r"\small",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Retention & Retained interactions & Users & $\Delta$ NDCG@10 & 95\% CI \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def run(
    config_path: Path,
    *,
    output_dir: Path,
    tex_dir: Path,
) -> list[Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(config["protocol"]["primary_relevance_threshold"])
    cached = {
        "popularity": output_dir / "popularity_tie_sensitivity.csv",
        "popularity_detail": output_dir / "popularity_tie_sensitivity_by_permutation.csv",
        "negative": output_dir / "bpr_negative_pool_diagnostics.csv",
        "cost": output_dir / "robustness_search_cost.csv",
    }
    if all(path.exists() for path in cached.values()):
        popularity = pd.read_csv(cached["popularity"])
        popularity_detail = pd.read_csv(cached["popularity_detail"])
        negative = pd.read_csv(cached["negative"])
        cost = pd.read_csv(cached["cost"])
    else:
        popularity, popularity_detail = analyze_popularity_ties(root, config, threshold)
        negative = analyze_negative_pools(root, config, threshold)
        cost = analyze_robustness_search_cost(root)
    outputs: list[Path] = []
    for name, table in (
        ("popularity_tie_sensitivity.csv", popularity),
        ("popularity_tie_sensitivity_by_permutation.csv", popularity_detail),
        ("bpr_negative_pool_diagnostics.csv", negative),
        ("robustness_search_cost.csv", cost),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)
    for name, text in (
        ("popularity_tie_sensitivity.tex", _tex_popularity(popularity)),
        ("bpr_negative_pool_diagnostics.tex", _tex_negative_pool(negative)),
        ("robustness_search_cost.tex", _tex_cost(cost)),
    ):
        path = tex_dir / name
        path.write_text(text, encoding="utf-8")
        outputs.append(path)
    secondary = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis"
        / "secondary_metric_differences.csv"
    )
    primary = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/confirmation/analysis"
        / "mf_selection_regret.csv"
    )
    baseline = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/baseline_confirmation/analysis"
        / "baseline_comparison.csv"
    )
    epoch_audit = analyze_primary_epochs(root)
    epoch_audit_path = output_dir / "primary_epoch_audit.csv"
    epoch_audit.to_csv(epoch_audit_path, index=False)
    outputs.append(epoch_audit_path)
    five_star = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/five_star_sensitivity/analysis"
        / "mf_selection_regret.csv"
    )
    activity = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis"
        / "activity_stratified_ndcg_regret.csv"
    )
    supplementary = (
        (
            "supplementary_ndcg20.tex",
            _tex_secondary_metric(
                secondary,
                "ndcg@20",
                "NDCG@20 cutoff sensitivity",
                "tab:supp-ndcg20",
            ),
        ),
        (
            "supplementary_recall10.tex",
            _tex_secondary_metric(
                secondary,
                "recall@10",
                "Recall@10 sensitivity",
                "tab:supp-recall10",
            ),
        ),
        (
            "supplementary_recall20.tex",
            _tex_secondary_metric(
                secondary,
                "recall@20",
                "Recall@20 sensitivity",
                "tab:supp-recall20",
            ),
        ),
        ("supplementary_clipping.tex", _tex_clipping(secondary, primary)),
        ("supplementary_five_star.tex", _tex_five_star(five_star)),
        ("supplementary_mae.tex", _tex_mae(secondary)),
        ("supplementary_primary_epochs.tex", _tex_epoch_audit(epoch_audit)),
        ("supplementary_bias_only.tex", _tex_bias_only(baseline)),
        *(
            (
                f"supplementary_activity_{dataset}.tex",
                _tex_activity(activity, dataset),
            )
            for dataset in DATASETS
        ),
    )
    for name, text in supplementary:
        path = tex_dir / name
        path.write_text(text, encoding="utf-8")
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/latent_factor_beyond_rmse/second_review_analysis"),
    )
    parser.add_argument(
        "--tex-dir",
        type=Path,
        default=Path("regenerated/tables"),
    )
    args = parser.parse_args()
    root = _repo_root(args.config)
    for path in run(
        args.config,
        output_dir=_resolve(root, args.output_dir),
        tex_dir=_resolve(root, args.tex_dir),
    ):
        print(path)


if __name__ == "__main__":
    main()
