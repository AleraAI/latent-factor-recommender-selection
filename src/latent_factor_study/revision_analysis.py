"""Post-confirmation diagnostics requested by the journal revision audit.

The primary selections and locked-test comparisons remain unchanged.  This
module derives additional, explicitly post-confirmation summaries from the
frozen development grids, per-user confirmation artifacts, and retained
training data.  It does not select a replacement primary model or overwrite a
selection lock.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve
from .inference import (
    bootstrap_mean_difference,
    holm_adjust,
    paired_sign_randomization_p_value,
)
from .selection_lock import DATASETS, RETENTIONS
from .plot_theme import (
    THESIS_PALETTE,
    apply_thesis_plot_style,
    style_axis,
    thesis_sequential_cmap,
)


DISPLAY = {
    "movielens_1m": "MovieLens 1M",
    "ciaodvd": "CiaoDVD",
    "yelp_tampa_restaurants": "Yelp--Tampa",
}
RETENTION_DISPLAY = {"r100": "100%", "r075": "75%", "r050": "50%", "r025": "25%"}


def practical_evidence_class(
    *,
    point: float,
    lower: float,
    upper: float,
    threshold: float,
    identical_selection: bool = False,
) -> str:
    """Classify evidence relative to a positive practical-effect threshold."""
    if identical_selection:
        return "identical selection"
    if lower >= threshold:
        return "clearly above"
    if upper < threshold:
        return "clearly below"
    return "inconclusive"


def gini(values: np.ndarray | pd.Series) -> float:
    """Return the finite-sample Gini coefficient for non-negative values."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("Gini input must be a non-empty vector.")
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Gini input must contain finite non-negative values.")
    total = float(array.sum())
    if total == 0.0:
        return 0.0
    ordered = np.sort(array)
    n = len(ordered)
    weights = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(weights, ordered) / (n * total)) - (n + 1.0) / n)


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    """Identify configurations non-dominated in RMSE, NDCG, and fitting time."""
    required = {"mean_validation_rmse", "mean_validation_ndcg_at_10", "mean_training_seconds"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pareto input is missing columns: {missing}")
    rmse = frame["mean_validation_rmse"].to_numpy(dtype=float)
    ndcg = frame["mean_validation_ndcg_at_10"].to_numpy(dtype=float)
    seconds = frame["mean_training_seconds"].to_numpy(dtype=float)
    keep = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        weakly_better = (rmse <= rmse[index]) & (ndcg >= ndcg[index]) & (seconds <= seconds[index])
        strictly_better = (rmse < rmse[index]) | (ndcg > ndcg[index]) | (seconds < seconds[index])
        if np.any(weakly_better & strictly_better):
            keep[index] = False
    return keep


def _grid(root: Path, dataset: str, retention: str, *, bpr: bool = False) -> pd.DataFrame:
    family = "bpr_development" if bpr else "development"
    return pd.read_csv(
        root / "outputs/latent_factor_beyond_rmse" / family / dataset / retention / "grid_runs.csv"
    )


def _aggregate_explicit(grid: pd.DataFrame) -> pd.DataFrame:
    return (
        grid.groupby(["dimension", "regularization"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_validation_rmse=("validation_rmse", "mean"),
            std_validation_rmse=("validation_rmse", "std"),
            mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
            std_validation_ndcg_at_10=("validation_ndcg@10", "std"),
            mean_training_seconds=("training_seconds", "mean"),
        )
        .sort_values(["dimension", "regularization"], kind="stable")
        .reset_index(drop=True)
    )


def _per_seed_winner_counts(grid: pd.DataFrame, metric: str, *, ascending: bool) -> dict[tuple[int, float], int]:
    winners: dict[tuple[int, float], int] = {}
    for _, subset in grid.groupby("seed", sort=True):
        row = subset.sort_values(
            [metric, "dimension", "regularization"],
            ascending=[ascending, True, False],
            kind="stable",
        ).iloc[0]
        key = (int(row["dimension"]), float(row["regularization"]))
        winners[key] = winners.get(key, 0) + 1
    return winners


def analyze_configuration_diagnostics(root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    pareto_records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            grid = _grid(root, dataset, retention)
            grouped = _aggregate_explicit(grid)
            rmse_order = grouped.sort_values(
                ["mean_validation_rmse", "dimension", "regularization"],
                ascending=[True, True, False],
                kind="stable",
            )
            ndcg_order = grouped.sort_values(
                ["mean_validation_ndcg_at_10", "dimension", "regularization"],
                ascending=[False, True, False],
                kind="stable",
            )
            rmse_best, rmse_second = rmse_order.iloc[0], rmse_order.iloc[1]
            ndcg_best, ndcg_second = ndcg_order.iloc[0], ndcg_order.iloc[1]
            rmse_key = (int(rmse_best["dimension"]), float(rmse_best["regularization"]))
            ndcg_key = (int(ndcg_best["dimension"]), float(ndcg_best["regularization"]))
            rmse_winners = _per_seed_winner_counts(grid, "validation_rmse", ascending=True)
            ndcg_winners = _per_seed_winner_counts(grid, "validation_ndcg@10", ascending=False)

            rmse_se = float(rmse_best["std_validation_rmse"] / math.sqrt(rmse_best["seeds"]))
            ndcg_se = float(
                ndcg_best["std_validation_ndcg_at_10"] / math.sqrt(ndcg_best["seeds"])
            )
            rmse_top_set = grouped[
                grouped["mean_validation_rmse"] <= float(rmse_best["mean_validation_rmse"]) + rmse_se
            ]
            ndcg_top_set = grouped[
                grouped["mean_validation_ndcg_at_10"]
                >= float(ndcg_best["mean_validation_ndcg_at_10"]) - ndcg_se
            ]
            rmse_top_keys = set(
                zip(rmse_top_set["dimension"].astype(int), rmse_top_set["regularization"].astype(float))
            )
            ndcg_top_keys = set(
                zip(ndcg_top_set["dimension"].astype(int), ndcg_top_set["regularization"].astype(float))
            )

            min_d, max_d = int(grouped["dimension"].min()), int(grouped["dimension"].max())
            min_l, max_l = float(grouped["regularization"].min()), float(
                grouped["regularization"].max()
            )
            mask = pareto_mask(grouped)
            frontier = grouped.loc[mask]
            for row in frontier.itertuples(index=False):
                pareto_records.append(
                    {
                        "dataset": dataset,
                        "retention_label": retention,
                        "dimension": int(row.dimension),
                        "regularization": float(row.regularization),
                        "mean_validation_rmse": float(row.mean_validation_rmse),
                        "mean_validation_ndcg_at_10": float(row.mean_validation_ndcg_at_10),
                        "mean_training_seconds": float(row.mean_training_seconds),
                    }
                )

            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "rmse_selected_dimension": rmse_key[0],
                    "rmse_selected_regularization": rmse_key[1],
                    "rmse_best_value": float(rmse_best["mean_validation_rmse"]),
                    "rmse_second_best_gap": float(
                        rmse_second["mean_validation_rmse"] - rmse_best["mean_validation_rmse"]
                    ),
                    "rmse_mean_winner_seed_wins": int(rmse_winners.get(rmse_key, 0)),
                    "rmse_unique_seed_winners": int(len(rmse_winners)),
                    "rmse_one_se_set_size": int(len(rmse_top_set)),
                    "rmse_boundary": bool(
                        rmse_key[0] in {min_d, max_d} or rmse_key[1] in {min_l, max_l}
                    ),
                    "ndcg_selected_dimension": ndcg_key[0],
                    "ndcg_selected_regularization": ndcg_key[1],
                    "ndcg_best_value": float(ndcg_best["mean_validation_ndcg_at_10"]),
                    "ndcg_second_best_gap": float(
                        ndcg_best["mean_validation_ndcg_at_10"]
                        - ndcg_second["mean_validation_ndcg_at_10"]
                    ),
                    "ndcg_mean_winner_seed_wins": int(ndcg_winners.get(ndcg_key, 0)),
                    "ndcg_unique_seed_winners": int(len(ndcg_winners)),
                    "ndcg_one_se_set_size": int(len(ndcg_top_set)),
                    "ndcg_boundary": bool(
                        ndcg_key[0] in {min_d, max_d} or ndcg_key[1] in {min_l, max_l}
                    ),
                    "ndcg_choice_in_rmse_one_se_set": ndcg_key in rmse_top_keys,
                    "rmse_choice_in_ndcg_one_se_set": rmse_key in ndcg_top_keys,
                    "exact_disagreement": rmse_key != ndcg_key,
                    "material_disagreement_one_se": bool(
                        rmse_key != ndcg_key
                        and (ndcg_key not in rmse_top_keys or rmse_key not in ndcg_top_keys)
                    ),
                    "pareto_configuration_count": int(mask.sum()),
                }
            )
    diagnostics = pd.DataFrame.from_records(records)
    pareto = pd.DataFrame.from_records(pareto_records)
    diagnostics.to_csv(output_dir / "configuration_diagnostics.csv", index=False)
    pareto.to_csv(output_dir / "pareto_configurations.csv", index=False)
    return diagnostics, pareto


def analyze_bpr_selection(root: Path, output_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            grid = _grid(root, dataset, retention, bpr=True)
            grouped = (
                grid.groupby(["dimension", "regularization"], as_index=False)
                .agg(
                    seeds=("seed", "nunique"),
                    mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
                    std_validation_ndcg_at_10=("validation_ndcg@10", "std"),
                    mean_training_seconds=("training_seconds", "mean"),
                )
            )
            order = grouped.sort_values(
                ["mean_validation_ndcg_at_10", "dimension", "regularization"],
                ascending=[False, True, False],
                kind="stable",
            )
            best, second = order.iloc[0], order.iloc[1]
            key = (int(best["dimension"]), float(best["regularization"]))
            winners = _per_seed_winner_counts(grid, "validation_ndcg@10", ascending=False)
            se = float(best["std_validation_ndcg_at_10"] / math.sqrt(best["seeds"]))
            one_se = grouped[
                grouped["mean_validation_ndcg_at_10"]
                >= float(best["mean_validation_ndcg_at_10"]) - se
            ]
            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "dimension": key[0],
                    "regularization": key[1],
                    "mean_validation_ndcg_at_10": float(best["mean_validation_ndcg_at_10"]),
                    "second_best_gap": float(
                        best["mean_validation_ndcg_at_10"]
                        - second["mean_validation_ndcg_at_10"]
                    ),
                    "mean_winner_seed_wins": int(winners.get(key, 0)),
                    "unique_seed_winners": int(len(winners)),
                    "one_se_set_size": int(len(one_se)),
                    "dimension_boundary": key[0]
                    in {int(grouped["dimension"].min()), int(grouped["dimension"].max())},
                    "regularization_boundary": key[1]
                    in {
                        float(grouped["regularization"].min()),
                        float(grouped["regularization"].max()),
                    },
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "bpr_selection_diagnostics.csv", index=False)
    return frame


def analyze_practical_evidence(root: Path, output_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        root / "outputs/latent_factor_beyond_rmse/confirmation/analysis/mf_selection_regret.csv"
    )
    lock = yaml.safe_load(
        (root / "configs/latent_factor_beyond_rmse/selection_lock.yaml").read_text(
            encoding="utf-8"
        )
    ) or {}
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        selected = lock["selections"][row["dataset"]][row["retention_label"]]
        identical = selected["rmse_selected"] == selected["ndcg_selected"]
        row["point_estimate_at_least_threshold"] = bool(
            row["delta_rank_ndcg@10"] >= row["practical_threshold"]
        )
        row["practical_evidence_class"] = practical_evidence_class(
            point=float(row["delta_rank_ndcg@10"]),
            lower=float(row["delta_rank_ci_lower"]),
            upper=float(row["delta_rank_ci_upper"]),
            threshold=float(row["practical_threshold"]),
            identical_selection=identical,
        )
        rows.append(row)
    output = pd.DataFrame.from_records(rows)
    output.to_csv(output_dir / "practical_evidence.csv", index=False)
    return output


def _seed_prediction_deltas(root: Path, dataset: str, retention: str) -> list[float]:
    manifest = json.loads(
        (
            root
            / "outputs/latent_factor_beyond_rmse/confirmation"
            / dataset
            / retention
            / "confirmation_manifest.json"
        ).read_text(encoding="utf-8")
    )
    by_seed: dict[int, dict[str, float]] = {}
    for run in manifest["runs"]:
        summary = json.loads((root / run["summary_path"]).read_text(encoding="utf-8"))
        value = float(summary["test"]["prediction"]["clipped"]["rmse"])
        for role in run["selection_roles"]:
            by_seed.setdefault(int(run["seed"]), {})[role] = value
    return [
        values["ndcg_selected"] - values["rmse_selected"]
        for _, values in sorted(by_seed.items())
    ]


def analyze_prediction_regret(root: Path, output_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        root / "outputs/latent_factor_beyond_rmse/confirmation/analysis/mf_selection_regret.csv"
    )
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        seed_deltas = _seed_prediction_deltas(root, row["dataset"], row["retention_label"])
        records.append(
            {
                "dataset": row["dataset"],
                "retention_label": row["retention_label"],
                "rmse_selected_global_rmse": row["rmse_selected_global_rmse"],
                "ndcg_selected_global_rmse": row["ndcg_selected_global_rmse"],
                "delta_prediction_global_rmse": row["delta_prediction_global_rmse"],
                "rmse_selected_mean_user_rmse": row["rmse_selected_mean_user_rmse"],
                "ndcg_selected_mean_user_rmse": row["ndcg_selected_mean_user_rmse"],
                "delta_mean_user_rmse": row["delta_mean_user_rmse"],
                "delta_mean_user_rmse_ci_lower": row["delta_mean_user_rmse_ci_lower"],
                "delta_mean_user_rmse_ci_upper": row["delta_mean_user_rmse_ci_upper"],
                "delta_mean_user_rmse_randomization_p": row[
                    "delta_mean_user_rmse_randomization_p"
                ],
                "seed_delta_global_rmse_min": min(seed_deltas),
                "seed_delta_global_rmse_max": max(seed_deltas),
            }
        )
    records_frame = pd.DataFrame.from_records(records)
    adjusted = holm_adjust(records_frame["delta_mean_user_rmse_randomization_p"].tolist())
    records_frame["delta_mean_user_rmse_holm_p"] = adjusted
    records_frame.to_csv(output_dir / "prediction_regret.csv", index=False)
    return records_frame


def analyze_retention_contrasts(root: Path, output_dir: Path) -> pd.DataFrame:
    analysis_dir = root / "outputs/latent_factor_beyond_rmse/confirmation/analysis"
    records: list[dict[str, Any]] = []
    contrast_index = 0
    for dataset in DATASETS:
        reference = pd.read_parquet(analysis_dir / f"{dataset}_r100_paired_users.parquet")[
            ["user_id", "delta_ndcg@10"]
        ].rename(columns={"delta_ndcg@10": "delta_reference"})
        for retention in RETENTIONS[1:]:
            target = pd.read_parquet(
                analysis_dir / f"{dataset}_{retention}_paired_users.parquet"
            )[["user_id", "delta_ndcg@10"]].rename(
                columns={"delta_ndcg@10": "delta_target"}
            )
            paired = reference.merge(target, on="user_id", validate="one_to_one").dropna()
            differences = (
                paired["delta_target"] - paired["delta_reference"]
            ).to_numpy(dtype=float)
            seed = 939393 + 10 * contrast_index
            lower, upper, _ = bootstrap_mean_difference(
                differences,
                resamples=10000,
                confidence_level=0.95,
                seed=seed,
            )
            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "reference_retention_label": "r100",
                    "users_paired": int(len(differences)),
                    "delta_rank_contrast": float(differences.mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "randomization_p": paired_sign_randomization_p_value(
                        differences, resamples=10000, seed=seed + 1
                    ),
                }
            )
            contrast_index += 1
    frame = pd.DataFrame.from_records(records)
    frame["holm_p"] = holm_adjust(frame["randomization_p"].tolist())
    frame.to_csv(output_dir / "retention_contrasts.csv", index=False)
    return frame


def _first_per_user(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No per-user artifact matching {path / pattern}")
    return matches[0]


def analyze_candidate_diagnostics(root: Path, output_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            condition = (
                root
                / "outputs/latent_factor_beyond_rmse/confirmation"
                / dataset
                / retention
            )
            frame = pd.read_parquet(_first_per_user(condition, "*/*/test_per_user.parquet"))
            evaluable = frame[frame["num_relevant"] > 0]
            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "users_total": int(len(frame)),
                    "users_ranking_evaluable": int(len(evaluable)),
                    "candidate_q05": float(frame["num_candidates"].quantile(0.05)),
                    "candidate_median": float(frame["num_candidates"].median()),
                    "candidate_q95": float(frame["num_candidates"].quantile(0.95)),
                    "relevant_median": float(evaluable["num_relevant"].median()),
                    "relevant_q95": float(evaluable["num_relevant"].quantile(0.95)),
                    "cold_relevant_events": int(frame["num_cold_relevant"].sum()),
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "candidate_diagnostics.csv", index=False)
    return frame


def analyze_selection_cost(root: Path, output_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            for family, folder in (("explicit_mf", "development"), ("bpr_mf", "bpr_development")):
                grid = pd.read_csv(
                    root
                    / "outputs/latent_factor_beyond_rmse"
                    / folder
                    / dataset
                    / retention
                    / "grid_runs.csv"
                )
                run_dir = (
                    root
                    / "outputs/latent_factor_beyond_rmse"
                    / folder
                    / dataset
                    / retention
                )
                per_user = pd.read_parquet(
                    _first_per_user(run_dir, "*/validation_per_user.parquet")
                )
                candidate_scores_per_run = int(per_user["num_candidates"].sum())
                records.append(
                    {
                        "dataset": dataset,
                        "retention_label": retention,
                        "family": family,
                        "runs": int(len(grid)),
                        "aggregate_fit_seconds": float(grid["training_seconds"].sum()),
                        "aggregate_fit_hours": float(grid["training_seconds"].sum() / 3600.0),
                        "validation_candidate_scores_per_run": candidate_scores_per_run,
                        "validation_candidate_scores_total": int(
                            candidate_scores_per_run * len(grid)
                        ),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "selection_cost.csv", index=False)
    return frame


def _popularity_groups(counts: pd.Series) -> pd.Series:
    ordered = (
        counts.rename("positive_count")
        .reset_index()
        .rename(columns={counts.index.name or "index": "item_id"})
        .assign(item_id=lambda value: value["item_id"].astype(str))
        .sort_values(["positive_count", "item_id"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    n_items = len(ordered)
    head_end = max(1, int(math.ceil(0.10 * n_items)))
    mid_end = max(head_end, int(math.ceil(0.50 * n_items)))
    ordered["group"] = "tail"
    ordered.loc[: head_end - 1, "group"] = "head"
    ordered.loc[head_end : mid_end - 1, "group"] = "mid"
    return ordered.set_index("item_id")["group"]


def analyze_popularity(root: Path, config_path: Path, output_dir: Path) -> pd.DataFrame:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sparsity_root = _resolve(root, config["controlled_sparsity"]["output_root"])
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        processed = _resolve(root, config["datasets"][dataset]["output_dir"])
        test = pd.read_parquet(processed / "test.parquet")
        relevant_test = test[test["rating"] >= 4.0].copy()
        relevant_test["item_id"] = relevant_test["item_id"].astype(str)
        for retention in RETENTIONS:
            train = pd.read_parquet(sparsity_root / dataset / f"train_{retention}.parquet")
            train["item_id"] = train["item_id"].astype(str)
            catalog = sorted(train["item_id"].unique())
            positive = (
                train[train["rating"] >= 4.0].groupby("item_id").size().reindex(catalog, fill_value=0)
            )
            test_counts = (
                relevant_test.groupby("item_id").size().reindex(catalog, fill_value=0)
            )
            groups = _popularity_groups(positive)
            mapped = relevant_test["item_id"].map(groups)
            known = mapped.notna()
            denominator = int(known.sum())
            rho = spearmanr(positive.to_numpy(dtype=float), test_counts.to_numpy(dtype=float))
            total_positive = float(positive.sum())
            shares = positive / total_positive if total_positive else positive.astype(float)
            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "catalog_items": int(len(catalog)),
                    "positive_interactions": int(total_positive),
                    "positive_count_gini": gini(positive),
                    "positive_count_hhi": float(np.square(shares).sum()),
                    "train_test_positive_count_spearman": float(rho.statistic),
                    "known_relevant_test_events": denominator,
                    "cold_relevant_test_events": int((~known).sum()),
                    "head_relevant_share": float((mapped[known] == "head").mean()),
                    "mid_relevant_share": float((mapped[known] == "mid").mean()),
                    "tail_relevant_share": float((mapped[known] == "tail").mean()),
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "popularity_diagnostics.csv", index=False)
    return frame


def _style() -> None:
    apply_thesis_plot_style(base_font_size=7.5)
    plt.rcParams.update({"figure.dpi": 140})


def plot_landscapes(root: Path, figure_dir: Path) -> list[Path]:
    _style()
    output: list[Path] = []
    for dataset in DATASETS:
        figure, axes = plt.subplots(2, 4, figsize=(7.2, 4.45), sharex=True, sharey=True)
        for column, retention in enumerate(RETENTIONS):
            grouped = _aggregate_explicit(_grid(root, dataset, retention))
            dimensions = sorted(grouped["dimension"].unique())
            regularizations = sorted(grouped["regularization"].unique())
            for row_index, (metric, label, cmap) in enumerate(
                (
                    (
                        "mean_validation_rmse",
                        "RMSE (lower is better)",
                        thesis_sequential_cmap(
                            name="thesis_rmse",
                            color=THESIS_PALETTE["blue"],
                            high_is_better=False,
                        ),
                    ),
                    (
                        "mean_validation_ndcg_at_10",
                        "NDCG@10 (higher is better)",
                        thesis_sequential_cmap(
                            name="thesis_ndcg",
                            color=THESIS_PALETTE["green"],
                            high_is_better=True,
                        ),
                    ),
                )
            ):
                pivot = grouped.pivot(
                    index="dimension", columns="regularization", values=metric
                ).reindex(index=dimensions, columns=regularizations)
                axis = axes[row_index, column]
                image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap=cmap)
                best_index = np.unravel_index(
                    np.nanargmin(pivot.to_numpy())
                    if metric.endswith("rmse")
                    else np.nanargmax(pivot.to_numpy()),
                    pivot.shape,
                )
                axis.scatter(
                    best_index[1],
                    best_index[0],
                    marker="s",
                    facecolors="white",
                    edgecolors=THESIS_PALETTE["orange"],
                    s=44,
                    linewidths=1.3,
                )
                axis.set_title(RETENTION_DISPLAY[retention], pad=5)
                if column == 0:
                    axis.set_ylabel(f"{label}\nLatent dimension")
                axis.set_xticks(range(len(regularizations)), [f"{value:g}" for value in regularizations], rotation=45, ha="right")
                axis.set_yticks(range(len(dimensions)), [str(value) for value in dimensions])
                style_axis(axis)
                colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
                colorbar.outline.set_edgecolor(THESIS_PALETTE["gray"])
                colorbar.outline.set_linewidth(0.6)
        figure.supxlabel("Regularization $\\lambda$")
        figure.suptitle(f"{DISPLAY[dataset].replace('--', '-')} validation configuration landscapes", y=0.995)
        figure.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.45, h_pad=0.55)
        for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
            path = figure_dir / f"configuration_landscape_{dataset}.{suffix}"
            figure.savefig(path, bbox_inches="tight", **kwargs)
            output.append(path)
        plt.close(figure)
    return output


def plot_rank_displacement(root: Path, figure_dir: Path) -> list[Path]:
    _style()
    figure, axes = plt.subplots(3, 4, figsize=(7.2, 6.15), sharex=True, sharey=True)
    for row, dataset in enumerate(DATASETS):
        for column, retention in enumerate(RETENTIONS):
            grouped = _aggregate_explicit(_grid(root, dataset, retention))
            grouped["rmse_rank"] = grouped["mean_validation_rmse"].rank(
                method="min", ascending=True
            )
            grouped["ndcg_rank"] = grouped["mean_validation_ndcg_at_10"].rank(
                method="min", ascending=False
            )
            axis = axes[row, column]
            axis.scatter(
                grouped["rmse_rank"],
                grouped["ndcg_rank"],
                s=13,
                alpha=0.78,
                color=THESIS_PALETTE["blue"],
                edgecolors="white",
                linewidths=0.25,
            )
            axis.plot(
                [1, 35],
                [1, 35],
                color=THESIS_PALETTE["gray"],
                linewidth=0.9,
                linestyle="--",
            )
            axis.set_title(RETENTION_DISPLAY[retention], pad=5)
            style_axis(axis, grid_axis="both")
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[dataset].replace('--', '–')}\nNDCG rank")
            if row == 2:
                axis.set_xlabel("RMSE rank")
    figure.tight_layout(w_pad=0.5, h_pad=0.65)
    output: list[Path] = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = figure_dir / f"configuration_rank_displacement.{suffix}"
        figure.savefig(path, bbox_inches="tight", **kwargs)
        output.append(path)
    plt.close(figure)
    return output


def _latex_configuration_table(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        for index, retention in enumerate(RETENTIONS):
            row = frame[(frame.dataset == dataset) & (frame.retention_label == retention)].iloc[0]
            boundary = "R/N" if row.rmse_boundary and row.ndcg_boundary else ("R" if row.rmse_boundary else ("N" if row.ndcg_boundary else "--"))
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention].replace("%", r"\%"),
                        f"{row.rmse_second_best_gap:.5f}",
                        f"{int(row.rmse_mean_winner_seed_wins)}/5",
                        f"{row.ndcg_second_best_gap:.7f}",
                        f"{int(row.ndcg_mean_winner_seed_wins)}/5",
                        boundary,
                        str(int(row.rmse_one_se_set_size)),
                        str(int(row.ndcg_one_se_set_size)),
                        "Yes" if row.material_disagreement_one_se else "No",
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
            r"\caption{Post-confirmation configuration-selection diagnostics on the frozen 35-point development grid. Gaps compare the seed-mean winner with the second-best configuration in the selector's metric. Win/5 is the number of development seeds on which the seed-mean winner also wins. Boundary marks the RMSE (R) and/or NDCG (N) winner. One-SE columns give the number of configurations within one standard error of the seed-mean winner. Material disagreement requires exact disagreement and failure of at least one cross-objective choice to lie in the other selector's one-SE set.}",
            r"\label{tab:configuration-stability}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & RMSE gap & Win/5 & NDCG gap & Win/5 & Boundary & RMSE 1-SE & NDCG 1-SE & Material \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _latex_prediction_table(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        for index, retention in enumerate(RETENTIONS):
            row = frame[(frame.dataset == dataset) & (frame.retention_label == retention)].iloc[0]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention].replace("%", r"\%"),
                        f"{row.rmse_selected_global_rmse:.4f}",
                        f"{row.ndcg_selected_global_rmse:.4f}",
                        f"{row.delta_prediction_global_rmse:+.4f}",
                        f"{row.delta_mean_user_rmse:+.4f}",
                        f"[{row.delta_mean_user_rmse_ci_lower:+.4f},{row.delta_mean_user_rmse_ci_upper:+.4f}]",
                        f"[{row.seed_delta_global_rmse_min:+.4f},{row.seed_delta_global_rmse_max:+.4f}]",
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
            r"\caption{Prediction cost of NDCG-based explicit-MF selection. Global RMSE weights held-out events; mean-user $\Delta$ gives equal weight to users and is accompanied by a user-cluster bootstrap interval. Seed range is the minimum and maximum global-RMSE difference across the five fresh initialization seeds. Positive differences favor the RMSE-selected configuration.}",
            r"\label{tab:prediction-regret}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & RMSE sel. & NDCG sel. & Global $\Delta$ & Mean-user $\Delta$ & 95\% CI & Seed range \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _latex_retention_table(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        subset = frame[frame.dataset == dataset]
        for index, retention in enumerate(RETENTIONS[1:]):
            row = subset[subset.retention_label == retention].iloc[0]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention].replace("%", r"\%"),
                        f"{row.delta_rank_contrast:+.4f}",
                        f"[{row.ci_lower:+.4f},{row.ci_upper:+.4f}]",
                        f"{row.holm_p:.4f}",
                        str(int(row.users_paired)),
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
            r"\caption{Post-confirmation within-dataset contrasts in ranking regret relative to the frozen 100\% retention condition. Positive values indicate that the NDCG-versus-RMSE selection advantage increased after retention was reduced. Intervals use paired user-cluster bootstrap resampling; Holm adjustment covers the nine contrasts. The primary conditions use retention-specific deterministic priorities and are not a nested path; these contrasts therefore quantify the declared retained subsets and do not average over alternative thinning realizations.}",
            r"\label{tab:retention-contrasts}",
            r"\small",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. vs. 100\% & Contrast & 95\% CI & $p_{\rm Holm}$ & Users \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def _latex_candidate_table(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        for index, retention in enumerate(RETENTIONS):
            row = frame[(frame.dataset == dataset) & (frame.retention_label == retention)].iloc[0]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention].replace("%", r"\%"),
                        f"{int(row.users_ranking_evaluable):,}",
                        f"{int(row.candidate_q05):,}/{int(row.candidate_median):,}/{int(row.candidate_q95):,}",
                        f"{row.relevant_median:.0f}/{row.relevant_q95:.0f}",
                        f"{int(row.cold_relevant_events):,}",
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
            r"\caption{Locked-test candidate diagnostics under the exact retained-training-catalog contract. Candidate counts are the 5th percentile/median/95th percentile over all evaluated users; relevant counts are the median/95th percentile among ranking-evaluable users. Item-cold relevant events are excluded from primary ranking and retained for the declared prediction fallback.}",
            r"\label{tab:candidate-diagnostics}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Evaluable users & Candidates $q_{.05}$/med./$q_{.95}$ & Relevant med./$q_{.95}$ & Cold relevant \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _latex_cost_table(frame: pd.DataFrame) -> str:
    pivot = frame.set_index(["dataset", "retention_label", "family"])
    rows: list[str] = []
    for dataset in DATASETS:
        for index, retention in enumerate(RETENTIONS):
            explicit = pivot.loc[(dataset, retention, "explicit_mf")]
            bpr = pivot.loc[(dataset, retention, "bpr_mf")]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention].replace("%", r"\%"),
                        f"{explicit.aggregate_fit_hours:.3f}",
                        f"{bpr.aggregate_fit_hours:.3f}",
                        f"{int(explicit.validation_candidate_scores_total):,}",
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
            r"\caption{Complete development-search work recorded by the frozen grids. Fit hours sum measured single-threaded training time over 175 runs per family and condition; they are not elapsed parallel wall time. Candidate scores are exact validation-catalog scoring operations for the explicit grid (the BPR grid has the same condition-level candidate count). Evaluation time and peak memory were not instrumented separately and are therefore not inferred from these work units.}",
            r"\label{tab:selection-cost}",
            r"\small",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Explicit fit h & BPR fit h & Explicit candidate scores \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def _latex_popularity_table(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for dataset in DATASETS:
        for index, retention in enumerate(RETENTIONS):
            row = frame[(frame.dataset == dataset) & (frame.retention_label == retention)].iloc[0]
            rows.append(
                " & ".join(
                    [
                        DISPLAY[dataset] if index == 0 else "",
                        RETENTION_DISPLAY[retention].replace("%", r"\%"),
                        f"{row.positive_count_gini:.3f}",
                        f"{row.train_test_positive_count_spearman:.3f}",
                        f"{100*row.head_relevant_share:.1f}\\%",
                        f"{100*row.mid_relevant_share:.1f}\\%",
                        f"{100*row.tail_relevant_share:.1f}\\%",
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
            r"\caption{Post-confirmation popularity diagnostics. Gini summarizes concentration of retained-training positive counts. $\rho_{\rm train,test}$ is the item-level Spearman association between retained-training positive counts and relevant-test-event counts. Head, mid, and tail are the top 10\%, next 40\%, and remaining 50\% of retained-catalog items ranked by training positive count; entries give their shares of train-known relevant test events.}",
            r"\label{tab:popularity-diagnostics}",
            r"\small",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Gini & $\rho_{\rm train,test}$ & Head & Mid & Tail \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def write_latex_tables(
    paper_generated_dir: Path,
    *,
    configurations: pd.DataFrame,
    prediction: pd.DataFrame,
    retention: pd.DataFrame,
    candidates: pd.DataFrame,
    cost: pd.DataFrame,
    popularity: pd.DataFrame,
) -> list[Path]:
    payloads = {
        "configuration_stability.tex": _latex_configuration_table(configurations),
        "prediction_regret.tex": _latex_prediction_table(prediction),
        "retention_contrasts.tex": _latex_retention_table(retention),
        "candidate_diagnostics.tex": _latex_candidate_table(candidates),
        "selection_cost.tex": _latex_cost_table(cost),
        "popularity_diagnostics.tex": _latex_popularity_table(popularity),
    }
    paths: list[Path] = []
    for name, text in payloads.items():
        path = paper_generated_dir / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def run_revision_analysis(config_path: Path) -> list[Path]:
    root = _repo_root(config_path)
    output_dir = root / "outputs/latent_factor_beyond_rmse/revision_analysis"
    figure_dir = root / "regenerated/figures"
    paper_generated_dir = root / "regenerated/tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paper_generated_dir.mkdir(parents=True, exist_ok=True)

    configurations, _ = analyze_configuration_diagnostics(root, output_dir)
    analyze_bpr_selection(root, output_dir)
    analyze_practical_evidence(root, output_dir)
    prediction = analyze_prediction_regret(root, output_dir)
    retention = analyze_retention_contrasts(root, output_dir)
    candidates = analyze_candidate_diagnostics(root, output_dir)
    cost = analyze_selection_cost(root, output_dir)
    popularity = analyze_popularity(root, config_path, output_dir)
    paths = write_latex_tables(
        paper_generated_dir,
        configurations=configurations,
        prediction=prediction,
        retention=retention,
        candidates=candidates,
        cost=cost,
        popularity=popularity,
    )
    paths.extend(plot_landscapes(root, figure_dir))
    paths.extend(plot_rank_displacement(root, figure_dir))
    manifest = {
        "schema_version": 1,
        "analysis_status": "post_confirmation",
        "primary_selection_locks_modified": False,
        "outputs": [str(path.relative_to(root)) for path in sorted(output_dir.glob("*.csv"))],
        "manuscript_artifacts": [str(path.relative_to(root)) for path in paths],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [manifest_path, *paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    for path in run_revision_analysis(args.config):
        print(path)


if __name__ == "__main__":
    main()
