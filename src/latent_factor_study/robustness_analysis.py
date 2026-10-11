"""Analyze the separately versioned post-confirmation robustness experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import DEFAULT_CONFIG_PATH, _repo_root
from .robustness_explicit import DATASETS, NON_FULL_RETENTIONS, RETENTIONS
from .plot_theme import (
    RETENTION_COLORS,
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


def _plot_style() -> None:
    apply_thesis_plot_style(base_font_size=9)
    plt.rcParams.update({"xtick.labelsize": 7})


def _seed_winner_count(
    condition: pd.DataFrame,
    *,
    metric: str,
    dimension: int,
    regularization: float,
    ascending: bool,
) -> int:
    count = 0
    for _, seed_frame in condition.groupby("seed", sort=True):
        winner = seed_frame.sort_values(
            [metric, "dimension", "regularization"],
            ascending=[ascending, True, False],
            kind="stable",
        ).iloc[0]
        count += int(
            int(winner.dimension) == dimension
            and math.isclose(float(winner.regularization), regularization)
        )
    return count


def analyze_expanded_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return selection, budget-change, and convergence diagnostics."""
    required = {
        "dataset",
        "retention_label",
        "dimension",
        "regularization",
        "seed",
        "epoch",
        "validation_rmse",
        "validation_ndcg@10",
        "convergence_epoch",
        "training_seconds",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Expanded runs are missing columns: {missing}")
    records: list[dict[str, Any]] = []
    roles = (
        ("rmse_selected", "validation_rmse", True),
        ("ndcg_selected", "validation_ndcg@10", False),
    )
    condition_columns = ["dataset", "retention_label", "epoch"]
    for keys, condition in frame.groupby(condition_columns, sort=False):
        dataset, retention, epoch = keys
        grouped = (
            condition.groupby(["dimension", "regularization"], as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                mean_validation_rmse=("validation_rmse", "mean"),
                std_validation_rmse=("validation_rmse", "std"),
                mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
                std_validation_ndcg_at_10=("validation_ndcg@10", "std"),
            )
        )
        dimensions = sorted(grouped["dimension"].astype(int).unique())
        regularizations = sorted(grouped["regularization"].astype(float).unique())
        for role, raw_metric, ascending in roles:
            mean_metric = (
                "mean_validation_rmse"
                if role == "rmse_selected"
                else "mean_validation_ndcg_at_10"
            )
            std_metric = (
                "std_validation_rmse"
                if role == "rmse_selected"
                else "std_validation_ndcg_at_10"
            )
            ordered = grouped.sort_values(
                [mean_metric, "dimension", "regularization"],
                ascending=[ascending, True, False],
                kind="stable",
            ).reset_index(drop=True)
            winner = ordered.iloc[0]
            runner_up = ordered.iloc[1]
            gap = (
                float(runner_up[mean_metric] - winner[mean_metric])
                if ascending
                else float(winner[mean_metric] - runner_up[mean_metric])
            )
            standard_error = float(winner[std_metric]) / math.sqrt(int(winner.seeds))
            if ascending:
                one_se = grouped[mean_metric] <= float(winner[mean_metric]) + standard_error
            else:
                one_se = grouped[mean_metric] >= float(winner[mean_metric]) - standard_error
            dimension = int(winner.dimension)
            regularization = float(winner.regularization)
            dimension_boundary = dimension in (dimensions[0], dimensions[-1])
            regularization_boundary = math.isclose(
                regularization, regularizations[0]
            ) or math.isclose(regularization, regularizations[-1])
            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "epoch": int(epoch),
                    "selection_role": role,
                    "dimension": dimension,
                    "regularization": regularization,
                    "mean_selection_metric": float(winner[mean_metric]),
                    "runner_up_gap": gap,
                    "winner_standard_error": standard_error,
                    "configurations_within_one_se": int(one_se.sum()),
                    "seed_wins": _seed_winner_count(
                        condition,
                        metric=raw_metric,
                        dimension=dimension,
                        regularization=regularization,
                        ascending=ascending,
                    ),
                    "selection_seed_count": int(winner.seeds),
                    "dimension_boundary": dimension_boundary,
                    "regularization_boundary": regularization_boundary,
                    "any_grid_boundary": dimension_boundary
                    or regularization_boundary,
                    "search_dimensions": len(dimensions),
                    "search_regularizations": len(regularizations),
                }
            )
    selections = pd.DataFrame.from_records(records)

    budget_records: list[dict[str, Any]] = []
    for keys, condition in selections.groupby(
        ["dataset", "retention_label", "selection_role"], sort=False
    ):
        ordered = condition.sort_values("epoch")
        configs = list(zip(ordered["dimension"], ordered["regularization"]))
        budget_records.append(
            {
                "dataset": keys[0],
                "retention_label": keys[1],
                "selection_role": keys[2],
                "budgets": ",".join(map(str, ordered["epoch"].astype(int))),
                "distinct_selected_configurations": len(set(configs)),
                "selection_changed_across_budgets": len(set(configs)) > 1,
                "selected_epochs_on_boundary": int(ordered["any_grid_boundary"].sum()),
            }
        )
    budgets = pd.DataFrame.from_records(budget_records)

    run_identity = frame.drop_duplicates(
        ["dataset", "retention_label", "dimension", "regularization", "seed"]
    )
    convergence = (
        run_identity.groupby(["dataset", "retention_label"], as_index=False)
        .agg(
            model_runs=("seed", "size"),
            convergence_rate=("convergence_epoch", lambda values: values.notna().mean()),
            median_convergence_epoch=("convergence_epoch", "median"),
            q25_convergence_epoch=("convergence_epoch", lambda values: values.quantile(0.25)),
            q75_convergence_epoch=("convergence_epoch", lambda values: values.quantile(0.75)),
            median_training_seconds=("training_seconds", "median"),
        )
    )
    return selections, budgets, convergence


def selected_repeated_observations(
    runs: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    """Extract initialization-level validation values at each retention-seed choice."""
    records: list[dict[str, Any]] = []
    role_specs = (
        (
            "rmse_selected",
            "rmse_dimension",
            "rmse_regularization",
            "validation_rmse",
        ),
        (
            "ndcg_selected",
            "ndcg_dimension",
            "ndcg_regularization",
            "validation_ndcg@10",
        ),
    )
    for selected in selections.itertuples(index=False):
        base = runs[
            (runs["dataset"] == selected.dataset)
            & (runs["retention_label"] == selected.retention_label)
            & (runs["retention_seed"] == selected.retention_seed)
            & (runs["epoch"] == selected.epoch)
        ]
        for role, dimension_column, regularization_column, metric in role_specs:
            dimension = int(getattr(selected, dimension_column))
            regularization = float(getattr(selected, regularization_column))
            chosen = base[
                (base["dimension"] == dimension)
                & np.isclose(base["regularization"], regularization)
            ]
            if chosen.empty:
                raise ValueError(
                    f"Missing selected runs for {selected.dataset}/"
                    f"{selected.retention_label}/{selected.retention_seed}/{role}"
                )
            for _, row in chosen.iterrows():
                records.append(
                    {
                        "dataset": selected.dataset,
                        "retention_label": selected.retention_label,
                        "retention_seed": int(selected.retention_seed),
                        "selection_role": role,
                        "dimension": dimension,
                        "regularization": regularization,
                        "initialization_seed": int(row["seed"]),
                        "metric_name": metric,
                        "validation_metric": float(row[metric]),
                    }
                )
    return pd.DataFrame.from_records(records)


def analyze_repeated_variance(observations: pd.DataFrame) -> pd.DataFrame:
    """Method-of-moments decomposition of retention and initialization variation."""
    records: list[dict[str, Any]] = []
    columns = ["dataset", "retention_label", "selection_role", "metric_name"]
    for keys, condition in observations.groupby(columns, sort=False):
        by_retention = condition.groupby("retention_seed")["validation_metric"]
        means = by_retention.mean()
        within_variances = by_retention.var(ddof=1)
        counts = by_retention.size()
        initialization_variance = float(within_variances.mean())
        variance_of_retention_means = float(means.var(ddof=1))
        mean_initializations = float(counts.mean())
        retention_variance = max(
            variance_of_retention_means - initialization_variance / mean_initializations,
            0.0,
        )
        total = retention_variance + initialization_variance
        records.append(
            {
                **dict(zip(columns, keys)),
                "retention_seed_count": int(len(means)),
                "initialization_seeds_per_retention_mean": mean_initializations,
                "distinct_selected_configurations": int(
                    condition[["dimension", "regularization"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "variance_of_retention_means": variance_of_retention_means,
                "initialization_variance": initialization_variance,
                "estimated_retention_sample_variance": retention_variance,
                "retention_variance_share": retention_variance / total if total else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def analyze_explicit_confirmation(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute post-confirmation prediction--ranking regret from run summaries."""
    identity = ["dataset", "retention_label"]
    if frame["retention_seed"].notna().any():
        identity.append("retention_seed")
    grouped = (
        frame.groupby([*identity, "selection_role"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_test_rmse=("test_rmse", "mean"),
            mean_test_ndcg_at_10=("test_ndcg@10", "mean"),
            std_test_ndcg_at_10=("test_ndcg@10", "std"),
        )
    )
    rmse = grouped[grouped["selection_role"] == "rmse_selected"].drop(
        columns="selection_role"
    )
    ndcg = grouped[grouped["selection_role"] == "ndcg_selected"].drop(
        columns="selection_role"
    )
    merged = rmse.merge(ndcg, on=identity, suffixes=("_rmse_selected", "_ndcg_selected"))
    merged["delta_rank_ndcg@10"] = (
        merged["mean_test_ndcg_at_10_ndcg_selected"]
        - merged["mean_test_ndcg_at_10_rmse_selected"]
    )
    merged["delta_prediction_rmse"] = (
        merged["mean_test_rmse_ndcg_selected"]
        - merged["mean_test_rmse_rmse_selected"]
    )
    return merged


def summarize_repeated_test_regret(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize post-confirmation regret over independent retention samples."""
    return (
        frame.groupby(["dataset", "retention_label"], as_index=False)
        .agg(
            retention_seed_count=("retention_seed", "nunique"),
            mean_delta_rank_ndcg_at_10=("delta_rank_ndcg@10", "mean"),
            std_delta_rank_ndcg_at_10=("delta_rank_ndcg@10", "std"),
            min_delta_rank_ndcg_at_10=("delta_rank_ndcg@10", "min"),
            max_delta_rank_ndcg_at_10=("delta_rank_ndcg@10", "max"),
            mean_delta_prediction_rmse=("delta_prediction_rmse", "mean"),
        )
    )


def analyze_bpr_sampler_confirmation(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize BPR test behavior under the three declared negative pools."""
    return (
        frame.groupby(
            ["dataset", "retention_label", "negative_mode"], as_index=False
        )
        .agg(
            seeds=("seed", "nunique"),
            mean_test_ndcg_at_10=("evaluation_ndcg@10", "mean"),
            std_test_ndcg_at_10=("evaluation_ndcg@10", "std"),
            mean_test_recall_at_10=("evaluation_recall@10", "mean"),
            mean_sampled_full_positive_rate=(
                "sampled_full_training_positive_rate",
                "mean",
            ),
        )
    )


def analyze_bpr_refine(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize independently tuned BPR optima, gaps, and one-SE plateaus."""
    records: list[dict[str, Any]] = []
    for keys, condition in frame.groupby(["dataset", "retention_label"], sort=False):
        grouped = (
            condition.groupby(
                [
                    "dimension",
                    "regularization",
                    "learning_rate",
                    "negatives_per_positive",
                    "negative_mode",
                    "epoch",
                ],
                as_index=False,
            )
            .agg(
                seeds=("seed", "nunique"),
                mean_validation_ndcg_at_10=("evaluation_ndcg@10", "mean"),
                std_validation_ndcg_at_10=("evaluation_ndcg@10", "std"),
                mean_false_negative_rate=(
                    "sampled_full_training_positive_rate",
                    "mean",
                ),
            )
        )
        ordered = grouped.sort_values(
            [
                "mean_validation_ndcg_at_10",
                "epoch",
                "dimension",
                "regularization",
                "negatives_per_positive",
                "learning_rate",
            ],
            ascending=[False, True, True, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        winner, runner_up = ordered.iloc[0], ordered.iloc[1]
        standard_error = float(winner.std_validation_ndcg_at_10) / math.sqrt(
            int(winner.seeds)
        )
        within = grouped["mean_validation_ndcg_at_10"] >= (
            float(winner.mean_validation_ndcg_at_10) - standard_error
        )
        records.append(
            {
                "dataset": keys[0],
                "retention_label": keys[1],
                "dimension": int(winner.dimension),
                "regularization": float(winner.regularization),
                "learning_rate": float(winner.learning_rate),
                "negatives_per_positive": int(winner.negatives_per_positive),
                "epoch": int(winner.epoch),
                "negative_mode": winner.negative_mode,
                "mean_validation_ndcg@10": float(
                    winner.mean_validation_ndcg_at_10
                ),
                "runner_up_gap": float(
                    winner.mean_validation_ndcg_at_10
                    - runner_up.mean_validation_ndcg_at_10
                ),
                "winner_standard_error": standard_error,
                "configurations_within_one_se": int(within.sum()),
                "mean_false_negative_rate": float(winner.mean_false_negative_rate),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_expanded_heatmaps(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot the 100-epoch explicit-MF response surface for every condition."""
    _plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_frame = frame[frame["epoch"] == 100]
    grouped = (
        epoch_frame.groupby(
            ["dataset", "retention_label", "dimension", "regularization"],
            as_index=False,
        )
        .agg(
            mean_validation_rmse=("validation_rmse", "mean"),
            mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
        )
    )
    outputs: list[Path] = []
    specifications = (
        (
            "rmse",
            "mean_validation_rmse",
            "Validation RMSE (lower is better)",
            thesis_sequential_cmap(
                name="expanded_thesis_rmse",
                color=THESIS_PALETTE["blue"],
                high_is_better=False,
            ),
        ),
        (
            "ndcg",
            "mean_validation_ndcg_at_10",
            "Validation NDCG@10 (higher is better)",
            thesis_sequential_cmap(
                name="expanded_thesis_ndcg",
                color=THESIS_PALETTE["green"],
                high_is_better=True,
            ),
        ),
    )
    for dataset in DATASETS:
        dataset_frame = grouped[grouped["dataset"] == dataset]
        for short_name, metric, color_map, cmap in [
            (spec[0], spec[1], spec[2], spec[3]) for spec in specifications
        ]:
            figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.25), sharex=True, sharey=True)
            values = dataset_frame[metric].to_numpy(float)
            minimum, maximum = float(values.min()), float(values.max())
            image = None
            for axis, retention in zip(axes.flat, RETENTIONS):
                condition = dataset_frame[
                    dataset_frame["retention_label"] == retention
                ]
                pivot = condition.pivot(
                    index="dimension", columns="regularization", values=metric
                ).sort_index().sort_index(axis=1)
                image = axis.imshow(
                    pivot.to_numpy(float),
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    vmin=minimum,
                    vmax=maximum,
                )
                winner_index = (
                    condition[metric].idxmin()
                    if short_name == "rmse"
                    else condition[metric].idxmax()
                )
                winner = condition.loc[winner_index]
                x = list(pivot.columns).index(float(winner.regularization))
                y = list(pivot.index).index(int(winner.dimension))
                axis.plot(
                    x,
                    y,
                    marker="o",
                    markersize=7,
                    markerfacecolor="none",
                    markeredgecolor="white",
                    markeredgewidth=1.4,
                )
                axis.plot(x, y, marker="x", markersize=5, color="black", markeredgewidth=1)
                axis.set_title(f"Training retention {RETENTION_DISPLAY[retention]}")
                axis.set_xticks(range(len(pivot.columns)))
                axis.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45, ha="right")
                axis.set_yticks(range(len(pivot.index)))
                axis.set_yticklabels([str(int(value)) for value in pivot.index])
                axis.set_xlabel(r"Regularization $\lambda$")
                axis.set_ylabel(r"Latent dimension $d$")
                style_axis(axis)
            assert image is not None
            colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
            colorbar.set_label(color_map)
            figure.suptitle(f"{DISPLAY[dataset]}: expanded explicit-MF surface at 100 epochs")
            figure.subplots_adjust(left=0.08, right=0.89, bottom=0.11, top=0.90, wspace=0.22, hspace=0.30)
            prefix = output_dir / f"expanded_surface_{dataset}_{short_name}"
            for suffix in (".pdf", ".png"):
                path = prefix.with_suffix(suffix)
                figure.savefig(path, dpi=240 if suffix == ".png" else None, bbox_inches="tight")
                outputs.append(path)
            plt.close(figure)
    return outputs


def plot_budget_selections(selections: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot the selected validation objective across 20/50/100-epoch budgets."""
    _plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for dataset in DATASETS:
        figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
        for axis, role, ylabel in (
            (axes[0], "rmse_selected", "Selected validation RMSE"),
            (axes[1], "ndcg_selected", "Selected validation NDCG@10"),
        ):
            subset = selections[
                (selections["dataset"] == dataset)
                & (selections["selection_role"] == role)
            ]
            for retention, color in zip(RETENTIONS, RETENTION_COLORS):
                condition = subset[subset["retention_label"] == retention].sort_values("epoch")
                axis.plot(
                    condition["epoch"],
                    condition["mean_selection_metric"],
                    marker="o",
                    color=color,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    label=RETENTION_DISPLAY[retention],
                )
            axis.set_xticks([20, 50, 100])
            axis.set_xlabel("Training budget (epochs)")
            axis.set_ylabel(ylabel)
            style_axis(axis, grid_axis="both")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.suptitle(
            f"{DISPLAY[dataset]}: validation optima across training budgets",
            y=0.985,
        )
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=4,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.78), w_pad=1.1)
        prefix = output_dir / f"budget_sensitivity_{dataset}"
        for suffix in (".pdf", ".png"):
            path = prefix.with_suffix(suffix)
            figure.savefig(path, dpi=240 if suffix == ".png" else None, bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)
    return outputs


def plot_combined_budget_selections(
    selections: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Plot all dataset budget sensitivities in one manuscript-ready panel."""
    _plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(7.15, 6.45))
    for row, dataset in enumerate(DATASETS):
        for column, role, ylabel in (
            (0, "rmse_selected", "Validation RMSE"),
            (1, "ndcg_selected", "Validation NDCG@10"),
        ):
            axis = axes[row, column]
            subset = selections[
                (selections["dataset"] == dataset)
                & (selections["selection_role"] == role)
            ]
            for retention, color in zip(RETENTIONS, RETENTION_COLORS):
                condition = subset[
                    subset["retention_label"] == retention
                ].sort_values("epoch")
                axis.plot(
                    condition["epoch"],
                    condition["mean_selection_metric"],
                    marker="o",
                    color=color,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    label=RETENTION_DISPLAY[retention],
                )
            axis.set_title(
                f"{DISPLAY[dataset]}: {'RMSE selection' if column == 0 else 'NDCG selection'}"
            )
            axis.set_xticks([20, 50, 100])
            axis.set_xlabel("Training budget (epochs)")
            axis.set_ylabel(ylabel)
            style_axis(axis, grid_axis="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Validation-optimum sensitivity to the training budget", y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90), h_pad=1.0, w_pad=1.3)
    outputs: list[Path] = []
    prefix = output_dir / "budget_sensitivity_all"
    for suffix in (".pdf", ".png"):
        path = prefix.with_suffix(suffix)
        figure.savefig(
            path,
            dpi=240 if suffix == ".png" else None,
            bbox_inches="tight",
        )
        outputs.append(path)
    plt.close(figure)
    return outputs


def load_selected_training_curves(
    robustness_root: Path, selections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load epoch histories for the two 100-epoch objective-specific winners."""
    training_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    selected = selections[selections["epoch"] == 100]
    for choice in selected.itertuples(index=False):
        condition_dir = (
            robustness_root
            / "explicit_expanded"
            / choice.dataset
            / choice.retention_label
            / "runs"
        )
        for seed in (101, 211, 307, 401, 503):
            path = (
                condition_dir
                / f"d{int(choice.dimension)}_lambda{float(choice.regularization):g}_seed{seed}"
                / "summary.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            identity = {
                "dataset": choice.dataset,
                "retention_label": choice.retention_label,
                "selection_role": choice.selection_role,
                "dimension": int(choice.dimension),
                "regularization": float(choice.regularization),
                "seed": seed,
            }
            for history in payload["training"]["epoch_history"]:
                training_records.append(
                    {
                        **identity,
                        "epoch": int(history["epoch"]),
                        "training_objective": float(history["training_objective"]),
                        "training_rmse": float(history["training_rmse"]),
                    }
                )
            for epoch, evaluation in payload["validation_checkpoints"].items():
                validation_records.append(
                    {
                        **identity,
                        "epoch": int(epoch),
                        "validation_rmse": float(
                            evaluation["prediction"]["clipped"]["rmse"]
                        ),
                        "validation_ndcg@10": float(
                            evaluation["ranking"]["ndcg@10"]
                        ),
                    }
                )
    return (
        pd.DataFrame.from_records(training_records),
        pd.DataFrame.from_records(validation_records),
    )


def plot_training_objective_composite(
    training: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Plot seed-mean objective trajectories for all selected explicit models."""
    _plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    role_style = {
        "rmse_selected": ("RMSE-selected", THESIS_PALETTE["orange"]),
        "ndcg_selected": ("NDCG-selected", THESIS_PALETTE["blue"]),
    }
    figure, axes = plt.subplots(3, 4, figsize=(7.15, 6.0), sharex=True)
    for row, dataset in enumerate(DATASETS):
        for column, retention in enumerate(RETENTIONS):
            axis = axes[row, column]
            for role, (label, color) in role_style.items():
                subset = training[
                    (training["dataset"] == dataset)
                    & (training["retention_label"] == retention)
                    & (training["selection_role"] == role)
                ]
                summary = subset.groupby("epoch", as_index=False).agg(
                    mean_objective=("training_objective", "mean"),
                    std_objective=("training_objective", "std"),
                    seeds=("seed", "nunique"),
                )
                standard_error = summary["std_objective"] / np.sqrt(
                    summary["seeds"]
                )
                axis.plot(
                    summary["epoch"],
                    summary["mean_objective"],
                    color=color,
                    linewidth=1.2,
                    label=label,
                )
                axis.fill_between(
                    summary["epoch"],
                    summary["mean_objective"] - standard_error,
                    summary["mean_objective"] + standard_error,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
            if row == 0:
                axis.set_title(f"Retention {RETENTION_DISPLAY[retention]}")
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[dataset]}\nTraining objective")
            if row == len(DATASETS) - 1:
                axis.set_xlabel("Epoch")
            axis.set_xlim(1, 100)
            axis.set_xticks([1, 20, 50, 100])
            style_axis(axis, grid_axis="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        "Training-objective trajectories of the 100-epoch selected models",
        y=0.99,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90), h_pad=0.9, w_pad=0.7)
    outputs: list[Path] = []
    prefix = output_dir / "training_objective_trajectories_all"
    for suffix in (".pdf", ".png"):
        path = prefix.with_suffix(suffix)
        figure.savefig(
            path,
            dpi=240 if suffix == ".png" else None,
            bbox_inches="tight",
        )
        outputs.append(path)
    plt.close(figure)
    return outputs


def plot_selected_training_curves(
    training: pd.DataFrame, validation: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Plot loss and validation trajectories for the 100-epoch selected models."""
    _plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    role_style = {
        "rmse_selected": ("RMSE-selected", THESIS_PALETTE["orange"]),
        "ndcg_selected": ("NDCG-selected", THESIS_PALETTE["blue"]),
    }
    for dataset in DATASETS:
        figure, axes = plt.subplots(3, 4, figsize=(7.15, 5.6), sharex="row")
        for column, retention in enumerate(RETENTIONS):
            for role, (label, color) in role_style.items():
                train_subset = training[
                    (training["dataset"] == dataset)
                    & (training["retention_label"] == retention)
                    & (training["selection_role"] == role)
                ]
                train_mean = train_subset.groupby("epoch", as_index=False).agg(
                    training_objective=("training_objective", "mean")
                )
                validation_subset = validation[
                    (validation["dataset"] == dataset)
                    & (validation["retention_label"] == retention)
                    & (validation["selection_role"] == role)
                ]
                validation_mean = validation_subset.groupby("epoch", as_index=False).agg(
                    validation_rmse=("validation_rmse", "mean"),
                    validation_ndcg_at_10=("validation_ndcg@10", "mean"),
                )
                axes[0, column].plot(
                    train_mean["epoch"],
                    train_mean["training_objective"],
                    color=color,
                    linewidth=1.1,
                    label=label,
                )
                axes[1, column].plot(
                    validation_mean["epoch"],
                    validation_mean["validation_rmse"],
                    color=color,
                    marker="o",
                    linewidth=1.1,
                )
                axes[2, column].plot(
                    validation_mean["epoch"],
                    validation_mean["validation_ndcg_at_10"],
                    color=color,
                    marker="o",
                    linewidth=1.1,
                )
            axes[0, column].set_title(RETENTION_DISPLAY[retention])
            axes[2, column].set_xticks([20, 50, 100])
            axes[2, column].set_xlabel("Epoch")
            for row in range(3):
                style_axis(axes[row, column], grid_axis="both")
        axes[0, 0].set_ylabel("Training objective")
        axes[1, 0].set_ylabel("Validation RMSE")
        axes[2, 0].set_ylabel("Validation NDCG@10")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.suptitle(
            f"{DISPLAY[dataset]}: selected-model convergence and budget sensitivity",
            y=0.99,
        )
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=2,
            frameon=False,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.89), h_pad=0.8, w_pad=0.7)
        prefix = output_dir / f"selected_convergence_{dataset}"
        for suffix in (".pdf", ".png"):
            path = prefix.with_suffix(suffix)
            figure.savefig(path, dpi=240 if suffix == ".png" else None, bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)
    return outputs


def _collect_csvs(paths: list[Path]) -> pd.DataFrame:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required result files; first: {missing[0]}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def run_analysis(root: Path) -> list[Path]:
    robustness_root = root / "outputs/latent_factor_beyond_rmse/robustness"
    output_dir = robustness_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_paths = [
        robustness_root / "explicit_expanded" / dataset / retention / "runs.csv"
        for dataset in DATASETS
        for retention in RETENTIONS
    ]
    expanded = _collect_csvs(expanded_paths)
    selections, budgets, convergence = analyze_expanded_frame(expanded)
    outputs: list[Path] = []
    for name, table in (
        ("expanded_selection_diagnostics.csv", selections),
        ("budget_selection_diagnostics.csv", budgets),
        ("explicit_convergence_diagnostics.csv", convergence),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)
    figure_dir = output_dir / "figures"
    outputs.extend(plot_expanded_heatmaps(expanded, figure_dir))
    outputs.extend(plot_budget_selections(selections, figure_dir))
    outputs.extend(plot_combined_budget_selections(selections, figure_dir))
    training_curves, validation_curves = load_selected_training_curves(
        robustness_root, selections
    )
    for name, table in (
        ("selected_training_curves.csv", training_curves),
        ("selected_validation_curves.csv", validation_curves),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)
    outputs.extend(plot_training_objective_composite(training_curves, figure_dir))
    outputs.extend(
        plot_selected_training_curves(training_curves, validation_curves, figure_dir)
    )

    repeated_runs = _collect_csvs(
        [
            robustness_root
            / "explicit_repeated_retention"
            / f"seed_{retention_seed}"
            / dataset
            / retention
            / "runs.csv"
            for retention_seed in (20260819, 20260829, 20260907)
            for dataset in DATASETS
            for retention in NON_FULL_RETENTIONS
        ]
    )
    repeated_selections = _collect_csvs(
        [path.parent / "selections.csv" for path in [
            robustness_root
            / "explicit_repeated_retention"
            / f"seed_{retention_seed}"
            / dataset
            / retention
            / "runs.csv"
            for retention_seed in (20260819, 20260829, 20260907)
            for dataset in DATASETS
            for retention in NON_FULL_RETENTIONS
        ]]
    )
    observations = selected_repeated_observations(repeated_runs, repeated_selections)
    variance = analyze_repeated_variance(observations)
    for name, table in (
        ("repeated_selected_observations.csv", observations),
        ("repeated_variance_components.csv", variance),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)

    expanded_confirmation = _collect_csvs(
        [
            robustness_root
            / "explicit_confirmation"
            / dataset
            / retention
            / "runs.csv"
            for dataset in DATASETS
            for retention in RETENTIONS
        ]
    )
    expanded_test = analyze_explicit_confirmation(expanded_confirmation)
    expanded_test_path = output_dir / "expanded_post_confirmation_regret.csv"
    expanded_test.to_csv(expanded_test_path, index=False)
    outputs.append(expanded_test_path)

    repeated_confirmation = _collect_csvs(
        [
            robustness_root
            / "explicit_repeated_confirmation"
            / f"seed_{retention_seed}"
            / dataset
            / retention
            / "runs.csv"
            for retention_seed in (20260819, 20260829, 20260907)
            for dataset in DATASETS
            for retention in NON_FULL_RETENTIONS
        ]
    )
    repeated_test = analyze_explicit_confirmation(repeated_confirmation)
    repeated_test_summary = summarize_repeated_test_regret(repeated_test)
    for name, table in (
        ("repeated_post_confirmation_regret_by_sample.csv", repeated_test),
        ("repeated_post_confirmation_regret_summary.csv", repeated_test_summary),
    ):
        path = output_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)

    bpr_refine = _collect_csvs(
        [
            robustness_root / "bpr_refine" / dataset / retention / "runs.csv"
            for dataset in DATASETS
            for retention in RETENTIONS
        ]
    )
    bpr = analyze_bpr_refine(bpr_refine)
    bpr_path = output_dir / "bpr_independent_selection_diagnostics.csv"
    bpr.to_csv(bpr_path, index=False)
    outputs.append(bpr_path)
    bpr_confirmation = _collect_csvs(
        [
            robustness_root / "bpr_confirmation" / dataset / retention / "runs.csv"
            for dataset in DATASETS
            for retention in RETENTIONS
        ]
    )
    bpr_sampler = analyze_bpr_sampler_confirmation(bpr_confirmation)
    sampler_path = output_dir / "bpr_negative_sampler_post_confirmation.csv"
    bpr_sampler.to_csv(sampler_path, index=False)
    outputs.append(sampler_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    root = _repo_root(args.config)
    for path in run_analysis(root):
        print(path)


if __name__ == "__main__":
    main()
