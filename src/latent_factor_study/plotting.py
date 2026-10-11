"""Generate publication figures from machine-readable journal-study results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import _repo_root
from .plot_theme import THESIS_PALETTE, apply_thesis_plot_style, style_axis


DATASETS = (
    ("movielens_1m", "MovieLens 1M"),
    ("ciaodvd", "CiaoDVD"),
    ("yelp_tampa_restaurants", "Yelp–Tampa"),
)
RETENTIONS = ("r100", "r075", "r050", "r025")
RETENTION_LABELS = ("100%", "75%", "50%", "25%")


def _style() -> None:
    apply_thesis_plot_style(base_font_size=9)


def plot_selection_regret(frame: pd.DataFrame, output_prefix: Path) -> list[Path]:
    """Plot primary NDCG@10 regret and user-clustered confidence intervals."""
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharex=True, sharey=True)
    y = np.arange(len(RETENTIONS))
    for axis, (dataset, title) in zip(axes, DATASETS):
        subset = frame.set_index(["dataset", "retention_label"]).loc[dataset]
        values = np.asarray(
            [subset.loc[retention, "delta_rank_ndcg@10"] for retention in RETENTIONS]
        )
        lower = np.asarray(
            [subset.loc[retention, "delta_rank_ci_lower"] for retention in RETENTIONS]
        )
        upper = np.asarray(
            [subset.loc[retention, "delta_rank_ci_upper"] for retention in RETENTIONS]
        )
        axis.errorbar(
            values,
            y,
            xerr=np.vstack([values - lower, upper - values]),
            fmt="o",
            markersize=4,
            color=THESIS_PALETTE["blue"],
            ecolor=THESIS_PALETTE["blue"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.4,
            capsize=3.0,
        )
        axis.axvline(0.0, color=THESIS_PALETTE["ink"], linewidth=0.9)
        axis.axvline(
            0.005,
            color=THESIS_PALETTE["orange"],
            linewidth=1.2,
            linestyle="--",
        )
        axis.set_title(title, pad=7)
        style_axis(axis, grid_axis="x")
    axes[0].set_yticks(y, RETENTION_LABELS)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Training retention")
    axes[1].set_xlabel(r"Test $\Delta$NDCG@10 (NDCG-selected $-$ RMSE-selected)")
    figure.tight_layout(w_pad=0.8)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def plot_model_boundary(
    baseline: pd.DataFrame, bpr: pd.DataFrame, output_prefix: Path
) -> list[Path]:
    """Compare test NDCG@10 for explicit MF, popularity, and BPR-MF."""
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharex=True)
    x = np.arange(len(RETENTIONS))
    for axis, (dataset, title) in zip(axes, DATASETS):
        base = baseline.set_index(["dataset", "retention_label"]).loc[dataset]
        pairwise = bpr.set_index(["dataset", "retention_label"]).loc[dataset]
        explicit = np.asarray(
            [
                base.loc[retention, "explicit_ndcg_selected_mean_ndcg@10"]
                for retention in RETENTIONS
            ]
        )
        popularity = np.asarray(
            [base.loc[retention, "popularity_mean_ndcg@10"] for retention in RETENTIONS]
        )
        bpr_values = np.asarray(
            [pairwise.loc[retention, "bpr_selected_mean_ndcg@10"] for retention in RETENTIONS]
        )
        axis.plot(
            x,
            explicit,
            marker="o",
            color=THESIS_PALETTE["blue"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Explicit MF",
        )
        axis.plot(
            x,
            popularity,
            marker="s",
            color=THESIS_PALETTE["orange"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Popularity",
        )
        axis.plot(
            x,
            bpr_values,
            marker="^",
            color=THESIS_PALETTE["green"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="BPR-MF",
        )
        axis.set_title(title, pad=7)
        axis.set_xticks(x, RETENTION_LABELS)
        style_axis(axis, grid_axis="y")
    axes[0].set_ylabel("Mean test NDCG@10")
    axes[1].set_xlabel("Training retention")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def plot_uncertainty_framework(
    construction: pd.DataFrame,
    prospective: pd.DataFrame,
    output_prefix: Path,
) -> list[Path]:
    """Separate retrospective intervals from the prospective decision bound."""
    _style()
    panels = (*DATASETS, ("epinions", "Epinions (prospective)"))
    figure, axes = plt.subplots(1, 4, figsize=(7.15, 2.75), sharex=True, sharey=True)
    y = np.arange(len(RETENTIONS))
    construction_indexed = construction.set_index(["dataset", "retention_label"])
    prospective_indexed = prospective.set_index(["dataset", "retention_label"])
    for axis, (dataset, title) in zip(axes, panels):
        source = prospective_indexed if dataset == "epinions" else construction_indexed
        subset = source.loc[dataset]
        values = np.asarray(
            [
                subset.loc[retention, "ndcg_regret_framework_vs_ndcg"]
                if dataset == "epinions"
                else subset.loc[retention, "ndcg_regret_framework_vs_ndcg_mean"]
                for retention in RETENTIONS
            ]
        )
        if dataset == "epinions":
            upper = np.asarray(
                [
                    subset.loc[retention, "ndcg_regret_one_sided_95_upper"]
                    for retention in RETENTIONS
                ]
            )
            xerr = np.vstack([np.zeros_like(values), np.maximum(0.0, upper - values)])
            color = THESIS_PALETTE["green"]
        else:
            lower = np.asarray(
                [
                    subset.loc[
                        retention, "ndcg_regret_framework_vs_ndcg_ci_lower"
                    ]
                    for retention in RETENTIONS
                ]
            )
            upper = np.asarray(
                [
                    subset.loc[
                        retention, "ndcg_regret_framework_vs_ndcg_ci_upper"
                    ]
                    for retention in RETENTIONS
                ]
            )
            xerr = np.vstack([values - lower, upper - values])
            color = THESIS_PALETTE["blue"]
        axis.errorbar(
            values,
            y,
            xerr=xerr,
            fmt="o",
            markersize=4,
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.4,
            capsize=3.0,
        )
        axis.axvline(0.0, color=THESIS_PALETTE["ink"], linewidth=0.9)
        axis.axvline(
            0.005,
            color=THESIS_PALETTE["orange"],
            linewidth=1.2,
            linestyle="--",
        )
        axis.set_title(title, pad=7, fontsize=8.5)
        style_axis(axis, grid_axis="x")
    axes[0].set_yticks(y, RETENTION_LABELS)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Training retention")
    figure.supxlabel(
        r"Test NDCG@10 regret (NDCG-selected $-$ uncertainty-aware)", y=0.01
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.65)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def plot_epinions_validation_surface(
    selections: dict[str, dict], output_prefix: Path
) -> list[Path]:
    """Show validation RMSE against the frozen operational regret screen."""
    _style()
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.1))
    flat_axes = axes.ravel()
    for axis, retention, label in zip(flat_axes, RETENTIONS, RETENTION_LABELS):
        selection = selections[retention]["selection"]
        frame = pd.DataFrame(selection["configuration_table"])
        axis.scatter(
            frame["mean_validation_rmse"],
            frame["paired_regret_upper_bound"],
            s=19,
            color=THESIS_PALETTE["blue"],
            alpha=0.62,
            edgecolors="white",
            linewidths=0.45,
            label="Candidate configuration",
            zorder=2,
        )

        def selected_row(role: str) -> pd.Series:
            chosen = selection[role]
            matches = frame[
                (frame["dimension"] == chosen["dimension"])
                & np.isclose(frame["regularization"], chosen["regularization"])
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one {role} row for {retention}.")
            return matches.iloc[0]

        ndcg_row = selected_row("ndcg_selected")
        framework_row = selected_row("framework_selected")
        axis.scatter(
            [ndcg_row["mean_validation_rmse"]],
            [ndcg_row["paired_regret_upper_bound"]],
            marker="D",
            s=46,
            color=THESIS_PALETTE["orange"],
            edgecolors="white",
            linewidths=0.8,
            label="NDCG-selected",
            zorder=4,
        )
        axis.scatter(
            [framework_row["mean_validation_rmse"]],
            [framework_row["paired_regret_upper_bound"]],
            marker="*",
            s=100,
            color=THESIS_PALETTE["green"],
            edgecolors=THESIS_PALETTE["ink"],
            linewidths=0.55,
            label="RMSE-selected = UA",
            zorder=5,
        )
        axis.axhline(
            0.005,
            color=THESIS_PALETTE["orange"],
            linewidth=1.2,
            linestyle="--",
            label=r"Admissibility margin $\delta=0.005$",
            zorder=1,
        )
        axis.set_ylim(bottom=-0.00015, top=0.00545)
        axis.set_title(f"{label} retention", pad=7)
        style_axis(axis, grid_axis="y")
    axes[1, 0].set_xlabel("Mean validation RMSE")
    axes[1, 1].set_xlabel("Mean validation RMSE")
    figure.supylabel("Operational paired-seed regret upper bound", x=0.012)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.tight_layout(rect=(0.04, 0, 1, 0.90), h_pad=1.0, w_pad=1.0)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/latent_factor_beyond_rmse/datasets.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("regenerated/figures"),
    )
    args = parser.parse_args()
    root = _repo_root(args.config)
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    primary = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/confirmation/analysis/mf_selection_regret.csv"
    )
    for path in plot_selection_regret(primary, output_dir / "selection_regret"):
        print(path)
    baseline_path = (
        root
        / "outputs/latent_factor_beyond_rmse/baseline_confirmation/analysis/baseline_comparison.csv"
    )
    bpr_path = (
        root
        / "outputs/latent_factor_beyond_rmse/bpr_confirmation/analysis/bpr_boundary_comparison.csv"
    )
    if baseline_path.exists() and bpr_path.exists():
        for path in plot_model_boundary(
            pd.read_csv(baseline_path),
            pd.read_csv(bpr_path),
            output_dir / "model_boundary",
        ):
            print(path)
    construction_path = (
        root
        / "outputs/latent_factor_beyond_rmse/framework_construction/analysis/"
        "construction_framework_results.csv"
    )
    prospective_path = (
        root
        / "outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/analysis/"
        "prospective_external_validation.csv"
    )
    if construction_path.exists() and prospective_path.exists():
        for path in plot_uncertainty_framework(
            pd.read_csv(construction_path),
            pd.read_csv(prospective_path),
            output_dir / "uncertainty_framework_validation",
        ):
            print(path)
    selection_root = (
        root
        / "outputs/latent_factor_beyond_rmse/prospective_epinions/development/epinions"
    )
    selection_paths = {
        retention: selection_root / retention / "framework_selection.json"
        for retention in RETENTIONS
    }
    if all(path.exists() for path in selection_paths.values()):
        selections = {
            retention: json.loads(path.read_text(encoding="utf-8"))
            for retention, path in selection_paths.items()
        }
        for path in plot_epinions_validation_surface(
            selections, output_dir / "epinions_validation_surface"
        ):
            print(path)


if __name__ == "__main__":
    main()
