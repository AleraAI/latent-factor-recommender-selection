"""Generate manuscript tables from machine-readable study artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve


DATASET_ORDER = ("movielens_1m", "ciaodvd", "yelp_tampa_restaurants")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _integer(value: int) -> str:
    return f"{value:,}"


def _percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}\\%"


def _regularization(value: float) -> str:
    labels = {
        0.0: "0",
        0.000001: r"10^{-6}",
        0.00001: r"10^{-5}",
        0.0001: r"10^{-4}",
        0.001: r"10^{-3}",
        0.01: r"10^{-2}",
        0.1: r"10^{-1}",
        0.3: "0.3",
        1.0: "1",
    }
    return labels[float(value)]


def _configuration(dimension: int, regularization: float) -> str:
    return rf"$({int(dimension)},{_regularization(regularization)})$"


def _learning_rate(value: float) -> str:
    labels = {
        0.001: r"10^{-3}",
        0.003: "0.003",
        0.01: r"10^{-2}",
        0.03: "0.03",
    }
    return labels[float(value)]


def _bpr_configuration(row: pd.Series) -> str:
    return (
        rf"$({int(row['dimension'])},{_regularization(float(row['regularization']))},"
        rf"{_learning_rate(float(row['learning_rate']))},{int(row['negatives_per_positive'])})$"
    )


def generate_dataset_table(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    rows: list[str] = []
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    for name in DATASET_ORDER:
        processed_dir = _resolve(root, config["datasets"][name]["output_dir"])
        manifest = _load(processed_dir / "manifest.json")
        raw = manifest["raw_or_scoped_statistics"]
        deduplicated = manifest["deduplicated_statistics"]
        final = manifest["processed_statistics"]
        split = manifest["split_statistics"]
        rows.append(
            " & ".join(
                [
                    display[name],
                    _integer(raw["rows"]),
                    _integer(deduplicated["rows"]),
                    _integer(final["users"]),
                    _integer(final["items"]),
                    _integer(final["rows"]),
                    _percent(final["density"], 3),
                    _integer(split["train"]["rows"]),
                    _integer(split["validation"]["rows"]),
                    _integer(split["test"]["rows"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Machine-generated preprocessing and chronological-split statistics. Raw denotes the full source for MovieLens and CiaoDVD and the predeclared Tampa restaurant scope for Yelp. Density is computed after iterative 10-core filtering.}",
            r"\label{tab:processed-datasets}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Raw & Dedup. & Users & Items & 10-core & Density & Train & Valid. & Test \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_sparsity_table(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    sparsity_root = _resolve(root, config["controlled_sparsity"]["output_root"])
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    labels = (("r100", "100\\%"), ("r075", "75\\%"), ("r050", "50\\%"), ("r025", "25\\%"))
    rows: list[str] = []
    for name in DATASET_ORDER:
        manifest = _load(sparsity_root / name / "sparsity_manifest.json")
        for index, (label, nominal) in enumerate(labels):
            values = manifest["conditions"][label]
            dataset_cell = display[name] if index == 0 else ""
            rows.append(
                " & ".join(
                    [
                        dataset_cell,
                        nominal,
                        _integer(values["realized_rows"]),
                        _percent(values["realized_retention"], 2),
                        _integer(values["items"]),
                        _integer(values["repair_rows_added"]),
                        _integer(values["minimum_realized_user_degree"]),
                    ]
                )
                + r" \\"
            )
        if name != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Realized controlled-sparsity conditions. Repair rows preserve the full-training item catalog without inspecting held-out identities.}",
            r"\label{tab:realized-sparsity}",
            r"\small",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Nominal & Train rows & Realized & Items & Repair rows & Min. user degree \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_agreement_table(config_path: Path) -> str:
    root = _repo_root(config_path)
    frame = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/development/development_landscape_summary.csv"
    )
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    rows: list[str] = []
    for dataset in DATASET_ORDER:
        subset = frame[frame["dataset"] == dataset]
        for index, retention in enumerate(("r100", "r075", "r050", "r025")):
            row = subset[subset["retention_label"] == retention].iloc[0]
            rows.append(
                " & ".join(
                    [
                        display[dataset] if index == 0 else "",
                        str(int(retention[1:])) + r"\%",
                        f"{row['spearman_rho']:.3f}",
                        f"{row['kendall_tau_b']:.3f}",
                        str(int(row["top3_overlap"])),
                        _configuration(
                            int(row["rmse_selected_dimension"]),
                            float(row["rmse_selected_regularization"]),
                        ),
                        _configuration(
                            int(row["ndcg_selected_dimension"]),
                            float(row["ndcg_selected_regularization"]),
                        ),
                        "Yes" if bool(row["same_top_configuration"]) else "No",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Agreement between validation prediction and ranking objectives across the 35 seed-mean MF configurations. Correlations compare negative RMSE with NDCG@10; Top-3 is the number of configurations shared by the two top-three sets. Configuration tuples are $(d,\lambda)$.}",
            r"\label{tab:configuration-agreement}",
            r"\small",
            r"\begin{tabular}{@{}lrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & $\rho$ & $\tau_b$ & Top-3 & RMSE choice & NDCG choice & Same \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_regret_table(config_path: Path) -> str:
    root = _repo_root(config_path)
    frame = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/confirmation/analysis/mf_selection_regret.csv"
    )
    evidence_path = (
        root
        / "outputs/latent_factor_beyond_rmse/revision_analysis/practical_evidence.csv"
    )
    if evidence_path.exists():
        evidence = pd.read_csv(evidence_path)[
            ["dataset", "retention_label", "practical_evidence_class"]
        ]
        frame = frame.merge(
            evidence,
            on=["dataset", "retention_label"],
            how="left",
            validate="one_to_one",
        )
    else:
        frame["practical_evidence_class"] = frame["practically_meaningful"].map(
            {True: "point above", False: "point below"}
        )
    lock = yaml.safe_load(
        (root / "configs/latent_factor_beyond_rmse/selection_lock.yaml").read_text(
            encoding="utf-8"
        )
    ) or {}
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    rows: list[str] = []
    for dataset in DATASET_ORDER:
        subset = frame[frame["dataset"] == dataset]
        for index, retention in enumerate(("r100", "r075", "r050", "r025")):
            row = subset[subset["retention_label"] == retention].iloc[0]
            selected = lock["selections"][dataset][retention]
            rows.append(
                " & ".join(
                    [
                        display[dataset] if index == 0 else "",
                        str(int(retention[1:])) + r"\%",
                        _configuration(**selected["rmse_selected"]),
                        _configuration(**selected["ndcg_selected"]),
                        f"{row['delta_rank_ndcg@10']:+.4f}",
                        (
                            f"[{row['delta_rank_ci_lower']:+.4f}, "
                            f"{row['delta_rank_ci_upper']:+.4f}]"
                        ),
                        f"{row['delta_rank_holm_p']:.4f}",
                        {
                            "clearly above": "Above",
                            "clearly below": "Below",
                            "inconclusive": "Uncertain",
                            "identical selection": "Identical",
                            "point above": "Point above",
                            "point below": "Point below",
                        }.get(str(row["practical_evidence_class"]), "Uncertain"),
                    ]
                )
                + r" \\"
            )
        if dataset != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Locked-test ranking regret from RMSE-based selection. $\Delta_{\rm rank}$ is NDCG@10 of the NDCG-selected configuration minus that of the RMSE-selected configuration after per-user averaging across five fresh seeds. Confidence intervals use 10,000 user-cluster bootstrap resamples; $p_{\rm Holm}$ adjusts the twelve primary paired randomization tests. Evidence is Above when the entire interval exceeds the predeclared 0.005 threshold, Below when the interval is entirely below it, Uncertain when it overlaps the threshold, and Identical when both selectors chose the same configuration.}",
            r"\label{tab:mf-selection-regret}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & RMSE choice & NDCG choice & $\Delta_{\rm rank}$ & 95\% CI & $p_{\rm Holm}$ & Evidence \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_boundary_table(config_path: Path) -> str:
    """Generate the locked-test model-family boundary comparison."""
    root = _repo_root(config_path)
    baseline = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/baseline_confirmation/analysis/baseline_comparison.csv"
    ).set_index(["dataset", "retention_label"])
    bpr = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/bpr_confirmation/analysis/bpr_boundary_comparison.csv"
    ).set_index(["dataset", "retention_label"])
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    rows: list[str] = []
    for dataset in DATASET_ORDER:
        for index, retention in enumerate(("r100", "r075", "r050", "r025")):
            base = baseline.loc[(dataset, retention)]
            ranker = bpr.loc[(dataset, retention)]
            rows.append(
                " & ".join(
                    [
                        display[dataset] if index == 0 else "",
                        str(int(retention[1:])) + r"\%",
                        f"{base['explicit_ndcg_selected_mean_ndcg@10']:.4f}",
                        f"{base['popularity_mean_ndcg@10']:.4f}",
                        f"{ranker['bpr_selected_mean_ndcg@10']:.4f}",
                        (
                            f"{ranker['delta_bpr_minus_explicit_ndcg@10']:+.4f} "
                            f"[{ranker['delta_ci_lower']:+.4f},"
                            f"{ranker['delta_ci_upper']:+.4f}]"
                        ),
                        (
                            f"{ranker['delta_bpr_minus_popularity_ndcg@10']:+.4f} "
                            f"[{ranker['delta_popularity_ci_lower']:+.4f},"
                            f"{ranker['delta_popularity_ci_upper']:+.4f}]"
                        ),
                    ]
                )
                + r" \\"
            )
        if dataset != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Locked-test ranking boundary. Values are mean per-user NDCG@10 under the common exact train-known-catalog protocol. $\Delta_E$ and $\Delta_P$ are BPR-MF minus the NDCG-selected explicit MF and popularity, respectively; brackets give 95\% user-cluster bootstrap intervals. Every BPR--explicit comparison has $p_{\rm Holm}=0.0012$; adjusted BPR--popularity outcomes are summarized in the text.}",
            r"\label{tab:model-boundary}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Explicit & Pop. & BPR & $\Delta_E$ [95\% CI] & $\Delta_P$ [95\% CI] \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def _timing(row: pd.Series) -> str:
    return (
        f"{row['median_training_seconds']:.2f} "
        f"[{row['q25_training_seconds']:.2f}--{row['q75_training_seconds']:.2f}]"
    )


def generate_cost_table(config_path: Path) -> str:
    """Generate confirmation-stage training-time summaries."""
    root = _repo_root(config_path)
    frame = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/computational_analysis/confirmation_training_cost_summary.csv"
    ).set_index(["dataset", "retention_label", "model", "selection_role"])
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    rows: list[str] = []
    for dataset in DATASET_ORDER:
        for index, retention in enumerate(("r100", "r075", "r050", "r025")):
            rows.append(
                " & ".join(
                    [
                        display[dataset] if index == 0 else "",
                        str(int(retention[1:])) + r"\%",
                        _timing(frame.loc[(dataset, retention, "bias_only", "ndcg_selected")]),
                        _timing(frame.loc[(dataset, retention, "explicit_mf", "rmse_selected")]),
                        _timing(frame.loc[(dataset, retention, "explicit_mf", "ndcg_selected")]),
                        _timing(frame.loc[(dataset, retention, "bpr_mf", "bpr_selected")]),
                    ]
                )
                + r" \\"
            )
        if dataset != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Confirmation-stage fitting time in seconds on the declared single-threaded CPU environment. Each entry is median [interquartile range] over the five fresh-seed fits. Explicit MF is shown for both objective-specific selections; bias and BPR-MF use their NDCG-selected configurations.}",
            r"\label{tab:training-cost}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Bias & Explicit (RMSE) & Explicit (NDCG) & BPR-MF \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_expanded_robustness_table(config_path: Path) -> str:
    """Generate the expanded-grid, 100-epoch post-confirmation summary."""
    root = _repo_root(config_path)
    analysis_dir = root / "outputs/latent_factor_beyond_rmse/robustness/analysis"
    selections = pd.read_csv(
        analysis_dir / "expanded_selection_diagnostics.csv"
    )
    selections = selections[selections["epoch"] == 100].set_index(
        ["dataset", "retention_label", "selection_role"]
    )
    regret = pd.read_csv(
        analysis_dir / "expanded_post_confirmation_regret.csv"
    ).set_index(["dataset", "retention_label"])
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    rows: list[str] = []
    for dataset in DATASET_ORDER:
        for index, retention in enumerate(("r100", "r075", "r050", "r025")):
            rmse = selections.loc[(dataset, retention, "rmse_selected")]
            ndcg = selections.loc[(dataset, retention, "ndcg_selected")]
            effect = regret.loc[(dataset, retention)]
            boundary_count = int(rmse["any_grid_boundary"]) + int(
                ndcg["any_grid_boundary"]
            )
            rows.append(
                " & ".join(
                    [
                        display[dataset] if index == 0 else "",
                        str(int(retention[1:])) + r"\%",
                        _configuration(
                            int(rmse["dimension"]), float(rmse["regularization"])
                        ),
                        _configuration(
                            int(ndcg["dimension"]), float(ndcg["regularization"])
                        ),
                        f"{effect['delta_rank_ndcg@10']:+.4f}",
                        f"{effect['delta_prediction_rmse']:+.4f}",
                        f"{boundary_count}/2",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post-confirmation robustness on the expanded explicit-MF surface at 100 epochs. Configuration tuples are $(d,\lambda)$. Effects are descriptive means over five fresh robustness seeds: $\Delta_{\rm rank}$ is NDCG-selected minus RMSE-selected test NDCG@10 and $\Delta_{\rm RMSE}$ is the corresponding increase in test RMSE. Edge reports how many of the two selectors remain on a dimension or regularization boundary of the expanded grid.}",
            r"\label{tab:expanded-robustness}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & RMSE choice & NDCG choice & $\Delta_{\rm rank}$ & $\Delta_{\rm RMSE}$ & Edge \\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_repeated_retention_robustness_table(config_path: Path) -> str:
    """Generate post-confirmation regret across independent retention samples."""
    root = _repo_root(config_path)
    frame = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/robustness/analysis/repeated_post_confirmation_regret_summary.csv"
    ).set_index(["dataset", "retention_label"])
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    rows: list[str] = []
    for dataset in DATASET_ORDER:
        for index, retention in enumerate(("r075", "r050", "r025")):
            row = frame.loc[(dataset, retention)]
            rank_summary = (
                f"{row['mean_delta_rank_ndcg_at_10']:+.4f} "
                f"[{row['min_delta_rank_ndcg_at_10']:+.4f}, "
                f"{row['max_delta_rank_ndcg_at_10']:+.4f}]"
            )
            rows.append(
                " & ".join(
                    [
                        display[dataset] if index == 0 else "",
                        str(int(retention[1:])) + r"\%",
                        str(int(row["retention_seed_count"])),
                        rank_summary,
                        f"{row['std_delta_rank_ndcg_at_10']:.4f}",
                        f"{row['mean_delta_prediction_rmse']:+.4f}",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post-confirmation robustness across three independently prioritized nested retention samples. Each sample reselects from the expanded 100-epoch explicit-MF grid with three development seeds and evaluates the two choices with five fresh seeds. Ranking regret is shown as the across-sample mean [minimum, maximum]; SD is the between-sample standard deviation.}",
            r"\label{tab:repeated-retention-robustness}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & Samples & $\Delta_{\rm rank}$ mean [range] & SD & Mean $\Delta_{\rm RMSE}$ \\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_framework_construction_table(config_path: Path) -> str:
    """Generate explicitly exploratory framework results on construction data."""
    root = _repo_root(config_path)
    frame = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/framework_construction/analysis/"
        "construction_framework_results.csv"
    ).set_index(["dataset", "retention_label"])
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    rows: list[str] = []
    for dataset in DATASET_ORDER:
        for index, retention in enumerate(("r100", "r075", "r050", "r025")):
            row = frame.loc[(dataset, retention)]
            match = (
                "RMSE"
                if bool(row["framework_matches_rmse"])
                else ("NDCG" if bool(row["framework_matches_ndcg"]) else "New")
            )
            rows.append(
                " & ".join(
                    [
                        display[dataset] if index == 0 else "",
                        str(int(retention[1:])) + r"\%",
                        _configuration(
                            int(row["framework_dimension"]),
                            float(row["framework_regularization"]),
                        ),
                        match,
                        str(int(row["admissible_configurations"])),
                        (
                            f"{row['ndcg_regret_framework_vs_ndcg_mean']:+.4f} "
                            f"[{row['ndcg_regret_framework_vs_ndcg_ci_lower']:+.4f},"
                            f"{row['ndcg_regret_framework_vs_ndcg_ci_upper']:+.4f}]"
                        ),
                        (
                            f"{row['rmse_framework_minus_ndcg_mean']:+.4f} "
                            f"[{row['rmse_framework_minus_ndcg_ci_lower']:+.4f},"
                            f"{row['rmse_framework_minus_ndcg_ci_upper']:+.4f}]"
                        ),
                    ]
                )
                + r" \\"
            )
        if dataset != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{Post-confirmation framework-construction results. The uncertainty-aware (UA) selector was applied retrospectively to validation grids whose primary test results were already known; these values are exploratory and are not external validation. Match identifies whether UA equals an original single-objective choice; $|\mathcal{A}|$ is the number of validation configurations admitted by the uncertainty screen. $\Delta_{N}$ is NDCG-selected minus UA test NDCG@10, and $\Delta_{R}$ is UA minus NDCG-selected mean-user test RMSE. Brackets are two-sided 95\% user-cluster bootstrap intervals after five-seed averaging.}",
            r"\label{tab:framework-construction}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & UA choice & Match & $|\mathcal{A}|$ & $\Delta_N$ [95\% CI] & $\Delta_R$ [95\% CI] \\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_prospective_framework_table(config_path: Path) -> str:
    """Generate the prospectively locked Epinions framework decision table."""
    root = _repo_root(config_path)
    frame = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/analysis/"
        "prospective_external_validation.csv"
    ).set_index("retention_label")
    selection_rows: list[str] = []
    outcome_rows: list[str] = []
    for retention in ("r100", "r075", "r050", "r025"):
        row = frame.loc[retention]
        label = str(int(retention[1:])) + r"\%"
        relative_ranking_loss = (
            100.0
            * float(row["ndcg_regret_framework_vs_ndcg"])
            / float(row["ndcg_selected_ndcg@10_mean"])
        )
        selection_rows.append(
            " & ".join(
                [
                    label,
                    _configuration(
                        int(row["rmse_selected_dimension"]),
                        float(row["rmse_selected_regularization"]),
                    ),
                    _configuration(
                        int(row["ndcg_selected_dimension"]),
                        float(row["ndcg_selected_regularization"]),
                    ),
                    _configuration(
                        int(row["framework_selected_dimension"]),
                        float(row["framework_selected_regularization"]),
                    ),
                    str(int(row["admissible_configurations"])),
                ]
            )
            + r" \\"
        )
        outcome_rows.append(
            " & ".join(
                [
                    label,
                    f"{row['framework_selected_ndcg@10_mean']:.5f}",
                    f"{row['ndcg_selected_ndcg@10_mean']:.5f}",
                    f"{row['ndcg_regret_framework_vs_ndcg']:+.5f}",
                    f"{row['ndcg_regret_one_sided_95_upper']:+.5f}",
                    f"{relative_ranking_loss:.1f}\\%",
                    (
                        f"{row['framework_minus_ndcg_user_rmse']:+.4f} "
                        f"[{row['framework_minus_ndcg_user_rmse_ci_lower']:+.4f},"
                        f"{row['framework_minus_ndcg_user_rmse_ci_upper']:+.4f}]"
                    ),
                    "Pass" if bool(row["condition_passes_noninferiority"]) else "Fail",
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Prospectively locked external validation on Epinions. Configuration tuples are $(d,\lambda)$ and $|\mathcal{A}|$ is the validation admissible-set size. $N_U$ and $N_N$ are the uncertainty-aware (UA) and NDCG-selected mean test NDCG@10; $\Delta_N=N_N-N_U$. $U_{0.95}$ is its predeclared one-sided 95\% user-cluster bootstrap upper bound. A condition satisfies the predeclared absolute-margin criterion when $U_{0.95}\leq0.005$; the family claim requires all four conditions to pass. Relative loss is the descriptive ratio $100\Delta_N/N_N$ and does not replace the frozen absolute-margin endpoint. $\Delta_R$ is UA minus NDCG-selected mean-user test RMSE with a two-sided 95\% interval.}",
            r"\label{tab:prospective-framework}",
            r"\small",
            r"\textit{Panel A: selections}\par\smallskip",
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            r"Ret. & RMSE choice & NDCG choice & UA choice & $|\mathcal{A}|$ \\",
            r"\midrule",
            *selection_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip",
            r"\textit{Panel B: test outcomes}\par\smallskip",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrr@{}}",
            r"\toprule",
            r"Ret. & $N_U$ & $N_N$ & $\Delta_N$ & $U_{0.95}$ & Rel. loss & $\Delta_R$ [95\% CI] & Decision \\",
            r"\midrule",
            *outcome_rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_prospective_secondary_table(config_path: Path) -> str:
    """Generate predeclared secondary Epinions contrasts versus NDCG selection."""
    root = _repo_root(config_path)
    frame = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/analysis/"
        "prospective_external_validation.csv"
    ).set_index("retention_label")

    def estimate(row: pd.Series, metric: str) -> str:
        prefix = f"{metric}_framework_minus_ndcg"
        return (
            f"{row[f'{prefix}_mean']:+.4f} "
            f"[{row[f'{prefix}_ci_lower']:+.4f},{row[f'{prefix}_ci_upper']:+.4f}]"
        )

    ranking_rows: list[str] = []
    prediction_rows: list[str] = []
    for retention in ("r100", "r075", "r050", "r025"):
        row = frame.loc[retention]
        label = str(int(retention[1:])) + r"\%"
        ranking_rows.append(
            " & ".join(
                [
                    label,
                    estimate(row, "ndcg@20"),
                    estimate(row, "recall@10"),
                    estimate(row, "recall@20"),
                ]
            )
            + r" \\"
        )
        prediction_rows.append(
            " & ".join(
                [
                    label,
                    estimate(row, "prediction_mae"),
                    estimate(row, "prediction_raw_rmse"),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Predeclared secondary prospective Epinions outcomes. Every effect is uncertainty-aware minus NDCG-selected after averaging each user across five fresh seeds; brackets are two-sided 95\% user-cluster bootstrap intervals. Positive ranking values favor the framework, whereas negative ranking values indicate lower ranking effectiveness for it; negative MAE and raw-RMSE values favor the framework. These outcomes do not alter the primary absolute-margin NDCG@10 decision.}",
            r"\label{tab:supp-prospective-secondary}",
            r"\small",
            r"\textit{Panel A: ranking outcomes}\par\smallskip",
            r"\begin{tabular}{@{}lrrr@{}}",
            r"\toprule",
            r"Ret. & $\Delta$NDCG@20 & $\Delta$Recall@10 & $\Delta$Recall@20 \\",
            r"\midrule",
            *ranking_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip",
            r"\textit{Panel B: prediction outcomes}\par\smallskip",
            r"\begin{tabular}{@{}lrr@{}}",
            r"\toprule",
            r"Ret. & $\Delta$MAE & $\Delta$raw RMSE \\",
            r"\midrule",
            *prediction_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    ) + "\n"


def generate_bpr_robustness_table(config_path: Path) -> str:
    """Generate independent BPR tuning and negative-pool robustness results."""
    root = _repo_root(config_path)
    analysis_dir = root / "outputs/latent_factor_beyond_rmse/robustness/analysis"
    selections = pd.read_csv(
        analysis_dir / "bpr_independent_selection_diagnostics.csv"
    ).set_index(["dataset", "retention_label"])
    samplers = pd.read_csv(
        analysis_dir / "bpr_negative_sampler_post_confirmation.csv"
    ).pivot(
        index=["dataset", "retention_label"],
        columns="negative_mode",
        values=["mean_test_ndcg_at_10", "mean_sampled_full_positive_rate"],
    )
    explicit = pd.read_csv(
        analysis_dir / "expanded_post_confirmation_regret.csv"
    ).set_index(["dataset", "retention_label"])
    display = {
        "movielens_1m": "MovieLens 1M",
        "ciaodvd": "CiaoDVD",
        "yelp_tampa_restaurants": "Yelp--Tampa",
    }
    rows: list[str] = []
    for dataset in DATASET_ORDER:
        for index, retention in enumerate(("r100", "r075", "r050", "r025")):
            selection = selections.loc[(dataset, retention)]
            sampler = samplers.loc[(dataset, retention)]
            unobserved = float(
                sampler[("mean_test_ndcg_at_10", "retained_unobserved")]
            )
            oracle = float(
                sampler[("mean_test_ndcg_at_10", "full_training_positive_oracle")]
            )
            explicit_negative = float(
                sampler[("mean_test_ndcg_at_10", "retained_explicit_negative")]
            )
            expanded_explicit = float(
                explicit.loc[(dataset, retention), "mean_test_ndcg_at_10_ndcg_selected"]
            )
            withheld_positive_rate = float(
                sampler[
                    (
                        "mean_sampled_full_positive_rate",
                        "retained_unobserved",
                    )
                ]
            )
            rows.append(
                " & ".join(
                    [
                        display[dataset] if index == 0 else "",
                        str(int(retention[1:])) + r"\%",
                        _bpr_configuration(selection),
                        f"{unobserved:.4f}",
                        f"{oracle:.4f}",
                        f"{explicit_negative:.4f}",
                        f"{unobserved - expanded_explicit:+.4f}",
                        _percent(withheld_positive_rate, 2),
                    ]
                )
                + r" \\"
            )
        if dataset != DATASET_ORDER[-1]:
            rows.append(r"\addlinespace")
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post-confirmation independent BPR tuning and fixed-configuration negative-pool sensitivity. Configuration tuples are $(d,\lambda,\eta,m)$ for dimension, regularization, learning rate, and negatives per positive; every selected checkpoint is at 100 epochs. The configuration is selected for retained-unobserved sampling and then reused, without retuning, for the full-training-positive oracle (Oracle) and retained explicit-negative (Exp.-neg.) pools. Test NDCG@10 is averaged over five fresh seeds. $\Delta_X$ is Unobs. BPR minus the expanded NDCG-selected explicit MF. Withheld $+$ is the sampled-negative rate during epoch 100 for Unobs. items that are positives in full training; it is not a cumulative rate across epochs.}",
            r"\label{tab:bpr-robustness}",
            r"\small",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrr@{}}",
            r"\toprule",
            r"Dataset & Ret. & BPR configuration & Unobs. & Oracle & Exp.-neg. & $\Delta_X$ & Withheld $+$ \\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    ) + "\n"


def write_manuscript_tables(config_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "dataset_characterization.tex",
        output_dir / "realized_sparsity.tex",
        output_dir / "configuration_agreement.tex",
        output_dir / "mf_selection_regret.tex",
    ]
    paths[0].write_text(generate_dataset_table(config_path), encoding="utf-8")
    paths[1].write_text(generate_sparsity_table(config_path), encoding="utf-8")
    paths[2].write_text(generate_agreement_table(config_path), encoding="utf-8")
    paths[3].write_text(generate_regret_table(config_path), encoding="utf-8")
    boundary_path = (
        _repo_root(config_path)
        / "outputs/latent_factor_beyond_rmse/bpr_confirmation/analysis/bpr_boundary_comparison.csv"
    )
    if boundary_path.exists():
        model_boundary_path = output_dir / "model_boundary.tex"
        model_boundary_path.write_text(
            generate_boundary_table(config_path), encoding="utf-8"
        )
        paths.append(model_boundary_path)
    cost_path = (
        _repo_root(config_path)
        / "outputs/latent_factor_beyond_rmse/computational_analysis/confirmation_training_cost_summary.csv"
    )
    if cost_path.exists():
        training_cost_path = output_dir / "training_cost.tex"
        training_cost_path.write_text(generate_cost_table(config_path), encoding="utf-8")
        paths.append(training_cost_path)
    robustness_path = (
        _repo_root(config_path)
        / "outputs/latent_factor_beyond_rmse/robustness/analysis/expanded_post_confirmation_regret.csv"
    )
    if robustness_path.exists():
        robustness_tables = (
            (
                "expanded_robustness.tex",
                generate_expanded_robustness_table(config_path),
            ),
            (
                "repeated_retention_robustness.tex",
                generate_repeated_retention_robustness_table(config_path),
            ),
            ("bpr_robustness.tex", generate_bpr_robustness_table(config_path)),
        )
        for filename, content in robustness_tables:
            path = output_dir / filename
            path.write_text(content, encoding="utf-8")
            paths.append(path)
    construction_path = (
        _repo_root(config_path)
        / "outputs/latent_factor_beyond_rmse/framework_construction/analysis/"
        "construction_framework_results.csv"
    )
    if construction_path.exists():
        path = output_dir / "framework_construction.tex"
        path.write_text(
            generate_framework_construction_table(config_path), encoding="utf-8"
        )
        paths.append(path)
    prospective_path = (
        _repo_root(config_path)
        / "outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/analysis/"
        "prospective_external_validation.csv"
    )
    if prospective_path.exists():
        path = output_dir / "prospective_framework.tex"
        path.write_text(
            generate_prospective_framework_table(config_path), encoding="utf-8"
        )
        paths.append(path)
        path = output_dir / "supplementary_prospective_secondary.tex"
        path.write_text(
            generate_prospective_secondary_table(config_path), encoding="utf-8"
        )
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("regenerated/tables"),
    )
    args = parser.parse_args()
    root = _repo_root(args.config)
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    for path in write_manuscript_tables(args.config, output_dir):
        print(path)


if __name__ == "__main__":
    main()
