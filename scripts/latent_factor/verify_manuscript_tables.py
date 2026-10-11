#!/usr/bin/env python3
"""Verify generated manuscript tables against their machine-readable sources.

The check is intentionally conservative: it verifies that every registered
table and source exists, that the generated table has the expected number of
data rows, records SHA-256 hashes, checks canonical primary values in the
clipping supplement, and optionally confirms sentinel values in compiled PDFs.
It does not replace the analysis functions that generate the tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TABLE_FIXTURES = Path("tests/fixtures/tables")


@dataclass(frozen=True)
class TableSpec:
    tex: str
    sources: tuple[str, ...]
    expected_rows: int


TABLES = (
    TableSpec("dataset_characterization.tex", ("data/processed/latent_factor_beyond_rmse/movielens_1m/manifest.json", "data/processed/latent_factor_beyond_rmse/ciaodvd/manifest.json", "data/processed/latent_factor_beyond_rmse/yelp_tampa_restaurants/manifest.json"), 3),
    TableSpec("realized_sparsity.tex", ("data/splits/latent_factor_beyond_rmse/movielens_1m/sparsity_manifest.json", "data/splits/latent_factor_beyond_rmse/ciaodvd/sparsity_manifest.json", "data/splits/latent_factor_beyond_rmse/yelp_tampa_restaurants/sparsity_manifest.json"), 12),
    TableSpec("candidate_diagnostics.tex", ("outputs/latent_factor_beyond_rmse/revision_analysis/candidate_diagnostics.csv",), 12),
    TableSpec("configuration_agreement.tex", ("outputs/latent_factor_beyond_rmse/development/development_landscape_summary.csv", "configs/latent_factor_beyond_rmse/selection_lock.yaml"), 12),
    TableSpec("configuration_stability.tex", ("outputs/latent_factor_beyond_rmse/revision_analysis/configuration_diagnostics.csv",), 12),
    TableSpec("mf_selection_regret.tex", ("outputs/latent_factor_beyond_rmse/confirmation/analysis/mf_selection_regret.csv",), 12),
    TableSpec("prediction_regret.tex", ("outputs/latent_factor_beyond_rmse/revision_analysis/prediction_regret.csv",), 12),
    TableSpec("framework_construction.tex", ("outputs/latent_factor_beyond_rmse/framework_construction/analysis/construction_framework_results.csv",), 12),
    TableSpec("prospective_framework.tex", ("outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/analysis/prospective_external_validation.csv",), 8),
    TableSpec("retention_contrasts.tex", ("outputs/latent_factor_beyond_rmse/revision_analysis/retention_contrasts.csv",), 9),
    TableSpec("model_boundary.tex", ("outputs/latent_factor_beyond_rmse/baseline_confirmation/analysis/baseline_comparison.csv", "outputs/latent_factor_beyond_rmse/bpr_confirmation/analysis/bpr_boundary_comparison.csv"), 12),
    TableSpec("popularity_diagnostics.tex", ("outputs/latent_factor_beyond_rmse/revision_analysis/popularity_diagnostics.csv",), 12),
    TableSpec("popularity_tie_sensitivity.tex", ("outputs/latent_factor_beyond_rmse/second_review_analysis/popularity_tie_sensitivity.csv",), 12),
    TableSpec("expanded_robustness.tex", ("outputs/latent_factor_beyond_rmse/robustness/analysis/expanded_selection_diagnostics.csv", "outputs/latent_factor_beyond_rmse/robustness/analysis/expanded_post_confirmation_regret.csv"), 12),
    TableSpec("repeated_retention_robustness.tex", ("outputs/latent_factor_beyond_rmse/robustness/analysis/repeated_post_confirmation_regret_summary.csv",), 9),
    TableSpec("bpr_robustness.tex", ("outputs/latent_factor_beyond_rmse/robustness/analysis/bpr_independent_selection_diagnostics.csv", "outputs/latent_factor_beyond_rmse/robustness/analysis/bpr_negative_sampler_post_confirmation.csv"), 12),
    TableSpec("bpr_negative_pool_diagnostics.tex", ("outputs/latent_factor_beyond_rmse/second_review_analysis/bpr_negative_pool_diagnostics.csv",), 12),
    TableSpec("training_cost.tex", ("outputs/latent_factor_beyond_rmse/computational_analysis/confirmation_training_cost_summary.csv",), 12),
    TableSpec("selection_cost.tex", ("outputs/latent_factor_beyond_rmse/revision_analysis/selection_cost.csv",), 12),
    TableSpec("robustness_search_cost.tex", ("outputs/latent_factor_beyond_rmse/second_review_analysis/robustness_search_cost.csv",), 3),
    TableSpec("supplementary_ndcg20.tex", ("outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/secondary_metric_differences.csv",), 12),
    TableSpec("supplementary_recall10.tex", ("outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/secondary_metric_differences.csv",), 12),
    TableSpec("supplementary_recall20.tex", ("outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/secondary_metric_differences.csv",), 12),
    TableSpec("supplementary_clipping.tex", ("outputs/latent_factor_beyond_rmse/confirmation/analysis/mf_selection_regret.csv", "outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/secondary_metric_differences.csv"), 12),
    TableSpec("supplementary_five_star.tex", ("outputs/latent_factor_beyond_rmse/five_star_sensitivity/analysis/mf_selection_regret.csv",), 12),
    TableSpec("supplementary_activity_movielens_1m.tex", ("outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/activity_stratified_ndcg_regret.csv",), 16),
    TableSpec("supplementary_activity_ciaodvd.tex", ("outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/activity_stratified_ndcg_regret.csv",), 16),
    TableSpec("supplementary_activity_yelp_tampa_restaurants.tex", ("outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/activity_stratified_ndcg_regret.csv",), 16),
    TableSpec("supplementary_mae.tex", ("outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/secondary_metric_differences.csv",), 12),
    TableSpec("supplementary_primary_epochs.tex", ("outputs/latent_factor_beyond_rmse/second_review_analysis/primary_epoch_audit.csv",), 4),
    TableSpec("supplementary_bias_only.tex", ("outputs/latent_factor_beyond_rmse/baseline_confirmation/analysis/baseline_comparison.csv",), 24),
    TableSpec("supplementary_prospective_secondary.tex", ("outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/analysis/prospective_external_validation.csv",), 8),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_rows(tex: str) -> int:
    headers = ("Dataset &", "Stage &", "Retention &", "Ret. &")
    return sum(
        1
        for line in tex.splitlines()
        if "&" in line
        and line.rstrip().endswith(r"\\")
        and not line.lstrip().startswith(headers)
    )


def verify_primary_clipping(root: Path, tex: str) -> None:
    primary = pd.read_csv(
        root / "outputs/latent_factor_beyond_rmse/confirmation/analysis/mf_selection_regret.csv"
    )
    for row in primary.itertuples(index=False):
        tokens = (
            f"{row.delta_mean_user_rmse:+.5f}",
            f"{row.delta_mean_user_rmse_ci_lower:+.5f}",
            f"{row.delta_mean_user_rmse_ci_upper:+.5f}",
        )
        if not all(token in tex for token in tokens):
            raise AssertionError(
                f"Canonical clipped RMSE values missing for {row.dataset}/{row.retention_label}"
            )


def verify_prospective_framework(root: Path, tex: str) -> None:
    """Check the externally validated NDCG values and descriptive loss ratios."""
    frame = pd.read_csv(
        root
        / "outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/analysis/"
        "prospective_external_validation.csv"
    )
    for _, row in frame.iterrows():
        relative_loss = (
            100.0
            * float(row["ndcg_regret_framework_vs_ndcg"])
            / float(row["ndcg_selected_ndcg@10_mean"])
        )
        tokens = (
            f"{row['framework_selected_ndcg@10_mean']:.5f}",
            f"{row['ndcg_selected_ndcg@10_mean']:.5f}",
            f"{row['ndcg_regret_framework_vs_ndcg']:+.5f}",
            f"{row['ndcg_regret_one_sided_95_upper']:+.5f}",
            f"{relative_loss:.1f}\\%",
        )
        if not all(token in tex for token in tokens):
            raise AssertionError(
                f"Canonical prospective values missing for {row['retention_label']}"
            )


def pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF sentinel checks require pypdf; install the repository test dependencies."
        ) from error
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--main-pdf", type=Path)
    parser.add_argument("--supplement-pdf", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("manifests/manuscript_table_verification.csv"),
    )
    parser.add_argument(
        "--check-report",
        action="store_true",
        help="Verify that the existing report is current without rewriting it.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    generated = root / TABLE_FIXTURES
    records: list[dict[str, str | int]] = []
    for spec in TABLES:
        tex_path = generated / spec.tex
        if not tex_path.exists():
            raise FileNotFoundError(tex_path)
        tex = tex_path.read_text(encoding="utf-8")
        observed = data_rows(tex)
        if observed != spec.expected_rows:
            raise AssertionError(
                f"{spec.tex}: expected {spec.expected_rows} data rows, observed {observed}"
            )
        for source in spec.sources:
            source_path = root / source
            if not source_path.exists():
                raise FileNotFoundError(source_path)
        if spec.tex == "supplementary_clipping.tex":
            verify_primary_clipping(root, tex)
        if spec.tex == "prospective_framework.tex":
            verify_prospective_framework(root, tex)
        records.append(
            {
                "table_tex": str(TABLE_FIXTURES / spec.tex),
                "table_sha256": sha256(tex_path),
                "data_rows": observed,
                "source_paths": ";".join(spec.sources),
                "source_sha256": ";".join(sha256(root / item) for item in spec.sources),
                "status": "verified",
            }
        )

    if args.main_pdf:
        text = pdf_text(args.main_pdf)
        for token in ("2,100-run development grid", "declared 0.005 tolerance"):
            if token not in re.sub(r"\s+", " ", text):
                raise AssertionError(f"Main PDF sentinel missing: {token}")
    if args.supplement_pdf:
        text = re.sub(r"\s+", " ", pdf_text(args.supplement_pdf))
        for token in ("Mean-user prediction-MAE sensitivity", "2,084", "Complete bias-only boundary"):
            if token not in text:
                raise AssertionError(f"Supplement PDF sentinel missing: {token}")

    report = args.report if args.report.is_absolute() else root / args.report
    if args.check_report:
        with report.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        expected = [
            {key: str(value) for key, value in record.items()} for record in records
        ]
        if existing != expected:
            raise AssertionError(f"Table-verification report is stale: {report}")
    else:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(records[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(records)
    print(f"Verified {len(records)} generated tables; report: {report}")


if __name__ == "__main__":
    main()
