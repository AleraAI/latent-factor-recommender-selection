#!/usr/bin/env python3
"""Report fixed September revision aggregates; never fit or resample models.

The source contrasts contain stored two-sided percentile intervals. Negating
the stored NDCG regret reverses both interval endpoints for UA-minus-mean
reporting. Relative percentages divide the signed point regret by the matching
NDCG-selected reference's held-out NDCG, not by a validation or oracle value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

DATASETS = {
    "movielens_1m": "MovieLens 1M",
    "ciaodvd": "CiaoDVD",
    "yelp_tampa_restaurants": "Yelp--Tampa",
    "epinions": "Epinions",
}
RETENTIONS = ("r100", "r075", "r050", "r025")
MARGINS = (.001, .0025, .005, .010)
REGRET = "ndcg_regret_comparator_minus_selected"
RMSE = "global_rmse_selected_minus_comparator"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: float, digits: int = 5, signed: bool = True) -> str:
    if abs(value) < .5 * 10 ** -digits:
        value = 0.0
    return format(value, f"{'+' if signed and value != 0 else ''}.{digits}f")


def interval(low: float, high: float) -> str:
    return f"$[{number(low)}, {number(high)}]$"


def table(caption: str, label: str, columns: str, headings: str, rows: list[str]) -> str:
    return "\n".join([
        r"\begin{table}[H]", r"\centering", "\\caption{" + caption + "}",
        "\\label{" + label + "}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}" + columns + "@{}}", r"\toprule", headings + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    destinations = parser.add_mutually_exclusive_group()
    destinations.add_argument("--tables-dir", type=Path)
    destinations.add_argument("--paper-dir", type=Path,
                              help="Legacy explicit destination; writes generated/coverage_review beneath it.")
    args = parser.parse_args()
    paths = {name: args.revision_dir / "selector_analysis" / name for name in
             ("selection_outcomes.csv", "paired_test_contrasts.csv")}
    initial_hashes = {name: digest(path) for name, path in paths.items()}
    outcomes = read_csv(paths["selection_outcomes.csv"])
    contrasts = read_csv(paths["paired_test_contrasts.csv"])
    assert len(outcomes) == 256 and len(contrasts) == 384
    oi = {(r["dataset"], r["retention"], float(r["margin"]), r["selector"]): r for r in outcomes}
    ci = {(r["dataset"], r["retention"], float(r["margin"]), r["selected"], r["comparator"]): r for r in contrasts}
    assert len(oi) == len(outcomes) and len(ci) == len(contrasts)
    assert all(r["test_available"] == "True" and int(r["test_seed_count"]) == 5 for r in outcomes)
    assert all(r["available"] == "True" for r in contrasts)
    derived = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            for margin in MARGINS:
                for selector in ("ua", "mean"):
                    o = oi[dataset, retention, margin, selector]
                    c = ci[dataset, retention, margin, selector, "ndcg"]
                    reference = oi[dataset, retention, margin, "ndcg"]
                    regret = float(c[REGRET])
                    denominator = float(reference["test_ndcg"])
                    assert denominator > 1e-8
                    assert c["selected_id"] == o["configuration_id"]
                    assert c["comparator_id"] == reference["configuration_id"]
                    assert math.isclose(regret, denominator - float(o["test_ndcg"]), abs_tol=1e-15)
                    assert math.isclose(float(c[RMSE]), float(o["test_global_rmse"]) -
                                        float(reference["test_global_rmse"]), abs_tol=1e-15)
                    relative = 100 * regret / denominator
                    assert math.isclose(relative, float(c["relative_ndcg_regret_percent"]), abs_tol=1e-12)
                    row = {
                        "dataset": dataset, "retention": retention, "margin": margin,
                        "selector": selector, "configuration_id": o["configuration_id"],
                        "reference_id": c["comparator_id"], "admissible_count": int(o["admissible_count"]),
                        "validation_mean_regret": float(o["paired_mean_ndcg_at_10_regret"]),
                        "validation_uncertainty_penalty": float(o["uncertainty_penalty"]),
                        "validation_upper_regret": float(o["paired_regret_upper_bound"]),
                        "test_ndcg": float(o["test_ndcg"]), "reference_test_ndcg": denominator,
                        "test_global_rmse": float(o["test_global_rmse"]),
                        "reference_test_global_rmse": float(reference["test_global_rmse"]),
                        "test_ndcg_regret_reference_minus_selected": regret,
                        "test_ndcg_regret_ci_low": float(c[REGRET + "_ci_low"]),
                        "test_ndcg_regret_ci_high": float(c[REGRET + "_ci_high"]),
                        "relative_ndcg_regret_percent": relative,
                        "point_regret_exceeds_margin": regret > margin,
                        "two_sided_95_ci_upper_exceeds_margin": float(c[REGRET + "_ci_high"]) > margin,
                        "global_rmse_selected_minus_reference": float(c[RMSE]),
                        "global_rmse_selected_minus_reference_ci_low": float(c[RMSE + "_ci_low"]),
                        "global_rmse_selected_minus_reference_ci_high": float(c[RMSE + "_ci_high"]),
                        "interval_target": c["interval_target"],
                        "evidence_stage": "revision_stage_sensitivity",
                    }
                    derived.append(row)
    assert len(derived) == 128
    summary = []
    for margin in MARGINS:
        row = {"margin": margin, "conditions_per_rule": 16}
        for selector in ("ua", "mean"):
            subset = [r for r in derived if r["margin"] == margin and r["selector"] == selector]
            assert len(subset) == 16
            row[selector + "_point_exceedances"] = sum(r["point_regret_exceeds_margin"] for r in subset)
            row[selector + "_two_sided_95_ci_upper_exceedances"] = sum(r["two_sided_95_ci_upper_exceeds_margin"] for r in subset)
        summary.append(row)
    assert [(r["ua_point_exceedances"], r["mean_point_exceedances"]) for r in summary] == [(0, 2), (0, 2), (0, 1), (1, 1)]
    assert [(r["ua_two_sided_95_ci_upper_exceedances"], r["mean_two_sided_95_ci_upper_exceedances"]) for r in summary] == [(0, 3), (0, 3), (1, 5), (1, 1)]
    ablation = []
    adverse = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            for margin in MARGINS:
                c = ci[dataset, retention, margin, "ua", "mean"]
                if float(c[REGRET]) > 0 and float(c[RMSE]) > 0:
                    adverse.append({"dataset": dataset, "retention": retention, "margin": margin,
                                    "ua_minus_mean_ndcg": -float(c[REGRET]),
                                    "ua_minus_mean_global_rmse": float(c[RMSE])})
                if margin != .005 or c["selected_id"] == c["comparator_id"]:
                    continue
                ablation.append({
                    "dataset": dataset, "retention": retention, "margin": margin,
                    "ua_id": c["selected_id"], "mean_id": c["comparator_id"],
                    "ua_minus_mean_ndcg": -float(c[REGRET]),
                    "ua_minus_mean_ndcg_ci_low": -float(c[REGRET + "_ci_high"]),
                    "ua_minus_mean_ndcg_ci_high": -float(c[REGRET + "_ci_low"]),
                    "ua_minus_mean_global_rmse": float(c[RMSE]),
                    "ua_minus_mean_global_rmse_ci_low": float(c[RMSE + "_ci_low"]),
                    "ua_minus_mean_global_rmse_ci_high": float(c[RMSE + "_ci_high"]),
                    "interval_target": c["interval_target"],
                    "evidence_stage": "revision_stage_sensitivity",
                })
    assert len(ablation) == 7
    assert {(r["dataset"], r["retention"], r["margin"]) for r in adverse} == {("ciaodvd", "r025", .001), ("ciaodvd", "r100", .010)}
    report = args.output_dir / "report_tables"
    tex = (args.paper_dir / "generated/coverage_review" if args.paper_dir else
           args.tables_dir if args.tables_dir else args.output_dir / "tables/coverage_review")
    report.mkdir(parents=True, exist_ok=True)
    tex.mkdir(parents=True, exist_ok=True)
    write_csv(report / "margin_reference_regrets.csv", derived)
    write_csv(report / "margin_exceedance_summary.csv", summary)
    write_csv(report / "ablation_paired_intervals.csv", ablation)
    write_csv(report / "adverse_ua_mean_point_comparisons.csv", adverse)
    caption = (r"Held-out exceedances of the declared absolute margin in revision-stage sensitivity analyses. "
               r"Each count is out of 16 dataset--retention conditions. Point exceedance means signed test regret "
               r"$N_{\rm ref}-N_{\rm selected}>\delta$. The separate interval columns count upper endpoints of "
               r"stored two-sided 95\% paired fixed-fit user-bootstrap intervals exceeding $\delta$; they are "
               r"not the original one-sided prospective Epinions criterion. Counts are descriptive and unadjusted, "
               r"not independent trials or deployment-risk estimates.")
    rows = [f"{r['margin']:.4f} & {r['ua_point_exceedances']} & {r['mean_point_exceedances']} & "
            f"{r['ua_two_sided_95_ci_upper_exceedances']} & {r['mean_two_sided_95_ci_upper_exceedances']} " + r"\\" for r in summary]
    (tex / "margin_exceedance_summary.tex").write_text(table(caption, "tab:coverage-margin-exceedance", "rrrrr",
        r"$\delta$ & UA point & Mean point & UA CI upper & Mean CI upper", rows), encoding="utf-8")
    caption = (r"Seven changed UA-versus-mean selections at the original margin $\delta=0.005$. "
               r"Both effects are UA minus mean; positive NDCG@10 and negative global RMSE favor UA. "
               r"Brackets give the existing two-sided 95\% paired fixed-fit user-bootstrap intervals, "
               r"conditional on the five fitted seeds and shared evaluation users. Global RMSE is event-weighted "
               r"within seed before averaging seeds, not mean-user RMSE. These revision-stage intervals are "
               r"unadjusted, not simultaneous or a new confirmatory family. The nine unchanged choices "
               r"have exactly zero paired differences and intervals.")
    rows = []
    for r in ablation:
        pct = int(r["retention"][1:])
        rows.append(f"{DATASETS[r['dataset']]} & {pct}\\% & NDCG@10 & {number(r['ua_minus_mean_ndcg'])} & "
                    f"{interval(r['ua_minus_mean_ndcg_ci_low'], r['ua_minus_mean_ndcg_ci_high'])} " + r"\\")
        rows.append(f" & & Global RMSE & {number(r['ua_minus_mean_global_rmse'])} & "
                    f"{interval(r['ua_minus_mean_global_rmse_ci_low'], r['ua_minus_mean_global_rmse_ci_high'])} " + r"\\")
        rows.append(r"\addlinespace")
    (tex / "ablation_paired_intervals.tex").write_text(table(caption, "tab:coverage-ablation-ci", "lllrr",
        r"Dataset & Retention & Endpoint & UA minus mean & 95\% CI", rows), encoding="utf-8")
    for dataset, display in DATASETS.items():
        caption = (display + r" revision-stage signed test regret relative to the validation-NDCG-selected reference. "
                   r"$\Delta=N_{\rm ref}-N_{\rm selected}$; relative regret is $100\Delta/N_{\rm ref}$. "
                   r"Brackets give existing two-sided 95\% paired fixed-fit user-bootstrap intervals for "
                   r"absolute $\Delta$, not ratio intervals. `Exceeds' means the unrounded point regret "
                   r"exceeds $\delta$; `within' is not a prospective-test result. Intervals are conditional "
                   r"and unadjusted. UA is uncertainty-aware; mean removes its uncertainty penalty.")
        rows = []
        previous = None
        for r in [x for x in derived if x["dataset"] == dataset]:
            if previous and r["retention"] != previous:
                rows.append(r"\addlinespace")
            previous = r["retention"]
            pct = int(r["retention"][1:])
            status = "exceeds" if r["point_regret_exceeds_margin"] else "within"
            rows.append(f"{pct}\\% & {r['margin']:.4f} & {'UA' if r['selector'] == 'ua' else 'Mean'} & "
                        f"{number(r['test_ndcg_regret_reference_minus_selected'])} & "
                        f"{interval(r['test_ndcg_regret_ci_low'], r['test_ndcg_regret_ci_high'])} & "
                        f"{number(r['relative_ndcg_regret_percent'], 1)} & {status} " + r"\\")
        (tex / f"margin_reference_regrets_{dataset}.tex").write_text(table(caption, "tab:coverage-regret-" + dataset.replace("_", "-"),
            "rrlrrrl", r"Retention & $\delta$ & Rule & $\Delta$ & 95\% CI & Relative (\%) & Point status", rows), encoding="utf-8")
    epinions = [r["test_ndcg"] for r in derived if r["dataset"] == "epinions" and r["margin"] == .001 and r["selector"] == "ua"]
    ciao = next(r for r in derived if r["dataset"] == "ciaodvd" and r["retention"] == "r100" and r["margin"] == .010 and r["selector"] == "ua")
    assert ciao["validation_upper_regret"] < ciao["margin"] < ciao["test_ndcg_regret_reference_minus_selected"]
    final_hashes = {name: digest(path) for name, path in paths.items()}
    assert final_hashes == initial_hashes
    verification = {
        "source_sha256": initial_hashes,
        "source_rows": {"selection_outcomes": len(outcomes), "paired_test_contrasts": len(contrasts)},
        "derived_rows": {"reference_regrets": len(derived), "ablation_intervals": len(ablation), "adverse_point_comparisons": len(adverse)},
        "source_inputs_unchanged": True, "fits_run": 0, "bootstraps_run": 0,
        "epinions_ua_margin_001_ndcg_range_unrounded": [min(epinions), max(epinions)],
        "epinions_ua_margin_001_ndcg_range_five_decimals": [number(min(epinions), signed=False), number(max(epinions), signed=False)],
        "ciao_full_retention_margin_010_ua": ciao,
        "exceedance_counts": summary,
        "interval_provenance": "Stored two-sided .025/.975 percentile endpoints copied from paired_test_contrasts.csv; NDCG endpoints reversed only when negating comparator-minus-UA to UA-minus-mean.",
        "relative_effect_definition": "100 * (NDCG-selected reference test NDCG - selected test NDCG) / reference test NDCG; signed point ratio, no new ratio inference.",
    }
    (report / "coverage_reporting_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_dir": str(report), "tex_dir": str(tex), "source_inputs_unchanged": True,
                      "reference_regret_rows": len(derived), "changed_ablation_rows": len(ablation),
                      "fits_run": 0, "bootstraps_run": 0, "exceedance_counts": summary}, indent=2))


if __name__ == "__main__":
    main()
