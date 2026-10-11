"""Generate source-derived protocol reporting for the 2026-09-14 journal revision.

Reads existing manifests and validation metadata, never fits or changes a model.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "regenerated/reports/protocol_audit"
TABLES = ROOT / "regenerated/tables/review_r1"
DATASETS = ("movielens_1m", "ciaodvd", "yelp_tampa_restaurants", "epinions")
NAMES = ("MovieLens 1M", "CiaoDVD", "Yelp--Tampa", "Epinions")
RETENTIONS = ("r100", "r075", "r050", "r025")


def read_json(path, inputs):
    inputs.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return json.loads(path.read_text())


def table(filename, caption, label, headers, rows, columns, *, note=None, resize=True):
    lines = [r"\begin{table}[H]", r"\centering", "\\caption{" + caption + "}",
             "\\label{" + label + "}", r"\small"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += ["\\begin{tabular}{@{}" + columns + "@{}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}%"]
    if resize:
        lines.append("}")
    if note:
        lines += [r"\par\smallskip\footnotesize " + note]
    lines += [r"\end{table}"]
    (TABLES / filename).write_text("\n".join(lines) + "\n")


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    inputs = []
    modeling_path = ROOT / "configs/latent_factor_beyond_rmse/modeling.yaml"
    modeling = yaml.safe_load(modeling_path.read_text())["explicit_mf"]
    inputs.append(dict(path=str(modeling_path.relative_to(ROOT)), sha256=hashlib.sha256(modeling_path.read_bytes()).hexdigest()))
    configs = list(itertools.product(modeling["dimensions"], modeling["regularizations"]))
    expected = set((*config, seed) for config, seed in itertools.product(configs, modeling["selection_seeds"]))
    registry = pd.DataFrame([dict(id=f"C{i+1:02d}", run_configuration_id=f"d{d}_lambda{reg:g}", dimension=d,
                                 regularization=reg) for i, (d, reg) in enumerate(configs)])
    registry.to_csv(OUTPUT / "configuration_registry.csv", index=False)
    datasets, conditions, grids = [], [], []
    for dataset, name in zip(DATASETS, NAMES):
        prospective = dataset == "epinions"
        data_slug = "latent_factor_framework_epinions" if prospective else "latent_factor_beyond_rmse"
        source = read_json(ROOT / "data/processed" / data_slug / dataset / "manifest.json", inputs)
        retained = read_json(ROOT / "data/splits" / data_slug / dataset / "sparsity_manifest.json", inputs)
        processed = source["processed_statistics"]
        datasets.append(dict(dataset=dataset, display_name=name, stage="prospective_external" if prospective else "construction",
                             raw_or_scoped_rows=source["duplicate_statistics"]["input_rows"],
                             deduplicated_rows=source["duplicate_statistics"]["deduplicated_rows"],
                             processed_rows=processed["rows"], processed_users=processed["users"],
                             processed_items=processed["items"], processed_density=processed["density"]))
        base = ROOT / "outputs/latent_factor_beyond_rmse"
        if prospective:
            base /= "prospective_epinions"
        for retention in RETENTIONS:
            gd = base / "development" / dataset / retention
            grid_path = gd / "grid_runs.csv"
            grid = pd.read_csv(grid_path)
            observed = list(map(tuple, grid[["dimension", "regularization", "seed"]].values))
            if len(observed) != len(expected) or set(observed) != expected:
                raise ValueError(f"Unexpected configuration registry: {grid_path}")
            inputs.append(dict(path=str(grid_path.relative_to(ROOT)), sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest()))
            schemas = {}
            for dim, reg, seed in expected:
                p = gd / f"d{dim}_lambda{reg:g}_seed{seed}" / "validation_per_user.parquet"
                fields = tuple(pq.read_schema(p).names)
                schemas[fields] = schemas.get(fields, 0) + 1
            grids.append(dict(dataset=dataset, retention=retention, verified_grid_rows=len(grid),
                              schemas=[dict(columns=list(columns), files=count) for columns, count in schemas.items()],
                              supports_end_to_end_user_resampling=all("prediction_rmse" in k and "num_prediction_rows" in k for k in schemas)))
            first = grid.iloc[0]
            val = read_json(gd / f"d{int(first.dimension)}_lambda{first.regularization:g}_seed{int(first.seed)}" / "summary.json", inputs)["validation"]
            confirmation = read_json(base / "confirmation" / dataset / retention / "confirmation_manifest.json", inputs)
            test = read_json(ROOT / confirmation["runs"][0]["summary_path"], inputs)["test"]
            r = retained["conditions"][retention]
            train_n = source["split_statistics"]["train"]["rows"]
            assert r["full_train_rows"] == train_n
            assert r["users"] == processed["users"]
            assert r["items"] == source["split_statistics"]["train"]["items"]
            assert val["prediction"]["num_rows"] == source["split_statistics"]["validation"]["rows"]
            assert test["prediction"]["num_rows"] == source["split_statistics"]["test"]["rows"]
            conditions.append(dict(dataset=dataset, display_name=name, retention=retention,
                                   stage="prospective_external" if prospective else "construction",
                                   nominal_retention=r["retention_nominal"], realized_retention=r["realized_retention"],
                                   train_rows=r["realized_rows"], validation_rows=val["prediction"]["num_rows"],
                                   test_rows=test["prediction"]["num_rows"], users=r["users"], train_catalog_items=r["items"],
                                   validation_ranking_users=val["ranking"]["users_evaluable"],
                                   test_ranking_users=test["ranking"]["users_evaluable"],
                                   validation_item_cold_events=val["prediction"]["item_cold_rows"],
                                   test_item_cold_events=test["prediction"]["item_cold_rows"],
                                   validation_cold_relevant_events=val["ranking"]["cold_relevant_events"],
                                   test_cold_relevant_events=test["ranking"]["cold_relevant_events"]))
    pd.DataFrame(datasets).to_csv(OUTPUT / "dataset_summary.csv", index=False)
    pd.DataFrame(conditions).to_csv(OUTPUT / "retention_counts.csv", index=False)
    reg_text = {0.: "$0$", .0001: "$10^{-4}$", .001: "$10^{-3}$", .01: "$10^{-2}$", .1: "$10^{-1}$"}
    # Two side-by-side blocks keep the complete 35-tuple registry compact.
    registry_rows = []
    records = registry.to_dict("records")
    for i in range(18):
        left = records[i]
        a = [left["id"], left["dimension"], reg_text[left["regularization"]]]
        if i + 18 < len(records):
            right = records[i + 18]
            a += [right["id"], right["dimension"], reg_text[right["regularization"]]]
        else:
            a += ["", "", ""]
        registry_rows.append(a)
    table("config_registry.tex", "Complete original 35-configuration explicit matrix-factorization registry. "
          "Identifiers C01--C35 are revision reporting aliases for the existing $(d,\\lambda)$ tuples, "
          "not new experimental configurations. All 16 original dataset--retention grids contain all "
          "35 tuples with the five declared development seeds. Fixed training settings and the separate "
          "expanded search are described in Methods.", "tab:config-registry",
          ["ID", "$d$", "$\\lambda$", "ID", "$d$", "$\\lambda$"], registry_rows, "lrrlrr", resize=False)
    dataset_rows = []
    for r in datasets:
        dataset_rows.append([r["display_name"], "Prospective" if r["stage"] == "prospective_external" else "Construction",
                             f'{r["raw_or_scoped_rows"]:,}', f'{r["deduplicated_rows"]:,}', f'{r["processed_users"]:,}',
                             f'{r["processed_items"]:,}', f'{r["processed_rows"]:,}', f'{100*r["processed_density"]:.3f}\\%'])
    table("dataset_summary.tex", "Consolidated source-derived dataset statistics. Raw/scoped uses the full "
          "rating source except for the predeclared Yelp--Tampa restaurant scope; the last four columns "
          "describe the final iterative 10-core. Construction denotes the three datasets later used "
          "for retrospective uncertainty-aware rule construction; Epinions was held out prospectively.",
          "tab:processed-datasets", ["Dataset", "Evidence stage", "Raw/scoped", "Dedup.", "Users", "Items", "10-core ratings", "Density"],
          dataset_rows, "llrrrrrr")
    retention_rows = []
    for r in conditions:
        retention_rows.append([r["display_name"], f'{100*r["nominal_retention"]:.0f}\\%', f'{100*r["realized_retention"]:.2f}\\%',
                               f'{r["train_rows"]:,}', f'{r["validation_rows"]:,}', f'{r["test_rows"]:,}',
                               f'{r["users"]:,}', f'{r["train_catalog_items"]:,}', f'{r["validation_ranking_users"]:,}',
                               f'{r["test_ranking_users"]:,}'])
    table("retention_counts.tex", "Exact interaction counts and ranking eligibility for every original "
          "condition, taken from frozen preprocessing/retention manifests and recorded evaluations. "
          "All retained users occur in both prediction partitions. Ranking includes only users with "
          "at least one eligible relevant item in the fixed train-known catalog after history exclusion. "
          "The Epinions rows belong to the prospective external protocol.", "tab:realized-sparsity",
          ["Dataset", "Nominal", "Realized", "Train", "Validation", "Test", "Users", "Train items", "Rank users V", "Rank users T"],
          retention_rows, "lrrrrrrrrr", note="The minimum-user-history constraint and item-catalog repair can increase realized retention "
          "above nominal retention. Validation and test interaction counts, train-known item catalogs, and "
          "ranking-eligibility sets are invariant across retention within a dataset. V and T denote validation and test.")
    resampling_root = ROOT / "output/latent_factor_major_revision_2026-09-14/user_resampling"
    stability_path = resampling_root / "selection_stability_summary.csv"
    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        agreement_path = resampling_root / "paired_selector_agreement.csv"
        agreement = pd.read_csv(agreement_path)
        for p in (stability_path, agreement_path):
            inputs.append(dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        rows = []
        for retention in RETENTIONS:
            selected = stability[(stability.retention == retention) & (stability.delta == .001)].set_index("selector")
            pair = agreement[(agreement.retention == retention) & (agreement.delta == .001)].iloc[0]
            rows.append([f'{int(retention[1:])}\\%',
                         f'{100*selected.loc["UA", "full_cohort_reference_frequency"]:.1f}\\%',
                         f'{100*selected.loc["UA", "full_cohort_selection_frequency"]:.1f}\\%',
                         f'{100*selected.loc["mean", "full_cohort_selection_frequency"]:.1f}\\%',
                         f'{100*pair.identical_selection_frequency:.1f}\\%'])
        table("user_resampling.tex", "Revision-stage Epinions validation-user resampling: 500 shared draws "
              "recompute the ranking reference and both selectors over all 35 configurations and five "
              "fixed fitted models per configuration. At $\\delta=0.001$, the columns give frequencies "
              "of retaining the full-cohort NDCG reference, retaining each selector's own full-cohort "
              "choice, and the two constrained selectors choosing identically within a draw. At "
              "$\\delta=0.005$ and $0.010$, both selectors admit all 35 configurations and choose the "
              "original RMSE configuration in every draw in every condition. Frequencies describe "
              "conditional stability, not confidence guarantees or retraining uncertainty.",
              "tab:validation-user-stability", ["Retention", "Same reference", "Same UA choice", "Same mean choice", "UA = mean"],
              rows, "lrrrr")
    audit = {
        "stage": "revision_reporting_audit_existing_artifacts", "new_model_fits": 0,
        "configurations": len(configs), "conditions": len(conditions), "grids": grids,
        "main_findings": {
            "chronology": "Per-user timestamp, stable source_row_id, then item_id; floor(.70n), floor(.85n). Invalid short histories raise an error, not silently assigned; final10core satisfies all users.",
            "deduplication": "Latest timestamp per user-item; largest stable source_row_id wins exact timestamp tie, before full-period iterative10core.",
            "retention": "round-half-up(q*n) with minimum3 plus deterministic full-training-item repair; original priorities include q and are nonnested.",
            "candidates": "All configurations/retentions share repaired train catalog and full-training history masks; test additionally excludes validation history. Item-ID score ties are lexicographic strings.",
            "prediction_inference": "Original global RMSE is event-weighted then averaged across seeds; intervals in prediction_regret.tex target mean-userRMSE differences, not globalRMSE. Existing table correctly separates them.",
            "seeds": "Manually fixed identifiers, reused across datasets/conditions/configurations; Python/NumPy/PyTorch seeded and epoch permutation uses NumPy default_rng(seed). No evidence of random seed-generation procedure.",
            "historical_runtime": "Manuscript declares AppleM1Max10core64GB, CPU1thread, Python3.10.9/PyTorch2.1.0.dev20230521/pandas1.5.3/NumPy1.21.5. No contemporaneous exact LF-run SciPy/Matplotlib/PyArrow/PyYAML pins located. Declared environment contains compatibility ranges, not exact historical version evidence.",
            "release": "Local GitHub-release artifacts and code/configs exist; DOI archive remains planned. Public visibility must be checked independently; a local checkout/remote URL does not establish anonymous access.",
        },
        # These are immutable source-era references, not active output paths.
        "source_locations": {
            "chronology": "src/latent_factor_study/data.py:168",
            "retention": "src/latent_factor_study/sparsity.py:28",
            "prediction_aggregation_and_schema": "src/latent_factor_study/explicit_mf.py:242",
            "prediction_inference": "src/latent_factor_study/inference.py:180",
            "training_rng": "src/latent_factor_study/explicit_mf.py:38",
            "runtime_declaration": "papers/latent_factor_beyond_rmse/sections/materials_methods.tex:464",
            "runtime_compatibility_ranges": "papers/latent_factor_beyond_rmse/repository/environment/declared-results-environment.yml",
            "deposit_plan": "papers/latent_factor_beyond_rmse/repository/ARTIFACT_DEPOSIT.md",
        },
        "inputs": inputs,
        "generator": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "revision_runtime_versions_not_historical_run_evidence": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scipy", "pyarrow", "PyYAML", "matplotlib")
        },
    }
    (OUTPUT / "protocol_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"dataset_rows": len(datasets), "retention_rows": len(conditions), "grid_runs_verified": sum(r["verified_grid_rows"] for r in grids),
                      "prediction_capable_conditions": sum(r["supports_end_to_end_user_resampling"] for r in grids), "tables": str(TABLES)}, indent=2))


if __name__ == "__main__":
    run()
