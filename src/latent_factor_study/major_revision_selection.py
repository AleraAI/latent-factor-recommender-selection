"""Revision-stage mean-constraint ablation and margin sweep; no model fitting.

Historical selectors, outputs and locks are read-only. ``prepare`` records the
choices and validation hashes before ``analyze`` is allowed to open test metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import scipy

from .grid import select_configurations
from .uncertainty_selector import select_uncertainty_aware_configuration

DATASETS = ("movielens_1m", "ciaodvd", "yelp_tampa_restaurants", "epinions")
RETENTIONS = ("r100", "r075", "r050", "r025")
DEVELOPMENT_SEEDS = (101, 211, 307, 401, 503)
TEST_SEEDS = (601, 701, 809, 907, 1009)
MARGINS = (0.001, 0.0025, 0.005, 0.010)
DEFAULT_OUTPUT = Path("output/latent_factor_major_revision_2026-09-14/selector_analysis")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def config_id(row: Any) -> str:
    return f"d{int(row['dimension'])}_lambda{float(row['regularization']):g}"


def grid_path(root: Path, dataset: str, retention: str) -> Path:
    stage = "prospective_epinions/development" if dataset == "epinions" else "development"
    return root / "outputs/latent_factor_beyond_rmse" / stage / dataset / retention / "grid_runs.csv"


def validate_grid(rows: pd.DataFrame) -> None:
    required = ["dimension", "regularization", "seed", "training_seconds", "validation_rmse", "validation_ndcg@10"]
    if any(c not in rows for c in required):
        raise ValueError("Missing selector inputs")
    if not np.isfinite(rows[required].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite selector inputs are not admissible revision data")
    expected = {(d, l, s) for d in (1, 2, 4, 8, 16, 32, 64)
                for l in (0., .0001, .001, .01, .1) for s in DEVELOPMENT_SEEDS}
    observed = list(rows[["dimension", "regularization", "seed"]].itertuples(index=False, name=None))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Incomplete, duplicate, or unexpected configuration/seed grid")


def constrained_choice(table: pd.DataFrame, margin: float, uncertainty: bool) -> pd.Series:
    """Inherit *all* UA ties; the mean ablation changes only the screen."""
    key = "paired_regret_upper_bound" if uncertainty else "paired_mean_ndcg_at_10_regret"
    admissible = table.loc[table[key] <= margin]
    if admissible.empty:
        raise ValueError("No valid admissible configuration; no fallback is invented")
    return admissible.sort_values(
        ["mean_validation_rmse", "mean_training_seconds", "dimension", "regularization"],
        ascending=[True, True, True, False], kind="stable").iloc[0]


def pareto_flags(table: pd.DataFrame) -> np.ndarray:
    rmse = table.mean_validation_rmse.to_numpy()
    ndcg = table["mean_validation_ndcg@10"].to_numpy()
    return np.array([not np.any((rmse <= rmse[i]) & (ndcg >= ndcg[i]) &
                               ((rmse < rmse[i]) | (ndcg > ndcg[i]))) for i in range(len(table))])


def validation_selection(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    choices, landscapes, audit = [], [], []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            path = grid_path(root, dataset, retention)
            rows = pd.read_csv(path)
            validate_grid(rows)
            original = select_configurations(rows)
            ua = select_uncertainty_aware_configuration(rows)
            old = json.loads((path.parent / "selection.json").read_text())["selection"]
            fpath = (path.parent / "framework_selection.json" if dataset == "epinions" else
                     root / "outputs/latent_factor_beyond_rmse/framework_construction/selections" /
                     dataset / retention / "framework_selection.json")
            artifact = json.loads(fpath.read_text())
            if artifact["grid_runs_sha256"] != sha(path):
                raise ValueError(f"Changed validation grid: {path}")
            for role in ("rmse_selected", "ndcg_selected"):
                if not config_id(original[role]) == config_id(old[role]) == config_id(ua[role]) == config_id(artifact["selection"][role]):
                    raise ValueError(f"Historical {role} did not reproduce: {dataset}/{retention}")
            if config_id(ua["framework_selected"]) != config_id(artifact["selection"]["framework_selected"]):
                raise ValueError("Historical UA identity did not reproduce")
            if ua["admissible_configurations"] != artifact["selection"]["admissible_configurations"]:
                raise ValueError("Historical UA admissible count changed")
            table = pd.DataFrame(ua["configuration_table"])
            table["configuration_id"] = table.apply(config_id, axis=1)
            table["uncertainty_penalty"] = table.paired_regret_upper_bound - table.paired_mean_ndcg_at_10_regret
            table["empirical_pareto"] = pareto_flags(table)
            table["dataset"], table["retention"] = dataset, retention
            table["ranking_reference_id"] = config_id(original["ndcg_selected"])
            unconstrained = constrained_choice(table, 1., True)
            indexed = table.set_index("configuration_id", drop=False)
            for margin in MARGINS:
                for role, chosen, size in (
                    ("rmse", indexed.loc[config_id(original["rmse_selected"])], 35),
                    ("ndcg", indexed.loc[config_id(original["ndcg_selected"])], 35),
                    ("mean", constrained_choice(table, margin, False), int((table.paired_mean_ndcg_at_10_regret <= margin).sum())),
                    ("ua", constrained_choice(table, margin, True), int((table.paired_regret_upper_bound <= margin).sum())),
                ):
                    choices.append({
                        **chosen.to_dict(), "selector": role, "margin": margin,
                        "admissible_count": size, "selection_stage": "revision_stage_sensitivity",
                        "is_intermediate": chosen.configuration_id not in
                            {config_id(original["rmse_selected"]), config_id(original["ndcg_selected"])},
                        "unconstrained_ua_id": unconstrained.configuration_id,
                        "unconstrained_ua_upper_regret": float(unconstrained.paired_regret_upper_bound),
                        "max_upper_regret": float(table.paired_regret_upper_bound.max()),
                        "max_mean_regret": float(table.paired_mean_ndcg_at_10_regret.max()),
                    })
            landscapes.extend(table.to_dict("records"))
            audit.append({"dataset": dataset, "retention": retention, "rows": len(rows),
                          "original_rmse": config_id(original["rmse_selected"]),
                          "original_ndcg": config_id(original["ndcg_selected"]),
                          "original_ua": config_id(ua["framework_selected"]),
                          "original_ua_admissible": ua["admissible_configurations"],
                          "grid_sha256": sha(path), "framework_artifact_sha256": sha(fpath)})
    return pd.DataFrame(choices), pd.DataFrame(landscapes), audit


def prepare(root: Path, output: Path) -> None:
    protocol_path = output / "revision_protocol.json"
    if protocol_path.exists():
        raise FileExistsError("Protocol exists: preserve it and run analyze; never silently replace it")
    inputs = [grid_path(root, d, r) for d in DATASETS for r in RETENTIONS]
    inputs += [root / "src/latent_factor_study" / name for name in
               ("grid.py", "uncertainty_selector.py", "explicit_mf.py", "evaluation.py")]
    inputs += [root / "configs/latent_factor_beyond_rmse" / name for name in
               ("selection_lock.yaml", "epinions_prospective_protocol_lock.yaml",
                "epinions_prospective_selection_lock.yaml", "modeling.yaml",
                "epinions_prospective_modeling.yaml")]
    protocol = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "revision_stage_sensitivity_on_previously_inspected_test_partitions",
        "new_test_summaries_accessed_before_recording": False,
        "margins": list(MARGINS), "original_margin": .005,
        "datasets": list(DATASETS), "retentions": list(RETENTIONS),
        "development_seeds": list(DEVELOPMENT_SEEDS), "evaluation_seeds": list(TEST_SEEDS),
        "mean_ablation": "remove only Student-t uncertainty adjustment; retain complete UA tie hierarchy",
        "ua_ties": ["lower_mean_validation_rmse", "lower_recorded_mean_training_seconds", "smaller_dimension", "larger_regularization"],
        "historical_single_objective_ties": ["objective", "smaller_dimension", "larger_regularization"],
        "test_reuse": "identical configuration/seed/run settings and primary data contract; no ensembles; no new fitting",
        "prediction_estimand": "mean across seeds of event-weighted global clipped RMSE; separately mean-user RMSE",
        "bootstrap": {"resamples": 2000, "seed": 20260914, "confidence": .95,
                      "unit": "all evaluated users, same draws across configurations/seeds; ranking averages eligible sampled users",
                      "target": "conditional fixed-fit user-sampling diagnostic, not seed/split/retraining uncertainty or simultaneous coverage"},
        "relative_regret": "100*(reference NDCG-selected minus selected)/reference; flag denominator <= 1e-8, never substitute epsilon",
        "missing_policy": "report missing runs; no partial-seed point estimates; no invented fallback",
        "input_sha256": {str(p.relative_to(root)): sha(p) for p in inputs},
        "analysis_module_sha256_at_recording": sha(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
    }
    # Written before any confirmation manifest, per-user test file or test summary is opened.
    write_json(protocol_path, protocol)
    selections, landscapes, audit = validation_selection(root)
    selections.to_csv(output / "validation_selections.csv", index=False)
    landscapes.to_csv(output / "validation_landscapes.csv", index=False)
    write_json(output / "original_identity_audit.json", audit)
    write_json(output / "selection_freeze.json", {
        "protocol_sha256": sha(protocol_path),
        "validation_selections_sha256": sha(output / "validation_selections.csv"),
        "validation_landscapes_sha256": sha(output / "validation_landscapes.csv"),
        "test_metrics_accessed": False,
    })


def test_records(root: Path, dataset: str, retention: str, revision_output: Path | None = None) -> dict[tuple[str, int], dict[str, Any]]:
    base = root / "outputs/latent_factor_beyond_rmse"
    stages = ["prospective_epinions/confirmation"] if dataset == "epinions" else ["confirmation", "framework_construction/confirmation"]
    records = {}
    for stage in stages:
        path = base / stage / dataset / retention / "confirmation_manifest.json"
        artifact = json.loads(path.read_text())
        if artifact["dataset"] != dataset or artifact["retention_label"] != retention:
            raise ValueError("Confirmation manifest condition mismatch")
        if artifact["expected_runs"] != artifact["observed_runs"] or artifact["observed_runs"] != len(artifact["runs"]):
            raise ValueError("Incomplete historical confirmation manifest")
        if "selection_lock_sha256" in artifact:
            lock_name = "epinions_prospective_selection_lock.yaml" if dataset == "epinions" else "selection_lock.yaml"
            if artifact["selection_lock_sha256"] != sha(root / "configs/latent_factor_beyond_rmse" / lock_name):
                raise ValueError("Historical confirmation selection-lock mismatch")
        else:
            original_manifest = base / "confirmation" / dataset / retention / "confirmation_manifest.json"
            selection_artifact = base / "framework_construction/selections" / dataset / retention / "framework_selection.json"
            if (artifact["historical_confirmation_manifest_sha256"] != sha(original_manifest) or
                    artifact["framework_selection_sha256"] != sha(selection_artifact)):
                raise ValueError("Framework confirmation provenance mismatch")
        for record in artifact["runs"]:
            key = (config_id(record), int(record["seed"]))
            if key in records and any(record[k] != records[key][k] for k in ("summary_sha256", "per_user_sha256")):
                raise ValueError("Conflicting duplicate historical run")
            if key not in records:
                records[key] = {**record, "dataset": dataset, "retention": retention,
                                "artifact_stage": "existing_historical_evaluation", "manifest_path": str(path.relative_to(root)),
                                "manifest_sha256": sha(path)}
    if revision_output is not None and (revision_output / "fits/confirmation_manifest.json").exists():
        path = revision_output / "fits/confirmation_manifest.json"
        artifact = json.loads(path.read_text())
        if artifact.get("runtime_replay_passed") is not True:
            raise ValueError("Revision fitting parity gate absent")
        for record in artifact["runs"]:
            if record["dataset"] == dataset and record["retention"] == retention:
                key = (config_id(record), int(record["seed"]))
                if key in records:
                    raise ValueError("New run would replace an existing historical evaluation")
                records[key] = {**record, "artifact_stage": "new_revision_stage_fit",
                                "manifest_path": str(path.relative_to(root)), "manifest_sha256": sha(path)}
    return records


def load_test(root: Path, record: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    for key in ("summary", "per_user"):
        if sha(root / record[f"{key}_path"]) != record[f"{key}_sha256"]:
            raise ValueError(f"Changed historical test artifact: {record[key + '_path']}")
    summary = json.loads((root / record["summary_path"]).read_text())
    if summary.get("dataset") != record["dataset"] or summary.get("retention_label") != record["retention"]:
        raise ValueError("Test summary condition mismatch")
    run = summary["run_config"]
    expected = {"dimension": int(record["dimension"]), "regularization": float(record["regularization"]),
                "seed": int(record["seed"]), "learning_rate": .01, "batch_size": 4096,
                "max_epochs": 20, "convergence_patience": 5, "convergence_relative_tolerance": 1e-5,
                "device": "cpu", "torch_num_threads": 1}
    if run != expected:
        raise ValueError("Available run does not match original fitting settings")
    frame = pd.read_parquet(root / record["per_user_path"]).sort_values("user_id").reset_index(drop=True)
    if frame.user_id.duplicated().any():
        raise ValueError("Duplicate test users")
    for col in ("prediction_rmse", "num_prediction_rows"):
        if col not in frame or not np.isfinite(frame[col]).all():
            raise ValueError("Missing prediction information")
    counts = frame.num_prediction_rows.to_numpy(dtype=float)
    if np.any(counts <= 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Prediction event counts must be positive integers")
    rebuilt = float(np.sqrt(np.average(frame.prediction_rmse ** 2, weights=frame.num_prediction_rows)))
    if not np.isclose(rebuilt, summary["test"]["prediction"]["clipped"]["rmse"], atol=1e-12, rtol=0):
        raise ValueError("Per-user SSE reconstruction disagrees with global RMSE")
    eligible = frame.num_relevant > 0
    if not eligible.any() or not np.isfinite(frame.loc[eligible, "ndcg@10"]).all():
        raise ValueError("No valid ranking-eligible test population")
    if (int(eligible.sum()) != summary["test"]["ranking"]["users_evaluable"] or
            not np.isclose(frame.loc[eligible, "ndcg@10"].mean(), summary["test"]["ranking"]["ndcg@10"], atol=1e-12, rtol=0)):
        raise ValueError("Per-user NDCG or eligible count disagrees with summary")
    return frame, summary


def bootstrap_metrics(frames: list[list[pd.DataFrame]], *, resamples: int, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Config x seed x user inputs; RMSE is taken before seed averaging."""
    reference = frames[0][0]
    fixed_cols = ["user_id", "num_prediction_rows", "num_candidates", "num_relevant", "num_cold_relevant"]
    for group in frames:
        if len(group) != len(TEST_SEEDS):
            raise ValueError("Every comparison requires all five seed instances")
        for frame in group:
            if not frame[fixed_cols].equals(reference[fixed_cols]):
                raise ValueError("Evaluation users, row counts, catalogs or relevance masks differ")
    nusers = len(reference)
    counts = reference.num_prediction_rows.to_numpy(dtype=float)
    eligible = reference.num_relevant.to_numpy() > 0
    if not eligible.any() or np.any(counts <= 0):
        raise ValueError("Invalid eligible population or nonpositive event count")
    rmse = np.array([[f.prediction_rmse.to_numpy() for f in group] for group in frames])
    ndcg = np.array([[f["ndcg@10"].to_numpy() for f in group] for group in frames])
    if not np.isfinite(ndcg[:, :, eligible]).all():
        raise ValueError("Nonfinite eligible-user NDCG")
    sse = rmse ** 2 * counts[None, None, :]
    rank = np.nanmean(ndcg[:, :, eligible], axis=1)
    user_rmse = rmse.mean(axis=1)
    point = {"global_rmse": np.sqrt(sse.sum(axis=2) / counts.sum()).mean(axis=1),
             "mean_user_rmse": user_rmse.mean(axis=1), "ndcg": rank.mean(axis=1)}
    samples = {key: np.empty((resamples, len(frames))) for key in point}
    rng = np.random.default_rng(seed)
    for start in range(0, resamples, 32):
        batch = min(32, resamples - start)
        draws = rng.integers(0, nusers, size=(batch, nusers))
        weights = np.array([np.bincount(row, minlength=nusers) for row in draws], dtype=float)
        denominator = weights @ counts
        if np.any(denominator <= 0) or np.any(weights[:, eligible].sum(axis=1) <= 0):
            raise ValueError("Bootstrap draw has no eligible population; no hidden fallback")
        numerator = weights @ sse.reshape(-1, nusers).T
        samples["global_rmse"][start:start + batch] = np.sqrt(
            numerator.reshape(batch, len(frames), len(TEST_SEEDS)) / denominator[:, None, None]).mean(axis=2)
        samples["mean_user_rmse"][start:start + batch] = weights @ user_rmse.T / nusers
        samples["ndcg"][start:start + batch] = weights[:, eligible] @ rank.T / weights[:, eligible].sum(axis=1)[:, None]
    return point, samples


def analyze(root: Path, output: Path) -> None:
    protocol = json.loads((output / "revision_protocol.json").read_text())
    freeze = json.loads((output / "selection_freeze.json").read_text())
    if sha(output / "revision_protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("Revision protocol changed")
    for name in ("validation_selections", "validation_landscapes"):
        if sha(output / f"{name}.csv") != freeze[f"{name}_sha256"]:
            raise ValueError("Frozen validation choices changed")
    for name, digest in protocol["input_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Historical input changed: {name}")
    new_manifest = output / "fits/confirmation_manifest.json"
    if new_manifest.exists():
        audit_path = output / "fits/data_integrity_audit_final.json"
        if not audit_path.exists():
            raise ValueError("New fits require a completed independent historical-data audit before integration")
        audit = json.loads(audit_path.read_text())
        if (audit.get("status") != "passed" or audit.get("expected_run_count") != 90 or
                audit.get("completed_run_count") != 90 or audit.get("all_planned_runs_complete") is not True or
                audit.get("confirmation_manifest_sha256") != sha(new_manifest) or
                audit.get("fitting_plan_sha256") != sha(output / "fits/fitting_plan.json")):
            raise ValueError("New-fit historical-data audit is incomplete or stale")
    selected = pd.read_csv(output / "validation_selections.csv")
    outcomes, contrasts, missing, provenance = [], [], [], []
    for condition_index, (dataset, retention) in enumerate((d, r) for d in DATASETS for r in RETENTIONS):
        condition = selected.loc[(selected.dataset == dataset) & (selected.retention == retention)]
        available = test_records(root, dataset, retention, revision_output=output)
        complete, allframes = [], []
        development = pd.read_csv(grid_path(root, dataset, retention))
        for identity in sorted(condition.configuration_id.unique()):
            example = condition.loc[condition.configuration_id == identity].iloc[0]
            absent = [s for s in TEST_SEEDS if (identity, s) not in available]
            if absent:
                fit_times = development.loc[(development.dimension == example.dimension) &
                                            (development.regularization == example.regularization), "training_seconds"]
                for seed in absent:
                    missing.append({"dataset": dataset, "retention": retention, "configuration_id": identity,
                                    "dimension": int(example.dimension), "regularization": float(example.regularization),
                                    "seed": seed, "reason": "no matching saved test evaluation or checkpoint",
                                    "estimated_fit_seconds_development_mean": float(fit_times.mean()),
                                    "runtime_caveat": "historical CPU fit-only estimate; excludes evaluation and scheduling"})
                continue
            group = []
            for seed in TEST_SEEDS:
                record = available[(identity, seed)]
                frame, summary = load_test(root, record)
                group.append(frame)
                provenance.append({"dataset": dataset, "retention": retention, "configuration_id": identity, **record})
            allframes.append(group)
            complete.append(identity)
        point, samples = bootstrap_metrics(allframes, resamples=protocol["bootstrap"]["resamples"],
                                          seed=protocol["bootstrap"]["seed"] + condition_index)
        indices = {identity: i for i, identity in enumerate(complete)}
        for _, choice in condition.iterrows():
            out = choice.to_dict()
            out["historical_ua_admissible_at_original_margin"] = out.pop("admissible")
            out["selected_admissible_at_current_margin"] = (
                bool(choice.paired_regret_upper_bound <= choice.margin) if choice.selector == "ua" else
                bool(choice.paired_mean_ndcg_at_10_regret <= choice.margin) if choice.selector == "mean" else None)
            identity = choice.configuration_id
            out["test_available"] = identity in indices
            out["test_seed_count"] = 5 if identity in indices else 0
            if identity in indices:
                out.update({"test_" + k: float(v[indices[identity]]) for k, v in point.items()})
            outcomes.append(out)
        for margin in MARGINS:
            margin_rows = condition.loc[condition.margin == margin].set_index("selector")
            pairs = [(a, b) for a, b in (("rmse", "ndcg"), ("mean", "ndcg"), ("ua", "ndcg"),
                                        ("ua", "mean"), ("ua", "rmse"), ("mean", "rmse"))]
            for left, right in pairs:
                lid, rid = margin_rows.loc[left, "configuration_id"], margin_rows.loc[right, "configuration_id"]
                row = {"dataset": dataset, "retention": retention, "margin": margin, "selected": left,
                       "comparator": right, "selected_id": lid, "comparator_id": rid,
                       "available": lid in indices and rid in indices,
                       "interval_target": "fixed-fit paired user bootstrap, not simultaneous or retraining uncertainty"}
                if row["available"]:
                    i, j = indices[lid], indices[rid]
                    for key in point:
                        sign = -1 if key == "ndcg" else 1
                        name = "ndcg_regret_comparator_minus_selected" if key == "ndcg" else key + "_selected_minus_comparator"
                        difference = sign * (samples[key][:, i] - samples[key][:, j])
                        lo, hi = np.quantile(difference, [.025, .975])
                        row[name] = float(sign * (point[key][i] - point[key][j]))
                        row[name + "_ci_low"], row[name + "_ci_high"] = float(lo), float(hi)
                    denominator = float(point["ndcg"][j])
                    row["comparator_test_ndcg"] = denominator
                    row["selected_test_ndcg"] = float(point["ndcg"][i])
                    row["relative_denominator_near_zero"] = denominator <= 1e-8
                    row["relative_ndcg_regret_percent"] = (100 * row["ndcg_regret_comparator_minus_selected"] / denominator
                                                            if denominator > 1e-8 else None)
                contrasts.append(row)
        print(f"{dataset}/{retention}: {len(complete)} evaluated configs, {len(condition.configuration_id.unique()) - len(complete)} missing", flush=True)
    pd.DataFrame(outcomes).to_csv(output / "selection_outcomes.csv", index=False)
    selected.rename(columns={"admissible": "historical_ua_admissible_at_original_margin"}).to_csv(
        output / "validation_selections_labeled.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "paired_test_contrasts.csv", index=False)
    pd.DataFrame(missing, columns=["dataset", "retention", "configuration_id", "dimension", "regularization", "seed",
                                  "reason", "estimated_fit_seconds_development_mean", "runtime_caveat"]).to_csv(output / "missing_runs.csv", index=False)
    write_json(output / "reused_test_provenance.json", provenance)
    write_json(output / "analysis_summary.json", {
        "stage": protocol["stage"], "protocol_sha256": sha(output / "revision_protocol.json"),
        "analysis_module_sha256": sha(Path(__file__)),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "condition_count": 16, "selection_rows": len(outcomes), "contrast_rows": len(contrasts),
        "missing_runs": len(missing),
        "missing_distinct_condition_configurations": len({(r["dataset"], r["retention"], r["configuration_id"]) for r in missing}),
        "estimated_missing_fit_seconds": sum(r["estimated_fit_seconds_development_mean"] for r in missing),
        "new_fitting_performed": any(r["artifact_stage"] == "new_revision_stage_fit" for r in provenance),
        "new_revision_runs_used": sum(r["artifact_stage"] == "new_revision_stage_fit" for r in provenance),
        "historical_artifacts_modified": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "analyze"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    (prepare if args.action == "prepare" else analyze)(root, output)


if __name__ == "__main__":
    main()
