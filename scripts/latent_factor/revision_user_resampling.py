"""Fixed-model, end-to-end validation-user resampling for the journal revision.

Reads development artifacts only. Historical sources and outputs are never changed.
See output/latent_factor_major_revision_2026-09-14/user_resampling/PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t


RETENTIONS = ("r100", "r075", "r050", "r025")
SEEDS = (101, 211, 307, 401, 503)
CONFIGS = tuple(itertools.product((1, 2, 4, 8, 16, 32, 64), (0., .0001, .001, .01, .1)))
MARGINS = (.001, .0025, .005, .010)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_id(index: int) -> str:
    dimension, regularization = CONFIGS[index]
    return f"d{dimension}_lambda{regularization:g}"


def aggregate_users(weights, ndcg, sse, counts, eligible):
    """Return per-draw configuration/seed NDCG and event-weighted global RMSE.

    weights is B x U; ndcg and sse are C x S x U. Ineligible ranking values
    must have been checked and replaced by zero, not included in denominator.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError("User multiplicities must be a finite nonnegative B x U matrix.")
    n_rank = weights @ np.asarray(eligible, dtype=np.float64)
    n_prediction = weights @ np.asarray(counts, dtype=np.float64)
    if (n_rank <= 0).any() or (n_prediction <= 0).any():
        raise ValueError("Every draw requires positive ranking and prediction denominators.")
    shape = (len(weights), *ndcg.shape[:2])
    sampled_ndcg = (weights @ ndcg.reshape(-1, ndcg.shape[-1]).T / n_rank[:, None]).reshape(shape)
    sampled_rmse = np.sqrt(
        weights @ sse.reshape(-1, sse.shape[-1]).T / n_prediction[:, None]
    ).reshape(shape)
    return sampled_ndcg, sampled_rmse, n_rank, n_prediction


def select_arrays(ndcg, rmse, fit_seconds, *, margins=MARGINS, configs=CONFIGS):
    """Original reference and tie hierarchy; ablation changes only uncertainty."""
    ndcg = np.asarray(ndcg, dtype=np.float64)
    rmse = np.asarray(rmse, dtype=np.float64)
    fit_seconds = np.asarray(fit_seconds, dtype=np.float64)
    if ndcg.shape != rmse.shape or ndcg.ndim != 2 or ndcg.shape[0] != len(configs):
        raise ValueError("Expected matching complete configuration x seed matrices.")
    if ndcg.shape[1] < 2 or not np.isfinite(ndcg).all() or not np.isfinite(rmse).all():
        raise ValueError("At least two complete finite paired seeds are required.")
    if fit_seconds.shape != (len(configs),) or not np.isfinite(fit_seconds).all():
        raise ValueError("Recorded configuration-level fit seconds are required.")
    if not all(np.isfinite(m) and m >= 0 for m in margins):
        raise ValueError("Margins must be finite and nonnegative.")
    mean_ndcg, mean_rmse = ndcg.mean(axis=1), rmse.mean(axis=1)
    reference = min(range(len(configs)), key=lambda i: (-mean_ndcg[i], configs[i][0], -configs[i][1]))
    rmse_reference = min(range(len(configs)), key=lambda i: (mean_rmse[i], configs[i][0], -configs[i][1]))
    regrets = ndcg[reference] - ndcg
    mean_regret = regrets.mean(axis=1)
    penalty = float(t.ppf(.95, ndcg.shape[1] - 1)) * regrets.std(axis=1, ddof=1) / np.sqrt(ndcg.shape[1])
    result = []
    for margin in margins:
        for selector, screen in (("mean", mean_regret), ("UA", mean_regret + penalty)):
            admissible = np.flatnonzero(screen <= margin)
            if len(admissible) == 0:
                raise AssertionError("The ranking reference must be admissible.")
            choice = min(admissible, key=lambda i: (mean_rmse[i], fit_seconds[i], configs[i][0], -configs[i][1]))
            result.append({
                "delta": float(margin), "selector": selector,
                "reference_index": int(reference), "rmse_reference_index": int(rmse_reference),
                "selected_index": int(choice), "admissible_count": int(len(admissible)),
                "selected_mean_global_rmse": float(mean_rmse[choice]),
                "selected_mean_ndcg_at_10": float(mean_ndcg[choice]),
                "reference_mean_ndcg_at_10": float(mean_ndcg[reference]),
                "selected_mean_regret": float(mean_regret[choice]),
                "selected_uncertainty_penalty": float(penalty[choice]),
                "selected_upper_bound": float(mean_regret[choice] + penalty[choice]),
                "all_admissible": bool(len(admissible) == len(configs)),
                "matches_draw_rmse_reference": bool(choice == rmse_reference),
            })
    return result


def load_condition(grid_dir: Path, provenance: list[dict]):
    """Verify all 175 runs, all user identities/masks, and historical metrics."""
    grid_path = grid_dir / "grid_runs.csv"
    grid = pd.read_csv(grid_path)
    expected = set((d, reg, seed) for (d, reg), seed in itertools.product(CONFIGS, SEEDS))
    keys = list(map(tuple, grid[["dimension", "regularization", "seed"]].values))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"Incomplete or duplicate 175-run grid: {grid_dir}")
    provenance.append({"path": str(grid_path), "sha256": digest(grid_path)})
    lookup = grid.set_index(["dimension", "regularization", "seed"])
    baseline = None
    ndcg = []
    sse = []
    fit_seconds = []
    max_rmse_error = 0.
    max_ndcg_error = 0.
    for dim, reg in CONFIGS:
        config_ndcg, config_sse, times = [], [], []
        for seed in SEEDS:
            path = grid_dir / f"d{dim}_lambda{reg:g}_seed{seed}" / "validation_per_user.parquet"
            frame = pd.read_parquet(path).sort_values("user_id").reset_index(drop=True)
            required = {"user_id", "num_candidates", "num_relevant", "num_cold_relevant",
                        "num_prediction_rows", "prediction_rmse", "ndcg@10"}
            if not required.issubset(frame):
                raise ValueError(f"Missing prediction/eligibility columns: {path}")
            if frame.user_id.duplicated().any():
                raise ValueError(f"Duplicate validation user: {path}")
            identity = frame[["user_id", "num_candidates", "num_relevant", "num_cold_relevant", "num_prediction_rows"]]
            if baseline is None:
                baseline = identity.copy()
            elif not identity.equals(baseline):
                raise ValueError(f"User identity, candidate, eligibility, or event-count mismatch: {path}")
            counts = frame.num_prediction_rows.to_numpy(dtype=np.float64)
            prediction = frame.prediction_rmse.to_numpy(dtype=np.float64)
            score = frame["ndcg@10"].to_numpy(dtype=np.float64)
            eligible = frame.num_relevant.to_numpy() > 0
            if not np.isfinite(counts).all() or (counts <= 0).any() or (counts != counts.astype(int)).any():
                raise ValueError(f"Invalid held-out event counts: {path}")
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise ValueError(f"Invalid prediction values: {path}")
            if not np.isfinite(score[eligible]).all() or not np.isnan(score[~eligible]).all():
                raise ValueError(f"Invalid ranking eligibility/missingness: {path}")
            if (score[eligible] < 0).any() or (score[eligible] > 1 + 1e-12).any():
                raise ValueError(f"Invalid NDCG bounds: {path}")
            row = lookup.loc[(dim, reg, seed)]
            total_sse = counts * prediction ** 2
            rmse_error = abs(np.sqrt(total_sse.sum() / counts.sum()) - row.validation_rmse)
            ndcg_error = abs(score[eligible].mean() - row["validation_ndcg@10"])
            max_rmse_error = max(max_rmse_error, rmse_error)
            max_ndcg_error = max(max_ndcg_error, ndcg_error)
            if max(rmse_error, ndcg_error) > 1e-12:
                raise ValueError(f"Historical grid metric cannot be reconstructed: {path}")
            config_ndcg.append(np.where(eligible, score, 0.))
            config_sse.append(total_sse)
            times.append(float(row.training_seconds))
            provenance.append({"path": str(path), "sha256": digest(path)})
        ndcg.append(config_ndcg)
        sse.append(config_sse)
        fit_seconds.append(np.mean(times))
    audit = {"grid_runs": len(grid), "per_user_files": len(grid), "users": len(baseline),
             "ranking_eligible_users": int((baseline.num_relevant > 0).sum()),
             "prediction_events": int(baseline.num_prediction_rows.sum()),
             "max_reconstructed_rmse_error": max_rmse_error,
             "max_reconstructed_ndcg_error": max_ndcg_error}
    return baseline, np.asarray(ndcg), np.asarray(sse), np.asarray(fit_seconds), audit


def run(root: Path, output: Path, *, resamples: int = 500, seed: int = 20260914):
    protocol_path = output / "PROTOCOL.md"
    if not protocol_path.is_file():
        raise ValueError("The saved pre-execution protocol is required.")
    if resamples != 500 or seed != 20260914:
        raise ValueError("This version implements only the saved 500-draw, seed-20260914 protocol.")
    for name in ("selection_draws.csv", "manifest.json"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite revision results: {output/name}")
    provenance = [{"path": str(protocol_path), "sha256": digest(protocol_path)}]
    first_identity = None
    weights = None
    records, baselines, audits = [], [], []
    for retention in RETENTIONS:
        grid_dir = root / "outputs/latent_factor_beyond_rmse/prospective_epinions/development/epinions" / retention
        identity, ndcg, sse, times, audit = load_condition(grid_dir, provenance)
        if first_identity is None:
            first_identity = identity
            rng = np.random.default_rng(seed)
            weights = np.empty((resamples, len(identity)), dtype=np.int64)
            for b in range(resamples):
                weights[b] = np.bincount(rng.integers(0, len(identity), size=len(identity)), minlength=len(identity))
        elif not identity.equals(first_identity):
            raise ValueError("Cross-retention user/catalog/eligibility/event-count mismatch.")
        counts = identity.num_prediction_rows.to_numpy(dtype=float)
        eligible = identity.num_relevant.to_numpy() > 0
        full_ndcg, full_rmse, _, _ = aggregate_users(np.ones((1, len(identity))), ndcg, sse, counts, eligible)
        baseline_rows = select_arrays(full_ndcg[0], full_rmse[0], times)
        frozen_path = grid_dir / "framework_selection.json"
        frozen = json.loads(frozen_path.read_text())["selection"]
        provenance.append({"path": str(frozen_path), "sha256": digest(frozen_path)})
        original = next(row for row in baseline_rows if row["selector"] == "UA" and row["delta"] == .005)
        for index_key, frozen_key in (("selected_index", "framework_selected"), ("reference_index", "ndcg_selected"), ("rmse_reference_index", "rmse_selected")):
            actual = CONFIGS[original[index_key]]
            expected = (int(frozen[frozen_key]["dimension"]), float(frozen[frozen_key]["regularization"]))
            if actual != expected:
                raise ValueError(f"Full-cohort selector identity mismatch: {retention}/{frozen_key}")
        if original["admissible_count"] != frozen["admissible_configurations"]:
            raise ValueError("Full-cohort admissible count mismatch.")
        for row in baseline_rows:
            row.update(retention=retention)
            baselines.append(row)
        sample_ndcg, sample_rmse, rank_n, pred_n = aggregate_users(weights, ndcg, sse, counts, eligible)
        lookup = {(row["selector"], row["delta"]): row for row in baseline_rows}
        for b in range(resamples):
            for row in select_arrays(sample_ndcg[b], sample_rmse[b], times):
                baseline = lookup[(row["selector"], row["delta"])]
                row.update(retention=retention, draw=b, sampled_ranking_users=int(rank_n[b]),
                           sampled_prediction_events=int(pred_n[b]),
                           matches_full_cohort_selection=bool(row["selected_index"] == baseline["selected_index"]),
                           matches_full_cohort_reference=bool(row["reference_index"] == baseline["reference_index"]))
                records.append(row)
        audits.append({"retention": retention, **audit, "original_identity_and_admissibility_verified": True})
        print(f"{retention}: verified 175 runs, completed {resamples} paired user draws", flush=True)
    all_draws, full = pd.DataFrame(records), pd.DataFrame(baselines)
    for frame in (all_draws, full):
        for prefix in ("reference", "rmse_reference", "selected"):
            frame[f"{prefix}_configuration"] = frame[f"{prefix}_index"].map(config_id)
    all_draws.to_csv(output / "selection_draws.csv", index=False)
    full.to_csv(output / "full_cohort_selections.csv", index=False)
    frequencies = (all_draws.groupby(["retention", "delta", "selector", "selected_configuration"], as_index=False)
                   .size().rename(columns={"size": "draw_count"}))
    frequencies["selection_frequency"] = frequencies.draw_count / resamples
    frequencies.to_csv(output / "selection_frequencies.csv", index=False)
    summary = all_draws.groupby(["retention", "delta", "selector"], as_index=False).agg(
        draws=("draw", "size"), distinct_selections=("selected_index", "nunique"),
        full_cohort_selection_frequency=("matches_full_cohort_selection", "mean"),
        full_cohort_reference_frequency=("matches_full_cohort_reference", "mean"),
        draw_rmse_reference_frequency=("matches_draw_rmse_reference", "mean"),
        all_admissible_frequency=("all_admissible", "mean"),
        admissible_min=("admissible_count", "min"), admissible_median=("admissible_count", "median"),
        admissible_max=("admissible_count", "max"))
    summary.to_csv(output / "selection_stability_summary.csv", index=False)
    paired = all_draws.pivot(index=["retention", "delta", "draw"], columns="selector", values="selected_index").reset_index()
    paired["selectors_identical"] = paired.UA == paired["mean"]
    paired.groupby(["retention", "delta"], as_index=False).agg(
        draws=("draw", "size"), identical_selection_frequency=("selectors_identical", "mean")
    ).to_csv(output / "paired_selector_agreement.csv", index=False)
    refs = all_draws[["retention", "draw", "reference_configuration"]].drop_duplicates()
    refs = refs.groupby(["retention", "reference_configuration"], as_index=False).size().rename(columns={"size": "draw_count"})
    refs["reference_frequency"] = refs.draw_count / resamples
    refs.to_csv(output / "reference_frequencies.csv", index=False)
    np.savez_compressed(output / "shared_user_multiplicities.npz", weights=weights)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "revision_stage_fixed_model_validation_user_resampling",
        "test_data_accessed": False, "model_fits": 0, "draws": resamples, "seed": seed,
        "seeds_selected_by": "manually fixed in pre-execution revision protocol",
        "margins": list(MARGINS), "retentions": list(RETENTIONS), "training_seeds": list(SEEDS),
        "same_user_draw_across_configurations_seeds_and_retentions": True,
        "ranking_reference_reselected_per_draw": True,
        "training_seconds": "original recorded grid values held fixed",
        "counts_interpretation": "conditional frequencies, not confidence guarantees",
        "construction_datasets_excluded": "Historical validation parquets lack per-user prediction errors and counts.",
        "audits": audits,
        "revision_environment": {"python": sys.version, "platform": platform.platform(),
                                 "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "script": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "inputs": provenance,
        "outputs": [{"path": str(p), "sha256": digest(p)} for p in sorted(output.iterdir()) if p.suffix in (".csv", ".npz")],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
