"""Paired user-level inference for BPR versus ranking-selected explicit MF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .bpr_grid import DATASETS, RETENTIONS
from .baseline_lock import DEFAULT_BASELINE_CONFIRMATION_PATH
from .bpr_lock import DEFAULT_BPR_CONFIRMATION_PATH
from .data import _repo_root, _resolve, sha256_file
from .inference import (
    bootstrap_mean_difference,
    holm_adjust,
    paired_sign_randomization_p_value,
)


def _mean_role_users(
    root: Path,
    manifest: dict[str, Any],
    *,
    role_key: str,
    role_value: str,
    roles_are_lists: bool,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for run in manifest["runs"]:
        roles = run[role_key] if roles_are_lists else [run[role_key]]
        if role_value not in roles:
            continue
        per_user_path = root / run["per_user_path"]
        if sha256_file(per_user_path) != run["per_user_sha256"]:
            raise ValueError(f"Per-user checksum mismatch: {per_user_path}")
        frame = pd.read_parquet(per_user_path, columns=["user_id", "ndcg@10"])
        frame["user_id"] = frame["user_id"].astype(str)
        frame["seed"] = int(run["seed"])
        parts.append(frame)
    if not parts:
        raise ValueError(f"Confirmation manifest contains no {role_value} runs.")
    combined = pd.concat(parts, ignore_index=True)
    seed_counts = combined.groupby("user_id")["seed"].nunique()
    if int(seed_counts.min()) != 5 or int(seed_counts.max()) != 5:
        raise ValueError(f"{role_value} does not have five fresh seeds for every user.")
    return combined.groupby("user_id", as_index=False)[["ndcg@10"]].mean()


def analyze_bpr_confirmation(confirmation_path: Path) -> Path:
    config = yaml.safe_load(confirmation_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(confirmation_path)
    bpr_output = _resolve(root, config["output_root"])
    explicit_config_path = _resolve(root, config["explicit_confirmation_config"])
    explicit_config = yaml.safe_load(explicit_config_path.read_text(encoding="utf-8")) or {}
    explicit_output = _resolve(root, explicit_config["output_root"])
    baseline_config = yaml.safe_load(
        (root / DEFAULT_BASELINE_CONFIRMATION_PATH).read_text(encoding="utf-8")
    ) or {}
    baseline_output = _resolve(root, baseline_config["output_root"])
    analysis_dir = bpr_output / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    inference = config["inference"]
    bootstrap_resamples = int(inference["user_cluster_bootstrap_resamples"])
    randomization_resamples = int(inference["paired_randomization_resamples"])
    confidence_level = float(inference["confidence_level"])
    practical_threshold = float(
        inference["minimum_practically_important_ndcg_at_10_difference"]
    )
    base_seed = int(inference["seed"])

    records: list[dict[str, Any]] = []
    condition_index = 0
    for dataset in DATASETS:
        for retention in RETENTIONS:
            bpr_manifest_path = (
                bpr_output / dataset / retention / "confirmation_manifest.json"
            )
            explicit_manifest_path = (
                explicit_output / dataset / retention / "confirmation_manifest.json"
            )
            baseline_manifest_path = (
                baseline_output / dataset / retention / "confirmation_manifest.json"
            )
            bpr_manifest = json.loads(bpr_manifest_path.read_text(encoding="utf-8"))
            explicit_manifest = json.loads(
                explicit_manifest_path.read_text(encoding="utf-8")
            )
            baseline_manifest = json.loads(
                baseline_manifest_path.read_text(encoding="utf-8")
            )
            for manifest_path, manifest in (
                (bpr_manifest_path, bpr_manifest),
                (explicit_manifest_path, explicit_manifest),
            ):
                if int(manifest["observed_runs"]) != int(manifest["expected_runs"]):
                    raise ValueError(f"Incomplete confirmation manifest: {manifest_path}")
            popularity_record = baseline_manifest["popularity"]
            popularity_path = root / popularity_record["per_user_path"]
            if sha256_file(popularity_path) != popularity_record["per_user_sha256"]:
                raise ValueError(f"Popularity checksum mismatch: {popularity_path}")

            bpr_users = _mean_role_users(
                root,
                bpr_manifest,
                role_key="selection_role",
                role_value="bpr_selected",
                roles_are_lists=False,
            ).rename(columns={"ndcg@10": "ndcg@10_bpr"})
            explicit_users = _mean_role_users(
                root,
                explicit_manifest,
                role_key="selection_roles",
                role_value="ndcg_selected",
                roles_are_lists=True,
            ).rename(columns={"ndcg@10": "ndcg@10_explicit"})
            paired = explicit_users.merge(
                bpr_users, on="user_id", how="inner", validate="one_to_one"
            )
            paired["delta_ndcg@10"] = (
                paired["ndcg@10_bpr"] - paired["ndcg@10_explicit"]
            )
            popularity = pd.read_parquet(
                popularity_path, columns=["user_id", "ndcg@10"]
            ).rename(columns={"ndcg@10": "ndcg@10_popularity"})
            popularity["user_id"] = popularity["user_id"].astype(str)
            paired = paired.merge(
                popularity, on="user_id", how="inner", validate="one_to_one"
            )
            paired["delta_bpr_minus_popularity_ndcg@10"] = (
                paired["ndcg@10_bpr"] - paired["ndcg@10_popularity"]
            )
            paired_path = analysis_dir / f"{dataset}_{retention}_paired_users.parquet"
            paired.to_parquet(paired_path, index=False)
            differences = paired["delta_ndcg@10"].dropna().to_numpy(dtype=float)
            popularity_differences = paired[
                "delta_bpr_minus_popularity_ndcg@10"
            ].dropna().to_numpy(dtype=float)
            condition_seed = base_seed + condition_index * 10
            lower, upper, bootstrap = bootstrap_mean_difference(
                differences,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=condition_seed,
            )
            resamples_path = analysis_dir / f"{dataset}_{retention}_resamples.npz"
            pop_lower, pop_upper, pop_bootstrap = bootstrap_mean_difference(
                popularity_differences,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=condition_seed + 2,
            )
            np.savez_compressed(
                resamples_path,
                bpr_minus_explicit_mean_differences=bootstrap,
                bpr_minus_popularity_mean_differences=pop_bootstrap,
            )
            explicit_mean = float(paired["ndcg@10_explicit"].mean())
            delta = float(differences.mean())
            records.append(
                {
                    "dataset": dataset,
                    "retention_label": retention,
                    "users_paired": int(len(differences)),
                    "explicit_ndcg_selected_mean_ndcg@10": explicit_mean,
                    "bpr_selected_mean_ndcg@10": float(paired["ndcg@10_bpr"].mean()),
                    "delta_bpr_minus_explicit_ndcg@10": delta,
                    "delta_relative": (
                        delta / explicit_mean if explicit_mean != 0.0 else float("nan")
                    ),
                    "delta_ci_lower": lower,
                    "delta_ci_upper": upper,
                    "delta_randomization_p": paired_sign_randomization_p_value(
                        differences,
                        resamples=randomization_resamples,
                        seed=condition_seed + 1,
                    ),
                    "popularity_mean_ndcg@10": float(
                        paired["ndcg@10_popularity"].mean()
                    ),
                    "delta_bpr_minus_popularity_ndcg@10": float(
                        popularity_differences.mean()
                    ),
                    "delta_popularity_ci_lower": pop_lower,
                    "delta_popularity_ci_upper": pop_upper,
                    "delta_popularity_randomization_p": paired_sign_randomization_p_value(
                        popularity_differences,
                        resamples=randomization_resamples,
                        seed=condition_seed + 3,
                    ),
                    "practical_threshold": practical_threshold,
                    "practically_meaningful": abs(delta) >= practical_threshold,
                    "paired_users_sha256": sha256_file(paired_path),
                    "resamples_sha256": sha256_file(resamples_path),
                }
            )
            condition_index += 1

    adjusted = holm_adjust([row["delta_randomization_p"] for row in records])
    for row, value in zip(records, adjusted):
        row["delta_holm_p"] = value
    popularity_adjusted = holm_adjust(
        [row["delta_popularity_randomization_p"] for row in records]
    )
    for row, value in zip(records, popularity_adjusted):
        row["delta_popularity_holm_p"] = value
    frame = pd.DataFrame.from_records(records)
    csv_path = analysis_dir / "bpr_boundary_comparison.csv"
    frame.to_csv(csv_path, index=False)
    (analysis_dir / "bpr_boundary_comparison.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_BPR_CONFIRMATION_PATH
    )
    args = parser.parse_args()
    print(analyze_bpr_confirmation(args.confirmation_config))


if __name__ == "__main__":
    main()
