"""Analyze validation configuration landscapes without accessing test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

from .data import _repo_root, _resolve


DATASETS = ("movielens_1m", "ciaodvd", "yelp_tampa_restaurants")
RETENTIONS = ("r100", "r075", "r050", "r025")


def configuration_agreement(grid_runs: pd.DataFrame) -> dict[str, Any]:
    grouped = (
        grid_runs.groupby(["dimension", "regularization"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_validation_rmse=("validation_rmse", "mean"),
            mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
        )
    )
    if len(grouped) < 2:
        raise ValueError("Agreement analysis requires at least two configurations.")
    spearman = spearmanr(
        -grouped["mean_validation_rmse"], grouped["mean_validation_ndcg_at_10"]
    )
    kendall = kendalltau(
        -grouped["mean_validation_rmse"],
        grouped["mean_validation_ndcg_at_10"],
        variant="b",
    )
    grouped["rmse_rank"] = grouped["mean_validation_rmse"].rank(
        method="min", ascending=True
    )
    grouped["ndcg_rank"] = grouped["mean_validation_ndcg_at_10"].rank(
        method="min", ascending=False
    )
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
    rmse_choice = rmse_order.iloc[0]
    ndcg_choice = ndcg_order.iloc[0]
    key = lambda row: (int(row["dimension"]), float(row["regularization"]))
    top3_rmse = {key(row) for _, row in rmse_order.head(3).iterrows()}
    top3_ndcg = {key(row) for _, row in ndcg_order.head(3).iterrows()}
    return {
        "configurations": int(len(grouped)),
        "minimum_seed_count": int(grouped["seed_count"].min()),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "kendall_tau_b": float(kendall.statistic),
        "kendall_p_value": float(kendall.pvalue),
        "top3_overlap": int(len(top3_rmse.intersection(top3_ndcg))),
        "same_top_configuration": key(rmse_choice) == key(ndcg_choice),
        "rmse_selected_dimension": int(rmse_choice["dimension"]),
        "rmse_selected_regularization": float(rmse_choice["regularization"]),
        "rmse_selected_ndcg_rank": int(rmse_choice["ndcg_rank"]),
        "ndcg_selected_dimension": int(ndcg_choice["dimension"]),
        "ndcg_selected_regularization": float(ndcg_choice["regularization"]),
        "ndcg_selected_rmse_rank": int(ndcg_choice["rmse_rank"]),
    }


def analyze_available(modeling_path: Path) -> Path:
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(modeling_path)
    development_root = _resolve(root, modeling["development_output_root"])
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for retention in RETENTIONS:
            grid_path = development_root / dataset / retention / "grid_runs.csv"
            if not grid_path.exists():
                continue
            grid = pd.read_csv(grid_path)
            agreement = configuration_agreement(grid)
            records.append(
                {"dataset": dataset, "retention_label": retention, **agreement}
            )
    output = development_root / "development_landscape_summary.csv"
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output, index=False)
    (development_root / "development_landscape_summary.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modeling-config",
        type=Path,
        default=Path("configs/latent_factor_beyond_rmse/modeling.yaml"),
    )
    args = parser.parse_args()
    print(analyze_available(args.modeling_config))


if __name__ == "__main__":
    main()
