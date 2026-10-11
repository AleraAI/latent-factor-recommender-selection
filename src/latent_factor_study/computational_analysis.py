"""Summarize measured confirmation-stage training costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import DEFAULT_CONFIG_PATH, _repo_root, sha256_file


SOURCES = (
    ("explicit_mf", "outputs/latent_factor_beyond_rmse/confirmation"),
    ("bias_only", "outputs/latent_factor_beyond_rmse/baseline_confirmation"),
    ("bpr_mf", "outputs/latent_factor_beyond_rmse/bpr_confirmation"),
)


def _roles(summary: dict[str, Any], model: str) -> list[str]:
    if model == "bpr_mf":
        return [str(summary.get("selection_role", "ndcg_selected"))]
    return [str(role) for role in summary.get("selection_roles", [])]


def collect_confirmation_costs(root: Path) -> pd.DataFrame:
    """Collect one row per model, selection role, and fresh-seed fit."""
    records: list[dict[str, Any]] = []
    for model, relative_root in SOURCES:
        for path in sorted((root / relative_root).glob("**/summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("model") == "popularity":
                continue
            training = summary.get("training", {})
            if "seconds" not in training:
                continue
            config = summary["run_config"]
            for role in _roles(summary, model):
                records.append(
                    {
                        "dataset": summary["dataset"],
                        "retention_label": summary["retention_label"],
                        "model": model,
                        "selection_role": role,
                        "dimension": int(config["dimension"]),
                        "regularization": float(config["regularization"]),
                        "seed": int(config["seed"]),
                        "epochs_completed": int(len(training["epoch_history"])),
                        "training_seconds": float(training["seconds"]),
                        "summary_path": str(path.relative_to(root)),
                        "summary_sha256": sha256_file(path),
                    }
                )
    return pd.DataFrame.from_records(records)


def aggregate_confirmation_costs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fresh-seed timing distributions by locked selection."""
    keys = [
        "dataset",
        "retention_label",
        "model",
        "selection_role",
        "dimension",
        "regularization",
    ]
    return (
        runs.groupby(keys, as_index=False)
        .agg(
            fresh_seed_fits=("seed", "nunique"),
            median_epochs=("epochs_completed", "median"),
            median_training_seconds=("training_seconds", "median"),
            q25_training_seconds=("training_seconds", lambda values: values.quantile(0.25)),
            q75_training_seconds=("training_seconds", lambda values: values.quantile(0.75)),
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/latent_factor_beyond_rmse/computational_analysis"),
    )
    args = parser.parse_args()
    root = _repo_root(args.config)
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_confirmation_costs(root)
    aggregate = aggregate_confirmation_costs(runs)
    runs_path = output_dir / "confirmation_training_cost_runs.csv"
    aggregate_path = output_dir / "confirmation_training_cost_summary.csv"
    runs.to_csv(runs_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    print(runs_path)
    print(aggregate_path)


if __name__ == "__main__":
    main()
