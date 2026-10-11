"""Resumable validation-only explicit-MF grid execution and selection."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve
from .explicit_mf import run_validation


def _execute_task(task: dict[str, Any]) -> str:
    return str(run_validation(**task))


def select_configurations(rows: pd.DataFrame) -> dict[str, Any]:
    required = {
        "dimension",
        "regularization",
        "seed",
        "validation_rmse",
        "validation_ndcg@10",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Grid rows are missing columns: {missing}")
    grouped = (
        rows.groupby(["dimension", "regularization"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_validation_rmse=("validation_rmse", "mean"),
            std_validation_rmse=("validation_rmse", "std"),
            mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
            std_validation_ndcg_at_10=("validation_ndcg@10", "std"),
        )
    )
    rmse_choice = grouped.sort_values(
        ["mean_validation_rmse", "dimension", "regularization"],
        ascending=[True, True, False],
        kind="stable",
    ).iloc[0]
    ndcg_choice = grouped.sort_values(
        ["mean_validation_ndcg_at_10", "dimension", "regularization"],
        ascending=[False, True, False],
        kind="stable",
    ).iloc[0]

    def choice(row: pd.Series) -> dict[str, Any]:
        return {
            "dimension": int(row["dimension"]),
            "regularization": float(row["regularization"]),
            "selection_seed_count": int(row["seeds"]),
            "mean_validation_rmse": float(row["mean_validation_rmse"]),
            "mean_validation_ndcg@10": float(row["mean_validation_ndcg_at_10"]),
        }

    return {
        "rmse_selected": choice(rmse_choice),
        "ndcg_selected": choice(ndcg_choice),
        "same_configuration": bool(
            rmse_choice["dimension"] == ndcg_choice["dimension"]
            and rmse_choice["regularization"] == ndcg_choice["regularization"]
        ),
        "configuration_table": grouped.sort_values(
            ["dimension", "regularization"], kind="stable"
        ).to_dict("records"),
    }


def _summary_matches(
    path: Path,
    *,
    dimension: int,
    regularization: float,
    seed: int,
    max_epochs: int,
) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    run = value.get("run_config", {})
    return bool(
        not value.get("smoke", True)
        and int(run.get("dimension", -1)) == dimension
        and float(run.get("regularization", -1.0)) == regularization
        and int(run.get("seed", -1)) == seed
        and int(run.get("max_epochs", -1)) == max_epochs
    )


def execute_grid(
    *,
    config_path: Path,
    modeling_path: Path,
    dataset_name: str,
    retention_label: str,
    force: bool = False,
    workers: int = 1,
) -> Path:
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    explicit = modeling["explicit_mf"]
    dimensions = [int(value) for value in explicit["dimensions"]]
    regularizations = [float(value) for value in explicit["regularizations"]]
    seeds = [int(value) for value in explicit["selection_seeds"]]
    max_epochs = int(explicit["max_epochs"])
    grid_dir = _resolve(root, modeling["development_output_root"]) / dataset_name / retention_label
    total = len(dimensions) * len(regularizations) * len(seeds)
    completed = 0
    pending: list[dict[str, Any]] = []

    for dimension in dimensions:
        for regularization in regularizations:
            for seed in seeds:
                run_id = f"d{dimension}_lambda{regularization:g}_seed{seed}"
                summary_path = grid_dir / run_id / "summary.json"
                if force or not _summary_matches(
                    summary_path,
                    dimension=dimension,
                    regularization=regularization,
                    seed=seed,
                    max_epochs=max_epochs,
                ):
                    pending.append(
                        {
                            "config_path": config_path,
                            "modeling_path": modeling_path,
                            "dataset_name": dataset_name,
                            "retention_label": retention_label,
                            "dimension": dimension,
                            "regularization": regularization,
                            "seed": seed,
                            "smoke": False,
                        }
                    )
                else:
                    completed += 1

    if workers < 1:
        raise ValueError("workers must be positive.")
    if workers == 1:
        for task in pending:
            _execute_task(task)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(f"{dataset_name}/{retention_label}: {completed}/{total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_execute_task, task) for task in pending]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total:
                    print(f"{dataset_name}/{retention_label}: {completed}/{total}", flush=True)
    if not pending:
        print(f"{dataset_name}/{retention_label}: {completed}/{total} already complete", flush=True)

    records: list[dict[str, Any]] = []
    for dimension in dimensions:
        for regularization in regularizations:
            for seed in seeds:
                run_id = f"d{dimension}_lambda{regularization:g}_seed{seed}"
                summary_path = grid_dir / run_id / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                records.append(
                    {
                        "dataset": dataset_name,
                        "retention_label": retention_label,
                        "dimension": dimension,
                        "regularization": regularization,
                        "seed": seed,
                        "epochs": summary["training"]["epochs_completed"],
                        "training_seconds": summary["training"]["seconds"],
                        "validation_rmse": summary["validation"]["prediction"]["clipped"]["rmse"],
                        "validation_mae": summary["validation"]["prediction"]["clipped"]["mae"],
                        "validation_ndcg@10": summary["validation"]["ranking"]["ndcg@10"],
                        "validation_recall@10": summary["validation"]["ranking"]["recall@10"],
                        "validation_ndcg@20": summary["validation"]["ranking"]["ndcg@20"],
                        "validation_recall@20": summary["validation"]["ranking"]["recall@20"],
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(grid_dir / "grid_runs.csv", index=False)
    selection = {
        "schema_version": 1,
        "stage": "development_validation_only",
        "dataset": dataset_name,
        "retention_label": retention_label,
        "expected_runs": total,
        "observed_runs": len(frame),
        "selection": select_configurations(frame),
    }
    selection_path = grid_dir / "selection.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--modeling-config",
        type=Path,
        default=Path("configs/latent_factor_beyond_rmse/modeling.yaml"),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["movielens_1m", "ciaodvd", "yelp_tampa_restaurants", "epinions"],
    )
    parser.add_argument(
        "--retention", required=True, choices=["all", "r100", "r075", "r050", "r025"]
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    retentions = ["r100", "r075", "r050", "r025"] if args.retention == "all" else [args.retention]
    for retention in retentions:
        print(
            execute_grid(
                config_path=args.config,
                modeling_path=args.modeling_config,
                dataset_name=args.dataset,
                retention_label=retention,
                force=args.force,
                workers=args.workers,
            )
        )


if __name__ == "__main__":
    main()
