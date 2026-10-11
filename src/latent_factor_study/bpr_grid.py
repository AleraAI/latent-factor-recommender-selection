"""Resumable validation-only BPR-MF grid execution and selection."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .bpr_mf import DEFAULT_BPR_MODELING_PATH, run_bpr_validation
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve


DATASETS = ("movielens_1m", "ciaodvd", "yelp_tampa_restaurants")
RETENTIONS = ("r100", "r075", "r050", "r025")


def _execute_task(task: dict[str, Any]) -> str:
    return str(run_bpr_validation(**task))


def select_bpr_configuration(rows: pd.DataFrame) -> dict[str, Any]:
    required = {"dimension", "regularization", "seed", "validation_ndcg@10"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"BPR grid rows are missing columns: {missing}")
    grouped = (
        rows.groupby(["dimension", "regularization"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_validation_ndcg_at_10=("validation_ndcg@10", "mean"),
            std_validation_ndcg_at_10=("validation_ndcg@10", "std"),
            mean_validation_recall_at_10=("validation_recall@10", "mean"),
            mean_training_seconds=("training_seconds", "mean"),
        )
    )
    choice = grouped.sort_values(
        ["mean_validation_ndcg_at_10", "dimension", "regularization"],
        ascending=[False, True, False],
        kind="stable",
    ).iloc[0]
    return {
        "bpr_selected": {
            "dimension": int(choice["dimension"]),
            "regularization": float(choice["regularization"]),
            "selection_seed_count": int(choice["seeds"]),
            "mean_validation_ndcg@10": float(choice["mean_validation_ndcg_at_10"]),
            "mean_validation_recall@10": float(choice["mean_validation_recall_at_10"]),
        },
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
        value.get("stage") == "bpr_development_validation_only"
        and not value.get("smoke", True)
        and int(run.get("dimension", -1)) == dimension
        and float(run.get("regularization", -1.0)) == regularization
        and int(run.get("seed", -1)) == seed
        and int(run.get("max_epochs", -1)) == max_epochs
    )


def execute_bpr_grid(
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
    bpr = modeling["bpr_mf"]
    dimensions = [int(value) for value in bpr["dimensions"]]
    regularizations = [float(value) for value in bpr["regularizations"]]
    seeds = [int(value) for value in bpr["selection_seeds"]]
    max_epochs = int(bpr["max_epochs"])
    grid_dir = (
        _resolve(root, modeling["development_output_root"])
        / dataset_name
        / retention_label
    )
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
                    print(
                        f"{dataset_name}/{retention_label}: {completed}/{total}",
                        flush=True,
                    )
    if not pending:
        print(
            f"{dataset_name}/{retention_label}: {completed}/{total} already complete",
            flush=True,
        )

    records: list[dict[str, Any]] = []
    for dimension in dimensions:
        for regularization in regularizations:
            for seed in seeds:
                run_id = f"d{dimension}_lambda{regularization:g}_seed{seed}"
                summary_path = grid_dir / run_id / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                ranking = summary["validation"]["ranking"]
                records.append(
                    {
                        "dataset": dataset_name,
                        "retention_label": retention_label,
                        "dimension": dimension,
                        "regularization": regularization,
                        "seed": seed,
                        "epochs": summary["training"]["epochs_completed"],
                        "training_seconds": summary["training"]["seconds"],
                        "positive_rows": summary["training"]["positive_rows"],
                        "users_without_positive_training": summary["training"][
                            "users_without_positive_training"
                        ],
                        "validation_ndcg@10": ranking["ndcg@10"],
                        "validation_recall@10": ranking["recall@10"],
                        "validation_ndcg@20": ranking["ndcg@20"],
                        "validation_recall@20": ranking["recall@20"],
                    }
                )
    frame = pd.DataFrame.from_records(records)
    grid_dir.mkdir(parents=True, exist_ok=True)
    grid_path = grid_dir / "grid_runs.csv"
    frame.to_csv(grid_path, index=False)
    selection = select_bpr_configuration(frame)
    selection_payload = {
        "schema_version": 1,
        "stage": "bpr_development_validation_only",
        "dataset": dataset_name,
        "retention_label": retention_label,
        "expected_runs": total,
        "observed_runs": int(len(frame)),
        "selection": selection,
    }
    (grid_dir / "selection.json").write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return grid_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--modeling-config", type=Path, default=DEFAULT_BPR_MODELING_PATH)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--retention", required=True, choices=RETENTIONS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        execute_bpr_grid(
            config_path=args.config,
            modeling_path=args.modeling_config,
            dataset_name=args.dataset,
            retention_label=args.retention,
            force=args.force,
            workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
