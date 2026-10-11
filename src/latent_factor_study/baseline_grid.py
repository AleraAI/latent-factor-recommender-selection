"""Resumable validation-only bias/popularity baseline execution and selection."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .baselines import (
    DEFAULT_BASELINE_MODELING_PATH,
    evaluate_popularity_stage,
    run_bias_validation,
)
from .bpr_grid import DATASETS, RETENTIONS
from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve
from .grid import select_configurations


def _execute_task(task: dict[str, Any]) -> str:
    return str(run_bias_validation(**task))


def _summary_matches(
    path: Path,
    *,
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
        value.get("stage") == "baseline_development_validation_only"
        and value.get("model") == "bias_only"
        and not value.get("smoke", True)
        and int(run.get("dimension", -1)) == 0
        and float(run.get("regularization", -1.0)) == regularization
        and int(run.get("seed", -1)) == seed
        and int(run.get("max_epochs", -1)) == max_epochs
    )


def _write_popularity_validation(
    *,
    config_path: Path,
    modeling_path: Path,
    dataset_name: str,
    retention_label: str,
    condition_dir: Path,
) -> Path:
    dataset_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    modeling = yaml.safe_load(modeling_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    dataset = dataset_config["datasets"][dataset_name]
    processed_dir = _resolve(root, dataset["output_dir"])
    sparsity_dir = _resolve(root, dataset_config["controlled_sparsity"]["output_root"]) / dataset_name
    sparse_train = pd.read_parquet(sparsity_dir / f"train_{retention_label}.parquet")
    full_train = pd.read_parquet(processed_dir / "train.parquet")
    validation = pd.read_parquet(processed_dir / "validation.parquet")
    summary, per_user = evaluate_popularity_stage(
        sparse_train=sparse_train,
        full_train=full_train,
        evaluation=validation,
        relevance_threshold=float(modeling["ranking"]["relevance_threshold"]),
    )
    output_dir = condition_dir / "popularity"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_user.to_parquet(output_dir / "validation_per_user.parquet", index=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "baseline_development_validation_only",
                "model": "popularity",
                "dataset": dataset_name,
                "retention_label": retention_label,
                "training": {
                    "rows": int(len(sparse_train)),
                    "users": int(sparse_train["user_id"].astype(str).nunique()),
                    "items": int(sparse_train["item_id"].astype(str).nunique()),
                },
                "validation": {"ranking": summary},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def execute_baseline_grid(
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
    settings = modeling["bias_only"]
    regularizations = [float(value) for value in settings["regularizations"]]
    seeds = [int(value) for value in settings["selection_seeds"]]
    max_epochs = int(settings["max_epochs"])
    condition_dir = (
        _resolve(root, modeling["development_output_root"])
        / dataset_name
        / retention_label
    )
    total = len(regularizations) * len(seeds)
    completed = 0
    pending: list[dict[str, Any]] = []
    for regularization in regularizations:
        for seed in seeds:
            run_id = f"lambda{regularization:g}_seed{seed}"
            summary_path = condition_dir / run_id / "summary.json"
            if force or not _summary_matches(
                summary_path,
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
            print(f"{dataset_name}/{retention_label}: {completed}/{total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_execute_task, task) for task in pending]
            for future in as_completed(futures):
                future.result()
                completed += 1
                print(f"{dataset_name}/{retention_label}: {completed}/{total}", flush=True)
    if not pending:
        print(
            f"{dataset_name}/{retention_label}: {completed}/{total} already complete",
            flush=True,
        )

    records: list[dict[str, Any]] = []
    for regularization in regularizations:
        for seed in seeds:
            run_id = f"lambda{regularization:g}_seed{seed}"
            summary = json.loads(
                (condition_dir / run_id / "summary.json").read_text(encoding="utf-8")
            )
            records.append(
                {
                    "dataset": dataset_name,
                    "retention_label": retention_label,
                    "dimension": 0,
                    "regularization": regularization,
                    "seed": seed,
                    "epochs": summary["training"]["epochs_completed"],
                    "training_seconds": summary["training"]["seconds"],
                    "validation_rmse": summary["validation"]["prediction"]["clipped"][
                        "rmse"
                    ],
                    "validation_mae": summary["validation"]["prediction"]["clipped"][
                        "mae"
                    ],
                    "validation_ndcg@10": summary["validation"]["ranking"]["ndcg@10"],
                    "validation_recall@10": summary["validation"]["ranking"][
                        "recall@10"
                    ],
                    "validation_ndcg@20": summary["validation"]["ranking"]["ndcg@20"],
                    "validation_recall@20": summary["validation"]["ranking"][
                        "recall@20"
                    ],
                }
            )
    frame = pd.DataFrame.from_records(records)
    condition_dir.mkdir(parents=True, exist_ok=True)
    grid_path = condition_dir / "grid_runs.csv"
    frame.to_csv(grid_path, index=False)
    popularity_path = _write_popularity_validation(
        config_path=config_path,
        modeling_path=modeling_path,
        dataset_name=dataset_name,
        retention_label=retention_label,
        condition_dir=condition_dir,
    )
    selection = select_configurations(frame)
    payload = {
        "schema_version": 1,
        "stage": "baseline_development_validation_only",
        "dataset": dataset_name,
        "retention_label": retention_label,
        "expected_bias_runs": total,
        "observed_bias_runs": int(len(frame)),
        "popularity_summary": str(popularity_path.relative_to(root)),
        "selection": selection,
    }
    (condition_dir / "selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return grid_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--modeling-config", type=Path, default=DEFAULT_BASELINE_MODELING_PATH
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--retention", required=True, choices=RETENTIONS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        execute_baseline_grid(
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
