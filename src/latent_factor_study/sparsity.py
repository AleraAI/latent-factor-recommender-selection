"""Controlled train-only sparsification with deterministic repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data import DEFAULT_CONFIG_PATH, _repo_root, _resolve, sha256_file


DEFAULT_RETENTIONS = (1.00, 0.75, 0.50, 0.25)


def _priority(seed: int, retention: float, interaction_id: str) -> str:
    payload = f"{seed}\0{retention:.8f}\0{interaction_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def thin_training_interactions(
    full_train: pd.DataFrame,
    *,
    retention: float,
    seed: int,
    minimum_per_user: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Thin training histories and restore representation of evaluable items.

    The initial sample retains ``max(minimum_per_user, round-half-up(q*n))``
    rows for every user, ordered by a cryptographic seed-derived priority.
    A repair pass then restores one deterministic training row for each item in
    the full-training catalog that disappeared from the sample. The repair is
    therefore independent of validation and test identities.
    """
    if not 0.0 < retention <= 1.0:
        raise ValueError("retention must be in (0, 1].")
    required = {"interaction_id", "user_id", "item_id"}
    missing = sorted(required - set(full_train.columns))
    if missing:
        raise ValueError(f"full_train is missing columns: {missing}")
    if full_train["interaction_id"].duplicated().any():
        raise ValueError("full_train contains duplicate interaction identifiers.")

    train = full_train.copy()
    train["_priority"] = train["interaction_id"].map(
        lambda value: _priority(seed, retention, str(value))
    )
    selected_indices: set[int] = set()
    requested_by_user: dict[str, int] = {}
    for user_id, rows in train.groupby("user_id", sort=False):
        n_rows = len(rows)
        if n_rows < minimum_per_user:
            raise ValueError(
                f"User {user_id!r} has {n_rows} training rows, below minimum {minimum_per_user}."
            )
        # round-half-up, avoiding Python's bankers' rounding.
        target = max(minimum_per_user, int(retention * n_rows + 0.5))
        target = min(n_rows, target)
        requested_by_user[str(user_id)] = target
        chosen = rows.sort_values(["_priority", "interaction_id"], kind="stable").head(target)
        selected_indices.update(int(index) for index in chosen.index)

    selected = train.loc[sorted(selected_indices)].copy()
    selected_item_ids = set(selected["item_id"].astype(str))
    full_train_item_ids = set(train["item_id"].astype(str))
    missing_items = sorted(full_train_item_ids - selected_item_ids)

    repaired_interaction_ids: list[str] = []
    for item_id in missing_items:
        candidate = (
            train.loc[train["item_id"].astype(str) == item_id]
            .sort_values(["_priority", "interaction_id"], kind="stable")
            .iloc[0]
        )
        selected_indices.add(int(candidate.name))
        repaired_interaction_ids.append(str(candidate["interaction_id"]))

    result = (
        train.loc[sorted(selected_indices)]
        .drop(columns="_priority")
        .sort_values(["user_id", "timestamp", "source_row_id", "item_id"], kind="stable")
        .reset_index(drop=True)
    )
    retained_ids = set(result["interaction_id"].astype(str))
    if not retained_ids.issubset(set(full_train["interaction_id"].astype(str))):
        raise AssertionError("Sparsification created interactions not present in full training.")
    if not full_train_item_ids.issubset(set(result["item_id"].astype(str))):
        raise AssertionError("Repair failed to preserve the full-training catalog.")
    min_degree = int(result.groupby("user_id").size().min())
    if min_degree < minimum_per_user:
        raise AssertionError("Sparsification violated the minimum user history.")

    requested_rows = sum(requested_by_user.values())
    audit = {
        "retention_nominal": float(retention),
        "seed": int(seed),
        "minimum_per_user": int(minimum_per_user),
        "full_train_rows": int(len(full_train)),
        "initial_requested_rows": int(requested_rows),
        "repair_rows_added": int(len(result) - requested_rows),
        "realized_rows": int(len(result)),
        "realized_retention": float(len(result) / len(full_train)),
        "users": int(result["user_id"].nunique()),
        "items": int(result["item_id"].nunique()),
        "minimum_realized_user_degree": min_degree,
        "full_training_catalog_items": int(len(full_train_item_ids)),
        "repaired_items": int(len(missing_items)),
        "repaired_interaction_ids": repaired_interaction_ids,
        "selection_rule": (
            "per user, retain max(minimum, floor(retention*n+0.5)) rows by "
            "SHA-256(seed, retention, interaction_id), then add one row for each "
            "otherwise-lost item in the full-training catalog; held-out identities "
            "are not used by the sampler"
        ),
    }
    return result, audit


def _label(retention: float) -> str:
    return f"r{int(round(retention * 100)):03d}"


def build_sparsity_conditions(
    *,
    dataset_name: str,
    processed_dir: Path,
    output_dir: Path,
    retentions: Sequence[float],
    seed: int,
    minimum_per_user: int,
) -> dict[str, Any]:
    full_train = pd.read_parquet(processed_dir / "train.parquet")
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions: dict[str, Any] = {}
    for retention in retentions:
        label = _label(retention)
        thinned, audit = thin_training_interactions(
            full_train,
            retention=retention,
            seed=seed,
            minimum_per_user=minimum_per_user,
        )
        path = output_dir / f"train_{label}.parquet"
        thinned.to_parquet(path, index=False)
        conditions[label] = {
            **audit,
            "artifact": {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            },
        }
    manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "conditions": conditions,
    }
    manifest_path = output_dir / "sparsity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_from_config(config_path: Path, dataset_names: list[str]) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    datasets: Mapping[str, Mapping[str, Any]] = config["datasets"]
    selected = list(datasets) if dataset_names == ["all"] else dataset_names
    experiment = config["controlled_sparsity"]
    output: dict[str, Any] = {}
    for name in selected:
        if name not in datasets:
            raise ValueError(f"Unknown dataset: {name}")
        processed_dir = _resolve(root, datasets[name]["output_dir"])
        output_dir = _resolve(root, experiment["output_root"]) / name
        output[name] = build_sparsity_conditions(
            dataset_name=name,
            processed_dir=processed_dir,
            output_dir=output_dir,
            retentions=[float(value) for value in experiment["retentions"]],
            seed=int(experiment["seed"]),
            minimum_per_user=int(experiment["minimum_per_user"]),
        )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=[
            "all",
            "movielens_1m",
            "ciaodvd",
            "yelp_tampa_restaurants",
            "epinions",
        ],
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifests = build_from_config(args.config, args.datasets or ["all"])
    summary = {
        dataset: {
            label: {
                key: values[key]
                for key in ("realized_rows", "realized_retention", "items", "repaired_items")
            }
            for label, values in manifest["conditions"].items()
        }
        for dataset, manifest in manifests.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
