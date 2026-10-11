"""Dataset preparation for the prediction--ranking model-selection study.

This module is deliberately independent of the precursor IEEE experiment.  It
loads the journal-study sources, keeps the complete 1--5 rating signal, and
writes immutable interaction and split identifiers plus a manifest.  Epinions
is supported through a separately versioned prospective-validation config so
the historical three-dataset locks remain byte-for-byte verifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CANONICAL_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "source_row_id",
]
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_CONFIG_PATH = Path("configs/latent_factor_beyond_rmse/datasets.yaml")


def _repo_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if parent.name == "configs":
            return parent.parent
    return resolved.parent


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _interaction_id(dataset_name: str, row: pd.Series) -> str:
    identity = "\0".join(
        [
            dataset_name,
            str(row["user_id"]),
            str(row["item_id"]),
            str(int(row["timestamp"])),
            str(int(row["source_row_id"])),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def canonicalize_interactions(
    interactions: pd.DataFrame,
    *,
    dataset_name: str,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate, deterministically deduplicate, and identify rating tuples."""
    missing = sorted(set(CANONICAL_COLUMNS) - set(interactions.columns))
    if missing:
        raise ValueError(f"Interactions are missing required columns: {missing}")

    frame = interactions[CANONICAL_COLUMNS].copy()
    input_rows = len(frame)
    if frame[CANONICAL_COLUMNS].isna().any().any():
        counts = frame[CANONICAL_COLUMNS].isna().sum()
        raise ValueError(f"Canonical columns contain nulls: {counts[counts > 0].to_dict()}")

    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["rating"] = pd.to_numeric(frame["rating"], errors="raise").astype("float64")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype("int64")
    frame["source_row_id"] = pd.to_numeric(
        frame["source_row_id"], errors="raise"
    ).astype("int64")

    invalid = ~frame["rating"].between(rating_min, rating_max, inclusive="both")
    if invalid.any():
        sample = frame.loc[invalid, ["user_id", "item_id", "rating"]].head(5)
        raise ValueError(f"Ratings outside [{rating_min}, {rating_max}]: {sample.to_dict('records')}")

    frame = frame.sort_values(
        ["user_id", "item_id", "timestamp", "source_row_id"], kind="stable"
    )
    frame = frame.drop_duplicates(["user_id", "item_id"], keep="last")
    frame = frame.sort_values(
        ["user_id", "timestamp", "source_row_id", "item_id"], kind="stable"
    ).reset_index(drop=True)
    frame.insert(
        0,
        "interaction_id",
        frame.apply(lambda row: _interaction_id(dataset_name, row), axis=1),
    )
    if not frame["interaction_id"].is_unique:
        raise AssertionError("Generated interaction identifiers are not unique.")
    return frame, {
        "input_rows": int(input_rows),
        "deduplicated_rows": int(len(frame)),
        "duplicate_rows_removed": int(input_rows - len(frame)),
    }


def iterative_k_core(
    interactions: pd.DataFrame,
    *,
    minimum_user_degree: int,
    minimum_item_degree: int,
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    """Apply simultaneous user/item degree filtering until convergence."""
    if minimum_user_degree < 1 or minimum_item_degree < 1:
        raise ValueError("Core degree thresholds must be positive.")

    frame = interactions.copy()
    audit: list[dict[str, int]] = []
    iteration = 0
    while True:
        iteration += 1
        rows_before = len(frame)
        user_degree = frame.groupby("user_id", sort=False)["item_id"].transform("size")
        item_degree = frame.groupby("item_id", sort=False)["user_id"].transform("size")
        frame = frame[
            (user_degree >= minimum_user_degree)
            & (item_degree >= minimum_item_degree)
        ].copy()
        audit.append(
            {
                "iteration": iteration,
                "rows_before": int(rows_before),
                "rows_after": int(len(frame)),
                "users_after": int(frame["user_id"].nunique()),
                "items_after": int(frame["item_id"].nunique()),
            }
        )
        if len(frame) == rows_before:
            break
        if frame.empty:
            raise ValueError(
                "Iterative core filtering removed every interaction; lower the thresholds."
            )

    user_min = int(frame.groupby("user_id").size().min())
    item_min = int(frame.groupby("item_id").size().min())
    if user_min < minimum_user_degree or item_min < minimum_item_degree:
        raise AssertionError("Core filtering did not reach the requested fixed point.")
    return frame.reset_index(drop=True), audit


def chronological_user_split(
    interactions: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> pd.DataFrame:
    """Assign deterministic within-user chronological 70/15/15-style splits.

    Boundaries are floor(0.70*n) and floor(0.85*n).  Consequently, the test
    window absorbs indivisible remainders.  Equal timestamps are ordered by the
    stable source-row identifier and then item identifier.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1).")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be below 1.")

    parts: list[pd.DataFrame] = []
    ordered = interactions.sort_values(
        ["user_id", "timestamp", "source_row_id", "item_id"], kind="stable"
    )
    for _, rows in ordered.groupby("user_id", sort=False):
        n_rows = len(rows)
        train_end = math.floor(train_ratio * n_rows)
        validation_end = math.floor((train_ratio + validation_ratio) * n_rows)
        if train_end < 3 or validation_end <= train_end or validation_end >= n_rows:
            raise ValueError(
                "A user history cannot satisfy train>=3, validation>=1, test>=1; "
                f"history length was {n_rows}."
            )
        assigned = rows.copy()
        assigned["user_position"] = range(n_rows)
        assigned["split"] = "test"
        split_col = assigned.columns.get_loc("split")
        assigned.iloc[:train_end, split_col] = "train"
        assigned.iloc[train_end:validation_end, split_col] = "validation"
        parts.append(assigned)

    result = pd.concat(parts, ignore_index=True)
    validate_split(result)
    return result


def validate_split(split_interactions: pd.DataFrame) -> None:
    """Fail on overlap, missing rows, or temporal leakage."""
    if split_interactions["interaction_id"].duplicated().any():
        raise ValueError("An interaction appears in more than one split.")
    observed_splits = set(split_interactions["split"].unique())
    if observed_splits != set(SPLIT_NAMES):
        raise ValueError(f"Unexpected split labels: {sorted(observed_splits)}")

    for user_id, rows in split_interactions.groupby("user_id", sort=False):
        times = {
            name: rows.loc[rows["split"] == name, "timestamp"] for name in SPLIT_NAMES
        }
        if any(values.empty for values in times.values()):
            raise ValueError(f"User {user_id!r} is absent from at least one split.")
        if times["train"].max() > times["validation"].min():
            raise ValueError(f"Training/validation temporal leakage for user {user_id!r}.")
        if times["validation"].max() > times["test"].min():
            raise ValueError(f"Validation/test temporal leakage for user {user_id!r}.")


def _parse_utc(value: str) -> int:
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def load_movielens_1m(archive_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("ml-1m/ratings.dat") as handle:
            for source_row_id, raw_line in enumerate(handle, start=1):
                user_id, item_id, rating, timestamp = raw_line.decode("latin-1").strip().split("::")
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rating": float(rating),
                        "timestamp": int(timestamp),
                        "source_row_id": source_row_id,
                    }
                )
    return pd.DataFrame.from_records(rows, columns=CANONICAL_COLUMNS)


def load_ciaodvd(archive_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("movie-ratings.txt") as handle:
            for source_row_id, raw_line in enumerate(handle, start=1):
                user_id, item_id, _genre_id, _review_id, rating, date = (
                    raw_line.decode("utf-8").strip().split(",")
                )
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rating": float(rating),
                        "timestamp": _parse_utc(date),
                        "source_row_id": source_row_id,
                    }
                )
    return pd.DataFrame.from_records(rows, columns=CANONICAL_COLUMNS)


def load_epinions_timestamped(archive_path: Path) -> pd.DataFrame:
    """Load the official six-column timestamped Epinions text archive.

    The author-supplied README defines the columns as user, product, category,
    rating, helpfulness, and Unix timestamp.  Category and helpfulness are not
    inputs to this explicit-feedback study.
    """
    member = "epinion_with_rating_timestamp_txt/rating_with_timestamp.txt"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"Epinions archive is missing {member!r}.")
        with archive.open(member) as handle:
            for source_row_id, raw_line in enumerate(handle, start=1):
                fields = raw_line.decode("ascii").split()
                if not fields:
                    continue
                if len(fields) != 6:
                    raise ValueError(
                        f"Epinions row {source_row_id} has {len(fields)} fields; expected 6."
                    )
                user_id, item_id, _category_id, rating, _helpfulness, timestamp = (
                    float(value) for value in fields
                )
                rows.append(
                    {
                        "user_id": str(int(user_id)),
                        "item_id": str(int(item_id)),
                        "rating": rating,
                        "timestamp": int(timestamp),
                        "source_row_id": source_row_id,
                    }
                )
    return pd.DataFrame.from_records(rows, columns=CANONICAL_COLUMNS)


def _iter_json_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for source_row_id, line in enumerate(handle, start=1):
            if line.strip():
                yield source_row_id, json.loads(line)


def _category_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    return {token.strip() for token in str(value).split(",") if token.strip()}


def load_yelp_scope(
    *,
    business_path: Path,
    review_path: Path,
    city: str,
    state: str,
    category: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    item_ids: set[str] = set()
    raw_business_rows = 0
    for _, row in _iter_json_lines(business_path):
        raw_business_rows += 1
        if (
            str(row.get("city", "")).strip() == city
            and str(row.get("state", "")).strip() == state
            and category in _category_tokens(row.get("categories"))
        ):
            item_id = str(row.get("business_id", "")).strip()
            if item_id:
                item_ids.add(item_id)
    if not item_ids:
        raise ValueError("The Yelp business scope selected no items.")

    rows: list[dict[str, Any]] = []
    raw_review_rows = 0
    for source_row_id, row in _iter_json_lines(review_path):
        raw_review_rows += 1
        item_id = str(row.get("business_id", "")).strip()
        if item_id not in item_ids:
            continue
        user_id = str(row.get("user_id", "")).strip()
        if not user_id:
            continue
        rows.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "rating": float(row["stars"]),
                "timestamp": _parse_utc(str(row["date"])),
                "source_row_id": source_row_id,
            }
        )
    return pd.DataFrame.from_records(rows, columns=CANONICAL_COLUMNS), {
        "raw_business_rows": raw_business_rows,
        "scoped_businesses": len(item_ids),
        "raw_review_rows": raw_review_rows,
        "scoped_review_rows": len(rows),
    }


def _frame_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    user_degree = frame.groupby("user_id").size()
    item_degree = frame.groupby("item_id").size()
    quantiles = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]

    def degree_summary(values: pd.Series) -> dict[str, float]:
        return {
            f"q{int(q * 100):02d}": float(values.quantile(q)) for q in quantiles
        }

    users = int(frame["user_id"].nunique())
    items = int(frame["item_id"].nunique())
    return {
        "rows": int(len(frame)),
        "users": users,
        "items": items,
        "density": float(len(frame) / (users * items)),
        "rating_histogram": {
            str(key): int(value)
            for key, value in sorted(Counter(frame["rating"].astype(int)).items())
        },
        "timestamp_min": int(frame["timestamp"].min()),
        "timestamp_max": int(frame["timestamp"].max()),
        "user_degree_quantiles": degree_summary(user_degree),
        "item_degree_quantiles": degree_summary(item_degree),
    }


def _split_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    train_items = set(frame.loc[frame["split"] == "train", "item_id"])
    stats: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_frame = frame[frame["split"] == split_name]
        stats[split_name] = {
            "rows": int(len(split_frame)),
            "users": int(split_frame["user_id"].nunique()),
            "items": int(split_frame["item_id"].nunique()),
            "relevant_rows_rating_ge_4": int((split_frame["rating"] >= 4.0).sum()),
            "users_with_relevant_rows": int(
                split_frame.loc[split_frame["rating"] >= 4.0, "user_id"].nunique()
            ),
            "item_cold_rows": int((~split_frame["item_id"].isin(train_items)).sum()),
        }
    return stats


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_raw_file(path: Path, expected_sha256: str | None) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Raw-file checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return actual


def prepare_dataset(
    dataset_name: str,
    dataset_config: Mapping[str, Any],
    *,
    root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare one dataset and write its frozen audit artifacts."""
    kind = str(dataset_config["loader"])
    source_files: dict[str, dict[str, Any]] = {}
    scope_stats: dict[str, int] = {}

    if kind in {"movielens_1m", "ciaodvd", "epinions_timestamped"}:
        archive_path = _resolve(root, dataset_config["archive_path"])
        actual_hash = _verify_raw_file(archive_path, dataset_config.get("sha256"))
        source_files["archive"] = {
            "path": str(archive_path.relative_to(root)),
            "sha256": actual_hash,
            "bytes": archive_path.stat().st_size,
        }
        if kind == "movielens_1m":
            raw = load_movielens_1m(archive_path)
        elif kind == "ciaodvd":
            raw = load_ciaodvd(archive_path)
        else:
            raw = load_epinions_timestamped(archive_path)
    elif kind == "yelp_scope":
        bundle_path = _resolve(root, dataset_config["archive_path"])
        bundle_hash = _verify_raw_file(bundle_path, dataset_config.get("sha256"))
        source_files["archive"] = {
            "path": str(bundle_path.relative_to(root)),
            "sha256": bundle_hash,
            "bytes": bundle_path.stat().st_size,
        }
        business_path = _resolve(root, dataset_config["business_path"])
        review_path = _resolve(root, dataset_config["review_path"])
        for label, path in (("business_json", business_path), ("review_json", review_path)):
            if not path.exists():
                raise FileNotFoundError(path)
            source_files[label] = {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
            }
        raw, scope_stats = load_yelp_scope(
            business_path=business_path,
            review_path=review_path,
            city=str(dataset_config["scope"]["city"]),
            state=str(dataset_config["scope"]["state"]),
            category=str(dataset_config["scope"]["category"]),
        )
    else:
        raise ValueError(f"Unsupported dataset loader: {kind!r}")

    canonical, duplicate_stats = canonicalize_interactions(
        raw,
        dataset_name=dataset_name,
        rating_min=float(protocol["rating_min"]),
        rating_max=float(protocol["rating_max"]),
    )
    core, core_audit = iterative_k_core(
        canonical,
        minimum_user_degree=int(protocol["minimum_user_degree"]),
        minimum_item_degree=int(protocol["minimum_item_degree"]),
    )
    split = chronological_user_split(
        core,
        train_ratio=float(protocol["train_ratio"]),
        validation_ratio=float(protocol["validation_ratio"]),
    )

    output_dir = _resolve(root, dataset_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "interactions": output_dir / "interactions_10core.parquet",
        "split_assignments": output_dir / "split_assignments.parquet",
        "train": output_dir / "train.parquet",
        "validation": output_dir / "validation.parquet",
        "test": output_dir / "test.parquet",
    }
    core.to_parquet(artifacts["interactions"], index=False)
    split[["interaction_id", "split", "user_position"]].to_parquet(
        artifacts["split_assignments"], index=False
    )
    for split_name in SPLIT_NAMES:
        split.loc[split["split"] == split_name].drop(columns="split").to_parquet(
            artifacts[split_name], index=False
        )

    artifact_manifest = {
        label: {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for label, path in artifacts.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset_name,
        "display_name": dataset_config["display_name"],
        "source_url": dataset_config["source_url"],
        "release": dataset_config["release"],
        "license_or_terms": dataset_config["license_or_terms"],
        "source_files": source_files,
        "scope": dataset_config.get("scope"),
        "scope_statistics": scope_stats,
        "protocol": dict(protocol),
        "deduplication_rule": "latest timestamp; source_row_id breaks ties",
        "split_boundary_rule": (
            "within user: train_end=floor(train_ratio*n), "
            "validation_end=floor((train_ratio+validation_ratio)*n); remainder=test"
        ),
        "raw_or_scoped_statistics": _frame_statistics(raw),
        "deduplicated_statistics": _frame_statistics(canonical),
        "duplicate_statistics": duplicate_stats,
        "core_iterations": core_audit,
        "processed_statistics": _frame_statistics(core),
        "split_statistics": _split_statistics(split),
        "artifacts": artifact_manifest,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def prepare_from_config(config_path: Path, dataset_names: list[str]) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _repo_root(config_path)
    available = config.get("datasets", {})
    selected = list(available) if dataset_names == ["all"] else dataset_names
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}; choose from {sorted(available)}")
    return {
        name: prepare_dataset(
            name,
            available[name],
            root=root,
            protocol=config["protocol"],
        )
        for name in selected
    }


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
        help="Dataset to prepare; repeat for several. Defaults to all.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifests = prepare_from_config(args.config, args.datasets or ["all"])
    summaries = {
        name: {
            "rows": manifest["processed_statistics"]["rows"],
            "users": manifest["processed_statistics"]["users"],
            "items": manifest["processed_statistics"]["items"],
            "split_rows": {
                split: values["rows"]
                for split, values in manifest["split_statistics"].items()
            },
        }
        for name, manifest in manifests.items()
    }
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
