#!/usr/bin/env python3
"""Capture or verify hashes for every data file used by the manuscript charts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SUMMARY_INPUTS = (
    "outputs/latent_factor_beyond_rmse/confirmation/analysis/mf_selection_regret.csv",
    "outputs/latent_factor_beyond_rmse/baseline_confirmation/analysis/baseline_comparison.csv",
    "outputs/latent_factor_beyond_rmse/bpr_confirmation/analysis/bpr_boundary_comparison.csv",
    "outputs/latent_factor_beyond_rmse/framework_construction/analysis/construction_framework_results.csv",
    "outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/analysis/prospective_external_validation.csv",
)
DATASETS = ("movielens_1m", "ciaodvd", "yelp_tampa_restaurants")
RETENTIONS = ("r100", "r075", "r050", "r025")


def source_paths(root: Path) -> list[Path]:
    paths = [root / relative for relative in SUMMARY_INPUTS]
    paths.extend(
        root
        / "outputs/latent_factor_beyond_rmse/development"
        / dataset
        / retention
        / "grid_runs.csv"
        for dataset in DATASETS
        for retention in RETENTIONS
    )
    paths.extend(
        root
        / "outputs/latent_factor_beyond_rmse/prospective_epinions/development/epinions"
        / retention
        / filename
        for retention in RETENTIONS
        for filename in ("grid_runs.csv", "framework_selection.json")
    )
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in source_paths(root):
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(root).as_posix()
        result[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return result


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    current = snapshot(root)
    timestamp = datetime.now(timezone.utc).isoformat()

    if args.check_report:
        report = json.loads(args.check_report.read_text(encoding="utf-8"))
        recorded = report.get("data_sources", {})
        changed = sorted(
            path
            for path in set(recorded) | set(current)
            if recorded.get(path) != current.get(path)
        )
        if changed:
            raise SystemExit(f"Chart data integrity failure: {changed}")
        print(f"Verified {len(current)} chart data sources.")
        return

    if args.capture:
        payload: dict[str, object] = {
            "captured_at_utc": timestamp,
            "data_sources": current,
            "status": "baseline_captured",
        }
        write_json(args.capture, payload)
        print(args.capture)
        return

    if not args.verify or not args.output:
        parser.error("use --capture, or provide both --verify and --output")
    baseline = json.loads(args.verify.read_text(encoding="utf-8"))["data_sources"]
    changed = sorted(
        path for path in set(baseline) | set(current) if baseline.get(path) != current.get(path)
    )
    payload = {
        "verified_at_utc": timestamp,
        "baseline_manifest": str(args.verify),
        "data_sources": current,
        "changed_sources": changed,
        "source_count": len(current),
        "status": "data_sources_unchanged" if not changed else "data_sources_changed",
    }
    write_json(args.output, payload)
    print(args.output)
    if changed:
        raise SystemExit(f"Chart data integrity failure: {changed}")


if __name__ == "__main__":
    main()
