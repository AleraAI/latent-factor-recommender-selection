"""Independent integrity audit for revision fits; records its actual audit time.

This audit was introduced after fitting began. It does not claim that its checks
predated the runs, and does not change the fitting queue, choices, or data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, destination: Path):
    if destination.exists():
        raise FileExistsError("Use a new audit filename; preserve previous audit chronology.")
    base = root / "output/latent_factor_major_revision_2026-09-14/selector_analysis"
    plan_path = base / "fits/fitting_plan.json"
    gate_path = base / "fits/runtime_replay_gate.json"
    plan, gate = json.loads(plan_path.read_text()), json.loads(gate_path.read_text())
    start = datetime.now(timezone.utc).isoformat()
    mismatches, data, config_checks, manifests = [], {}, {}, {}
    for run in plan["runs"]:
        dataset, retention = run["dataset"], run["retention"]
        prospective = dataset == "epinions"
        config_path = root / "configs/latent_factor_beyond_rmse" / (
            "epinions_prospective_datasets.yaml" if prospective else "datasets.yaml")
        config = yaml.safe_load(config_path.read_text())
        lock_path = config_path.parent / (
            "epinions_prospective_protocol_lock.yaml" if prospective else "selection_lock.yaml")
        expected_config = yaml.safe_load(lock_path.read_text())["locked_input_sha256"]["datasets_yaml"]
        if config_path not in config_checks:
            actual = sha(config_path)
            config_checks[config_path] = dict(path=str(config_path.relative_to(root)), expected_sha256=expected_config,
                                            actual_sha256=actual, matches=actual == expected_config,
                                            reference_lock=str(lock_path.relative_to(root)), reference_lock_sha256=sha(lock_path))
            if actual != expected_config:
                mismatches.append(f"Dataset configuration differs from original lock: {config_path}")
        processed = root / config["datasets"][dataset]["output_dir"]
        split = root / config["controlled_sparsity"]["output_root"] / dataset
        for manifest_path in (processed / "manifest.json", split / "sparsity_manifest.json"):
            if manifest_path not in manifests:
                manifests[manifest_path] = dict(path=str(manifest_path.relative_to(root)), sha256=sha(manifest_path))
        original = json.loads((processed / "manifest.json").read_text())
        retained = json.loads((split / "sparsity_manifest.json").read_text())
        expected = {processed / f"{stage}.parquet": (original["artifacts"][stage]["sha256"], processed / "manifest.json")
                    for stage in ("train", "validation", "test")}
        expected[split / f"train_{retention}.parquet"] = (retained["conditions"][retention]["artifact"]["sha256"], split / "sparsity_manifest.json")
        for path, (digest, manifest_path) in expected.items():
            if path in data:
                continue
            actual = sha(path)
            data[path] = dict(path=str(path.relative_to(root)), expected_sha256=digest, actual_sha256=actual,
                              matches=actual == digest, reference_manifest=str(manifest_path.relative_to(root)))
            if actual != digest:
                mismatches.append(f"Data file differs from historical manifest: {path}")
    source_checks = []
    for relative, digest in gate["source_sha256"].items():
        actual = sha(root / relative)
        source_checks.append(dict(path=relative, replay_gate_sha256=digest, actual_sha256=actual, matches=actual == digest))
        if actual != digest:
            mismatches.append(f"Fit implementation changed after replay gate: {relative}")
    manifest_path = base / "fits/confirmation_manifest.json"
    manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
    current_manifest = json.loads(manifest_bytes) if manifest_bytes is not None else {"runs": []}
    manifest_runs = {record["summary_path"]: record for record in current_manifest["runs"]}
    expected_identities = {(r["dataset"], r["retention"], r["configuration_id"], int(r["seed"])) for r in plan["runs"]}
    observed_identities = set()
    summaries = []
    for path in sorted((base / "fits/runs").glob("*/*/*/summary.json")):
        summary = json.loads(path.read_text())
        errors = []
        if summary["fitting_plan_sha256"] != sha(plan_path):
            errors.append("fitting_plan_sha256")
        if summary["source_sha256"] != gate["source_sha256"]:
            errors.append("source_sha256")
        if summary["runtime"] != gate["runtime"]:
            errors.append("runtime")
        # Path is .../dataset/retention/run/summary.json.
        if summary["dataset"] != path.parents[2].name or summary["retention_label"] != path.parents[1].name:
            errors.append("dataset_or_retention")
        cfg = summary["run_config"]
        identity = (summary["dataset"], summary["retention_label"],
                    f'd{int(cfg["dimension"])}_lambda{float(cfg["regularization"]):g}', int(cfg["seed"]))
        if identity not in expected_identities or identity in observed_identities:
            errors.append("unexpected_or_duplicate_planned_run")
        observed_identities.add(identity)
        for relative, digest in summary["data_sha256"].items():
            inp = root / relative
            expected = data[inp]["expected_sha256"] if inp in data else config_checks[inp]["expected_sha256"] if inp in config_checks else None
            if expected is None or digest != expected:
                errors.append("data_sha256:" + relative)
        if summary["checkpoint_sha256"] != sha(path.parent / "checkpoint.pt"):
            errors.append("checkpoint_sha256")
        relative = str(path.relative_to(root))
        record = manifest_runs.get(relative)
        if record:
            if sha(path) != record["summary_sha256"]:
                errors.append("manifest_summary_sha256")
            if sha(root / record["per_user_path"]) != record["per_user_sha256"]:
                errors.append("manifest_test_parquet_sha256")
        summaries.append(dict(path=relative, sha256=sha(path), matches=not errors,
                              in_manifest_snapshot=record is not None, errors=errors))
        mismatches.extend(f"{relative}: {error}" for error in errors)
    result = dict(
        status="passed" if not mismatches else "failed", audit_started_utc=start,
        audit_completed_utc=datetime.now(timezone.utc).isoformat(),
        chronology="Independent data-integrity audit introduced after revision fitting had begun; not a prefit declaration.",
        fitting_plan_path=str(plan_path.relative_to(root)), fitting_plan_sha256=sha(plan_path),
        plan_recorded_utc=plan["created_utc"], planned_runs=len(plan["runs"]),
        expected_run_count=len(plan["runs"]), completed_run_count=len(summaries),
        confirmation_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None,
        runtime_replay_gate_sha256=sha(gate_path), unique_data_files=len(data),
        completed_summary_files_checked=len(summaries), manifest_snapshot_runs=len(manifest_runs),
        all_planned_runs_complete=(len(summaries) == len(plan["runs"]) == len(manifest_runs)
                                   and observed_identities == expected_identities), mismatches=mismatches,
        data=list(data.values()), dataset_configurations=list(config_checks.values()),
        historical_manifests=list(manifests.values()), source_checks=source_checks, summaries=summaries,
        auditor=dict(path=str(Path(__file__).relative_to(root)), sha256=sha(Path(__file__))),
        scope="File integrity and agreement with saved manifests/locked dataset configurations; not proof that every historical trainer source was cryptographically archived.",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "unique_data_files", "completed_summary_files_checked", "manifest_snapshot_runs", "all_planned_runs_complete", "mismatches")}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve())
