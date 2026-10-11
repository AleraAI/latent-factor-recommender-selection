"""Bounded revision-only missing-run completion, gated by historical replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .explicit_mf import MFRunConfig, fit_explicit_mf, evaluate_stage, predict_frame
from .major_revision_selection import DEFAULT_OUTPUT, sha, write_json, grid_path, test_records


def environment():
    return {"python": platform.python_version(), "executable": sys.executable,
            "torch": torch.__version__, "pandas": pd.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "torch_num_threads": 1, "device": "cpu"}


def data_paths(root, dataset, retention):
    config = root / "configs/latent_factor_beyond_rmse" / (
        "epinions_prospective_datasets.yaml" if dataset == "epinions" else "datasets.yaml")
    values = yaml.safe_load(config.read_text())
    processed = root / values["datasets"][dataset]["output_dir"]
    return {"sparse_train": root / values["controlled_sparsity"]["output_root"] / dataset / f"train_{retention}.parquet",
            "full_train": processed / "train.parquet", "validation": processed / "validation.parquet",
            "test": processed / "test.parquet", "dataset_config": config}


def run_config(dimension, regularization, seed):
    return MFRunConfig(dimension=int(dimension), regularization=float(regularization), seed=int(seed),
                       learning_rate=.01, batch_size=4096, max_epochs=20, convergence_patience=5,
                       convergence_relative_tolerance=1e-5, device="cpu", torch_num_threads=1)


def same_metrics(current, historical, label):
    differences = {}
    for kind in ("clipped", "raw"):
        for metric in ("rmse", "mae"):
            differences[f"prediction.{kind}.{metric}"] = current["prediction"][kind][metric] - historical["prediction"][kind][metric]
    for metric in ("ndcg@10", "ndcg@20", "recall@10", "recall@20"):
        differences["ranking." + metric] = current["ranking"][metric] - historical["ranking"][metric]
    if max(abs(v) for v in differences.values()) > 1e-12:
        raise ValueError(f"Historical replay scalar mismatch for {label}: {differences}")
    return differences


def gate(root, output):
    path = output / "fits/runtime_replay_gate.json"
    runtime = environment()
    expected = {"python": "3.10.9", "torch": "2.1.0.dev20230521", "pandas": "1.5.3", "numpy": "1.21.5"}
    if any(runtime[k] != v for k, v in expected.items()):
        raise ValueError(f"Original runtime not active: {runtime}")
    source = {str(p.relative_to(root)): sha(p) for p in
              [root / "src/latent_factor_study" / name for name in ("explicit_mf.py", "evaluation.py", "data.py")]}
    if path.exists():
        result = json.loads(path.read_text())
        if result["source_sha256"] != source or result["runtime"] != runtime or result["status"] != "passed":
            raise ValueError("Existing replay gate no longer matches source/runtime")
        for name, digest in result["data_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("Replay data changed")
        return result
    paths = data_paths(root, "epinions", "r025")
    frames = {key: pd.read_parquet(value) for key, value in paths.items() if key != "dataset_config"}
    records = []
    for seed, stage in [(101, "validation"), (601, "test")]:
        cfg = run_config(64, .1, seed)
        started = time.monotonic()
        fitted = fit_explicit_mf(frames["sparse_train"], cfg)
        metrics, users = evaluate_stage(fitted, full_train=frames["full_train"], evaluation=frames[stage],
                                       prior_evaluation=frames["validation"] if stage == "test" else None,
                                       ranking_batch_size=256)
        if stage == "validation":
            historical = grid_path(root, "epinions", "r025").parent / "d64_lambda0.1_seed101/summary.json"
        else:
            historical = root / test_records(root, "epinions", "r025")[("d64_lambda0.1", 601)]["summary_path"]
        previous = json.loads(historical.read_text())
        differences = same_metrics(metrics, previous[stage], f"{stage}/seed{seed}")
        old_users = pd.read_parquet(historical.parent / f"{stage}_per_user.parquet").sort_values("user_id").reset_index(drop=True)
        current_users = users.sort_values("user_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(current_users, old_users, check_exact=False, atol=1e-12, rtol=0)
        if len(fitted.epoch_history) != previous["training"]["epochs_completed"]:
            raise ValueError("Historical stopping epoch mismatch")
        dest = output / "fits/runtime_replay" / f"{stage}_seed{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        users.to_parquet(dest / f"{stage}_per_user.parquet", index=False)
        torch.save({"state_dict": fitted.model.state_dict(), "user_ids": fitted.user_ids,
                    "item_ids": fitted.item_ids, "run_config": asdict(cfg)}, dest / "checkpoint.pt")
        write_json(dest / "summary.json", {"stage": "revision_runtime_parity_replay", "run_config": asdict(cfg),
                   stage: metrics, "training": {"epochs_completed": len(fitted.epoch_history), "seconds": fitted.training_seconds},
                   "historical_path": str(historical.relative_to(root)), "historical_sha256": sha(historical),
                   "scalar_differences": differences, "per_user_parity": True,
                   "elapsed_seconds": time.monotonic() - started})
        records.append({"stage": stage, "seed": seed, "scalar_differences": differences, "per_user_parity": True})
    result = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
              "source_sha256": source, "data_sha256": {str(p.relative_to(root)): sha(p) for p in paths.values()},
              "records": records, "claim_scope": "two fixed-data exact-metric/per-user replay checks; not proof of every historical fit"}
    write_json(path, result)
    return result


def complete(root, output):
    verified = gate(root, output)
    queue_path = output / "missing_runs.csv"
    plan_path = output / "fits/fitting_plan.json"
    if not plan_path.exists():
        queue = pd.read_csv(queue_path)
        if len(queue) != 90:
            raise ValueError("This bounded completion is authorized for the initial 90 runs only")
        write_json(plan_path, {"stage": "revision_stage_missing_selected_configurations", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "queue_sha256": sha(queue_path), "selection_freeze_sha256": sha(output / "selection_freeze.json"),
                   "runtime_gate_sha256": sha(output / "fits/runtime_replay_gate.json"),
                   "runs": queue.to_dict("records"), "runtime": verified["runtime"],
                   "runner_sha256": sha(Path(__file__)), "tuning_permitted": False})
    plan = json.loads(plan_path.read_text())
    if plan["selection_freeze_sha256"] != sha(output / "selection_freeze.json"):
        raise ValueError("Frozen selections changed")
    records, cache = [], {}
    started = time.monotonic()
    for index, row in enumerate(plan["runs"]):
        dataset, retention = row["dataset"], row["retention"]
        cfg = run_config(row["dimension"], row["regularization"], row["seed"])
        dest = output / "fits/runs" / dataset / retention / f"{row['configuration_id']}_seed{cfg.seed}"
        summary_path = dest / "summary.json"
        if not summary_path.exists():
            paths = data_paths(root, dataset, retention)
            for name, path in paths.items():
                if name != "dataset_config" and str(path) not in cache:
                    cache[str(path)] = pd.read_parquet(path)
            frames = {key: cache[str(path)] for key, path in paths.items() if key != "dataset_config"}
            run_start = time.monotonic()
            fitted = fit_explicit_mf(frames["sparse_train"], cfg)
            test, per_user = evaluate_stage(fitted, full_train=frames["full_train"], evaluation=frames["test"],
                                            prior_evaluation=frames["validation"], ranking_batch_size=256)
            validation, validation_users = evaluate_stage(fitted, full_train=frames["full_train"],
                                                          evaluation=frames["validation"], ranking_batch_size=256)
            dest.mkdir(parents=True, exist_ok=True)
            per_user.to_parquet(dest / "test_per_user.parquet", index=False)
            validation_users.to_parquet(dest / "validation_per_user.parquet", index=False)
            torch.save({"state_dict": fitted.model.state_dict(), "user_ids": fitted.user_ids,
                        "item_ids": fitted.item_ids, "run_config": asdict(cfg)}, dest / "checkpoint.pt")
            for stage in ("validation", "test"):
                predictions = frames[stage][["user_id", "item_id", "rating"]].copy()
                predictions["raw_prediction"] = predict_frame(fitted, frames[stage])
                predictions.to_parquet(dest / f"{stage}_event_predictions.parquet", index=False)
            summary = {"stage": "revision_stage_missing_selected_configuration", "dataset": dataset, "retention_label": retention,
                       "run_config": asdict(cfg), "test": test, "validation": validation,
                       "training": {"seconds": fitted.training_seconds, "epochs_completed": len(fitted.epoch_history),
                                    "epoch_history": fitted.epoch_history, "rows": len(frames["sparse_train"]),
                                    "users": len(fitted.user_ids), "items": len(fitted.item_ids)},
                       "runtime": verified["runtime"], "source_sha256": verified["source_sha256"],
                       "data_sha256": {str(path.relative_to(root)): sha(path) for path in paths.values()},
                       "fitting_plan_sha256": sha(plan_path), "selection_freeze_sha256": sha(output / "selection_freeze.json"),
                       "checkpoint_sha256": sha(dest / "checkpoint.pt"), "elapsed_seconds": time.monotonic() - run_start}
            write_json(summary_path, summary)
        existing = json.loads(summary_path.read_text())
        if existing["run_config"] != asdict(cfg) or existing["fitting_plan_sha256"] != sha(plan_path):
            raise ValueError("Existing revision run has incompatible provenance")
        record = {"dimension": cfg.dimension, "regularization": cfg.regularization, "seed": cfg.seed,
                  "dataset": dataset, "retention": retention, "selection_roles": ["revision_selected"],
                  "summary_path": str(summary_path.relative_to(root)), "summary_sha256": sha(summary_path),
                  "per_user_path": str((dest / "test_per_user.parquet").relative_to(root)),
                  "per_user_sha256": sha(dest / "test_per_user.parquet")}
        records.append(record)
        write_json(output / "fits/confirmation_manifest.json", {"stage": "revision_stage_sensitivity", "runs": records,
                   "expected_runs": len(plan["runs"]), "observed_runs": len(records), "runtime_replay_passed": True})
        print(f"Revision fits: {index + 1}/{len(plan['runs'])}: {dataset}/{retention}/{row['configuration_id']}/seed{cfg.seed}", flush=True)
    write_json(output / "fits/completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
               "runs": len(records), "execution_elapsed_seconds": time.monotonic() - started,
               "summed_run_elapsed_seconds": sum(json.loads((root / r["summary_path"]).read_text())["elapsed_seconds"] for r in records),
               "fitting_plan_sha256": sha(plan_path), "historical_artifacts_modified": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["gate", "complete"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    (gate if args.action == "gate" else complete)(root, output)


if __name__ == "__main__":
    main()
