"""Guarded resume for the already fixed 90-run revision queue.

Introduced after the original revision fitting process began. This wrapper does
not replace its runner or retrospectively change the timing of its plan.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.latent_factor.revision_fit_data_audit import audit
from src.latent_factor_study.major_revision_selection import DEFAULT_OUTPUT, sha, write_json
from src.latent_factor_study.major_revision_fits import complete, environment


def main():
    output = ROOT / DEFAULT_OUTPUT
    plan_path = output / "fits/fitting_plan.json"
    plan = json.loads(plan_path.read_text())
    runner = ROOT / "src/latent_factor_study/major_revision_fits.py"
    gate_path = output / "fits/runtime_replay_gate.json"
    if sha(runner) != plan["runner_sha256"]:
        raise ValueError("Bounded-fit runner differs from the implementation fixed in the fitting plan")
    if sha(gate_path) != plan["runtime_gate_sha256"] or environment() != plan["runtime"]:
        raise ValueError("Replay gate or runtime changed")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    audit_path = output / "fits" / f"resume_preflight_{timestamp}.json"
    audit(ROOT, audit_path)
    preflight = json.loads(audit_path.read_text())
    write_json(output / "fits" / f"resume_guard_{timestamp}.json", {
        "guard_recorded_utc": datetime.now(timezone.utc).isoformat(),
        "chronology": "Resume guard added after initial fitting began; applies before any resumed fitting only",
        "plan_sha256": sha(plan_path), "preflight_sha256": sha(audit_path),
        "runner_sha256": sha(runner), "wrapper_sha256": sha(Path(__file__)),
        "frozen_current_data_sha256": {row["path"]: row["actual_sha256"] for row in preflight["data"]},
        "remaining_runs_at_guard": preflight["expected_run_count"] - preflight["completed_run_count"],
    })
    if preflight["all_planned_runs_complete"]:
        print("All 90 planned fits are already complete and verified; no artifact regenerated.")
        return
    complete(ROOT, output)
    audit(ROOT, output / "fits" / f"resume_postflight_{timestamp}.json")


if __name__ == "__main__":
    main()
