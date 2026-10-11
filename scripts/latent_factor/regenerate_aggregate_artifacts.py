"""Regenerate revision tables and Pareto figures from included aggregates only.

No data download, fitting, per-user resampling, checkpoint loading, or test
scoring occurs. Outputs are isolated from included evidence and manuscripts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REVISION = Path("output/latent_factor_major_revision_2026-09-14")
COVERAGE = Path("output/latent_factor_coverage_revision_2026-09-19")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("regenerated"))
    args = parser.parse_args()
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("Choose a fresh output directory to preserve previous regenerated artifacts")
    if ROOT not in destination.parents:
        raise ValueError("Place regenerated artifacts beneath this package root")
    inputs = {
        "selector_analysis": ["validation_selections.csv", "selection_outcomes.csv", "paired_test_contrasts.csv", "validation_landscapes.csv"],
        "protocol_audit": ["configuration_registry.csv"],
        "search_boundary": ["matched_budget_summary.csv"],
    }
    snapshots = {}
    evidence = destination / "evidence"
    for folder, names in inputs.items():
        for name in names:
            original = ROOT / REVISION / folder / name
            snapshots[original] = sha(original)
            copied = evidence / folder / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copied)
    environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(destination / "matplotlib-cache"),
                       XDG_CACHE_HOME=str(destination / "cache"),
                       PYTHONDONTWRITEBYTECODE="1")
    commands = []
    tables = destination / "tables"
    figures = destination / "figures"
    reports = destination / "reports"
    for script, report_name in [("revision_report_tables.py", "review_r1"),
                                ("revision_pareto_plots.py", "pareto_plots")]:
        command = [sys.executable, str(ROOT / "scripts/latent_factor" / script),
                   "--revision-dir", str(evidence), "--output-dir", str(reports / report_name),
                   "--tables-dir", str(tables / "review_r1")]
        if script == "revision_pareto_plots.py":
            command.extend(["--figures-dir", str(figures / "review_r1")])
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        commands.append(["python", f"scripts/latent_factor/{script}", "--revision-dir",
                         str(evidence.relative_to(ROOT)), "--output-dir", str((reports / report_name).relative_to(ROOT)),
                         "--tables-dir", str((tables / "review_r1").relative_to(ROOT))])
        if script == "revision_pareto_plots.py":
            commands[-1].extend(["--figures-dir", str((figures / "review_r1").relative_to(ROOT))])
    coverage_output = destination / "coverage"
    coverage_script = "scripts/latent_factor/coverage_report_tables.py"
    subprocess.run(
        [sys.executable, str(ROOT / coverage_script), "--revision-dir", str(evidence),
         "--output-dir", str(coverage_output), "--tables-dir", str(tables / "coverage_review")],
        cwd=ROOT, env=environment, check=True,
    )
    commands.append(["python", coverage_script, "--revision-dir", str(evidence.relative_to(ROOT)),
                     "--output-dir", str(coverage_output.relative_to(ROOT)),
                     "--tables-dir", str((tables / "coverage_review").relative_to(ROOT))])
    checked_csvs = []
    for published in sorted((ROOT / COVERAGE / "report_tables").glob("*.csv")):
        regenerated = coverage_output / "report_tables" / published.name
        if regenerated.read_bytes() != published.read_bytes():
            raise AssertionError(f"Coverage CSV differs from included evidence: {published.name}")
        checked_csvs.append(published.name)
    if len(checked_csvs) != 4:
        raise AssertionError("Expected four complete coverage CSV reports")
    checked_tex = []
    for published in sorted((ROOT / "tests/fixtures/tables/coverage_review").glob("*.tex")):
        regenerated = tables / "coverage_review" / published.name
        if regenerated.read_bytes() != published.read_bytes():
            raise AssertionError(f"Coverage TeX differs from included evidence: {published.name}")
        checked_tex.append(published.name)
    if len(checked_tex) != 6:
        raise AssertionError("Expected six complete coverage TeX tables")
    if any(sha(path) != digest for path, digest in snapshots.items()):
        raise AssertionError("Included source aggregates changed during regeneration")
    results = dict(status="passed", scope="table and figure regeneration from fixed aggregates; no fitting or statistical resampling",
                   commands=commands, input_sha256={str(p.relative_to(ROOT)): digest for p, digest in snapshots.items()},
                   coverage_csvs_byte_identical=checked_csvs, coverage_tex_byte_identical=checked_tex,
                   output_files=[str(p.relative_to(ROOT)) for folder in (tables, figures, reports, coverage_output)
                                 for p in sorted(folder.rglob("*")) if p.is_file()])
    (destination / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Regeneration completed: {len(results['output_files'])} table/figure files; included aggregates unchanged.")


if __name__ == "__main__":
    main()
