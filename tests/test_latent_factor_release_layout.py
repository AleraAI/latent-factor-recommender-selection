"""Presentation regeneration must not require or recreate private papers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import yaml

from scripts.latent_factor import verify_manuscript_tables
from src.latent_factor_study import plotting, reporting, revision_analysis, second_review_analysis


ROOT = Path(__file__).resolve().parents[1]
REVISION = Path("output/latent_factor_major_revision_2026-09-14")
COVERAGE = Path("output/latent_factor_coverage_revision_2026-09-19")


def test_workflow_cache_paths_are_configured_on_runner(tmp_path: Path) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    for job in workflow["jobs"].values():
        # GitHub evaluates job env before the runner context is available.
        assert all("runner." not in str(value) for value in job.get("env", {}).values())
        step = next(step for step in job["steps"]
                    if step.get("name") == "Configure cache directories")
        runner_temp = tmp_path / "runner temp"
        environment_file = tmp_path / "github_env"
        subprocess.run(
            ["bash", "-e", "-c", step["run"]],
            env=dict(os.environ, RUNNER_TEMP=str(runner_temp),
                     GITHUB_ENV=str(environment_file)),
            check=True,
        )
        assert environment_file.read_text().splitlines() == [
            f"MPLCONFIGDIR={runner_temp}/latent-factor-matplotlib",
            f"XDG_CACHE_HOME={runner_temp}/latent-factor-cache",
        ]


def copy_relative(destination: Path, relative: Path | str) -> None:
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)


def run_script(root: Path, script: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(root / "scripts/latent_factor" / script), *args],
        cwd=root,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                 MPLCONFIGDIR=str(root / ".matplotlib-cache"),
                 XDG_CACHE_HOME=str(root / ".cache")),
        check=True,
        capture_output=True,
        text=True,
    )


def test_manuscript_verification_uses_exact_fixtures_without_papers(tmp_path: Path) -> None:
    paths = {
        Path("scripts/latent_factor/verify_manuscript_tables.py"),
        Path("manifests/manuscript_table_verification.csv"),
    }
    for spec in verify_manuscript_tables.TABLES:
        paths.add(verify_manuscript_tables.TABLE_FIXTURES / spec.tex)
        paths.update(Path(source) for source in spec.sources)
    for path in paths:
        copy_relative(tmp_path, path)

    run_script(tmp_path, "verify_manuscript_tables.py", "--check-report")

    assert not (tmp_path / "papers").exists()
    assert len(list((tmp_path / "tests/fixtures/tables").glob("*.tex"))) == 32


def test_aggregate_regeneration_without_papers_preserves_fixed_evidence(tmp_path: Path) -> None:
    for script in ("regenerate_aggregate_artifacts.py", "revision_report_tables.py",
                   "revision_pareto_plots.py", "coverage_report_tables.py"):
        copy_relative(tmp_path, Path("scripts/latent_factor") / script)
    copy_relative(tmp_path, "src/latent_factor_study/plot_theme.py")
    sources = [
        REVISION / "selector_analysis" / name
        for name in ("validation_selections.csv", "selection_outcomes.csv",
                     "paired_test_contrasts.csv", "validation_landscapes.csv")
    ] + [
        REVISION / "protocol_audit/configuration_registry.csv",
        REVISION / "search_boundary/matched_budget_summary.csv",
    ]
    sources.extend(path.relative_to(ROOT) for path in
                   (ROOT / COVERAGE / "report_tables").glob("*.csv"))
    sources.extend(path.relative_to(ROOT) for path in
                   (ROOT / "tests/fixtures/tables").rglob("*.tex"))
    for path in sources:
        copy_relative(tmp_path, path)
    before = {path: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
              for path in sources}

    run_script(tmp_path, "regenerate_aggregate_artifacts.py")

    regenerated = tmp_path / "regenerated"
    result = json.loads((regenerated / "verification.json").read_text())
    assert result["status"] == "passed"
    assert len(result["coverage_tex_byte_identical"]) == 6
    assert len(result["coverage_csvs_byte_identical"]) == 4
    assert len(list((regenerated / "figures/review_r1").glob("*.pdf"))) == 5
    assert len(list((regenerated / "figures/review_r1").glob("*.png"))) == 5
    for table in (regenerated / "tables").rglob("*.tex"):
        fixture = tmp_path / "tests/fixtures/tables" / table.relative_to(regenerated / "tables")
        assert table.read_bytes() == fixture.read_bytes(), table.name
    assert all(hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == digest
               for path, digest in before.items())
    assert not (tmp_path / "papers").exists()
    assert not (regenerated / "publication-artifacts").exists()

    # Direct presentation commands also use isolated public defaults.
    run_script(tmp_path, "revision_report_tables.py")
    run_script(tmp_path, "revision_pareto_plots.py")
    assert (regenerated / "reports/review_r1/report_summary.json").is_file()
    assert (regenerated / "reports/pareto_plots/provenance.json").is_file()
    assert not (tmp_path / REVISION / "report_tables").exists()
    assert not (tmp_path / REVISION / "pareto_plots").exists()
    assert not (tmp_path / "papers").exists()


def test_reporting_default_table_destination(tmp_path: Path, monkeypatch) -> None:
    destinations = []
    monkeypatch.setattr(sys, "argv", ["reporting"])
    monkeypatch.setattr(reporting, "_repo_root", lambda config: tmp_path)
    monkeypatch.setattr(reporting, "write_manuscript_tables",
                        lambda config, output: destinations.append(output) or [])
    reporting.main()
    assert destinations == [tmp_path / "regenerated/tables"]


def test_plotting_default_figure_destination(tmp_path: Path, monkeypatch) -> None:
    destinations = []
    monkeypatch.setattr(sys, "argv", ["plotting"])
    monkeypatch.setattr(plotting, "_repo_root", lambda config: tmp_path)
    monkeypatch.setattr(plotting.pd, "read_csv", lambda path: pd.DataFrame())
    monkeypatch.setattr(plotting, "plot_selection_regret",
                        lambda frame, output: destinations.append(output) or [])
    plotting.main()
    assert destinations == [tmp_path / "regenerated/figures/selection_regret"]


def test_second_review_default_tables_preserve_analysis_contract(tmp_path: Path, monkeypatch) -> None:
    destinations = []
    monkeypatch.setattr(sys, "argv", ["second-review"])
    monkeypatch.setattr(second_review_analysis, "_repo_root", lambda config: tmp_path)
    monkeypatch.setattr(second_review_analysis, "run",
                        lambda config, **kwargs: destinations.append(kwargs) or [])
    second_review_analysis.main()
    assert destinations == [{
        "output_dir": tmp_path / "outputs/latent_factor_beyond_rmse/second_review_analysis",
        "tex_dir": tmp_path / "regenerated/tables",
    }]


def test_revision_default_presentations_preserve_analysis_contract(tmp_path: Path, monkeypatch) -> None:
    destinations = []
    monkeypatch.setattr(revision_analysis, "_repo_root", lambda config: tmp_path)
    monkeypatch.setattr(revision_analysis, "analyze_configuration_diagnostics",
                        lambda *args: (pd.DataFrame(), None))
    for name in ("analyze_bpr_selection", "analyze_practical_evidence", "analyze_prediction_regret",
                 "analyze_retention_contrasts", "analyze_candidate_diagnostics",
                 "analyze_selection_cost", "analyze_popularity"):
        monkeypatch.setattr(revision_analysis, name, lambda *args: pd.DataFrame())
    monkeypatch.setattr(revision_analysis, "write_latex_tables",
                        lambda output, **kwargs: destinations.append(output) or [])
    for name in ("plot_landscapes", "plot_rank_displacement"):
        monkeypatch.setattr(revision_analysis, name,
                            lambda root, output: destinations.append(output) or [])

    paths = revision_analysis.run_revision_analysis(tmp_path / "datasets.yaml")

    assert destinations == [tmp_path / "regenerated/tables",
                            tmp_path / "regenerated/figures", tmp_path / "regenerated/figures"]
    assert paths == [tmp_path / "outputs/latent_factor_beyond_rmse/revision_analysis/manifest.json"]
    assert not (tmp_path / "papers").exists()
