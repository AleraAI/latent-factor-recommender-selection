"""Release guard tests use synthetic temporary files, never saved evidence."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.latent_factor.build_github_release import build
from scripts.latent_factor.verify_release import (
    MANIFEST,
    collect_files,
    public_path,
    scan_metadata,
    scan_npz,
    scan_public_files,
    sha256,
    verify_release_manifest,
    write_manifest,
)


ARCHIVE = "outputs/latent_factor_beyond_rmse/confirmation/secondary_analysis/ciaodvd_r025_prediction_rmse.npz"
METADATA = "outputs/latent_factor_beyond_rmse/revision_analysis/manifest.json"


def put(root: Path, relative: str, value: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode("utf-8") if isinstance(value, str) else value)
    return path


@pytest.fixture
def release(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    put(root, "README.md", "Public test release\n")
    put(root, "pyproject.toml", "[project]\nname = 'release-test'\n")
    return root


@pytest.mark.parametrize("relative", [
    "tests/test_latent_factor_major_revision_selection.py",
    "tests/test_latent_factor_revision_user_resampling.py",
    "tests/test_any_new_feature.py",
    "tests/nested/test_other_feature.py",
    "tests/conftest.py",
    "tests/fixtures/tables/review_r1/table.tex",
    "tests/fixtures/tables/coverage_review/table.tex",
    "scripts/latent_factor/verify_release.py",
    "docs/RESULTS_INDEX.md",
    ARCHIVE,
])
def test_allowlist_includes_all_test_families_and_fixtures(relative: str) -> None:
    assert public_path(relative)


@pytest.mark.parametrize("relative", [
    "../README.md", "/README.md", "docs/../README.md", "./README.md",
    "docs//README.md", "docs\\README.md", "papers/main.tex",
    "release-artifacts/manuscript.pdf", ".git/config", ".env",
    "data/raw/ratings.csv", "outputs/latent_factor_beyond_rmse/revision_analysis/ratings.parquet",
    "outputs/latent_factor_beyond_rmse/revision_analysis/prediction_users.csv",
    "outputs/latent_factor_beyond_rmse/revision_analysis/user_ids.json",
    "outputs/latent_factor_beyond_rmse/revision_analysis/paired_users.csv",
    "outputs/latent_factor_beyond_rmse/revision_analysis/per_user.json",
    "outputs/latent_factor_beyond_rmse/revision_analysis/model.pt",
    "docs/private/NOTES.md", "private_notes.md", "unreviewed/result.json",
    "docs/PRIVATE_NOTES.md", "docs/DRAFT.md",
    "tests/generated/report.json", "tests/fixtures/tables/main.pdf",
])
def test_allowlist_excludes_private_and_out_of_scope_files(relative: str) -> None:
    assert not public_path(relative)


def test_manifest_exact_membership_size_hash_and_stability(release: Path) -> None:
    assert write_manifest(release) == 2
    original = (release / MANIFEST).read_bytes()
    assert verify_release_manifest(release) == 2
    write_manifest(release)
    assert (release / MANIFEST).read_bytes() == original
    readme = release / "README.md"
    readme.write_bytes(b"x" * readme.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_release_manifest(release)
    readme.write_text("larger content than the original test release\n")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_release_manifest(release)


def test_missing_and_extra_files_are_reported(release: Path) -> None:
    write_manifest(release)
    (release / "README.md").unlink()
    with pytest.raises(ValueError, match="missing=.*README"):
        verify_release_manifest(release)
    put(release, "README.md", "Public test release\n")
    put(release, "tests/test_new.py", "pass\n")
    with pytest.raises(ValueError, match="unexpected=.*test_new"):
        verify_release_manifest(release)


@pytest.mark.parametrize("entry", [
    ("../outside", "1", "0" * 64),
    (MANIFEST, "1", "0" * 64),
    ("README.md", "-1", "0" * 64),
    ("README.md", "1", "invalid"),
])
def test_malformed_manifest_rows_fail(release: Path, entry: tuple[str, ...]) -> None:
    path = put(release, MANIFEST, "")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "bytes", "sha256"))
        writer.writerow(entry)
    with pytest.raises(ValueError):
        verify_release_manifest(release)


def test_duplicate_manifest_rows_fail(release: Path) -> None:
    write_manifest(release)
    path = release / MANIFEST
    lines = path.read_text().splitlines()
    path.write_text("\n".join([*lines, lines[1]]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        verify_release_manifest(release)


@pytest.mark.parametrize("relative", [
    ".git/config", ".pytest_cache/v/cache/nodeids", "scripts/latent_factor/__pycache__/example.pyc",
    ".venv/bin/python", "venv/bin/python", "regenerated/reports/check.csv",
    "project.egg-info/PKG-INFO", ".DS_Store", "dist/example.whl",
])
def test_local_files_ignored_only_in_checkout_mode(release: Path, relative: str) -> None:
    write_manifest(release)
    put(release, relative, "local only")
    assert verify_release_manifest(release) == 2
    with pytest.raises(ValueError, match="membership mismatch"):
        verify_release_manifest(release, strict=True)


def test_manifest_refresh_cannot_bless_out_of_scope_data(release: Path) -> None:
    write_manifest(release)
    old = (release / MANIFEST).read_bytes()
    put(release, "data/raw/ratings.csv", "user_id,item_id,rating\n1,2,3\n")
    with pytest.raises(ValueError, match="allowlist"):
        write_manifest(release)
    assert (release / MANIFEST).read_bytes() == old


@pytest.mark.parametrize("value", [
    {"user_id": 123}, {"records": [{"userId": "person"}]},
    {"USER_IDS": [1, 2]}, {"users": [1, 2]}, {"users": {"1": {"score": 0.5}}},
    {"per_user": [{"score": 0.5}]}, {"email": "example"},
    {"interaction_id": "hashed-row"}, {"interaction_ids": ["hashed-row"]},
])
def test_actual_user_metadata_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="User-level"):
        scan_metadata(value, METADATA)


def test_aggregate_counts_schema_names_and_provenance_paths_are_not_user_records() -> None:
    scan_metadata({
        "users": 12168,
        "users_paired": 100,
        "users_after": 100,
        "user_degree_quantiles": {"q50": 30},
        "schemas": [{"columns": ["user_id", "rating"], "files": 175}],
        "per_user_path": "external/per_user.parquet",
        "paired_users_sha256": "0" * 64,
        "source_path": "papers/historical/main.tex",
        "repaired_interaction_ids_omitted_count": 3,
    }, METADATA)


def test_nonempty_hashed_interaction_repair_lists_are_private() -> None:
    with pytest.raises(ValueError, match="Row-level interaction"):
        scan_metadata({"repaired_interaction_ids": ["a" * 64]}, METADATA)
    scan_metadata({"repaired_interaction_ids": []}, METADATA)


def test_csv_identifier_columns_are_rejected(release: Path) -> None:
    relative = "outputs/latent_factor_beyond_rmse/revision_analysis/practical_evidence.csv"
    put(release, relative, "user_id,score\n1,0.5\n")
    with pytest.raises(ValueError, match="identifier column"):
        scan_public_files(release, collect_files(release))


@pytest.mark.parametrize("secret", [
    "ghp_" + "a" * 36,
    "-----BEGIN " + "PRIVATE KEY-----",
    "AKIA" + "A" * 16,
    "api_key = " + "a" * 30,
    "https://" + "name:password@example.invalid",
])
def test_likely_secrets_fail_without_disclosing_value(release: Path, secret: str) -> None:
    put(release, "README.md", secret)
    with pytest.raises(ValueError) as error:
        scan_public_files(release, collect_files(release))
    assert "value suppressed" in str(error.value)
    assert secret not in str(error.value)


def test_machine_specific_home_path_fails_without_rewriting(release: Path) -> None:
    content = "/" + "Users" + "/researcher/secret.csv"
    path = put(release, "README.md", content)
    with pytest.raises(ValueError, match="home path"):
        scan_public_files(release, collect_files(release))
    assert path.read_text() == content


def test_aggregate_npz_is_allowed_and_unmodified(release: Path) -> None:
    path = put(release, ARCHIVE, b"")
    np.savez_compressed(path, mean_differences=np.linspace(-0.1, 0.1, 10000))
    before = path.read_bytes()
    scan_npz(path, ARCHIVE)
    write_manifest(release)
    verify_release_manifest(release)
    assert path.read_bytes() == before


@pytest.mark.parametrize("values", [
    np.arange(10000), np.zeros((100, 100)), np.zeros(9999),
    np.full(10000, np.nan), np.array(["person"] * 10000),
    np.array([{"id": 1}], dtype=object),
])
def test_npz_rejects_nonaggregate_arrays(release: Path, values: np.ndarray) -> None:
    path = put(release, ARCHIVE, b"")
    np.savez_compressed(path, mean_differences=values)
    with pytest.raises(ValueError, match="Unsafe aggregate"):
        scan_npz(path, ARCHIVE)


def test_npz_rejects_user_id_members(release: Path) -> None:
    path = put(release, ARCHIVE, b"")
    np.savez_compressed(path, mean_differences=np.zeros(10000), user_id=np.arange(10000))
    with pytest.raises(ValueError, match="Unsafe aggregate"):
        scan_npz(path, ARCHIVE)


def test_immutable_runtime_source_hash_cannot_be_refreshed_away(release: Path) -> None:
    source = put(release, "src/latent_factor_study/data.py", "# historical\n")
    put(release, METADATA, json.dumps({"source_sha256": {
        "src/latent_factor_study/data.py": sha256(source),
    }}))
    write_manifest(release)
    source.write_text("# changed\n")
    with pytest.raises(ValueError, match="Immutable source/configuration"):
        write_manifest(release)


def test_locked_config_hashes_include_prospective_aliases(release: Path) -> None:
    config = put(release, "configs/latent_factor_beyond_rmse/epinions_prospective_datasets.yaml", "version: 1\n")
    put(release, "configs/latent_factor_beyond_rmse/epinions_prospective_selection_lock.yaml", f"locked_input_sha256:\n  datasets_yaml: {sha256(config)}\n")
    write_manifest(release)
    config.write_text("version: 2\n")
    with pytest.raises(ValueError, match="Immutable source/configuration"):
        write_manifest(release)


def test_historical_source_map_is_not_misrepresented_as_current_hashes(release: Path) -> None:
    put(release, "provenance/source_file_map.json", json.dumps({
        "scope": "historical snapshot", "exact_copies": [
            {"path": "src/latent_factor_study/reporting.py", "sha256": "0" * 64},
        ],
    }))
    write_manifest(release)


def test_historical_external_inventory_does_not_require_complete_current_coverage(release: Path) -> None:
    path = put(release, METADATA, '{"users": 20}\n')
    put(release, "manifests/external_artifact_inventory.csv", (
        "path,bytes,sha256,included_in_github\n"
        f"{METADATA},{path.stat().st_size},{sha256(path)},True\n"
        f"data/raw/ratings.parquet,100,{'0' * 64},False\n"
    ))
    write_manifest(release)
    assert verify_release_manifest(release) == 4
    path.write_text('{"users": 21}\n')
    with pytest.raises(ValueError, match="Historical included artifact hash/size"):
        write_manifest(release)


@pytest.mark.parametrize("included", [True, False])
def test_historical_external_inventory_membership_claims_are_checked(release: Path, included: bool) -> None:
    put(release, "manifests/external_artifact_inventory.csv", (
        "path,bytes,sha256,included_in_github\n"
        f"{METADATA},100,{'0' * 64},{included}\n"
    ))
    if not included:
        put(release, METADATA, "{}\n")
    with pytest.raises(ValueError, match="Historical (included|external-only) artifact"):
        write_manifest(release)


def redaction_fixture(release: Path) -> tuple[Path, dict[str, object]]:
    relative = "data/splits/latent_factor_beyond_rmse/ciaodvd/sparsity_manifest.json"
    removed = ["conditions.r025.repaired_interaction_ids"]
    projection = {
        "conditions": {"r025": {"users": 442, "repaired_interaction_ids_omitted_count": 3}},
        "_public_export": {
            "source_sha256": "1" * 64, "not_canonical_original": True,
            "removed_fields": removed,
        },
    }
    path = put(release, relative, json.dumps(projection))
    entry = {
        "path": relative, "original_release_sha256": "1" * 64,
        "original_release_bytes": 100, "canonical_sha256": "2" * 64,
        "canonical_bytes": 200, "exported_sha256": sha256(path),
        "exported_bytes": path.stat().st_size, "removed_fields": removed,
    }
    put(release, "provenance/public_metadata_redactions.json", json.dumps({"entries": [entry]}))
    put(release, "manifests/external_artifact_inventory.csv", (
        "path,bytes,sha256,included_in_github\n"
        f"{relative},200,{'2' * 64},True\n"
    ))
    return path, entry


def test_explicit_metadata_redactions_preserve_canonical_inventory_hashes(release: Path) -> None:
    path, entry = redaction_fixture(release)
    write_manifest(release)
    verify_release_manifest(release)
    assert sha256(path) == entry["exported_sha256"]


@pytest.mark.parametrize("change", ["canonical", "exported", "source", "fields", "missing_row"])
def test_redaction_map_mismatches_fail_closed(release: Path, change: str) -> None:
    path, entry = redaction_fixture(release)
    if change == "canonical":
        entry["canonical_sha256"] = "3" * 64
    elif change == "exported":
        path.write_text("{}")
    elif change == "source":
        entry["original_release_sha256"] = "3" * 64
    elif change == "fields":
        entry["removed_fields"] = ["different.field"]
    else:
        put(release, "manifests/external_artifact_inventory.csv", "path,bytes,sha256,included_in_github\n")
    put(release, "provenance/public_metadata_redactions.json", json.dumps({"entries": [entry]}))
    with pytest.raises(ValueError, match="metadata"):
        write_manifest(release)


def test_symlinks_are_never_followed(release: Path, tmp_path: Path) -> None:
    outside = put(tmp_path, "outside.txt", "private material")
    (release / "NOTICE").symlink_to(outside)
    with pytest.raises(ValueError, match="Non-regular"):
        write_manifest(release)
    with pytest.raises(ValueError, match="Non-regular"):
        build(release, tmp_path / "export")


def test_builder_is_standalone_complete_byte_exact_and_excludes_private_files(release: Path, tmp_path: Path) -> None:
    files = {
        "tests/test_latent_factor_major_revision_selection.py": b"# major revision\n",
        "tests/test_latent_factor_revision_user_resampling.py": b"# user resampling\n",
        "tests/test_other.py": b"# unrelated name included\n",
        "tests/fixtures/tables/review_r1/table.tex": b"% exact fixture\r\n",
        METADATA: b'{"users": 20, "schema": ["user_id"]}\n',
    }
    for relative, value in files.items():
        put(release, relative, value)
    write_manifest(release)
    for relative in (
        "papers/main.tex", "release-artifacts/manuscript.pdf", ".git/config",
        "regenerated/tables/table.tex", "data/raw/ratings.csv",
        "outputs/latent_factor_beyond_rmse/revision_analysis/paired_users.parquet",
        "private_notes.md", "tests/generated/report.json",
    ):
        put(release, relative, "must not be exported")
    destination = tmp_path / "export"
    count = build(release, destination)
    assert count == len(files) + 2
    assert verify_release_manifest(destination, strict=True) == count
    assert set(collect_files(destination)) == {*files, "README.md", "pyproject.toml", MANIFEST}
    for relative, value in files.items():
        assert (release / relative).read_bytes() == value
        assert (destination / relative).read_bytes() == value


def test_builder_refuses_existing_nested_and_invalid_sources(release: Path, tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        build(release, existing)
    with pytest.raises(ValueError, match="outside"):
        build(release, release / "nested")
    with pytest.raises(ValueError, match="standalone"):
        build(existing, tmp_path / "export")


@pytest.mark.parametrize("change", ["missing", "modified", "new"])
def test_builder_cannot_bless_unreviewed_changes_to_source_inventory(release: Path, tmp_path: Path, change: str) -> None:
    evidence = put(release, METADATA, '{"users": 20}\n')
    write_manifest(release)
    if change == "missing":
        evidence.unlink()
    elif change == "modified":
        evidence.write_text('{"users": 21}\n')
    else:
        put(release, "tests/test_new.py", "# not yet in inventory\n")
    target = tmp_path / "export"
    with pytest.raises(ValueError, match="membership mismatch|SHA-256 mismatch"):
        build(release, target)
    assert not target.exists()


def test_builder_requires_an_existing_source_inventory(release: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest is missing"):
        build(release, tmp_path / "export")


def test_exported_commands_work_without_parent_repository(release: Path, tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    for name in ("build_github_release.py", "verify_release.py"):
        source = repository / "scripts/latent_factor" / name
        target = release / "scripts/latent_factor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_manifest(release)
    exported = tmp_path / "exported"
    build(release, exported)
    verified = subprocess.run(
        [sys.executable, "-B", str(exported / "scripts/latent_factor/verify_release.py"), "--strict"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert "clean export" in verified.stdout
    rebuilt = tmp_path / "rebuilt"
    result = subprocess.run(
        [sys.executable, "-B", str(exported / "scripts/latent_factor/build_github_release.py"), "--output", str(rebuilt)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (exported / MANIFEST).read_bytes() == (rebuilt / MANIFEST).read_bytes()
