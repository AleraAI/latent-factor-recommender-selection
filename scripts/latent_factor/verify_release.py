#!/usr/bin/env python3
"""Check exact release membership, byte sizes, SHA-256, and public-data scope.

By default a checkout may contain .git, caches, virtual environments,
build/install products, and regenerated/. These are never release inputs.
Use --strict for a clean export: even those local files are unexpected.
--write-manifest explicitly refreshes the inventory after safety checks.
The inventory excludes only itself and is not a signed trust anchor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST = "manifests/release_files_sha256.csv"
REDACTION_MAP = "provenance/public_metadata_redactions.json"
ROOT_FILES = frozenset({
    "README.md", "pyproject.toml", "requirements-ci.txt", ".gitignore",
    ".gitattributes", "CITATION.cff", "LICENSE", "NOTICE",
    "THIRD_PARTY_NOTICES.md", "ARTIFACT_DEPOSIT.md",
})
LOCAL_ROOTS = frozenset({".git", ".venv", "venv", "regenerated", "build", "dist"})
CACHE_PARTS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".cache", ".tox", ".nox", ".ipynb_checkpoints", "__MACOSX",
})
FORBIDDEN_PARTS = frozenset({
    "papers", "release-artifacts", "raw", "checkpoints", "checkpoint",
    "private", "private_notes", "private-notes", "credentials", "secrets",
    "paired_users", "per_user", "per-user",
})
FORBIDDEN_SUFFIXES = frozenset({
    ".parquet", ".pq", ".arrow", ".feather", ".h5", ".hdf5", ".npy",
    ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".bin", ".safetensors",
    ".pem", ".key", ".p12", ".pfx", ".zip", ".tar", ".gz",
})
DATASETS = r"(?:movielens_1m|ciaodvd|yelp_tampa_restaurants)"
ALL_DATASETS = rf"(?:{DATASETS}|epinions)"
RETENTION = r"r(?:025|050|075|100)"
ANALYSIS_SCOPES = (
    "confirmation/analysis", "confirmation/secondary_analysis",
    "baseline_confirmation/analysis", "bpr_confirmation/analysis",
    "five_star_sensitivity/analysis", "robustness/analysis", "revision_analysis",
    "second_review_analysis", "computational_analysis",
    "framework_construction/analysis", "prospective_epinions/confirmation/analysis",
)
# Anchored expressions avoid recursive wildcards for evidence. New scientific
# artifact families require a reviewed policy change.
PUBLIC_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml",
    r"LICENSES/[A-Za-z0-9_.-]+\.txt",
    r"docs/[A-Z0-9_]+\.md",
    r"environment/(?:README\.md|declared-results-environment\.yml|report-smoke-environment\.json|revision-run-runtime-inventory\.csv|revision-runtime-evidence\.json)",
    r"src/latent_factor_study/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py",
    r"scripts/latent_factor/[A-Za-z0-9_]+\.(?:py|sh)",
    r"configs/latent_factor_beyond_rmse/[A-Za-z0-9_]+\.ya?ml",
    r"tests/(?:[A-Za-z0-9_]+/)*(?:test_[A-Za-z0-9_]+|conftest|__init__)\.py",
    r"tests/fixtures/tables/(?:review_r1/|coverage_review/)?[A-Za-z0-9_.-]+\.tex",
    r"manifests/(?:release_files_sha256|external_artifact_inventory|manuscript_table_verification)\.csv",
    rf"data/processed/(?:latent_factor_beyond_rmse|latent_factor_framework_epinions)/{ALL_DATASETS}/manifest\.json",
    rf"data/splits/(?:latent_factor_beyond_rmse|latent_factor_framework_epinions)/{ALL_DATASETS}/sparsity_manifest\.json",
    rf"data/splits/latent_factor_beyond_rmse_robustness/seed_[0-9]+/{DATASETS}/nested_retention_manifest\.json",
    r"provenance/(?:source_file_map|public_metadata_redactions)\.json",
    rf"provenance/relocated/data/splits/(?:latent_factor_beyond_rmse|latent_factor_framework_epinions)/{ALL_DATASETS}/sparsity_manifest\.json",
    r"provenance/relocated/output/latent_factor_major_revision_2026-09-14/(?:selector_analysis/fits/(?:runtime_replay_gate|fitting_plan)|report_tables/report_summary|user_resampling/manifest)\.json",
    r"output/latent_factor_coverage_revision_2026-09-19/report_tables/[A-Za-z0-9_]+\.(?:csv|json)",
    r"output/latent_factor_major_revision_2026-09-14/(?:pareto_plots|protocol_audit|report_tables|search_boundary|selector_analysis|user_resampling)/[A-Za-z0-9_]+\.(?:csv|json)",
    r"output/latent_factor_major_revision_2026-09-14/selector_analysis/fits/(?:completion|confirmation_manifest|data_integrity_audit_final)\.json",
    r"output/latent_factor_major_revision_2026-09-14/search_boundary/analyze_matched_budget\.py",
    rf"outputs/latent_factor_beyond_rmse/(?:{'|'.join(ANALYSIS_SCOPES)})/[A-Za-z0-9_@.-]+\.(?:csv|json|npz)",
    rf"outputs/latent_factor_beyond_rmse/(?:development|bpr_development|baseline_development)/{DATASETS}/{RETENTION}/(?:grid_runs\.csv|selection\.json)",
    r"outputs/latent_factor_beyond_rmse/development/development_landscape_summary\.(?:csv|json)",
    rf"outputs/latent_factor_beyond_rmse/baseline_development/{DATASETS}/{RETENTION}/popularity/summary\.json",
    rf"outputs/latent_factor_beyond_rmse/(?:confirmation|baseline_confirmation|bpr_confirmation|five_star_sensitivity|framework_construction/confirmation)/{DATASETS}/{RETENTION}/confirmation_manifest\.json",
    rf"outputs/latent_factor_beyond_rmse/framework_construction/selections/{DATASETS}/{RETENTION}/framework_selection\.json",
    rf"outputs/latent_factor_beyond_rmse/prospective_epinions/development/epinions/{RETENTION}/(?:grid_runs\.csv|selection\.json|framework_selection\.json)",
    rf"outputs/latent_factor_beyond_rmse/prospective_epinions/confirmation/epinions/{RETENTION}/confirmation_manifest\.json",
    rf"outputs/latent_factor_beyond_rmse/robustness/explicit_expanded/{DATASETS}/{RETENTION}/runs\.csv",
))

# Counts, hashes/path references, and schema descriptions containing a string
# user_id are not the same thing as populated identifier fields.
IDENTIFIER_KEYS = frozenset({
    "user", "userid", "userids", "userindex", "userindices", "useridx",
    "uid", "uids", "username", "usernames", "reviewerid", "reviewerids",
    "customerid", "customerids", "email", "emailaddress", "emails",
    "interactionid", "interactionids",
})
USER_COLLECTION_KEYS = frozenset({"users", "pairedusers", "peruser", "userrecords"})
SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")),
    ("service token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|xox[baprs]-[A-Za-z0-9-]{20,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,})\b")),
    ("cloud access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("embedded credential", re.compile(r"(?i)https?://[^\s/:@]+:[^\s/@]+@")),
    ("assigned credential", re.compile(r"(?im)[\"']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,}")),
)
PRIVATE_PATH = re.compile(r"(?:/(?:Users|home)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\[^\\\r\n]+\\)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return bool(parts) and (
        parts[0] in LOCAL_ROOTS
        or any(part in CACHE_PARTS or part.endswith(".egg-info") for part in parts)
        or parts[-1] in {".DS_Store", ".coverage"}
        or parts[-1].endswith((".pyc", ".pyo"))
    )


def public_path(relative: str) -> bool:
    """Explicit release allowlist, independent of Git and ignore rules."""
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or "\\" in relative or str(path) != relative:
        return False
    if any(part in {".", ".."} for part in path.parts) or local_path(relative):
        return False
    if any(part.lower() in FORBIDDEN_PARTS for part in path.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    # Preserve aggregate names such as mean_user_rmse and user_resampling.
    if re.search(r"(?:^|[_-])(?:paired[_-]users|per[_-]user|user[_-](?:ids?|records)|prediction[_-]users|ranking[_-]users)(?:[_.-]|$)", path.name.lower()):
        return False
    if any(part.startswith(".env") or part.lower() in {"id_rsa", "id_ed25519"} for part in path.parts):
        return False
    if re.search(r"(?:^|[_-])(?:private|secrets?|credentials?|notes|draft)(?:[_.-]|$)", path.name.lower()):
        return False
    return relative in ROOT_FILES or any(pattern.fullmatch(relative) for pattern in PUBLIC_PATTERNS)


def collect_files(root: Path, *, strict: bool = False) -> dict[str, Path]:
    """Walk without following links; non-local links are always rejected."""
    files: dict[str, Path] = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if not strict and local_path(relative):
                dirs.remove(name)
            elif path.is_symlink():
                raise ValueError(f"Symlink is not a public release file: {relative}")
            elif strict and not any(path.iterdir()):
                if local_path(relative) or any(part in FORBIDDEN_PARTS for part in path.parts):
                    raise ValueError(f"Unexpected release directory: {relative}")
        for name in names:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if not strict and local_path(relative):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Non-regular release file: {relative}")
            files[relative] = path
    return files


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def scan_metadata(value: Any, relative: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = normalized_key(str(key))
            if normalized in IDENTIFIER_KEYS:
                raise ValueError(f"User-level identifier field in public metadata: {relative}")
            if normalized in USER_COLLECTION_KEYS and isinstance(item, (list, dict)):
                raise ValueError(f"User-level record collection in public metadata: {relative}")
            if normalized == "repairedinteractionids" and isinstance(item, list) and item:
                raise ValueError(f"Row-level interaction identifiers in public metadata: {relative}")
            scan_metadata(item, relative)
    elif isinstance(value, list):
        for item in value:
            scan_metadata(item, relative)


def npz_keys(relative: str) -> set[str]:
    """Only the five audited aggregate bootstrap schemas are permitted."""
    base = "outputs/latent_factor_beyond_rmse/"
    if re.fullmatch(rf"{base}(?:confirmation|five_star_sensitivity)/analysis/{DATASETS}_{RETENTION}_resamples\.npz", relative):
        return {"ranking_mean_differences", "prediction_rmse_mean_differences"}
    if re.fullmatch(rf"{base}baseline_confirmation/analysis/{DATASETS}_{RETENTION}_resamples\.npz", relative):
        return {"prediction_rmse_mean_differences", "explicit_minus_bias_ndcg_mean_differences", "explicit_minus_popularity_ndcg_mean_differences"}
    if re.fullmatch(rf"{base}bpr_confirmation/analysis/{DATASETS}_{RETENTION}_resamples\.npz", relative):
        return {"bpr_minus_explicit_mean_differences", "bpr_minus_popularity_mean_differences"}
    if re.fullmatch(rf"{base}confirmation/secondary_analysis/{DATASETS}_{RETENTION}_(?:prediction_(?:raw_)?(?:rmse|mae)|recall_at_(?:10|20)|ndcg_at_20)\.npz", relative):
        return {"mean_differences"}
    if re.fullmatch(rf"{base}prospective_epinions/confirmation/analysis/epinions_{RETENTION}_bootstrap_resamples\.npz", relative):
        return {
            "ndcg_regret_framework_vs_ndcg", "rmse_framework_minus_ndcg",
            "ndcg_framework_minus_rmse", "rmse_framework_minus_rmse",
            *(f"{metric}_framework_minus_{reference}"
              for metric in ("ndcg@10", "recall@10", "ndcg@20", "recall@20", "prediction_rmse", "prediction_mae", "prediction_raw_rmse", "prediction_raw_mae")
              for reference in ("ndcg", "rmse")),
        }
    raise ValueError(f"Unapproved resampling archive path: {relative}")


def scan_npz(path: Path, relative: str) -> None:
    import numpy as np

    expected = npz_keys(relative)
    try:
        with zipfile.ZipFile(path) as zipped:
            entries = zipped.infolist()
            if len(entries) != len(expected) or {entry.filename for entry in entries} != {key + ".npy" for key in expected}:
                raise ValueError("unexpected archive members")
            if any(entry.file_size > 1024 * 1024 for entry in entries):
                raise ValueError("oversized archive member")
        with np.load(path, allow_pickle=False) as archive:
            for key in expected:
                values = archive[key]
                if values.dtype.kind != "f" or values.shape != (10000,) or not np.isfinite(values).all():
                    raise ValueError("not a finite 10000-draw aggregate float vector")
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        # Never print values or untrusted archive member names.
        raise ValueError(f"Unsafe aggregate resampling archive: {relative}") from exc


def bound_source_hashes(value: Any, *, config_prefix: str = "") -> list[tuple[str, str]]:
    """Find path/hash pairs, not mutable historical snapshot inventories."""
    bindings: list[tuple[str, str]] = []
    if isinstance(value, dict):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            bindings.append((path, digest))
        for key, item in value.items():
            if isinstance(item, str) and key.startswith(("src/", "configs/")):
                bindings.append((key, item))
            if key == "locked_input_sha256" and isinstance(item, dict):
                bindings.extend(
                    (f"configs/latent_factor_beyond_rmse/{config_prefix}{alias[:-5]}.yaml", digest)
                    for alias, digest in item.items()
                    if alias.endswith("_yaml") and isinstance(digest, str)
                )
                if "selector_py" in item:
                    bindings.append(("src/latent_factor_study/uncertainty_selector.py", item["selector_py"]))
            bindings.extend(bound_source_hashes(item, config_prefix=config_prefix))
    elif isinstance(value, list):
        for item in value:
            bindings.extend(bound_source_hashes(item, config_prefix=config_prefix))
    return bindings


def metadata_redactions(files: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Validate explicit provenance for sanitized public metadata projections."""
    mapping = files.get(REDACTION_MAP)
    if mapping is None:
        return {}
    payload = json.loads(mapping.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Public metadata redaction map must contain an entries list")
    indexed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed public metadata redaction entry")
        relative = entry.get("path")
        if not isinstance(relative, str) or not public_path(relative) or not relative.startswith("data/splits/") or PurePosixPath(relative).name not in {"sparsity_manifest.json", "nested_retention_manifest.json"} or relative in indexed:
            raise ValueError("Unsafe or duplicate metadata redaction path")
        for prefix in ("original_release", "canonical", "exported"):
            digest, size = entry.get(f"{prefix}_sha256"), entry.get(f"{prefix}_bytes")
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) or type(size) is not int or size < 0:
                raise ValueError(f"Malformed metadata redaction hash/size: {relative}")
        removed = entry.get("removed_fields")
        if not isinstance(removed, list) or not removed or any(not isinstance(field, str) for field in removed):
            raise ValueError(f"Malformed metadata redaction field list: {relative}")
        path = files.get(relative)
        if path is None or path.stat().st_size != entry["exported_bytes"] or sha256(path) != entry["exported_sha256"]:
            raise ValueError(f"Public metadata projection hash/size mismatch: {relative}")
        public = json.loads(path.read_text(encoding="utf-8")).get("_public_export", {})
        if not isinstance(public, dict) or public.get("source_sha256") != entry["original_release_sha256"] or public.get("not_canonical_original") is not True or public.get("removed_fields") != removed:
            raise ValueError(f"Public metadata projection provenance mismatch: {relative}")
        indexed[relative] = entry
    return indexed


def verify_external_inventory(files: dict[str, Path]) -> None:
    """Check retained historical inventory claims, not current completeness."""
    inventory = files.get("manifests/external_artifact_inventory.csv")
    redactions = metadata_redactions(files)
    if inventory is None:
        if redactions:
            raise ValueError("Public metadata redactions require the historical external inventory")
        return
    seen: set[str] = set()
    mapped: set[str] = set()
    with inventory.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "bytes", "sha256", "included_in_github"]:
            raise ValueError("Malformed historical external artifact inventory")
        for row in reader:
            relative = row.get("path") or ""
            path = PurePosixPath(relative)
            if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative or str(path) != relative or relative in seen:
                raise ValueError("Unsafe or duplicate historical inventory path")
            seen.add(relative)
            flag = row.get("included_in_github")
            if flag not in {"True", "False"}:
                raise ValueError(f"Invalid historical inclusion flag: {relative}")
            if flag == "False":
                if relative in redactions:
                    raise ValueError(f"Public metadata projection is marked external-only: {relative}")
                if relative in files:
                    raise ValueError(f"Historical external-only artifact is present: {relative}")
                continue
            if relative not in files:
                raise ValueError(f"Historical included artifact is missing: {relative}")
            if not re.fullmatch(r"[0-9]+", row.get("bytes") or "") or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256") or ""):
                raise ValueError(f"Malformed historical artifact digest: {relative}")
            if relative in redactions:
                projection = redactions[relative]
                if projection["canonical_bytes"] != int(row["bytes"]) or projection["canonical_sha256"] != row["sha256"]:
                    raise ValueError(f"Public metadata canonical provenance mismatch: {relative}")
                mapped.add(relative)
            elif files[relative].stat().st_size != int(row["bytes"]) or sha256(files[relative]) != row["sha256"]:
                raise ValueError(f"Historical included artifact hash/size mismatch: {relative}")
    if mapped != set(redactions):
        raise ValueError("Public metadata redaction entry lacks a historical included artifact")


def scan_public_files(root: Path, files: dict[str, Path]) -> None:
    """Fail closed on scope violations, likely credentials, or user records."""
    bindings: set[tuple[str, str]] = set()
    for relative, path in sorted(files.items()):
        if not public_path(relative):
            raise ValueError(f"File is outside the public release allowlist: {relative}")
        if path.suffix == ".npz":
            scan_npz(path, relative)
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Non-text content outside an approved archive: {relative}") from exc
        for category, pattern in SECRET_PATTERNS:
            if pattern.search(content):
                raise ValueError(f"Possible {category} in public file: {relative} (value suppressed)")
        if PRIVATE_PATH.search(content):
            raise ValueError(f"Machine-specific home path in public file: {relative} (value suppressed)")
        parsed: Any = None
        if path.suffix == ".json":
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON metadata: {relative}") from exc
        elif path.suffix in {".yaml", ".yml"}:
            import yaml
            try:
                parsed = yaml.safe_load(content)
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML metadata: {relative}") from exc
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                fields = next(csv.reader(handle), [])
            if any(normalized_key(field) in IDENTIFIER_KEYS for field in fields):
                raise ValueError(f"User-level identifier column in public CSV: {relative}")
        if parsed is not None:
            scan_metadata(parsed, relative)
            # This file describes a historical snapshot, not current files;
            # actual scientific locks and runtime provenance remain binding.
            if relative != "provenance/source_file_map.json":
                prefix = "epinions_prospective_" if path.name.startswith("epinions_prospective_") else ""
                bindings.update(bound_source_hashes(parsed, config_prefix=prefix))
    for relative, digest in sorted(bindings):
        if not relative.startswith(("src/latent_factor_study/", "configs/latent_factor_beyond_rmse/")):
            continue
        if not public_path(relative) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid immutable source/configuration hash binding")
        if relative not in files or sha256(root / relative) != digest:
            raise ValueError(f"Immutable source/configuration hash mismatch: {relative}")
    verify_external_inventory(files)


def write_manifest(root: Path, *, strict: bool = False) -> int:
    """Refresh inventory only after successful public-scope checks."""
    root = root.resolve()
    files = collect_files(root, strict=strict)
    files.pop(MANIFEST, None)
    scan_public_files(root, files)
    report = root / MANIFEST
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("path", "bytes", "sha256"))
        for relative, path in sorted(files.items()):
            writer.writerow((relative, path.stat().st_size, sha256(path)))
    return len(files)


def verify_release_manifest(root: Path, *, strict: bool = False) -> int:
    root = root.resolve()
    return verify_manifest_files(root, collect_files(root, strict=strict))


def verify_manifest_files(root: Path, files: dict[str, Path]) -> int:
    """Verify an exact selected set, including the inventory file itself.

    The builder uses this before copying so local excluded files do not enter
    the export, but missing or modified public evidence cannot be re-blessed.
    """
    files = files.copy()
    report = files.pop(MANIFEST, None)
    if report is None:
        raise ValueError(f"Missing release manifest: {MANIFEST}")
    indexed: dict[str, tuple[int, str]] = {}
    with report.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["path", "bytes", "sha256"]:
            raise ValueError("Release manifest must have exactly path,bytes,sha256 columns")
        for row in reader:
            relative = row["path"]
            if relative is None or not public_path(relative) or relative == MANIFEST:
                raise ValueError("Release manifest contains a non-public or unsafe path")
            if relative in indexed:
                raise ValueError(f"Duplicate release manifest path: {relative}")
            if None in row or not re.fullmatch(r"[0-9]+", row.get("bytes") or "") or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256") or ""):
                raise ValueError(f"Malformed release manifest entry: {relative}")
            indexed[relative] = (int(row["bytes"]), row["sha256"])
    unexpected = sorted(set(files) - set(indexed))
    missing = sorted(set(indexed) - set(files))
    if unexpected or missing:
        raise ValueError(f"Release membership mismatch; unexpected={unexpected}, missing={missing}")
    for relative, (size, digest) in indexed.items():
        path = files[relative]
        if path.stat().st_size != size:
            raise ValueError(f"Release size mismatch: {relative}")
        if sha256(path) != digest:
            raise ValueError(f"Release SHA-256 mismatch: {relative}")
    scan_public_files(root, files)
    return len(indexed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--strict", action="store_true", help="Reject caches, .git, venvs, and regenerated files.")
    parser.add_argument("--write-manifest", action="store_true", help="Refresh inventory after public-scope checks.")
    args = parser.parse_args()
    try:
        if args.write_manifest:
            write_manifest(args.project_root, strict=args.strict)
        count = verify_release_manifest(args.project_root, strict=args.strict)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Release verification failed: {exc}\n")
    mode = "clean export" if args.strict else "checkout"
    print(f"Verified {count} public files: exact paths, sizes, SHA-256, and scope ({mode}).")


if __name__ == "__main__":
    main()
