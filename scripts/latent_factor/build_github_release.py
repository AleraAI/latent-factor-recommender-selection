#!/usr/bin/env python3
"""Export this public code-and-aggregate repository into a new clean directory.

No authoring checkout, paper sources, private data, or installed Git is needed.
The current source inventory must verify before copying. Files are copied
byte-for-byte; evidence is never sanitized or regenerated.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from .verify_release import (
        MANIFEST, collect_files, public_path,
        verify_manifest_files, verify_release_manifest, write_manifest,
    )
except ImportError:  # Direct invocation, without installing the package.
    from verify_release import (
        MANIFEST, collect_files, public_path,
        verify_manifest_files, verify_release_manifest, write_manifest,
    )


def build(root: Path, target: Path) -> int:
    """Copy approved public files; refuse existing or nested destinations."""
    root = root.resolve()
    target = target.absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing destination: {target}")
    if target.resolve().is_relative_to(root):
        raise ValueError("The export destination must be outside the source repository.")
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Source must be the standalone public repository root.")
    # Do not depend on Git, authoring templates, or one historical test glob.
    files = {
        relative: path
        for relative, path in collect_files(root).items()
        if relative != MANIFEST and public_path(relative)
    }
    inventory = root / MANIFEST
    if not inventory.is_file():
        raise ValueError("Source release manifest is missing; verify the source inventory first.")
    verify_manifest_files(root, {**files, MANIFEST: inventory})
    target.mkdir(parents=True)
    for relative, path in sorted(files.items()):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_manifest(target, strict=True)
    return verify_release_manifest(target, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the source repository; never overwritten.")
    args = parser.parse_args()
    try:
        count = build(args.project_root, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Release export failed: {exc}\n")
    print(f"Verified clean export: {count} files at {args.output.absolute()}")


if __name__ == "__main__":
    main()
