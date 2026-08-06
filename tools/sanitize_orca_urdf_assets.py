#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanitize asset filenames referenced by an ORCA URDF for Isaac Sim.

Purpose
-------
Isaac Sim's URDF importer may use the mesh filename stem as a USD prim
identifier. Characters such as '-' can therefore produce an invalid
SdfPath and eventually cause:

    RuntimeError: Used null prim

This script:

1. Reads all filename="..." references in the target URDF.
2. Resolves package://orcahand_description/... paths.
3. Renames referenced files whose basenames contain invalid characters.
4. Updates the URDF references.
5. Sanitizes optional visual/collision/material name attributes.
6. Preserves a .bak copy of the original working URDF.

Run this script only on a copied working package, not on the official
external/orcahand_description Git repository.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


FILENAME_PATTERN = re.compile(r'filename="([^"]+)"')

NAMED_ELEMENT_PATTERN = re.compile(
    r'(<(?:visual|collision|material)\b[^>]*\bname=")'
    r'([^"]+)'
    r'(")'
)


def sanitize_identifier(value: str) -> str:
    """Convert a string into a conservative USD-compatible identifier."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", value)

    # USD identifiers should not begin with a number.
    if not sanitized:
        sanitized = "_unnamed"
    elif sanitized[0].isdigit():
        sanitized = "_" + sanitized

    return sanitized


def resolve_asset_path(
    uri: str,
    urdf_path: Path,
    package_dir: Path,
    package_name: str,
) -> Path:
    package_prefix = f"package://{package_name}/"

    if uri.startswith(package_prefix):
        relative = uri[len(package_prefix):]
        return (package_dir / relative).resolve()

    if uri.startswith("file://"):
        return Path(uri[len("file://"):]).resolve()

    return (urdf_path.parent / uri).resolve()


def replace_filename_basename(uri: str, new_basename: str) -> str:
    if "/" not in uri:
        return new_basename

    directory = uri.rsplit("/", maxsplit=1)[0]
    return f"{directory}/{new_basename}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        required=True,
        type=Path,
        help="Path to the copied working URDF.",
    )
    parser.add_argument(
        "--package-dir",
        required=True,
        type=Path,
        help="Path corresponding to package://orcahand_description/.",
    )
    parser.add_argument(
        "--package-name",
        default="orcahand_description",
    )
    args = parser.parse_args()

    urdf_path = args.urdf.expanduser().resolve()
    package_dir = args.package_dir.expanduser().resolve()

    if not urdf_path.is_file():
        print(f"[ERROR] URDF does not exist: {urdf_path}", file=sys.stderr)
        return 2

    if not package_dir.is_dir():
        print(
            f"[ERROR] Package directory does not exist: {package_dir}",
            file=sys.stderr,
        )
        return 2

    original_text = urdf_path.read_text(encoding="utf-8")
    updated_text = original_text

    uris = sorted(set(FILENAME_PATTERN.findall(original_text)))

    renamed_files: dict[Path, Path] = {}
    uri_updates: dict[str, str] = {}
    missing_assets: list[tuple[str, Path]] = []

    for uri in uris:
        old_path = resolve_asset_path(
            uri=uri,
            urdf_path=urdf_path,
            package_dir=package_dir,
            package_name=args.package_name,
        )

        if not old_path.exists():
            missing_assets.append((uri, old_path))
            continue

        sanitized_stem = sanitize_identifier(old_path.stem)
        new_basename = sanitized_stem + old_path.suffix
        new_path = old_path.with_name(new_basename)

        if new_path == old_path:
            continue

        if new_path.exists() and new_path.resolve() != old_path.resolve():
            raise FileExistsError(
                "Sanitized filename collision:\n"
                f"  old: {old_path}\n"
                f"  new: {new_path}"
            )

        old_path.rename(new_path)
        renamed_files[old_path] = new_path

        new_uri = replace_filename_basename(uri, new_basename)
        uri_updates[uri] = new_uri

    for old_uri, new_uri in uri_updates.items():
        updated_text = updated_text.replace(
            f'filename="{old_uri}"',
            f'filename="{new_uri}"',
        )

    # Sanitize optional names that the importer may also convert to prim names.
    def replace_named_element(match: re.Match[str]) -> str:
        prefix, old_name, suffix = match.groups()
        new_name = sanitize_identifier(old_name)
        return f"{prefix}{new_name}{suffix}"

    updated_text = NAMED_ELEMENT_PATTERN.sub(
        replace_named_element,
        updated_text,
    )

    backup_path = urdf_path.with_suffix(urdf_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(urdf_path, backup_path)

    urdf_path.write_text(updated_text, encoding="utf-8")

    print(f"[INFO] URDF       : {urdf_path}")
    print(f"[INFO] Backup     : {backup_path}")
    print(f"[INFO] References : {len(uris)}")
    print(f"[INFO] Renamed    : {len(renamed_files)}")

    for old_path, new_path in renamed_files.items():
        print(f"[RENAME] {old_path.name} -> {new_path.name}")

    if missing_assets:
        print()
        print("[WARNING] Missing referenced assets:")

        for uri, resolved in missing_assets:
            print(f"  URI      : {uri}")
            print(f"  Resolved : {resolved}")

        return 1

    print("[PASS] Asset filename sanitization completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
