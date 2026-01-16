#!/usr/bin/env python3
"""
src/mgc/submission_cli.py

Submission bundle packaging (deterministic).

This module is imported by mgc.main via:
  from mgc.submission_cli import register_submission_subcommand

So we MUST provide register_submission_subcommand(subparsers).

Design goals:
- No DB mutation
- Deterministic ZIP bytes (stable ordering, timestamps, perms)
- Deterministic README
- Validate bundle before packaging
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from mgc.bundle_validate import validate_bundle


# -----------------------------
# Determinism constants
# -----------------------------

FIXED_ISO_TS = "2020-01-01T00:00:00+00:00"
FIXED_ZIP_DT = (2020, 1, 1, 0, 0, 0)
FIXED_FILE_MODE = 0o100644  # -rw-r--r--


# -----------------------------
# JSON helpers
# -----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _emit_json(obj: Dict[str, Any]) -> None:
    # In --json mode, caller contract expects exactly one JSON object on stdout.
    print(_stable_json_dumps(obj))


# -----------------------------
# Deterministic ZIP writer
# -----------------------------

def _write_zip_deterministic(zip_path: Path, root_dir: Path, rel_paths: Iterable[str]) -> None:
    rels = sorted(set(rel_paths))
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in rels:
            src = root_dir / rel
            data = src.read_bytes()

            zi = zipfile.ZipInfo(filename=rel.replace(os.sep, "/"), date_time=FIXED_ZIP_DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (FIXED_FILE_MODE & 0xFFFF) << 16

            zf.writestr(zi, data)


# -----------------------------
# Bundle collection
# -----------------------------

def _collect_files_strict(root: Path) -> List[str]:
    """
    Collect files under `root` with basic hygiene rules.
    This prevents volatile junk from sneaking into the submission.
    """
    out: List[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        rel = p.relative_to(root).as_posix()

        # hygiene / determinism
        if rel.startswith("."):
            continue
        if "/." in rel:
            continue
        if "__pycache__" in rel:
            continue
        if rel.endswith(".pyc"):
            continue
        if rel.endswith(".tmp"):
            continue
        if rel.endswith(".DS_Store") or rel.endswith("DS_Store"):
            continue

        out.append(rel)

    return sorted(set(out))


def _stable_readme(drop_id: str) -> str:
    return (
        "Music Generator Company – Submission\n"
        "\n"
        f"Drop ID: {drop_id}\n"
        f"Generated: {FIXED_ISO_TS}\n"
        "\n"
        "This archive is generated deterministically.\n"
    )


# -----------------------------
# Core build
# -----------------------------

def build_submission_zip_from_bundle_dir(bundle_dir: Path, out_zip: Path) -> Dict[str, Any]:
    """
    Validate bundle, stage a deterministic view, then write a deterministic ZIP.
    Returns a small receipt dict (caller may persist separately).
    """
    validate_bundle(bundle_dir)

    drop_json = bundle_dir / "drop.json"
    drop = _read_json(drop_json)
    drop_id = str(drop.get("drop_id") or "")

    if not drop_id:
        raise SystemExit(f"bundle missing drop_id in {drop_json}")

    staging = out_zip.parent / f".staging_submission_{drop_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # Copy bundle files into staging (exact bytes preserved)
    bundle_files = _collect_files_strict(bundle_dir)
    for rel in bundle_files:
        src = bundle_dir / rel
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    # Deterministic README
    (staging / "README.txt").write_text(_stable_readme(drop_id), encoding="utf-8")

    # Deterministic ZIP (sorted rel list)
    staging_files = _collect_files_strict(staging)
    if "README.txt" not in staging_files:
        staging_files.append("README.txt")
    _write_zip_deterministic(out_zip, staging, staging_files)

    shutil.rmtree(staging)

    return {
        "ok": True,
        "drop_id": drop_id,
        "bundle_dir": str(bundle_dir),
        "out_zip": str(out_zip),
    }


def _bundle_dir_from_db(drop_id: str, db_path: Path, evidence_root: Optional[Path] = None) -> Path:
    """
    Find the bundle directory for a drop. We try a couple common DB schemas.
    If your schema differs, we adjust this query, but this is a sensible default.
    """
    con = sqlite3.connect(str(db_path))
    try:
        # Common case: drops table has bundle_dir column
        row = con.execute(
            "SELECT bundle_dir FROM drops WHERE drop_id = ?",
            (drop_id,),
        ).fetchone()
        if row and row[0]:
            return Path(row[0])

        # Alternate: drop_bundles table
        row2 = con.execute(
            "SELECT bundle_dir FROM drop_bundles WHERE drop_id = ?",
            (drop_id,),
        ).fetchone()
        if row2 and row2[0]:
            return Path(row2[0])
    finally:
        con.close()

    # As a fallback, if evidence_root is provided, try evidence_root/<drop_id>
    if evidence_root:
        cand = evidence_root / drop_id
        if cand.exists():
            return cand

    raise SystemExit(f"Could not locate bundle_dir for drop_id={drop_id} in db={db_path}")


# -----------------------------
# CLI commands (registered by mgc.main)
# -----------------------------

def cmd_submission_build(args: argparse.Namespace) -> int:
    out_zip = Path(args.out).resolve()

    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir).resolve()
    else:
        if not args.db:
            raise SystemExit("--db is required with --drop-id")
        bundle_dir = _bundle_dir_from_db(
            drop_id=str(args.drop_id),
            db_path=Path(args.db).resolve(),
            evidence_root=Path(args.evidence_root).resolve() if args.evidence_root else None,
        )

    receipt = build_submission_zip_from_bundle_dir(bundle_dir=bundle_dir, out_zip=out_zip)

    if getattr(args, "json", False):
        _emit_json({"cmd": "submission.build", **receipt})
    else:
        print(f"[submission.build] ok drop_id={receipt['drop_id']} out={receipt['out_zip']}")

    return 0


def register_submission_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """
    Hook for mgc.main. Adds:
      mgc submission build ...
    """
    p = subparsers.add_parser("submission", help="Build deterministic submission ZIPs")
    sub = p.add_subparsers(dest="submission_cmd", required=True)

    b = sub.add_parser("build", help="Build a submission ZIP from a bundle dir or drop_id")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle-dir", dest="bundle_dir", type=str, help="Path to bundle directory")
    src.add_argument("--drop-id", dest="drop_id", type=str, help="Drop id to resolve via DB")

    b.add_argument("--db", type=str, default=None, help="SQLite DB path (required with --drop-id)")
    b.add_argument("--evidence-root", type=str, default=None, help="Optional evidence root fallback")
    b.add_argument("--out", type=str, required=True, help="Output ZIP path")

    # mgc.main usually plumbs global --json onto args; keep it compatible.
    b.set_defaults(fn=cmd_submission_build)
