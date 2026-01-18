#!/usr/bin/env python3
"""
src/mgc/submission_cli.py

Deterministic submission bundle packaging.

Commands (via mgc.main):
  mgc submission build  --bundle-dir <dir> --out submission.zip [--evidence-root <dir>]
  mgc submission build  --drop-id <id> --db <db> --out submission.zip [--evidence-root <dir>]
  mgc submission latest --db <db> --out submission.zip [--evidence-root <dir>]
  mgc submission verify --zip <submission.zip>

Design goals:
- No DB mutation.
- Deterministic ZIP output: stable ordering + stable timestamps + stable manifest bytes.
- Validate bundle schema before packaging (mgc.bundle_validate.validate_bundle).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from mgc.bundle_validate import validate_bundle


# -----------------------------
# Determinism helpers
# -----------------------------

_EPOCH_ZIP_DT = (1980, 1, 1, 0, 0, 0)  # stable ZIP timestamp


def _stable_zip_write(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    zi = zipfile.ZipInfo(filename=arcname, date_time=_EPOCH_ZIP_DT)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = (0o100644 & 0xFFFF) << 16  # rw-r--r--
    zi.create_system = 3  # force Unix
    zf.writestr(zi, data)


def _sorted_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    files.sort(key=lambda x: x.as_posix())
    return files


def _write_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _snapshot_bundle_bytes(bundle_dir: Path) -> Dict[str, bytes]:
    bundle_dir = bundle_dir.resolve()
    files = _sorted_files(bundle_dir)
    snap: Dict[str, bytes] = {}
    for f in files:
        rel = f.relative_to(bundle_dir).as_posix()
        snap[rel] = f.read_bytes()
    return snap


def _validate_bundle_without_mutating(bundle_dir: Path) -> None:
    bundle_dir = bundle_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="mgc_submission_validate_") as td:
        tmp_bundle = Path(td) / "drop_bundle"
        shutil.copytree(bundle_dir, tmp_bundle)
        validate_bundle(str(tmp_bundle))


# -----------------------------
# Bundle resolution
# -----------------------------

def _resolve_bundle_dir_from_evidence_root(evidence_root: Optional[str]) -> Optional[Path]:
    if not evidence_root:
        return None
    p = Path(evidence_root).expanduser().resolve()
    bd = p / "drop_bundle"
    return bd if bd.exists() and bd.is_dir() else None


def _db_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _db_latest_drop_id(con: sqlite3.Connection) -> str:
    for table in ("drops", "drop"):
        try:
            row = con.execute(f"SELECT drop_id FROM {table} ORDER BY ts DESC LIMIT 1").fetchone()
            if row and row["drop_id"]:
                return str(row["drop_id"])
        except sqlite3.Error:
            continue
    raise RuntimeError("Could not determine latest drop_id from DB (expected drops/drop table).")


def _db_bundle_dir_for_drop_id(con: sqlite3.Connection, drop_id: str, evidence_root: Optional[str]) -> Path:
    if evidence_root:
        er = Path(evidence_root).expanduser().resolve()
        cand = er / drop_id / "drop_bundle"
        if cand.exists() and cand.is_dir():
            return cand

    for table in ("drops", "drop"):
        try:
            row = con.execute(f"SELECT bundle_dir FROM {table} WHERE drop_id = ?", (drop_id,)).fetchone()
            if row and row["bundle_dir"]:
                bd = Path(str(row["bundle_dir"])).expanduser().resolve()
                if bd.exists() and bd.is_dir():
                    return bd
        except sqlite3.Error:
            continue

    raise RuntimeError("Could not resolve bundle_dir for drop_id (no evidence-root bundle and no DB bundle_dir).")


# -----------------------------
# Core operations
# -----------------------------

def build_submission_zip(bundle_dir: Path, out_zip: Path) -> Dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    out_zip = out_zip.expanduser().resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"bundle_dir not found: {bundle_dir}")

    # Snapshot before any validation/side effects
    snap = _snapshot_bundle_bytes(bundle_dir)
    rels = sorted(snap.keys())

    # Validate on a temp copy (prevent mutations of real bundle)
    _validate_bundle_without_mutating(bundle_dir)

    # Deterministic manifest:
    # - no timestamps
    # - no absolute paths
    # - no output-zip name (because CI builds submission.zip and submission_2.zip)
    manifest: Dict[str, Any] = {
        "format": "mgc_submission_manifest_v1",
        "bundle_root": "drop_bundle",
        "files": rels,
    }

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w") as zf:
        for rel in rels:
            _stable_zip_write(zf, rel, snap[rel])
        _stable_zip_write(zf, "submission_manifest.json", _write_json_bytes(manifest))

    return {"ok": True, "out": str(out_zip), "file_count": len(rels)}


def verify_submission_zip(zip_path: Path) -> Dict[str, Any]:
    zip_path = zip_path.expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        ok = "submission_manifest.json" in names
        return {"ok": bool(ok), "zip": str(zip_path), "entries": len(names)}


# -----------------------------
# CLI handlers
# -----------------------------

def cmd_submission_build(args: argparse.Namespace) -> int:
    bundle_dir: Optional[Path] = None

    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    else:
        bundle_dir = _resolve_bundle_dir_from_evidence_root(args.evidence_root)

    if bundle_dir is None and args.drop_id:
        if not args.db:
            raise SystemExit("submission build: --drop-id requires --db")
        con = _db_connect(args.db)
        try:
            bundle_dir = _db_bundle_dir_for_drop_id(con, args.drop_id, args.evidence_root)
        finally:
            con.close()

    if bundle_dir is None:
        raise SystemExit("submission build: need --bundle-dir OR (--drop-id and --db) OR evidence-root with drop_bundle")

    out = Path(args.out).expanduser().resolve()
    res = build_submission_zip(bundle_dir=bundle_dir, out_zip=out)
    if getattr(args, "json", False):
        print(json.dumps(res, sort_keys=True))
    return 0


def cmd_submission_latest(args: argparse.Namespace) -> int:
    if not args.db:
        raise SystemExit("submission latest: requires --db")

    con = _db_connect(args.db)
    try:
        drop_id = _db_latest_drop_id(con)
        bundle_dir = _db_bundle_dir_for_drop_id(con, drop_id, args.evidence_root)
    finally:
        con.close()

    out = Path(args.out).expanduser().resolve()
    res = build_submission_zip(bundle_dir=bundle_dir, out_zip=out)
    res["drop_id"] = drop_id
    if getattr(args, "json", False):
        print(json.dumps(res, sort_keys=True))
    return 0


def cmd_submission_verify(args: argparse.Namespace) -> int:
    res = verify_submission_zip(Path(args.zip))
    if getattr(args, "json", False):
        print(json.dumps(res, sort_keys=True))
    else:
        if not res["ok"]:
            print(json.dumps(res, sort_keys=True))
            return 2
    return 0


# -----------------------------
# Registrar
# -----------------------------

def register_submission_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("submission", help="Build deterministic submission ZIPs")
    sp = p.add_subparsers(dest="submission_cmd", required=True)

    b = sp.add_parser("build", help="Build a submission ZIP from a drop bundle")
    b.add_argument("--bundle-dir", default=None, help="Path to drop_bundle directory")
    b.add_argument("--drop-id", default=None, help="Drop id to resolve bundle via DB/evidence-root")
    b.add_argument("--out", required=True, help="Output ZIP path")
    b.add_argument("--evidence-root", default=None, help="Evidence root (may contain drop_bundle)")
    b.set_defaults(fn=cmd_submission_build)

    l = sp.add_parser("latest", help="Build submission ZIP for latest drop in DB")
    l.add_argument("--db", required=True, help="SQLite DB path")
    l.add_argument("--out", required=True, help="Output ZIP path")
    l.add_argument("--evidence-root", default=None, help="Evidence root")
    l.set_defaults(fn=cmd_submission_latest)

    v = sp.add_parser("verify", help="Verify a submission ZIP structure")
    v.add_argument("--zip", required=True, help="ZIP path to verify")
    v.set_defaults(fn=cmd_submission_verify)
