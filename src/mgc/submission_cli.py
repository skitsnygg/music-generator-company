#!/usr/bin/env python3
"""\
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

Extension:
- If marketing receipts exist for the bundle, include them deterministically under:
    marketing/receipts/...

  Resolution order:
  1) If drop_evidence.json exists and declares paths.marketing_receipts_dir, use it.
  2) Otherwise, infer sibling directory <bundle_dir_parent>/marketing/receipts if present.

This keeps older bundles working while allowing "release package" submission zips to
carry promotion artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _snapshot_dir_bytes(root: Path) -> Dict[str, bytes]:
    """Return a mapping of relative-posix-path -> file bytes for a directory."""
    root = root.resolve()
    files = _sorted_files(root)
    snap: Dict[str, bytes] = {}
    for f in files:
        rel = f.relative_to(root).as_posix()
        snap[rel] = f.read_bytes()
    return snap


def _validate_bundle_without_mutating(bundle_dir: Path) -> None:
    bundle_dir = bundle_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="mgc_submission_validate_") as td:
        tmp_bundle = Path(td) / "drop_bundle"
        shutil.copytree(bundle_dir, tmp_bundle)
        validate_bundle(str(tmp_bundle))


def _resolve_marketing_receipts_root(bundle_dir: Path) -> Tuple[Optional[Path], bool, Optional[str]]:
    """Return (receipts_root, inferred, source).

    inferred=True means we did not find an explicit declaration in drop_evidence.json.
    source is a short string for the manifest.
    """
    bundle_dir = bundle_dir.resolve()

    declared: Optional[str] = None
    drop_evidence = bundle_dir / "drop_evidence.json"
    if drop_evidence.exists() and drop_evidence.is_file():
        try:
            obj = json.loads(drop_evidence.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                paths = obj.get("paths")
                if isinstance(paths, dict):
                    v = paths.get("marketing_receipts_dir")
                    if isinstance(v, str) and v.strip():
                        declared = v.strip()
        except Exception:
            declared = None

    # 1) declared path
    if declared:
        p = Path(declared)
        if p.is_absolute():
            cand = p
            if cand.exists() and cand.is_dir():
                return (cand.resolve(), False, "drop_evidence.paths.marketing_receipts_dir")
            return (None, False, "drop_evidence.paths.marketing_receipts_dir")

        # Try relative to bundle_dir first, then bundle_dir.parent (weekly layout)
        cand1 = (bundle_dir / p).resolve()
        if cand1.exists() and cand1.is_dir():
            return (cand1, False, "drop_evidence.paths.marketing_receipts_dir")
        cand2 = (bundle_dir.parent / p).resolve()
        if cand2.exists() and cand2.is_dir():
            return (cand2, False, "drop_evidence.paths.marketing_receipts_dir")
        return (None, False, "drop_evidence.paths.marketing_receipts_dir")

    # 2) inferred sibling default
    inferred_root = (bundle_dir.parent / "marketing" / "receipts").resolve()
    if inferred_root.exists() and inferred_root.is_dir():
        return (inferred_root, True, "sibling:marketing/receipts")

    return (None, True, None)


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
    bundle_snap = _snapshot_dir_bytes(bundle_dir)
    bundle_rels = sorted(bundle_snap.keys())

    receipts_root, receipts_inferred, receipts_source = _resolve_marketing_receipts_root(bundle_dir)
    receipts_snap: Dict[str, bytes] = {}
    receipts_rels: list[str] = []
    if receipts_root is not None:
        receipts_snap = _snapshot_dir_bytes(receipts_root)
        receipts_rels = sorted(receipts_snap.keys())

    # Validate on a temp copy (prevent mutations of real bundle)
    _validate_bundle_without_mutating(bundle_dir)

    # Deterministic manifest:
    # - no timestamps
    # - no absolute paths
    # - no output-zip name
    manifest: Dict[str, Any] = {
        "format": "mgc_submission_manifest_v1",
        "bundle_root": "drop_bundle",
        "files": bundle_rels,
    }

    if receipts_root is not None:
        manifest["extra_roots"] = {
            "marketing_receipts": {
                "root": "marketing/receipts",
                "files": receipts_rels,
                "inferred": bool(receipts_inferred),
                "source": receipts_source,
            }
        }

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w") as zf:
        # Bundle files at zip root (existing behavior)
        for rel in bundle_rels:
            _stable_zip_write(zf, rel, bundle_snap[rel])

        # Optional marketing receipts under marketing/receipts
        if receipts_root is not None:
            for rel in receipts_rels:
                arc = f"marketing/receipts/{rel}"
                _stable_zip_write(zf, arc, receipts_snap[rel])

        _stable_zip_write(zf, "submission_manifest.json", _write_json_bytes(manifest))

    res: Dict[str, Any] = {"ok": True, "out": str(out_zip), "file_count": len(bundle_rels)}
    if receipts_root is not None:
        res["marketing_receipts"] = {
            "included": True,
            "count": len(receipts_rels),
            "inferred": bool(receipts_inferred),
        }
    else:
        res["marketing_receipts"] = {"included": False, "count": 0, "inferred": True}
    return res


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
    return 0 if bool(res.get("ok")) else 2


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
    return 0 if bool(res.get("ok")) else 2


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


# -----------------------------
# Standalone entrypoint
# -----------------------------

def _build_standalone_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mgc.submission_cli",
        description="Deterministic submission bundle packaging (standalone).",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build", help="Build submission ZIP from a bundle directory or a DB-resolved drop")
    b.add_argument("--bundle-dir", default=None, help="Drop bundle directory (may contain drop_bundle)")
    b.add_argument("--drop-id", default=None, help="Drop ID (requires --db)")
    b.add_argument("--db", default=None, help="SQLite DB path (required with --drop-id)")
    b.add_argument("--out", required=True, help="Output ZIP path")
    b.add_argument("--evidence-root", default=None, help="Evidence root")
    b.add_argument("--json", action="store_true", help="Emit JSON result")
    b.set_defaults(fn=cmd_submission_build)

    l = sp.add_parser("latest", help="Build submission ZIP for latest drop in DB")
    l.add_argument("--db", required=True, help="SQLite DB path")
    l.add_argument("--out", required=True, help="Output ZIP path")
    l.add_argument("--evidence-root", default=None, help="Evidence root")
    l.add_argument("--json", action="store_true", help="Emit JSON result")
    l.set_defaults(fn=cmd_submission_latest)

    v = sp.add_parser("verify", help="Verify a submission ZIP structure")
    v.add_argument("--zip", required=True, help="ZIP path to verify")
    v.add_argument("--json", action="store_true", help="Emit JSON result")
    v.set_defaults(fn=cmd_submission_verify)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_standalone_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.fn(args))
    except SystemExit:
        raise
    except Exception as e:
        # Don't mask failures: return non-zero and emit a minimal JSON error if requested.
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(e)}, sort_keys=True))
        else:
            raise
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
