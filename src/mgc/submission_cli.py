#!/usr/bin/env python3
"""
src/mgc/submission_cli.py

Submission packaging CLI.

Goals:
- Deterministic ZIP outputs (stable file ordering + fixed timestamps).
- Flexible inputs:
  - --bundle-dir <dir>            (points at a bundle directory)
  - --evidence-root <dir>         (points at a run output dir; auto-detects drop_bundle/ and drop_evidence.json)
  - latest: auto-find newest evidence root under an evidence directory

Commands:
  mgc submission build   --bundle-dir <dir> --out submission.zip
  mgc submission build   --evidence-root <run_out_dir> --out submission.zip
  mgc submission latest  [--evidence-dir <dir>] --out submission.zip
  mgc submission verify  --zip submission.zip

Notes:
- If mgc.bundle_validate.validate_bundle is available, `submission verify` will use it.
- This file intentionally keeps dependencies light so it works in CI and local dev.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Deterministic zip helpers
# ---------------------------------------------------------------------------

_FIXED_ZIP_DT = (1980, 1, 1, 0, 0, 0)


def _stable_sorted_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    # sort by POSIX-style relative path for cross-platform stability
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files


def _zip_write_file(zf: zipfile.ZipFile, *, arcname: str, src_path: Path) -> None:
    # zipfile.write() uses filesystem mtimes; we want deterministic timestamps
    info = zipfile.ZipInfo(filename=arcname, date_time=_FIXED_ZIP_DT)
    info.compress_type = zipfile.ZIP_DEFLATED
    with src_path.open("rb") as f:
        data = f.read()
    zf.writestr(info, data)


def _make_zip_deterministic(src_dir: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w") as zf:
        for p in _stable_sorted_files(src_dir):
            arc = p.relative_to(src_dir).as_posix()
            _zip_write_file(zf, arcname=arc, src_path=p)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def _resolve_bundle_from_evidence_root(evidence_root: Path) -> Tuple[Path, Optional[Path]]:
    """
    Given a run out_dir (evidence_root), determine:
      - bundle_dir: either evidence_root/drop_bundle if it exists, else evidence_root
      - drop_evidence: evidence_root/drop_evidence.json if present, else None
    """
    evidence_root = evidence_root.resolve()
    drop_evidence = evidence_root / "drop_evidence.json"
    bundle_dir = evidence_root / "drop_bundle"
    use_bundle = bundle_dir if bundle_dir.exists() and bundle_dir.is_dir() else evidence_root
    return use_bundle, (drop_evidence if drop_evidence.exists() else None)


def _stage_submission_tree(*, bundle_dir: Path, drop_evidence: Optional[Path]) -> Path:
    """
    Creates a temporary directory containing:
      - contents of bundle_dir at the root
      - drop_evidence.json at the root (if provided and not already present)
    Returns the temp dir path (caller must clean up).
    """
    tmp = Path(tempfile.mkdtemp(prefix="mgc_submission_"))
    # copy bundle contents
    for p in _stable_sorted_files(bundle_dir):
        rel = p.relative_to(bundle_dir)
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)

    if drop_evidence is not None:
        dst = tmp / "drop_evidence.json"
        if not dst.exists():
            shutil.copyfile(drop_evidence, dst)

    return tmp


def _find_latest_evidence_root(evidence_dir: Path) -> Path:
    """
    Find the most recently modified drop_evidence.json anywhere under evidence_dir.
    Returns the parent directory (the evidence root).
    """
    evidence_dir = evidence_dir.resolve()
    if not evidence_dir.exists() or not evidence_dir.is_dir():
        raise SystemExit(f"submission latest: evidence dir not found: {evidence_dir}")

    best_path: Optional[Path] = None
    best_mtime: float = -1.0

    # First: check if evidence_dir itself is a root
    direct = evidence_dir / "drop_evidence.json"
    if direct.exists() and direct.is_file():
        best_path = direct
        best_mtime = direct.stat().st_mtime

    # Then: scan subdirs
    for p in evidence_dir.rglob("drop_evidence.json"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > best_mtime:
            best_mtime = m
            best_path = p

    if best_path is None:
        raise SystemExit(f"submission latest: no drop_evidence.json found under {evidence_dir}")

    return best_path.parent


# ---------------------------------------------------------------------------
# Optional validation
# ---------------------------------------------------------------------------

def _try_validate_bundle(bundle_dir: Path) -> Dict[str, Any]:
    """
    If mgc.bundle_validate.validate_bundle exists, call it.
    Otherwise perform a minimal sanity check.
    """
    notes: List[str] = []
    ok = True

    req = ["playlist.json", "daily_evidence.json"]
    for name in req:
        if not (bundle_dir / name).exists():
            ok = False
            notes.append(f"missing:{name}")

    tracks_dir = bundle_dir / "tracks"
    if not tracks_dir.exists():
        ok = False
        notes.append("missing:tracks_dir")

    try:
        from mgc.bundle_validate import validate_bundle  # type: ignore
        res = validate_bundle(str(bundle_dir))
        if isinstance(res, dict):
            return {"ok": bool(res.get("ok", ok)), "validator": "mgc.bundle_validate", "details": res, "notes": notes}
        return {"ok": bool(res), "validator": "mgc.bundle_validate", "notes": notes}
    except Exception as e:
        notes.append(f"validator_unavailable:{type(e).__name__}")
        return {"ok": ok, "validator": "minimal", "notes": notes}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _build_zip_from_bundle_or_evidence(*, bundle_dir: Optional[Path], evidence_root: Optional[Path], out_zip: Path,
                                      include_drop_evidence: bool, json_out: bool) -> int:
    if bundle_dir is None and evidence_root is None:
        raise SystemExit("submission build: provide --bundle-dir or --evidence-root")

    drop_evidence: Optional[Path] = None
    if evidence_root is not None:
        use_bundle, drop_ev = _resolve_bundle_from_evidence_root(evidence_root)
        if bundle_dir is None:
            bundle_dir = use_bundle
        if include_drop_evidence:
            drop_evidence = drop_ev

    assert bundle_dir is not None
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise SystemExit(f"submission build: bundle dir not found: {bundle_dir}")

    tmp = _stage_submission_tree(bundle_dir=bundle_dir, drop_evidence=drop_evidence)
    try:
        _make_zip_deterministic(tmp, out_zip)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if json_out:
        print(json.dumps({"ok": True, "out": str(out_zip), "bundle_dir": str(bundle_dir), "evidence_root": str(evidence_root) if evidence_root else None},
                         sort_keys=True))
    else:
        print(f"[submission.build] ok out={out_zip}")
    return 0


def cmd_submission_build(args: argparse.Namespace) -> int:
    out_zip = Path(args.out).resolve()
    bundle_dir = Path(args.bundle_dir).resolve() if getattr(args, "bundle_dir", None) else None
    evidence_root = Path(args.evidence_root).resolve() if getattr(args, "evidence_root", None) else None
    return _build_zip_from_bundle_or_evidence(
        bundle_dir=bundle_dir,
        evidence_root=evidence_root,
        out_zip=out_zip,
        include_drop_evidence=bool(getattr(args, "include_drop_evidence", True)),
        json_out=bool(getattr(args, "json", False)),
    )


def cmd_submission_latest(args: argparse.Namespace) -> int:
    out_zip = Path(args.out).resolve()
    evidence_dir = Path(args.evidence_dir).resolve()
    evidence_root = _find_latest_evidence_root(evidence_dir)

    return _build_zip_from_bundle_or_evidence(
        bundle_dir=None,
        evidence_root=evidence_root,
        out_zip=out_zip,
        include_drop_evidence=True,
        json_out=bool(getattr(args, "json", False)),
    )


def cmd_submission_verify(args: argparse.Namespace) -> int:
    zpath = Path(args.zip).resolve()
    if not zpath.exists():
        raise SystemExit(f"submission verify: zip not found: {zpath}")

    tmp = Path(tempfile.mkdtemp(prefix="mgc_submission_verify_"))
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmp)

        # if the zip contains a drop_bundle/ folder, validate that; else validate root
        bundle_dir = tmp / "drop_bundle"
        if not bundle_dir.exists():
            bundle_dir = tmp

        res = _try_validate_bundle(bundle_dir)
        ok = bool(res.get("ok", False))

        if getattr(args, "json", False):
            out = {"ok": ok, "zip": str(zpath), "bundle_dir": str(bundle_dir), "validation": res}
            print(json.dumps(out, indent=2, sort_keys=True))
        else:
            status = "ok" if ok else "FAIL"
            print(f"[submission.verify] {status} zip={zpath}")
            if not ok:
                for n in res.get("notes", []):
                    print(f"  - {n}")

        return 0 if ok else 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Argparse registrar
# ---------------------------------------------------------------------------

def register_submission_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("submission", help="Build + verify deterministic submission ZIPs")
    sp = p.add_subparsers(dest="submission_cmd", required=True)

    b = sp.add_parser("build", help="Build a deterministic submission.zip from a bundle or evidence root")
    b.add_argument("--bundle-dir", default=None, help="Directory containing bundle files (playlist.json, daily_evidence.json, tracks/...)")
    b.add_argument(
        "--evidence-root",
        default=None,
        help="Run output directory containing drop_evidence.json and optionally drop_bundle/ (auto-detected)",
    )
    b.add_argument(
        "--include-drop-evidence",
        action="store_true",
        default=True,
        help="If --evidence-root is set and drop_evidence.json exists, include it in the ZIP (default: true)",
    )
    b.add_argument("--out", required=True, help="Output ZIP path (e.g. submission.zip)")
    b.set_defaults(func=cmd_submission_build)

    l = sp.add_parser("latest", help="Build a submission ZIP from the newest drop_evidence.json under an evidence directory")
    l.add_argument(
        "--evidence-dir",
        default="data/evidence",
        help="Directory to search for the newest drop_evidence.json (default: data/evidence)",
    )
    l.add_argument("--out", required=True, help="Output ZIP path (e.g. submission.zip)")
    l.set_defaults(func=cmd_submission_latest)

    v = sp.add_parser("verify", help="Verify a submission ZIP (and optionally validate bundle schema)")
    v.add_argument("--zip", required=True, help="Path to submission.zip")
    v.set_defaults(func=cmd_submission_verify)


__all__ = ["register_submission_subcommand"]
