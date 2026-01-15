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
  mgc submission verify  --zip submission.zip [--strict]

Notes:
- If mgc.bundle_validate.validate_bundle is available, `submission verify` will *attempt* to use it.
- By default verification is permissive: it ensures the ZIP contains the core portable artifacts
  (tracks + playlist + some evidence JSON). This is the "submission.zip is usable" bar.
- Use `--strict` to require the bundle validator to pass, if/when your bundle schema is stricter.
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
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files


def _zip_write_file(zf: zipfile.ZipFile, *, arcname: str, src_path: Path) -> None:
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
    evidence_root = evidence_root.resolve()
    drop_evidence = evidence_root / "drop_evidence.json"
    bundle_dir = evidence_root / "drop_bundle"
    use_bundle = bundle_dir if bundle_dir.exists() and bundle_dir.is_dir() else evidence_root
    return use_bundle, (drop_evidence if drop_evidence.exists() else None)


def _stage_submission_tree(*, bundle_dir: Path, drop_evidence: Optional[Path]) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="mgc_submission_"))
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
    evidence_dir = evidence_dir.resolve()
    if not evidence_dir.exists() or not evidence_dir.is_dir():
        raise SystemExit(f"submission latest: evidence dir not found: {evidence_dir}")

    best_path: Optional[Path] = None
    best_mtime: float = -1.0

    direct = evidence_dir / "drop_evidence.json"
    if direct.exists() and direct.is_file():
        best_path = direct
        best_mtime = direct.stat().st_mtime

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
# Validation
# ---------------------------------------------------------------------------

def _extract_notes_from_validator_dict(d: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    for key in ("errors", "error", "problems", "problem", "missing", "warnings", "warning", "details"):
        if key not in d:
            continue
        v = d.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            notes.append(f"{key}:{v}")
        elif isinstance(v, (list, tuple)):
            for item in v:
                notes.append(f"{key}:{item}")
        elif isinstance(v, dict):
            # summarize shallowly to avoid huge dumps
            for k2, v2 in v.items():
                notes.append(f"{key}.{k2}:{v2}")
        else:
            notes.append(f"{key}:{repr(v)}")
    return notes


def _minimal_validate(bundle_dir: Path) -> Dict[str, Any]:
    """
    Permissive "is this zip usable?" checks.
    """
    notes: List[str] = []
    ok = True

    tracks_ok = (bundle_dir / "tracks").exists() or (bundle_dir / "data" / "tracks").exists()
    if not tracks_ok:
        ok = False
        notes.append("missing:tracks_dir (tracks/ or data/tracks/)")

    playlist_ok = False
    if (bundle_dir / "playlist.json").exists():
        playlist_ok = True
    if (bundle_dir / "playlists").exists():
        if any(p.is_file() and p.suffix == ".json" for p in (bundle_dir / "playlists").rglob("*.json")):
            playlist_ok = True
    if not playlist_ok:
        ok = False
        notes.append("missing:playlist (playlist.json or playlists/*.json)")

    evidence_ok = any(
        (bundle_dir / name).exists()
        for name in ("daily_evidence.json", "drop_evidence.json", "manifest.json", "weekly_manifest.json")
    )
    if not evidence_ok:
        ok = False
        notes.append("missing:evidence_json (daily_evidence.json/drop_evidence.json/manifest.json)")

    return {"ok": ok, "validator": "minimal", "notes": notes}


def _try_validate_bundle(bundle_dir: Path, *, strict: bool) -> Dict[str, Any]:
    """
    Try mgc.bundle_validate.validate_bundle if available.
    If it fails without diagnostics, fall back to minimal validation unless --strict.
    """
    minimal = _minimal_validate(bundle_dir)

    try:
        from mgc.bundle_validate import validate_bundle  # type: ignore
        res = validate_bundle(str(bundle_dir))

        if isinstance(res, dict):
            v_ok = bool(res.get("ok", False))
            v_notes = _extract_notes_from_validator_dict(res)

            # If validator provides no actionable diagnostics, treat it as a soft signal unless strict.
            if (not v_ok) and (len(v_notes) == 0) and bool(minimal.get("ok", False)) and (not strict):
                return {
                    "ok": True,
                    "validator": "minimal (bundle_validate soft-fail)",
                    "notes": ["bundle_validate returned ok=false without details; passing minimal checks"],
                    "details": res,
                }

            # Merge notes
            merged: List[str] = []
            for n in (minimal.get("notes", []) or []) + v_notes:
                if n and n not in merged:
                    merged.append(str(n))

            if strict:
                return {"ok": v_ok, "validator": "mgc.bundle_validate", "notes": merged, "details": res}

            # permissive mode: require minimal OK, and treat validator failure as warning unless it includes notes
            if v_ok:
                return {"ok": True, "validator": "mgc.bundle_validate", "notes": merged, "details": res}

            if bool(minimal.get("ok", False)):
                warn = ["bundle_validate failed; passing minimal checks"] + merged
                return {"ok": True, "validator": "minimal+warning", "notes": warn, "details": res}

            # minimal already failed -> fail
            return {"ok": False, "validator": "minimal+bundle_validate", "notes": merged, "details": res}

        # Non-dict result (bool / truthy)
        v_ok = bool(res)
        if strict:
            return {"ok": v_ok, "validator": "mgc.bundle_validate", "notes": minimal.get("notes", [])}
        if v_ok:
            return {"ok": True, "validator": "mgc.bundle_validate", "notes": minimal.get("notes", [])}
        # permissive: allow if minimal passes
        if bool(minimal.get("ok", False)):
            return {"ok": True, "validator": "minimal (bundle_validate falsey)", "notes": ["bundle_validate returned falsey; passing minimal checks"]}
        return {"ok": False, "validator": "minimal+bundle_validate", "notes": minimal.get("notes", [])}
    except Exception as e:
        # Validator missing/errored -> use minimal
        out = dict(minimal)
        out["notes"] = (out.get("notes", []) or []) + [f"validator_unavailable:{type(e).__name__}"]
        return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _build_zip_from_bundle_or_evidence(
    *,
    bundle_dir: Optional[Path],
    evidence_root: Optional[Path],
    out_zip: Path,
    include_drop_evidence: bool,
    json_out: bool,
) -> int:
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
        print(json.dumps({"ok": True, "out": str(out_zip), "bundle_dir": str(bundle_dir), "evidence_root": str(evidence_root) if evidence_root else None}, sort_keys=True))
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

        bundle_dir = tmp / "drop_bundle"
        if not bundle_dir.exists():
            bundle_dir = tmp

        res = _try_validate_bundle(bundle_dir, strict=bool(getattr(args, "strict", False)))
        ok = bool(res.get("ok", False))

        if getattr(args, "json", False):
            print(json.dumps({"ok": ok, "zip": str(zpath), "bundle_dir": str(bundle_dir), "validation": res}, indent=2, sort_keys=True))
        else:
            status = "ok" if ok else "FAIL"
            print(f"[submission.verify] {status} zip={zpath}")
            if (not ok) or (res.get("notes") and getattr(args, "verbose", False)):
                for n in res.get("notes", []) or []:
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
    b.add_argument("--bundle-dir", default=None, help="Directory containing bundle files (playlist.json, tracks/...)")
    b.add_argument("--evidence-root", default=None, help="Run output dir containing drop_evidence.json and optionally drop_bundle/")
    b.add_argument("--include-drop-evidence", action="store_true", default=True, help="Include drop_evidence.json when using --evidence-root (default: true)")
    b.add_argument("--out", required=True, help="Output ZIP path")
    b.set_defaults(func=cmd_submission_build)

    l = sp.add_parser("latest", help="Build a submission ZIP from the newest drop_evidence.json under an evidence directory")
    l.add_argument("--evidence-dir", default="data/evidence", help="Directory to search for newest drop_evidence.json (default: data/evidence)")
    l.add_argument("--out", required=True, help="Output ZIP path")
    l.set_defaults(func=cmd_submission_latest)

    v = sp.add_parser("verify", help="Verify a submission ZIP")
    v.add_argument("--zip", required=True, help="Path to submission.zip")
    v.add_argument("--strict", action="store_true", help="Require mgc.bundle_validate to pass (if available)")
    v.add_argument("--verbose", action="store_true", help="Print notes even on success")
    v.set_defaults(func=cmd_submission_verify)


__all__ = ["register_submission_subcommand"]
