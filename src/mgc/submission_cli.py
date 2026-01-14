#!/usr/bin/env python3
"""
src/mgc/submission_cli.py

Build a submission ZIP for a specific drop bundle.

Design goals:
- No DB mutation.
- Deterministic outputs (stable JSON/README, stable zip file ordering + timestamps).
- Bundle validation before packaging.
- Works with either:
    (a) explicit --bundle-dir, or
    (b) --drop-id + DB lookup (best-effort).

Expected bundle layout (v1):
  <bundle>/
    tracks/
      <track_id>.<ext>
    playlist.json
    daily_evidence.json
    daily_evidence_<drop_id>.json (optional)

CLI:
  mgc submission build --bundle-dir <dir> --out submission.zip
  mgc submission build --drop-id <drop_id> --db data/db.sqlite --out submission.zip
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mgc.bundle_validate import validate_bundle


# -----------------------------
# Utilities
# -----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _posix_rel(p: Path) -> str:
    return str(p).replace("\\", "/")


def _zip_add_dir(zf: zipfile.ZipFile, root: Path, arc_root: str) -> None:
    """
    Add directory contents to zip with deterministic ordering and fixed timestamps.
    """
    root = root.resolve()
    entries = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        fn.sort()
        dp_path = Path(dp)
        for name in fn:
            file_path = (dp_path / name).resolve()
            rel = file_path.relative_to(root)
            arcname = f"{arc_root}/{_posix_rel(rel)}"
            entries.append((arcname, file_path))

    # Deterministic ordering
    entries.sort(key=lambda x: x[0])

    fixed_dt = (2020, 1, 1, 0, 0, 0)

    for arcname, file_path in entries:
        data = file_path.read_bytes()
        zi = zipfile.ZipInfo(filename=arcname, date_time=fixed_dt)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16  # permissions
        zf.writestr(zi, data)


def _find_bundle_dir_by_drop_id(con: sqlite3.Connection, drop_id: str) -> Optional[Path]:
    """
    Best-effort lookup: tries to find a usable bundle directory for a drop_id.

    This is intentionally conservative because repo layouts vary. We check common patterns:
      - drops table contains evidence_path / bundle_dir / out_dir columns (if present)
      - evidence JSON path referenced in drops.meta (if you store it)
      - fallback: data/evidence/daily_evidence_<drop_id>.json (common)
    """
    drop_id = drop_id.strip()
    if not drop_id:
        return None

    # 1) Try reading drops.meta for an evidence/bundle hint.
    try:
        cur = con.execute("SELECT meta FROM drops WHERE id = ? OR drop_id = ? LIMIT 1", (drop_id, drop_id))
        row = cur.fetchone()
        if row and row[0]:
            # meta may be JSON string
            import json
            try:
                meta = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                if isinstance(meta, dict):
                    for k in ("bundle_dir", "out_dir", "evidence_dir", "evidence_path"):
                        v = meta.get(k)
                        if isinstance(v, str) and v:
                            p = Path(v)
                            if p.is_file():
                                # evidence_path -> bundle dir is parent
                                bd = p.parent
                                if (bd / "playlist.json").exists() and (bd / "daily_evidence.json").exists():
                                    return bd.resolve()
                            if p.is_dir():
                                if (p / "playlist.json").exists() and (p / "daily_evidence.json").exists():
                                    return p.resolve()
            except Exception:
                pass
    except Exception:
        # drops schema may differ; ignore
        pass

    # 2) Common default evidence location: data/evidence/daily_evidence_<drop_id>.json
    candidate = Path.cwd() / "data" / "evidence" / f"daily_evidence_{drop_id}.json"
    if candidate.exists() and candidate.is_file():
        bd = candidate.parent
        # In some layouts evidence might be separate; still attempt parent-as-bundle only if required files exist.
        if (bd / "playlist.json").exists() and (bd / "daily_evidence.json").exists() and (bd / "tracks").is_dir():
            return bd.resolve()

    # 3) Nothing found
    return None


def _build_readme(bundle_dir: Path, evidence_obj: Dict[str, Any]) -> str:
    ids = evidence_obj.get("ids") if isinstance(evidence_obj.get("ids"), dict) else {}
    paths = evidence_obj.get("paths") if isinstance(evidence_obj.get("paths"), dict) else {}
    sha = evidence_obj.get("sha256") if isinstance(evidence_obj.get("sha256"), dict) else {}

    drop_id = ids.get("drop_id", "")
    run_id = ids.get("run_id", "")
    track_id = ids.get("track_id", "")
    provider = evidence_obj.get("provider", "")
    context = evidence_obj.get("context", "")
    ts = evidence_obj.get("ts", "")
    deterministic = evidence_obj.get("deterministic", "")

    bundle_track = paths.get("bundle_track", "")
    playlist = paths.get("playlist", "playlist.json")

    return "\n".join(
        [
            "# Music Generator Company – Drop Submission",
            "",
            "## Identifiers",
            f"- drop_id: {drop_id}",
            f"- run_id: {run_id}",
            f"- track_id: {track_id}",
            "",
            "## Run metadata",
            f"- ts: {ts}",
            f"- context: {context}",
            f"- provider: {provider}",
            f"- deterministic: {deterministic}",
            "",
            "## Contents",
            f"- {playlist}: playlist pointing at bundled audio",
            f"- {bundle_track}: bundled audio asset",
            "- daily_evidence.json: provenance + sha256 hashes",
            "",
            "## How to review",
            "1) Open playlist.json in the static web player (if included) OR inspect it directly.",
            "2) Confirm hashes in daily_evidence.json match the files in the bundle.",
            "",
            "## Notes",
            f"- Generated at: {_utc_now_iso()}",
        ]
    ) + "\n"


# -----------------------------
# Command implementation
# -----------------------------

def cmd_submission_build(args: argparse.Namespace) -> int:
    out_path = Path(args.out).expanduser().resolve()
    _safe_mkdir(out_path.parent)

    bundle_dir: Optional[Path] = None

    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir).expanduser().resolve()
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            raise SystemExit(f"Bundle dir not found: {bundle_dir}")
    else:
        if not args.drop_id:
            raise SystemExit("Either --bundle-dir or --drop-id is required.")
        db_path = Path(args.db).expanduser().resolve()
        if not db_path.exists():
            raise SystemExit(f"DB not found: {db_path}")
        con = sqlite3.connect(str(db_path))
        try:
            bundle_dir = _find_bundle_dir_by_drop_id(con, args.drop_id)
        finally:
            con.close()
        if bundle_dir is None:
            raise SystemExit(
                "Could not locate bundle dir from --drop-id. "
                "Re-run with --bundle-dir pointing at the portable bundle directory."
            )

    # Validate bundle before packaging
    validate_bundle(bundle_dir)

    # Load evidence for README
    import json
    evidence_main = bundle_dir / "daily_evidence.json"
    evidence_obj = json.loads(_read_text(evidence_main))

    # Build staging directory
    with tempfile.TemporaryDirectory(prefix="mgc_submission_") as td:
        stage = Path(td).resolve()
        pkg_root = stage / "submission"
        _safe_mkdir(pkg_root)

        # Copy bundle into submission/drop_bundle
        drop_bundle_dst = pkg_root / "drop_bundle"
        shutil.copytree(bundle_dir, drop_bundle_dst)

        # Write README.md
        readme_text = _build_readme(bundle_dir, evidence_obj)
        (pkg_root / "README.md").write_text(readme_text, encoding="utf-8")

        # Optional: include web build directory if user points at one
        if args.web_dir:
            web_src = Path(args.web_dir).expanduser().resolve()
            if not web_src.exists() or not web_src.is_dir():
                raise SystemExit(f"--web-dir not found: {web_src}")
            shutil.copytree(web_src, pkg_root / "web")

        # Build zip deterministically
        if out_path.exists():
            out_path.unlink()

        with zipfile.ZipFile(str(out_path), mode="w") as zf:
            # Add README first for nicer browsing order
            fixed_dt = (2020, 1, 1, 0, 0, 0)
            zi = zipfile.ZipInfo(filename="submission/README.md", date_time=fixed_dt)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, (pkg_root / "README.md").read_text(encoding="utf-8"))

            _zip_add_dir(zf, drop_bundle_dst, arc_root="submission/drop_bundle")
            if (pkg_root / "web").exists():
                _zip_add_dir(zf, pkg_root / "web", arc_root="submission/web")

    if getattr(args, "json", False):
        import json as _json
        print(
            _json.dumps(
                {
                    "ok": True,
                    "out": str(out_path),
                    "bundle_dir": str(bundle_dir),
                    "included_web": bool(args.web_dir),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        print(f"[submission] wrote: {out_path}")
        print(f"[submission] bundle: {bundle_dir}")
        if args.web_dir:
            print(f"[submission] web: {Path(args.web_dir).expanduser().resolve()}")

    return 0


def register_submission_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """
    Hook this into main.py's parser wiring.

    Example (in main.py):
        sub = parser.add_subparsers(dest="cmd", required=True)
        ...
        register_submission_subcommand(sub)
    """
    p = subparsers.add_parser("submission", help="Build submission bundles")
    sp = p.add_subparsers(dest="submission_cmd", required=True)

    b = sp.add_parser("build", help="Build a submission zip for a drop bundle")
    b.add_argument("--out", required=True, help="Output zip path (e.g. submission.zip)")

    # Choose one:
    b.add_argument("--bundle-dir", help="Path to an existing portable bundle directory")
    b.add_argument("--drop-id", help="Drop id to locate bundle from DB (best-effort)")
    b.add_argument("--db", default="data/db.sqlite", help="DB path for --drop-id lookup")

    # Optional extras
    b.add_argument("--web-dir", help="Optional static web build directory to include under submission/web")

    # If your CLI supports global --json, we still accept a local flag too
    b.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")

    b.set_defaults(fn=cmd_submission_build)
