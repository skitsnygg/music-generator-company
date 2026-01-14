#!/usr/bin/env python3
"""
src/mgc/submission_cli.py

Build + verify deterministic submission ZIPs.

Design goals:
- No DB mutation.
- Deterministic ZIP outputs (stable ordering + fixed timestamps + fixed compression).
- Validate bundle before packaging.
- Verify one or more submission.zip files (unzip + validate_bundle).
- Optional: embed a static web player build inside the submission ZIP (submission/web).

Commands:
  mgc submission build   --bundle-dir <dir> --out submission.zip [--with-web]
  mgc submission build   --drop-id <drop_id> --db data/db.sqlite --out submission.zip [--with-web]
  mgc submission latest  --db data/db.sqlite --out submission.zip [--evidence-root data/evidence] [--with-web]
  mgc submission verify  <zip1> [zip2 ...]
  mgc submission verify  --zip <zip1> --zip <zip2> ...

JSON behavior:
- This module does NOT define any per-subcommand --json flags.
- It relies exclusively on mgc.main's GLOBAL --json (args.json).
- In JSON mode, commands emit valid JSON to stdout and nothing else.

Notes on --with-web:
- Builds a static web bundle from the portable bundle’s playlist.json and bundled audio.
- Runs `mgc.main web build` in a subprocess with cwd=bundle_dir so relative paths resolve.
- Includes output under submission/web in the resulting ZIP.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

from mgc.bundle_validate import validate_bundle


# -----------------------------
# JSON + time utilities
# -----------------------------


def _json_mode(args: argparse.Namespace) -> bool:
    # Only global --json is honored (provided by mgc.main).
    return bool(getattr(args, "json", False))


def _json_dumps(obj: Any) -> str:
    # Deterministic, pipe-friendly JSON output (no external dependency).
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _emit_json(obj: Any) -> None:
    # JSON mode must always emit valid JSON to stdout.
    print(_json_dumps(obj))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _posix_rel(p: Path) -> str:
    return str(p).replace("\\", "/")


# -----------------------------
# ZIP utilities (deterministic)
# -----------------------------

_FIXED_ZIP_DT = (2020, 1, 1, 0, 0, 0)  # YYYY,MM,DD,hh,mm,ss
_ZIP_COMPRESSLEVEL = 9


def _zipinfo_for_file(arcname: str, *, is_executable: bool = False) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(filename=arcname, date_time=_FIXED_ZIP_DT)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.create_system = 3  # Unix

    # Stable permissions
    mode = 0o755 if is_executable else 0o644
    zi.external_attr = (mode & 0xFFFF) << 16
    return zi


def _zip_add_dir(zf: zipfile.ZipFile, root: Path, arc_root: str) -> None:
    """
    Add directory contents to zip with deterministic ordering and fixed timestamps.

    - stable file list (sorted arcname)
    - fixed ZipInfo date_time
    - fixed compression + compresslevel
    - normalized permissions
    """
    root = root.resolve()
    entries: list[tuple[str, Path, bool]] = []

    for dp, dn, fn in os.walk(root):
        dn.sort()
        fn.sort()
        dp_path = Path(dp)
        for name in fn:
            file_path = (dp_path / name).resolve()
            rel = file_path.relative_to(root)
            arcname = f"{arc_root}/{_posix_rel(rel)}"

            try:
                st = file_path.stat()
                is_exec = bool(st.st_mode & stat.S_IXUSR)
            except Exception:
                is_exec = False

            entries.append((arcname, file_path, is_exec))

    entries.sort(key=lambda x: x[0])  # deterministic by archive path

    for arcname, file_path, is_exec in entries:
        data = file_path.read_bytes()
        zi = _zipinfo_for_file(arcname, is_executable=is_exec)
        zf.writestr(zi, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=_ZIP_COMPRESSLEVEL)


def _zip_write_text(zf: zipfile.ZipFile, arcname: str, text: str) -> None:
    zi = _zipinfo_for_file(arcname, is_executable=False)
    zf.writestr(zi, text.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED, compresslevel=_ZIP_COMPRESSLEVEL)


# -----------------------------
# DB + bundle discovery helpers
# -----------------------------


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(r[1]) for r in rows}  # (cid, name, type, notnull, dflt_value, pk)
    except Exception:
        return set()


def _is_bundle_dir(p: Path) -> bool:
    return (
        p.is_dir()
        and (p / "tracks").is_dir()
        and (p / "playlist.json").is_file()
        and (p / "daily_evidence.json").is_file()
    )


def _find_bundle_dir_by_drop_id(con: sqlite3.Connection, drop_id: str) -> Optional[Path]:
    """
    Best-effort lookup: tries to find a usable bundle directory for a drop_id.

    Conservative heuristics:
    - drops.meta/meta_json may contain bundle_dir/out_dir/evidence_path/evidence_dir pointers
    - fallback: data/evidence/daily_evidence_<drop_id>.json (common pattern)
    """
    drop_id = drop_id.strip()
    if not drop_id:
        return None

    # 1) Try reading drops.meta for a hint
    try:
        if _table_exists(con, "drops"):
            cols = _columns(con, "drops")
            id_col = "id" if "id" in cols else ("drop_id" if "drop_id" in cols else None)
            meta_col = "meta" if "meta" in cols else ("meta_json" if "meta_json" in cols else None)
            if id_col and meta_col:
                cur = con.execute(f"SELECT {meta_col} FROM drops WHERE {id_col} = ? LIMIT 1", (drop_id,))
                row = cur.fetchone()
                if row and row[0]:
                    try:
                        meta = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                        if isinstance(meta, dict):
                            for k in ("bundle_dir", "out_dir", "evidence_dir", "evidence_path"):
                                v = meta.get(k)
                                if isinstance(v, str) and v:
                                    p = Path(v).expanduser()
                                    if p.is_file():
                                        bd = p.parent
                                        if _is_bundle_dir(bd):
                                            return bd.resolve()
                                    if p.is_dir():
                                        if _is_bundle_dir(p):
                                            return p.resolve()
                    except Exception:
                        pass
    except Exception:
        pass

    # 2) Common default evidence location
    candidate = Path.cwd() / "data" / "evidence" / f"daily_evidence_{drop_id}.json"
    if candidate.exists() and candidate.is_file():
        bd = candidate.parent
        if _is_bundle_dir(bd):
            return bd.resolve()

    return None


def _latest_drop_id(con: sqlite3.Connection) -> Optional[str]:
    """
    Best-effort: fetch the latest drop id from drops table.
    Handles column drift: id vs drop_id, created_at vs ts vs created_ts.
    """
    if not _table_exists(con, "drops"):
        return None

    cols = _columns(con, "drops")
    id_col = "id" if "id" in cols else ("drop_id" if "drop_id" in cols else None)
    if not id_col:
        return None

    ts_col = None
    for c in ("created_at", "ts", "created_ts", "published_ts"):
        if c in cols:
            ts_col = c
            break

    try:
        if ts_col:
            row = con.execute(
                f"SELECT {id_col} AS did FROM drops ORDER BY {ts_col} DESC LIMIT 1"
            ).fetchone()
        else:
            row = con.execute(
                f"SELECT {id_col} AS did FROM drops ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        return None

    return None


def _find_latest_bundle_dir_by_scan(evidence_root: Path) -> Optional[Path]:
    """
    Scan evidence_root for the most recently modified directory that looks like a bundle.
    Only scans 2 levels deep.
    """
    evidence_root = evidence_root.resolve()
    if not evidence_root.exists() or not evidence_root.is_dir():
        return None

    candidates: list[Path] = []

    if _is_bundle_dir(evidence_root):
        candidates.append(evidence_root)

    for child in sorted(evidence_root.iterdir()):
        if child.is_dir() and _is_bundle_dir(child):
            candidates.append(child)
        if child.is_dir():
            for grand in sorted(child.iterdir()):
                if grand.is_dir() and _is_bundle_dir(grand):
                    candidates.append(grand)

    if not candidates:
        return None

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0

    candidates.sort(key=mtime, reverse=True)
    return candidates[0].resolve()


# -----------------------------
# README + packaging
# -----------------------------


def _build_readme(evidence_obj: Dict[str, Any], *, included_web: bool) -> str:
    ids = evidence_obj.get("ids") if isinstance(evidence_obj.get("ids"), dict) else {}
    paths = evidence_obj.get("paths") if isinstance(evidence_obj.get("paths"), dict) else {}

    drop_id = ids.get("drop_id", "")
    run_id = ids.get("run_id", "")
    track_id = ids.get("track_id", "")
    provider = evidence_obj.get("provider", "")
    context = evidence_obj.get("context", "")
    ts = evidence_obj.get("ts", "")
    deterministic = evidence_obj.get("deterministic", "")

    bundle_track = paths.get("bundle_track_path", paths.get("bundle_track", ""))
    playlist = paths.get("playlist_path", paths.get("playlist", "playlist.json"))

    # Determinism: never use wall-clock time here.
    packaged_at = ts or "2020-01-01T00:00:00+00:00"

    lines = [
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
    ]

    if included_web:
        lines += [
            "- web/: static web player bundle (index.html + tracks/)",
        ]

    lines += [
        "",
        "## How to review",
        "1) Inspect playlist.json (it references the bundled track under tracks/).",
        "2) Confirm hashes in daily_evidence.json match the files in the bundle.",
    ]

    if included_web:
        lines += [
            "3) Open web/index.html (or serve web/ with any static server) to listen.",
        ]

    lines += [
        "",
        "## Notes",
        f"- Packaged at: {packaged_at}",
    ]

    return "\n".join(lines) + "\n"


def _build_web_bundle_from_portable_bundle(
    *,
    bundle_dir: Path,
    web_out_dir: Path,
) -> Dict[str, Any]:
    """
    Build a web bundle from a *portable bundle dir* using mgc.main web build.

    Critical detail:
    - We run with cwd=bundle_dir so any relative audio paths inside playlist.json
      resolve against the bundle directory (not the repo root).
    """
    playlist_path = bundle_dir / "playlist.json"
    if not playlist_path.exists():
        raise FileNotFoundError(f"bundle missing playlist.json: {playlist_path}")

    # Use the same python running this process
    cmd = [
        sys.executable,
        "-m",
        "mgc.main",
        "web",
        "build",
        "--playlist",
        str(playlist_path),
        "--out-dir",
        str(web_out_dir),
        "--clean",
        "--fail-if-empty",
        "--strip-paths",  # NEW: makes playlist.json + web_manifest.json portable/deterministic
        "--json",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(bundle_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )

    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0:
        msg = proc.stderr.strip() or "web build failed"
        raise RuntimeError(f"web build failed (rc={proc.returncode}): {msg}")

    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        raise RuntimeError(
            "web build did not emit valid JSON. "
            f"stdout={stdout[:200]!r} stderr={(proc.stderr or '')[:200]!r}"
        )

    if not isinstance(payload, dict) or not payload.get("ok", False):
        raise RuntimeError(f"web build returned ok=false: {payload}")

    copied = int(payload.get("copied_count", 0) or 0)
    if copied < 1:
        raise RuntimeError(f"web build copied_count={copied} (expected >= 1): {payload}")

    idx = web_out_dir / "index.html"
    pl = web_out_dir / "playlist.json"
    tr = web_out_dir / "tracks"
    if not idx.exists():
        raise FileNotFoundError(f"web bundle missing index.html: {idx}")
    if not pl.exists():
        raise FileNotFoundError(f"web bundle missing playlist.json: {pl}")
    if not tr.exists() or not tr.is_dir():
        raise FileNotFoundError(f"web bundle missing tracks/: {tr}")

    return payload


def _package_bundle_to_zip(
    *,
    bundle_dir: Path,
    out_path: Path,
    include_web: bool,
) -> Dict[str, Any]:
    validate_bundle(bundle_dir)

    evidence_main = bundle_dir / "daily_evidence.json"
    evidence_obj = json.loads(_read_text(evidence_main))

    with tempfile.TemporaryDirectory(prefix="mgc_submission_") as td:
        stage = Path(td).resolve()
        pkg_root = stage / "submission"
        _safe_mkdir(pkg_root)

        drop_bundle_dst = pkg_root / "drop_bundle"
        shutil.copytree(bundle_dir, drop_bundle_dst)

        web_build_meta: Optional[Dict[str, Any]] = None
        if include_web:
            web_dst = pkg_root / "web"
            web_build_meta = _build_web_bundle_from_portable_bundle(
                bundle_dir=bundle_dir,
                web_out_dir=web_dst,
            )

        readme_text = _build_readme(evidence_obj, included_web=include_web)
        (pkg_root / "README.md").write_text(readme_text, encoding="utf-8")

        if out_path.exists():
            out_path.unlink()
        _safe_mkdir(out_path.parent)

        with zipfile.ZipFile(
            str(out_path),
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=_ZIP_COMPRESSLEVEL,
            strict_timestamps=False,
        ) as zf:
            _zip_write_text(zf, "submission/README.md", readme_text)
            _zip_add_dir(zf, drop_bundle_dst, arc_root="submission/drop_bundle")

            if include_web and (pkg_root / "web").exists():
                _zip_add_dir(zf, pkg_root / "web", arc_root="submission/web")

    return {"included_web": include_web, "web_build": web_build_meta}


# -----------------------------
# Command implementations
# -----------------------------


def cmd_submission_build(args: argparse.Namespace) -> int:
    out_path = Path(args.out).expanduser().resolve()
    bundle_dir: Optional[Path] = None

    if getattr(args, "bundle_dir", None):
        bundle_dir = Path(args.bundle_dir).expanduser().resolve()
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            raise SystemExit(f"Bundle dir not found: {bundle_dir}")
    else:
        drop_id = str(getattr(args, "drop_id", "") or "").strip()
        if not drop_id:
            raise SystemExit("Either --bundle-dir or --drop-id is required.")
        db_arg = getattr(args, "db", None)
        if not db_arg:
            raise SystemExit("--db is required when using --drop-id")
        db_path = Path(db_arg).expanduser().resolve()
        if not db_path.exists():
            raise SystemExit(f"DB not found: {db_path}")

        con = sqlite3.connect(str(db_path))
        try:
            bundle_dir = _find_bundle_dir_by_drop_id(con, drop_id)
        finally:
            con.close()

        if bundle_dir is None:
            raise SystemExit(
                "Could not locate bundle dir from --drop-id. "
                "Re-run with --bundle-dir pointing at the portable bundle directory."
            )

    include_web = bool(getattr(args, "with_web", False))

    try:
        meta = _package_bundle_to_zip(
            bundle_dir=bundle_dir,
            out_path=out_path,
            include_web=include_web,
        )
    except Exception as e:
        if _json_mode(args):
            _emit_json({"ok": False, "cmd": "submission.build", "error": str(e)})
            return 2
        raise

    if _json_mode(args):
        payload: Dict[str, Any] = {
            "ok": True,
            "cmd": "submission.build",
            "out": str(out_path),
            "bundle_dir": str(bundle_dir),
            "included_web": bool(meta.get("included_web", False)),
        }
        if meta.get("web_build") is not None:
            payload["web_build"] = meta["web_build"]
        _emit_json(payload)
    else:
        print(f"[submission] wrote: {out_path}")
        print(f"[submission] bundle: {bundle_dir}")
        if include_web:
            print("[submission] included web bundle: submission/web")

    return 0


def cmd_submission_latest(args: argparse.Namespace) -> int:
    out_path = Path(args.out).expanduser().resolve()

    db_arg = getattr(args, "db", None) or "data/db.sqlite"
    db_path = Path(db_arg).expanduser().resolve()

    evidence_root_arg = getattr(args, "evidence_root", None) or "data/evidence"
    evidence_root = Path(evidence_root_arg).expanduser().resolve()

    bundle_dir: Optional[Path] = None

    if db_path.exists():
        con = sqlite3.connect(str(db_path))
        try:
            did = _latest_drop_id(con)
            if did:
                bundle_dir = _find_bundle_dir_by_drop_id(con, did)
        finally:
            con.close()

    if bundle_dir is None:
        bundle_dir = _find_latest_bundle_dir_by_scan(evidence_root)

    if bundle_dir is None:
        msg = (
            "Could not find a bundle to package.\n"
            f"- DB tried: {db_path}\n"
            f"- evidence_root scanned: {evidence_root}\n"
            "Tip: run a daily/drop that writes a portable bundle (tracks/ + playlist.json + daily_evidence.json), "
            "or use `mgc submission build --bundle-dir <dir>`."
        )
        if _json_mode(args):
            _emit_json({"ok": False, "cmd": "submission.latest", "error": msg})
            return 2
        raise SystemExit(msg)

    include_web = bool(getattr(args, "with_web", False))

    try:
        meta = _package_bundle_to_zip(
            bundle_dir=bundle_dir,
            out_path=out_path,
            include_web=include_web,
        )
    except Exception as e:
        if _json_mode(args):
            _emit_json({"ok": False, "cmd": "submission.latest", "error": str(e)})
            return 2
        raise

    if _json_mode(args):
        payload: Dict[str, Any] = {
            "ok": True,
            "cmd": "submission.latest",
            "out": str(out_path),
            "bundle_dir": str(bundle_dir),
            "included_web": bool(meta.get("included_web", False)),
        }
        if meta.get("web_build") is not None:
            payload["web_build"] = meta["web_build"]
        _emit_json(payload)
    else:
        print(f"[submission] wrote: {out_path}")
        print(f"[submission] bundle: {bundle_dir}")
        if include_web:
            print("[submission] included web bundle: submission/web")

    return 0


def _verify_optional_web(extracted_root: Path) -> None:
    """
    If submission/web exists, verify it looks like a valid web bundle.
    """
    web_dir = extracted_root / "submission" / "web"
    if not web_dir.exists():
        return  # optional

    idx = web_dir / "index.html"
    pl = web_dir / "playlist.json"
    tracks = web_dir / "tracks"

    if not idx.exists():
        raise FileNotFoundError(f"web bundle missing index.html: {idx}")
    if not pl.exists():
        raise FileNotFoundError(f"web bundle missing playlist.json: {pl}")
    if not tracks.exists() or not tracks.is_dir():
        raise FileNotFoundError(f"web bundle missing tracks/: {tracks}")

    audio = list(tracks.glob("*.mp3")) + list(tracks.glob("*.wav"))
    if not audio:
        raise ValueError(f"web bundle has no audio files under {tracks} (expected *.mp3 or *.wav)")


def cmd_submission_verify(args: argparse.Namespace) -> int:
    """
    Verify one or more submission.zip files by:
      - unzipping to a temp dir
      - validating submission/drop_bundle via validate_bundle()
      - if submission/web exists, validate it is usable (index.html + playlist.json + tracks/*.(mp3|wav))

    Accepts:
      - positional zips: mgc submission verify a.zip b.zip
      - flag zips:       mgc submission verify --zip a.zip --zip b.zip

    Exit codes:
      0 = all valid
      2 = any invalid OR no zips provided
    """
    zips_pos = list(getattr(args, "zips", []) or [])
    zips_flag = list(getattr(args, "zip_flags", []) or [])
    zip_inputs = [str(z) for z in (zips_flag + zips_pos) if z]

    if not zip_inputs:
        payload = {"ok": False, "cmd": "submission.verify", "error": "missing zip paths", "results": []}
        if _json_mode(args):
            _emit_json(payload)
            return 2
        raise SystemExit("submission verify: provide at least one zip (positional or --zip)")

    results: list[dict] = []
    all_ok = True

    for zp in zip_inputs:
        zip_path = Path(zp).expanduser().resolve()
        item = {"zip": str(zip_path), "ok": False, "error": None}

        try:
            if not zip_path.exists():
                raise FileNotFoundError(f"zip not found: {zip_path}")

            with zipfile.ZipFile(str(zip_path), "r") as zf:
                names = zf.namelist()
                expected_prefix = "submission/drop_bundle/"
                if not any(n.startswith(expected_prefix) for n in names):
                    raise ValueError("invalid layout: expected files under submission/drop_bundle/")

                with tempfile.TemporaryDirectory(prefix="mgc_verify_") as td:
                    td_path = Path(td).resolve()
                    zf.extractall(td_path)

                    bundle_dir = td_path / "submission" / "drop_bundle"
                    if not bundle_dir.exists() or not bundle_dir.is_dir():
                        raise FileNotFoundError(f"extracted bundle missing: {bundle_dir}")

                    validate_bundle(bundle_dir)
                    _verify_optional_web(td_path)

            item["ok"] = True

        except Exception as e:
            all_ok = False
            item["error"] = str(e)

        results.append(item)

    payload = {"ok": bool(all_ok), "cmd": "submission.verify", "count": len(results), "results": results}

    if _json_mode(args):
        _emit_json(payload)
    else:
        for r in results:
            if r["ok"]:
                print(f"OK: {r['zip']}")
            else:
                print(f"INVALID: {r['zip']}\n{r['error']}")

    return 0 if all_ok else 2


# -----------------------------
# CLI wiring (mgc.main imports this)
# -----------------------------


def register_submission_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("submission", help="Build submission bundles")
    p.set_defaults(_mgc_group="submission")

    s = p.add_subparsers(dest="submission_cmd", required=True)

    build = s.add_parser("build", help="Build a submission zip")
    build.add_argument("--out", required=True, help="Output zip path (e.g. submission.zip)")
    build.add_argument("--bundle-dir", default=None, help="Path to an existing portable bundle directory")
    build.add_argument("--drop-id", default=None, help="Drop id to locate bundle from DB (best-effort)")
    build.add_argument("--db", default=None, help="DB path for --drop-id lookup")
    build.add_argument(
        "--with-web",
        action="store_true",
        help="Include a static web player bundle under submission/web (built from bundle playlist.json)",
    )
    build.set_defaults(func=cmd_submission_build)

    latest = s.add_parser("latest", help="Build submission zip from the latest drop bundle")
    latest.add_argument("--out", required=True, help="Output zip path")
    latest.add_argument("--db", default="data/db.sqlite", help="DB path")
    latest.add_argument("--evidence-root", default="data/evidence", help="Evidence root to scan if DB lookup fails")
    latest.add_argument(
        "--with-web",
        action="store_true",
        help="Include a static web player bundle under submission/web (built from bundle playlist.json)",
    )
    latest.set_defaults(func=cmd_submission_latest)

    verify = s.add_parser("verify", help="Verify one or more submission.zip (unzip + bundle validation)")
    verify.add_argument("zips", nargs="*", help="One or more paths to submission.zip")
    verify.add_argument("--zip", dest="zip_flags", action="append", default=[], help="Repeatable zip path flag")
    verify.set_defaults(func=cmd_submission_verify)
