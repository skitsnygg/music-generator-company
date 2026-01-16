#!/usr/bin/env python3
"""
src/mgc/web_cli.py

Static web bundle builder + simple dev server.

Design goals:
- Keep this module ONLY about "web" commands.
- Provide a registrar function that mgc.main can discover:
    register_web_subcommand(subparsers)
  (plus a few aliases for backwards compatibility)
- Billing gate for web commands when --token is provided:
    Requires an active entitlement tier == "pro".

CRITICAL UX SAFETY FIX:
- Never silently use MGC_DB (which may point at fixtures/CI) as the billing DB.
- Billing DB resolution for gating is STRICT:
    1) --billing-db if provided
    2) env MGC_BILLING_DB if set
    3) --db only if it does NOT equal env MGC_DB (i.e., not implicitly coming from env default)
    4) otherwise -> error: billing_db_ambiguous / billing_db_missing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Small utilities
# -----------------------------

def _eprint(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _resolve_path_maybe(p: str) -> str:
    if not p:
        return ""
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return p


# -----------------------------
# Billing gate (pro only)
# -----------------------------

def _billing_require_pro(*, db_path: str, token: str) -> Dict[str, Any]:
    """
    Validate token and require an active entitlement with tier == "pro".

    Expected tables:
      billing_users(user_id, ...)
      billing_tokens(token_sha256, user_id, created_ts, label, ...)
      billing_entitlements(user_id, tier, starts_ts, ends_ts, source, ...)
    """
    if not db_path:
        raise ValueError("billing_db_missing")

    if not token.strip():
        raise ValueError("token_missing")

    dbp = Path(db_path).expanduser()
    if not dbp.exists():
        raise ValueError("billing_db_not_found")

    con = _connect(str(dbp))
    try:
        for t in ("billing_users", "billing_tokens", "billing_entitlements"):
            if not _table_exists(con, t):
                raise ValueError(f"billing_tables_missing:{t}")

        token_sha = _sha256_hex(token.strip())
        tok = con.execute(
            "SELECT token_sha256, user_id, created_ts, label "
            "FROM billing_tokens WHERE token_sha256 = ? LIMIT 1",
            (token_sha,),
        ).fetchone()
        if tok is None:
            raise ValueError("invalid_token")

        user_id = str(tok["user_id"])
        now_ts = _utc_now_iso()

        ent = con.execute(
            """
            SELECT user_id, tier, starts_ts, ends_ts, source
              FROM billing_entitlements
             WHERE user_id = ?
               AND starts_ts <= ?
               AND (ends_ts IS NULL OR ends_ts > ?)
             ORDER BY starts_ts DESC
             LIMIT 1
            """,
            (user_id, now_ts, now_ts),
        ).fetchone()

        tier = str(ent["tier"]) if ent is not None else "free"
        if tier != "pro":
            raise ValueError("requires_pro")

        return {
            "ok": True,
            "user_id": user_id,
            "tier": tier,
            "token_sha256": token_sha,
            "now": now_ts,
        }
    finally:
        try:
            con.close()
        except Exception:
            pass


def _resolve_billing_db_strict(args: argparse.Namespace) -> str:
    """
    STRICT billing DB resolution to prevent CI fixture bleed-through.

    Order:
      1) --billing-db (web subcommand arg)
      2) env MGC_BILLING_DB
      3) --db ONLY IF it does NOT equal env MGC_DB (i.e. not coming from implicit env default)
      4) error

    Notes:
    - mgc.main may define a global --db whose default is os.environ["MGC_DB"].
      That is exactly the footgun we prevent: if args.db == env MGC_DB and the
      user didn't explicitly provide a billing DB, we refuse with billing_db_ambiguous.
    """
    billing_db = str(getattr(args, "billing_db", "") or "").strip()
    if billing_db:
        return billing_db

    env_billing = str(os.environ.get("MGC_BILLING_DB", "") or "").strip()
    if env_billing:
        return env_billing

    db = str(getattr(args, "db", "") or "").strip()
    if not db:
        raise ValueError("billing_db_missing")

    env_mgc_db = str(os.environ.get("MGC_DB", "") or "").strip()

    # If MGC_DB is set and args.db matches it, treat as ambiguous (likely implicit default).
    if env_mgc_db:
        db_res = _resolve_path_maybe(db)
        env_res = _resolve_path_maybe(env_mgc_db)
        if db_res and env_res and db_res == env_res:
            raise ValueError("billing_db_ambiguous")

    # Otherwise, allow args.db as the billing DB.
    return db


def _require_pro_if_token(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    token = str(getattr(args, "token", "") or "").strip()
    if not token:
        return None
    billing_db = _resolve_billing_db_strict(args)
    return _billing_require_pro(db_path=billing_db, token=token)


# -----------------------------
# Playlist helpers
# -----------------------------

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}


def _looks_like_audio_path(s: str) -> bool:
    try:
        p = Path(s)
    except Exception:
        return False
    return p.suffix.lower() in _AUDIO_EXTS


def _walk_collect_strings(obj: Any, out: List[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
        return
    if isinstance(obj, list):
        for it in obj:
            _walk_collect_strings(it, out)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_collect_strings(v, out)
        return


def _collect_track_paths_from_playlist(playlist_obj: Any) -> List[str]:
    """
    Extract audio file paths from common playlist shapes.

    Supported patterns:
      - top-level list[str]
      - {"tracks": [str | dict]}
      - dict entries containing keys like:
          artifact_path, audio_path, path, file, uri, src, url
      - any string anywhere that "looks like" an audio path
    """
    out: List[str] = []

    def add_path(v: Any) -> None:
        if not v:
            return
        if isinstance(v, str) and _looks_like_audio_path(v.strip()):
            out.append(v.strip())

    if isinstance(playlist_obj, list):
        for it in playlist_obj:
            if isinstance(it, str):
                add_path(it)
            elif isinstance(it, dict):
                for k in ("artifact_path", "audio_path", "path", "file", "uri", "src", "url"):
                    if k in it:
                        add_path(it.get(k))
                        break

    if isinstance(playlist_obj, dict):
        tracks = playlist_obj.get("tracks")
        if isinstance(tracks, list):
            for it in tracks:
                if isinstance(it, str):
                    add_path(it)
                elif isinstance(it, dict):
                    for k in ("artifact_path", "audio_path", "path", "file", "uri", "src", "url"):
                        if k in it:
                            add_path(it.get(k))
                            break

    if not out:
        all_strs: List[str] = []
        _walk_collect_strings(playlist_obj, all_strs)
        for s in all_strs:
            if _looks_like_audio_path(s):
                out.append(s)

    seen: set[str] = set()
    deduped: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _collect_track_ids_from_playlist(playlist_obj: Any) -> List[str]:
    ids: List[str] = []

    def add_id(v: Any) -> None:
        if not v:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                ids.append(s)

    if isinstance(playlist_obj, dict):
        if isinstance(playlist_obj.get("track_ids"), list):
            for it in playlist_obj["track_ids"]:
                add_id(it)

        tracks = playlist_obj.get("tracks")
        if isinstance(tracks, list):
            for it in tracks:
                if isinstance(it, dict) and "track_id" in it:
                    add_id(it.get("track_id"))

    seen: set[str] = set()
    out: List[str] = []
    for t in ids:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _resolve_track_paths_from_db(*, db_path: str, track_ids: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "resolved_from": "db",
        "db": db_path,
        "requested_ids": len(track_ids),
        "resolved": 0,
        "missing": 0,
        "reason": None,
    }

    if not db_path:
        meta["reason"] = "db_missing"
        return [], meta

    p = Path(db_path).expanduser()
    if not p.exists():
        meta["reason"] = "db_not_found"
        return [], meta

    con = _connect(str(p))
    try:
        if not _table_exists(con, "tracks"):
            meta["reason"] = "tracks_table_missing"
            return [], meta

        cols = [r["name"] for r in con.execute("PRAGMA table_info(tracks)").fetchall()]
        path_col = None
        for c in ("artifact_path", "audio_path", "path", "file_path", "uri"):
            if c in cols:
                path_col = c
                break

        id_col = None
        for c in ("track_id", "id"):
            if c in cols:
                id_col = c
                break

        if not path_col or not id_col:
            meta["reason"] = "tracks_schema_unexpected"
            return [], meta

        resolved: List[str] = []
        missing = 0
        CHUNK = 800

        for i in range(0, len(track_ids), CHUNK):
            chunk = track_ids[i : i + CHUNK]
            qmarks = ",".join(["?"] * len(chunk))
            rows = con.execute(
                f"SELECT {id_col} AS track_id, {path_col} AS p FROM tracks WHERE {id_col} IN ({qmarks})",
                tuple(chunk),
            ).fetchall()
            got = {str(r["track_id"]): (str(r["p"]) if r["p"] is not None else "") for r in rows}

            for tid in chunk:
                path = got.get(tid, "").strip()
                if path:
                    resolved.append(path)
                else:
                    missing += 1

        meta["resolved"] = len(resolved)
        meta["missing"] = missing
        return resolved, meta
    finally:
        try:
            con.close()
        except Exception:
            pass


def _resolve_input_path(raw: str, *, playlist_dir: Path) -> Path:
    rp = Path(raw).expanduser()
    if rp.is_absolute():
        return rp
    return (playlist_dir / rp).resolve()


def _prefer_mp3_path(p: Path) -> Path:
    if p.suffix.lower() == ".wav":
        alt = p.with_suffix(".mp3")
        if alt.exists():
            return alt
    return p


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Web build
# -----------------------------

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MGC Web Player</title>
  <style>
    body { font-family: sans-serif; margin: 20px; line-height: 1.4; }
    .track { margin: 10px 0; }
    audio { width: 100%; max-width: 760px; }
    .muted { color: #666; font-size: 0.9em; }
  </style>
</head>
<body>
  <h1>MGC Web Player</h1>
  <div class="muted">Loads <code>playlist.json</code> from this folder.</div>
  <div id="app"></div>

<script>
async function main() {
  const app = document.getElementById("app");
  let playlist = null;

  try {
    const res = await fetch("./playlist.json", { cache: "no-store" });
    playlist = await res.json();
  } catch (e) {
    app.innerHTML = "<p>Failed to load playlist.json</p>";
    return;
  }

  const tracks = (playlist && playlist.tracks) ? playlist.tracks : [];
  if (!Array.isArray(tracks) || tracks.length === 0) {
    app.innerHTML = "<p>No tracks found in playlist.json</p>";
    return;
  }

  for (const t of tracks) {
    const row = document.createElement("div");
    row.className = "track";
    const title = document.createElement("div");
    title.textContent = t.title || t.track_id || t.path || t.artifact_path || "Track";
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = t.web_path || t.path || "";
    row.appendChild(title);
    row.appendChild(audio);
    app.appendChild(row);
  }
}
main();
</script>
</body>
</html>
"""


def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(str(args.playlist)).expanduser().resolve()
    if not playlist_path.exists():
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "playlist_not_found", "playlist": str(playlist_path)}) + "\n")
        return 2

    # Billing gate (strict DB resolution)
    try:
        _require_pro_if_token(args)
    except Exception as e:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "billing_denied",
            "error": str(e),
            "hint": "Pass --billing-db explicitly or set MGC_BILLING_DB (billing will not silently use MGC_DB).",
        }) + "\n")
        return 2

    playlist_dir = playlist_path.parent
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    tracks_dir = out_dir / "tracks"
    manifest_path = out_dir / "web_manifest.json"
    out_playlist_path = out_dir / "playlist.json"
    index_path = out_dir / "index.html"

    if bool(getattr(args, "clean", False)) and out_dir.exists():
        shutil.rmtree(out_dir)

    _safe_mkdir(tracks_dir)

    prefer_mp3 = bool(getattr(args, "prefer_mp3", False))
    strip_paths = bool(getattr(args, "strip_paths", False))
    fail_if_empty = bool(getattr(args, "fail_if_empty", False))
    fail_if_none_copied = bool(getattr(args, "fail_if_none_copied", False))
    fail_on_missing = bool(getattr(args, "fail_on_missing", False))

    playlist_obj = json.loads(playlist_path.read_text(encoding="utf-8"))

    raw_paths = _collect_track_paths_from_playlist(playlist_obj)
    resolved_from = "playlist_path"

    # If no paths, try DB via track_ids
    db_meta: Dict[str, Any] = {}
    if not raw_paths:
        track_ids = _collect_track_ids_from_playlist(playlist_obj)
        if track_ids:
            db_path = str(getattr(args, "db", "") or "").strip()
            raw_paths, db_meta = _resolve_track_paths_from_db(db_path=db_path, track_ids=track_ids)
            if raw_paths:
                resolved_from = "db"

    if fail_if_empty and not raw_paths:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "empty_playlist",
            "playlist": str(playlist_path),
            "resolved_from": resolved_from,
            "db_meta": db_meta,
        }) + "\n")
        return 2

    copied = 0
    missing = 0
    bundled: List[Dict[str, Any]] = []
    out_tracks: List[Dict[str, Any]] = []

    for i, raw in enumerate(raw_paths):
        rp = _resolve_input_path(raw, playlist_dir=playlist_dir)
        if prefer_mp3:
            rp = _prefer_mp3_path(rp)

        if not rp.exists() or not rp.is_file():
            missing += 1
            bundled.append({
                "index": i,
                "source": raw if not strip_paths else str(Path(raw)).as_posix(),
                "resolved": str(rp) if not strip_paths else rp.name,
                "ok": False,
                "reason": "missing",
                "resolved_from": resolved_from,
            })
            continue

        dest = tracks_dir / rp.name
        if dest.exists():
            stem = dest.stem
            suf = dest.suffix
            n = 1
            while True:
                cand = tracks_dir / f"{stem}_{n}{suf}"
                if not cand.exists():
                    dest = cand
                    break
                n += 1

        shutil.copy2(str(rp), str(dest))
        copied += 1
        rel = f"tracks/{dest.name}"

        bundled.append({
            "index": i,
            "source": str(rp) if not strip_paths else rp.name,
            "dest": str(dest),
            "web_path": rel,
            "ok": True,
            "resolved_from": resolved_from,
        })

        out_tracks.append({
            "title": rp.stem,
            "path": str(rp) if not strip_paths else rp.name,
            "web_path": rel,
        })

    if fail_if_none_copied and copied == 0:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "none_copied",
            "copied_count": copied,
            "missing_count": missing,
            "out_dir": str(out_dir),
        }) + "\n")
        return 2

    if fail_on_missing and missing > 0:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "missing_tracks",
            "copied_count": copied,
            "missing_count": missing,
            "out_dir": str(out_dir),
        }) + "\n")
        return 2

    out_playlist = {
        "source_playlist": str(playlist_path),
        "resolved_from": resolved_from,
        "tracks": out_tracks,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_playlist_path.write_text(json.dumps(out_playlist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path.write_text(_INDEX_HTML, encoding="utf-8")

    manifest = {
        "ok": True,
        "cmd": "web.build",
        "out_dir": str(out_dir),
        "playlist": str(playlist_path),
        "playlist_out": str(out_playlist_path),
        "index": str(index_path),
        "manifest_path": str(manifest_path),
        "resolved_from": resolved_from,
        "copied_count": copied,
        "missing_count": missing,
        "bundled": bundled,
        "db_meta": db_meta,
    }
    manifest_path.write_text(_stable_json_dumps(manifest) + "\n", encoding="utf-8")

    sys.stdout.write(_stable_json_dumps(manifest) + "\n")
    return 0


# -----------------------------
# Web serve
# -----------------------------

def cmd_web_serve(args: argparse.Namespace) -> int:
    directory = Path(str(args.dir)).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "dir_not_found", "dir": str(directory)}) + "\n")
        return 2

    try:
        _require_pro_if_token(args)
    except Exception as e:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "billing_denied",
            "error": str(e),
            "hint": "Pass --billing-db explicitly or set MGC_BILLING_DB (billing will not silently use MGC_DB).",
        }) + "\n")
        return 2

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, directory=str(directory), **kw)

    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((host, port), Handler)
    _eprint(f"[mgc.web] serving {directory} on http://{host}:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            httpd.server_close()
        except Exception:
            pass
    return 0


# -----------------------------
# Registrar
# -----------------------------

def register_web_subcommand(subparsers: argparse._SubParsersAction) -> None:
    web = subparsers.add_parser("web", help="Static web player build/serve")
    ws = web.add_subparsers(dest="web_cmd", required=True)

    build = ws.add_parser("build", help="Build a static web bundle")
    build.add_argument("--playlist", required=True, help="Playlist JSON path")
    build.add_argument("--out-dir", required=True, help="Output directory for the web bundle")

    # Keep --db because builds may need it to resolve track IDs -> paths
    # NOTE: billing DB is resolved STRICTLY and will not silently use env MGC_DB.
    build.add_argument("--db", default=os.environ.get("MGC_DB", ""), help="DB path (used only if playlist contains IDs)")
    build.add_argument("--token", default=None, help="Billing access token (requires pro)")
    build.add_argument(
        "--billing-db",
        dest="billing_db",
        default=None,
        help="DB path for billing/token validation (recommended). If omitted, uses MGC_BILLING_DB or --db (but will NOT silently use MGC_DB).",
    )
    build.add_argument("--prefer-mp3", action="store_true", help="Prefer .mp3 when a .wav sibling exists")
    build.add_argument("--strip-paths", action="store_true", help="Strip absolute paths in manifest output")
    build.add_argument("--clean", action="store_true", help="Delete out-dir before building")
    build.add_argument("--fail-if-empty", action="store_true", help="Fail if playlist resolves to 0 tracks")
    build.add_argument("--fail-if-none-copied", action="store_true", help="Fail if no files were copied")
    build.add_argument("--fail-on-missing", action="store_true", help="Fail if any referenced tracks are missing")
    build.set_defaults(func=cmd_web_build)

    serve = ws.add_parser("serve", help="Serve a built web bundle")
    serve.add_argument("--dir", required=True, help="Directory to serve (output of web build)")
    serve.add_argument("--db", default=os.environ.get("MGC_DB", ""), help="DB path (required only if --token is used)")
    serve.add_argument("--token", default=None, help="Billing access token (requires pro)")
    serve.add_argument(
        "--billing-db",
        dest="billing_db",
        default=None,
        help="DB path for billing/token validation (recommended). If omitted, uses MGC_BILLING_DB or --db (but will NOT silently use MGC_DB).",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_web_serve)


# Compatibility aliases for mgc.main registrar probing
def register_web_subparser(subparsers: argparse._SubParsersAction) -> None:
    register_web_subcommand(subparsers)


def register_web_cli(subparsers: argparse._SubParsersAction) -> None:
    register_web_subcommand(subparsers)


def register_web(subparsers: argparse._SubParsersAction) -> None:
    register_web_subcommand(subparsers)


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    register_web_subcommand(subparsers)


def register(subparsers: argparse._SubParsersAction) -> None:
    register_web_subcommand(subparsers)
