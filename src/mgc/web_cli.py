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

def _argv_get_flag_value(flag: str) -> str:
    """Best-effort extraction of a flag value from sys.argv.

    mgc.main defines some global flags (e.g. --db) and may consume them depending on position.
    Web commands still need to honor `--db ...` even when provided after `web build`.
    """
    try:
        argv = list(sys.argv or [])
    except Exception:
        return ""
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            nxt = argv[i + 1]
            if not str(nxt).startswith("-"):
                return str(nxt)
        if str(tok).startswith(flag + "="):
            return str(tok).split("=", 1)[1]
    return ""

def _eprint(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _connect(db_path: str) -> sqlite3.Connection:
    # Always resolve to an absolute path (avoid surprises with CWD).
    p = Path(str(db_path or "")).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass
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


def _iter_track_dicts(playlist_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a flat list of dict-like track entries from multiple playlist schema variants."""
    candidates: List[Any] = []
    for k in ("tracks", "items", "entries", "playlist_items"):
        v = playlist_obj.get(k)
        if isinstance(v, list):
            candidates.extend(v)
    out: List[Dict[str, Any]] = []
    for x in candidates:
        if isinstance(x, dict):
            out.append(x)
    return out


def _dig_first_str(obj: Any, keys: List[str]) -> str:
    """Try keys at this level and common nested shapes."""
    if not isinstance(obj, dict):
        return ""
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for nest in ("track", "meta", "payload"):
        nv = obj.get(nest)
        if isinstance(nv, dict):
            for k in keys:
                v = nv.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ""

def _collect_track_paths_from_playlist(obj: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for t in _iter_track_dicts(obj):
        v = _dig_first_str(t, ["web_path", "artifact_path", "repo_artifact_path", "path", "file", "filename"])
        if v:
            out.append(v)
    return out

def _collect_track_ids_from_playlist(obj: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for t in _iter_track_dicts(obj):
        v = _dig_first_str(t, ["track_id", "trackId", "id", "uuid"])
        if v:
            out.append(v)
    return out

def _resolve_track_paths_from_db(*, db_path: str, track_ids: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    db_raw = str(db_path or "").strip()
    if not db_raw:
        return [], {"ok": False, "reason": "db_missing", "db": ""}

    dbp = Path(db_raw).expanduser()
    try:
        dbp = dbp.resolve()
    except Exception:
        pass

    if not dbp.exists():
        return [], {"ok": False, "reason": "db_not_found", "db": str(dbp)}

    if not track_ids:
        return [], {"ok": False, "reason": "no_track_ids", "db": str(dbp)}

    try:
        con = _connect(str(dbp))
    except Exception as e:
        return [], {"ok": False, "reason": "db_open_failed", "db": str(dbp), "error": str(e)}

    try:
        if not _table_exists(con, "tracks"):
            return [], {"ok": False, "reason": "tracks_table_missing", "db": str(dbp)}

        cols = {r["name"] for r in con.execute("PRAGMA table_info(tracks)").fetchall()}

        # Determine ID column
        if "track_id" in cols:
            id_col = "track_id"
        elif "id" in cols:
            id_col = "id"
        elif "uuid" in cols:
            id_col = "uuid"
        else:
            cand = sorted([c for c in cols if c.endswith("_id")])
            id_col = cand[0] if cand else None

        # Determine artifact/path column (schema drift tolerant)
        if "artifact_path" in cols:
            path_col = "artifact_path"
        elif "full_path" in cols:
            path_col = "full_path"
        elif "preview_path" in cols:
            path_col = "preview_path"
        elif "path" in cols:
            path_col = "path"
        elif "file_path" in cols:
            path_col = "file_path"
        elif "filename" in cols:
            path_col = "filename"
        else:
            path_col = None

        if not id_col or not path_col:
            return [], {
                "ok": False,
                "reason": "tracks_schema_unexpected",
                "db": str(dbp),
                "id_col": id_col,
                "path_col": path_col,
                "cols": sorted(list(cols)),
            }

        q = f"SELECT {id_col} AS tid, {path_col} AS ap FROM tracks WHERE {id_col} IN ({','.join(['?'] * len(track_ids))})"
        rows = con.execute(q, tuple(track_ids)).fetchall()
        found = {str(r["tid"]): str(r["ap"]) for r in rows if r["tid"] is not None}

        out: List[str] = []
        missing = 0
        for tid in track_ids:
            p = found.get(tid)
            if p:
                out.append(p)
            else:
                missing += 1

        return out, {"ok": True, "found": len(out), "missing": missing, "db": str(dbp), "id_col": id_col, "path_col": path_col}
    finally:
        try:
            con.close()
        except Exception:
            pass

def _resolve_input_path(raw: str, *, playlist_dir: Path, repo_root: Path) -> Path:
    """Resolve an audio path from playlist metadata.

    This repo needs to handle several playlist shapes:
      - repo paths:        data/tracks/<id>.wav   (often relative)
      - absolute paths:    /.../data/tracks/<id>.wav (CI runner, local dev)
      - bundle paths:      tracks/<id>.wav or tracks/<file> relative to playlist dir
      - older variants:    arbitrary relative paths next to playlist.json

    Behavior:
      - Prefer exact existing paths.
      - If an absolute path is missing, fall back to bundle/playlist-dir candidates by filename.
      - If relative, prefer repo_root first (for DB-returned data/tracks/...), then fall back to
        deterministic search within data/tracks/*/<file>, then to playlist_dir.
    """
    raw_s = str(raw or "").strip()
    rp = Path(raw_s).expanduser()

    # Absolute path: accept if it exists, otherwise fall back to bundle-style lookup.
    if rp.is_absolute():
        if rp.exists():
            return rp.resolve()

        # Try filename-based fallbacks (portable bundle layout)
        needle = rp.name
        cand = (playlist_dir / needle)
        if cand.exists():
            return cand.resolve()
        cand = (playlist_dir / "tracks" / needle)
        if cand.exists():
            return cand.resolve()

        # If this absolute path actually points inside repo_root, try to relativize.
        try:
            rel = rp.relative_to(repo_root)
            cand = (repo_root / rel)
            if cand.exists():
                return cand.resolve()
        except Exception:
            pass

        # Give back the original absolute path (caller will treat as missing)
        return rp

    # Relative path: prefer repo root for DB-returned relative paths like data/tracks/...
    cand1 = (repo_root / rp)
    if cand1.exists():
        return cand1.resolve()

    # Fallback 1: if DB says data/tracks/<file> but files live in daily folders:
    # try <repo_root>/data/tracks/*/<file>
    try:
        parts = rp.parts
        if len(parts) >= 3 and parts[0] == "data" and parts[1] == "tracks":
            needle = parts[-1]
            base = repo_root / "data" / "tracks"
            # one directory level deep, deterministic ordering
            hits = sorted(p for p in base.glob(f"*/{needle}") if p.is_file())
            if len(hits) == 1:
                return hits[0].resolve()
            # if multiple matches, do NOT guess—fall through to playlist_dir behavior
    except Exception:
        pass

    # Fallback 2: resolve relative to the playlist directory (run-bundled playlists)
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
    # Billing gate: only enforce when a token-gated feature is requested.
    if getattr(args, "token", None):
        try:
            _require_pro_if_token(args)
        except Exception as e:
            sys.stdout.write(_stable_json_dumps({
                'ok': False,
                'reason': 'billing_denied',
                'error': str(e),
                'hint': 'If using --token, pass --billing-db or set MGC_BILLING_DB.',
            }) + '\n')
            return 2

    playlist_dir = playlist_path.parent
    # Prefer deriving repo_root from the DB path when available (db usually lives at <repo>/data/db.sqlite).
    repo_root = Path.cwd().resolve()
    try:
        _db_raw = str(getattr(args, "db", "") or "").strip() or _argv_get_flag_value("--db").strip() or str(os.environ.get("MGC_DB", "") or "").strip()
        if _db_raw:
            _dbp = Path(_db_raw).expanduser().resolve()
            # If DB is in <repo>/data/, treat <repo> as repo_root.
            if _dbp.parent.name == "data":
                repo_root = _dbp.parent.parent
            else:
                repo_root = _dbp.parent
    except Exception:
        pass
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
            if not db_path:
                db_path = _argv_get_flag_value("--db").strip()
            if not db_path:
                db_path = str(os.environ.get("MGC_DB", "") or "").strip()
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
        rp = _resolve_input_path(raw, playlist_dir=playlist_dir, repo_root=repo_root)
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
            "source": str(Path("tracks") / rp.name) if not strip_paths else rp.name,
            "dest": rel,
            "web_path": rel,
            "ok": True,
            "resolved_from": resolved_from,
        })

        out_tracks.append({
            "title": rp.stem,
            "path": rel,
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
        "source_playlist": str(playlist_path.name),
        "resolved_from": resolved_from,
        "tracks": out_tracks,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_playlist_path.write_text(_stable_json_dumps(out_playlist) + "\n", encoding="utf-8")
    index_path.write_text(_INDEX_HTML, encoding="utf-8")

    ok = True
    if missing > 0:
        ok = False
    if raw_paths and copied == 0:
        ok = False

    manifest = {
        "ok": ok,
        "cmd": "web.build",

        # Contract-stable relative paths only
        "paths": {
            "playlist": "playlist.json",
            "index": "index.html",
            "web_manifest": "web_manifest.json",
        },

        # Counts are deterministic
        "copied_count": int(copied),
        "missing_count": int(missing),

        # Bundled files: ensure stable ordering
        "bundled": sorted(bundled) if isinstance(bundled, list) else bundled,

        # DB metadata must already be deterministic upstream
        "db_meta": db_meta,

        # Resolution source should be semantic, not path-based
        "resolved_from": resolved_from,
    }

    # Write manifest deterministically
    manifest_text = _stable_json_dumps(manifest) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")

    # Emit the same object to stdout (CLI contract preserved)
    sys.stdout.write(manifest_text)
    return 0


# -----------------------------
# Web serve
# -----------------------------

def cmd_web_serve(args: argparse.Namespace) -> int:
    directory = Path(str(args.dir)).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "dir_not_found", "dir": str(directory)}) + "\n")
        return 2

    # Only require billing if a token-gated feature is used.
    if getattr(args, "token", None):
        try:
            _require_pro_if_token(args)
        except Exception as e:
            sys.stdout.write(
                _stable_json_dumps(
                    {
                        "ok": False,
                        "reason": "billing_denied",
                        "error": str(e),
                        "hint": "If using --token, pass --billing-db or set MGC_BILLING_DB.",
                    }
                )
                + "\n"
            )
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

def register_web_subcommand(subparsers: argparse._SubParsersAction) -> None:
    web = subparsers.add_parser("web", help="Static web player build/serve")
    ws = web.add_subparsers(dest="web_cmd", required=True)

    build = ws.add_parser("build", help="Build a static web bundle")
    build.add_argument("--playlist", required=True, help="Playlist JSON path")
    build.add_argument("--out-dir", required=True, help="Output directory for the web bundle")

    # Keep --db because builds may need it to resolve track IDs -> paths
    # NOTE: billing DB is resolved STRICTLY and will not silently use env MGC_DB.
    build.add_argument("--db", default="", help="DB path (used only if playlist contains IDs)")
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
    serve.add_argument("--db", default="", help="DB path (required only if --token is used)")
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
