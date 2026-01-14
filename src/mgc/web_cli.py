#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg")


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _as_posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json_dumps(obj) + "\n", encoding="utf-8", newline="\n")


def _rm_tree_contents(root: Path) -> None:
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _looks_like_audio_path(s: str) -> bool:
    s2 = (s or "").strip().lower()
    return any(s2.endswith(ext) for ext in AUDIO_EXTS)


def _prefer_mp3_path(p: Path) -> Path:
    # If a WAV is referenced but an MP3 sibling exists, prefer it.
    if p.suffix.lower() == ".wav":
        mp3 = p.with_suffix(".mp3")
        if mp3.exists():
            return mp3
    return p


def _walk_collect_strings(obj: Any, out: List[str], limit: int = 20000) -> None:
    """Collect all strings in a nested JSON-ish structure (best-effort)."""
    if len(out) >= limit:
        return
    if obj is None:
        return
    if isinstance(obj, str):
        out.append(obj)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_collect_strings(v, out, limit=limit)
        return
    if isinstance(obj, list):
        for v in obj:
            _walk_collect_strings(v, out, limit=limit)
        return


def _extract_track_paths(playlist_obj: Any) -> List[str]:
    """
    Extract audio file paths from many possible playlist shapes.
    Strategy:
      1) Check common keys first (tracks/items/playlist.items).
      2) Fall back to recursive string scan for anything ending with audio ext.
    """
    candidates: List[Any] = []

    if isinstance(playlist_obj, dict):
        if "tracks" in playlist_obj:
            candidates = playlist_obj.get("tracks")  # type: ignore[assignment]
        elif "items" in playlist_obj:
            candidates = playlist_obj.get("items")  # type: ignore[assignment]
        elif isinstance(playlist_obj.get("playlist"), dict):
            pl = playlist_obj.get("playlist")
            if isinstance(pl, dict) and "items" in pl:
                candidates = pl.get("items")  # type: ignore[assignment]
    elif isinstance(playlist_obj, list):
        candidates = playlist_obj

    out: List[str] = []

    def add_path(v: Any) -> None:
        if v is None:
            return
        s = str(v).strip()
        if s and _looks_like_audio_path(s):
            out.append(s)

    if isinstance(candidates, list):
        for it in candidates:
            if isinstance(it, str):
                add_path(it)
            elif isinstance(it, dict):
                for k in ("artifact_path", "audio_path", "path", "file", "uri", "src", "url"):
                    if k in it:
                        add_path(it.get(k))
                        break

    # fallback: recursive scan
    if not out:
        all_strs: List[str] = []
        _walk_collect_strings(playlist_obj, all_strs)
        for s in all_strs:
            if _looks_like_audio_path(s):
                out.append(s)

    # De-dupe preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _extract_track_ids(playlist_obj: Any) -> List[str]:
    """
    If playlist doesn't include file paths, it may include track IDs.
    We look for:
      - top-level "track_ids": [...]
      - items containing {"track_id": "..."} or {"id": "..."}
      - a recursive scan for UUID-like strings is intentionally NOT done (too risky).
    """
    ids: List[str] = []

    def add_id(v: Any) -> None:
        if v is None:
            return
        s = str(v).strip()
        if s:
            ids.append(s)

    if isinstance(playlist_obj, dict):
        if isinstance(playlist_obj.get("track_ids"), list):
            for v in playlist_obj["track_ids"]:
                add_id(v)
        if isinstance(playlist_obj.get("tracks"), list):
            for it in playlist_obj["tracks"]:
                if isinstance(it, dict):
                    if "track_id" in it:
                        add_id(it.get("track_id"))
                    elif "id" in it:
                        add_id(it.get("id"))
                elif isinstance(it, str) and not _looks_like_audio_path(it):
                    add_id(it)
        if isinstance(playlist_obj.get("items"), list):
            for it in playlist_obj["items"]:
                if isinstance(it, dict):
                    if "track_id" in it:
                        add_id(it.get("track_id"))
                    elif "id" in it:
                        add_id(it.get("id"))
    elif isinstance(playlist_obj, list):
        for it in playlist_obj:
            if isinstance(it, dict):
                if "track_id" in it:
                    add_id(it.get("track_id"))
                elif "id" in it:
                    add_id(it.get("id"))
            elif isinstance(it, str) and not _looks_like_audio_path(it):
                add_id(it)

    # De-dupe preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for t in ids:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _db_connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _resolve_track_paths_from_db(db_path: Path, track_ids: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "used_db": str(db_path),
        "tracks_table": False,
        "path_column": None,
        "resolved": 0,
        "missing": 0,
    }
    if not db_path.exists():
        meta["reason"] = "db_not_found"
        return [], meta

    con = _db_connect(db_path)
    try:
        if not _table_exists(con, "tracks"):
            meta["reason"] = "tracks_table_missing"
            return [], meta
        meta["tracks_table"] = True

        cols = [r["name"] for r in con.execute("PRAGMA table_info(tracks)").fetchall()]
        candidates = [
            "artifact_path",
            "audio_path",
            "path",
            "file_path",
            "filepath",
            "src_path",
            "wav_path",
            "mp3_path",
        ]
        path_col = next((c for c in candidates if c in cols), None)
        meta["path_column"] = path_col
        if not path_col:
            meta["reason"] = "no_path_column_found"
            return [], meta

        id_col = "track_id" if "track_id" in cols else ("id" if "id" in cols else None)
        if not id_col:
            meta["reason"] = "no_id_column_found"
            return [], meta

        resolved_paths: List[str] = []
        missing = 0
        CHUNK = 800

        for i in range(0, len(track_ids), CHUNK):
            chunk = track_ids[i : i + CHUNK]
            qs = ",".join("?" for _ in chunk)
            rows = con.execute(
                f"SELECT {id_col} AS tid, {path_col} AS p FROM tracks WHERE {id_col} IN ({qs})",
                tuple(chunk),
            ).fetchall()
            got = {str(r["tid"]): (r["p"] if r else None) for r in rows}

            for tid in chunk:
                p = got.get(str(tid))
                if p is None:
                    missing += 1
                    continue
                s = str(p).strip()
                if not s:
                    missing += 1
                    continue
                resolved_paths.append(s)

        meta["resolved"] = len(resolved_paths)
        meta["missing"] = missing
        return resolved_paths, meta
    finally:
        try:
            con.close()
        except Exception:
            pass


def _resolve_input_path(raw: str, *, playlist_dir: Path) -> Path:
    """
    Resolve a raw track reference into a filesystem Path.

    Key behavior:
    - Relative paths are resolved relative to the playlist file directory (NOT cwd).
    - Absolute paths are used as-is.
    """
    rp = Path(str(raw).strip())
    if rp.is_absolute():
        return rp.resolve()
    return (playlist_dir / rp).resolve()


def _display_path_strip(p: Path, *, base_dir: Path) -> str:
    """
    Make a stable, portable display path for JSON outputs that will be written to disk
    (playlist.json, web_manifest.json), avoiding temp dirs/absolute paths when possible.
    """
    try:
        rp = p.resolve()
    except Exception:
        rp = p
    try:
        # If under base_dir, make it relative.
        rel = rp.relative_to(base_dir.resolve())
        return _as_posix(rel)
    except Exception:
        # Otherwise, fall back to basename only (portable; avoids temp paths).
        return rp.name


def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(str(args.playlist)).expanduser().resolve()
    if not playlist_path.exists():
        out = {"ok": False, "reason": "playlist_not_found", "playlist": str(playlist_path)}
        sys.stdout.write(_stable_json_dumps(out) + "\n")
        return 2

    playlist_dir = playlist_path.parent

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    tracks_dir = out_dir / "tracks"
    manifest_path = out_dir / "web_manifest.json"
    out_playlist_path = out_dir / "playlist.json"
    index_path = out_dir / "index.html"

    if bool(getattr(args, "clean", False)):
        out_dir.mkdir(parents=True, exist_ok=True)
        _rm_tree_contents(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    playlist_obj = _read_json(playlist_path)

    raw_paths = _extract_track_paths(playlist_obj)
    resolved_from = "playlist_path"
    db_resolution: Optional[Dict[str, Any]] = None

    if not raw_paths:
        track_ids = _extract_track_ids(playlist_obj)
        if track_ids:
            db_path = Path(str(getattr(args, "db", ""))).expanduser().resolve()
            raw_paths, db_resolution = _resolve_track_paths_from_db(db_path, track_ids)
            if raw_paths:
                resolved_from = "db"

    if bool(getattr(args, "fail_if_empty", False)) and not raw_paths:
        out: Dict[str, Any] = {
            "ok": False,
            "playlist": str(playlist_path),
            "reason": "playlist_empty",
            "track_count": 0,
        }
        if db_resolution is not None:
            out["db_resolution"] = db_resolution
        sys.stdout.write(_stable_json_dumps(out) + "\n")
        return 2

    if bool(getattr(args, "require_bundled_tracks", False)) and (resolved_from == "db"):
        out = {
            "ok": False,
            "playlist": str(playlist_path),
            "out_dir": str(out_dir),
            "reason": "db_resolution_disallowed",
            "track_count": len(raw_paths),
            "db_resolution": db_resolution,
        }
        sys.stdout.write(_stable_json_dumps(out) + "\n")
        return 2

    copied = 0
    missing = 0
    bundled: List[Dict[str, Any]] = []

    prefer_mp3 = bool(getattr(args, "prefer_mp3", False))
    strip_paths = bool(getattr(args, "strip_paths", False))

    # For deterministic outputs written to disk, never embed absolute temp paths:
    playlist_display = (
        _display_path_strip(playlist_path, base_dir=playlist_dir) if strip_paths else str(playlist_path)
    )

    for i, raw in enumerate(raw_paths):
        rp = _resolve_input_path(raw, playlist_dir=playlist_dir)

        if prefer_mp3:
            rp = _prefer_mp3_path(rp)

        if not rp.exists() or not rp.is_file():
            missing += 1
            bundled.append(
                {
                    "index": i,
                    "source": raw if not strip_paths else str(Path(raw)).replace("\\", "/"),
                    "resolved": (str(rp) if not strip_paths else _display_path_strip(rp, base_dir=playlist_dir)),
                    "ok": False,
                    "reason": "missing",
                    "resolved_from": resolved_from,
                }
            )
            continue

        dest = tracks_dir / rp.name
        if dest.exists():
            # Collision resolution is deterministic given:
            # - clean output dir
            # - stable raw_paths order
            stem = dest.stem
            suf = dest.suffix
            n = 1
            while True:
                cand = tracks_dir / f"{stem}_{n}{suf}"
                if not cand.exists():
                    dest = cand
                    break
                n += 1

        # Copy bytes; metadata not important for our determinism gates (hash is on bytes).
        shutil.copyfile(rp, dest)
        copied += 1
        bundled.append(
            {
                "index": i,
                "source": raw if not strip_paths else str(Path(raw)).replace("\\", "/"),
                "resolved": (str(rp) if not strip_paths else _display_path_strip(rp, base_dir=playlist_dir)),
                "ok": True,
                "bundle_path": _as_posix(dest.relative_to(out_dir)),
                "resolved_from": resolved_from,
            }
        )

    if bool(getattr(args, "fail_on_missing", False)) and missing > 0:
        out = {
            "ok": False,
            "playlist": str(playlist_path),
            "out_dir": str(out_dir),
            "reason": "missing_tracks",
            "track_count": len(raw_paths),
            "copied_count": copied,
            "missing_count": missing,
        }
        if db_resolution is not None:
            out["db_resolution"] = db_resolution
        sys.stdout.write(_stable_json_dumps(out) + "\n")
        return 2

    if bool(getattr(args, "fail_if_none_copied", False)) and copied == 0:
        out = {
            "ok": False,
            "playlist": str(playlist_path),
            "out_dir": str(out_dir),
            "reason": "no_tracks_copied",
            "track_count": len(raw_paths),
            "copied_count": copied,
            "missing_count": missing,
        }
        if db_resolution is not None:
            out["db_resolution"] = db_resolution
        sys.stdout.write(_stable_json_dumps(out) + "\n")
        return 2

    # Files written to disk must be stable across runs (esp. when built under temp dirs).
    web_playlist = {
        "version": 1,
        "source_playlist": playlist_display,
        "resolved_from": resolved_from,
        "tracks": [t for t in bundled if t.get("ok")],
    }
    _write_json(out_playlist_path, web_playlist)

    index_html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MGC Web Player</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; }
    button { padding: 8px 12px; }
    .row { margin: 8px 0; }
    .small { color: #666; font-size: 12px; }
    select { padding: 6px; min-width: 320px; }
  </style>
</head>
<body>
  <h1>MGC Web Player</h1>
  <div class="row">
    <select id="sel"></select>
    <button id="play">Play</button>
  </div>
  <audio id="audio" controls style="width: 100%; max-width: 800px;"></audio>
  <div class="row small" id="meta"></div>
<script>
(async function () {
  const res = await fetch('./playlist.json');
  const pl = await res.json();
  const tracks = (pl.tracks || []).map(t => t.bundle_path);
  const sel = document.getElementById('sel');
  const audio = document.getElementById('audio');
  const meta = document.getElementById('meta');

  function setSrc(p) {
    audio.src = './' + p;
    meta.textContent = p;
  }

  tracks.forEach((p, i) => {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = (i+1) + ' - ' + p.split('/').pop();
    sel.appendChild(opt);
  });

  if (tracks.length) setSrc(tracks[0]);

  sel.addEventListener('change', () => setSrc(sel.value));
  document.getElementById('play').addEventListener('click', () => {
    if (sel.value) setSrc(sel.value);
    audio.play();
  });
})();
</script>
</body>
</html>
"""
    index_path.write_text(index_html, encoding="utf-8", newline="\n")

    manifest_obj: Dict[str, Any] = {
        "ok": True,
        "playlist": playlist_display,
        # For portability/determinism, never embed temp out_dir paths in files on disk.
        "out_dir": ("." if strip_paths else str(out_dir)),
        "track_count": len(raw_paths),
        "copied_count": copied,
        "missing_count": missing,
        "tracks_dir": _as_posix(tracks_dir.relative_to(out_dir)),
        "playlist_json": _as_posix(out_playlist_path.relative_to(out_dir)),
        "index_html": _as_posix(index_path.relative_to(out_dir)),
        "resolved_from": resolved_from,
        "items": bundled,
        "db_resolution": db_resolution,
        "strip_paths": strip_paths,
    }
    _write_json(manifest_path, manifest_obj)

    # CLI JSON output can remain absolute (debugging), but keep it stable JSON.
    out_obj: Dict[str, Any] = {
        "ok": True,
        "playlist": str(playlist_path),
        "out_dir": str(out_dir),
        "track_count": len(raw_paths),
        "copied_count": copied,
        "missing_count": missing,
        "manifest_path": str(manifest_path),
        "resolved_from": resolved_from,
        "strip_paths": strip_paths,
    }
    if db_resolution is not None:
        out_obj["db_resolution"] = db_resolution

    sys.stdout.write(_stable_json_dumps(out_obj) + "\n")
    return 0


def cmd_web_serve(args: argparse.Namespace) -> int:
    directory = Path(str(args.dir)).expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        _eprint(f"[mgc.web] ERROR: directory not found: {directory}")
        return 2

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any):
            super().__init__(*a, directory=str(directory), **kw)

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
    build.add_argument("--db", default=os.environ.get("MGC_DB", ""), help="DB path (used only if playlist contains IDs)")
    build.add_argument("--prefer-mp3", action="store_true", help="Prefer .mp3 when a .wav sibling exists")
    build.add_argument("--clean", action="store_true", help="Delete out-dir contents before writing")
    build.add_argument("--fail-if-empty", action="store_true", help="Exit nonzero if playlist has zero tracks")
    build.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit nonzero if any referenced track path cannot be found",
    )
    build.add_argument(
        "--fail-if-none-copied",
        action="store_true",
        help="Exit nonzero if zero audio files were copied into the bundle",
    )
    build.add_argument(
        "--require-bundled-tracks",
        dest="require_bundled_tracks",
        action="store_true",
        help="Exit nonzero if DB resolution was used (forces portable bundle playlists with file paths)",
    )
    build.add_argument(
        "--strip-paths",
        dest="strip_paths",
        action="store_true",
        help="Write portable outputs: avoid embedding absolute/temp paths in playlist.json and web_manifest.json",
    )
    build.set_defaults(func=cmd_web_build)

    serve = ws.add_parser("serve", help="Serve a built web bundle")
    serve.add_argument("--dir", required=True, help="Directory to serve (output of web build)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_web_serve)
