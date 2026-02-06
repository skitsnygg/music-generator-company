#!/usr/bin/env python3
"""
src/mgc/web_cli.py

Static web bundle builder + simple dev server.

This version:
- `web build` supports `--no-audio` (default remains bundling audio).
- Robustly resolves audio when CI playlists contain placeholder track_id/path:
    If playlist has exactly 1 track and drop_bundle/tracks contains exactly 1 audio file,
    web build will use that audio file and override effective track_id to the file stem.
- Always writes a minimal index.html into the bundle if none was provided/found.
- `web validate` enforces bundle_audio true/false contract.
- `web serve` exposes /api/health /api/me /api/catalog /api/stream/<track_id>
  with GET + HEAD support and library-db-first streaming.

Tokens:
- Authorization: Bearer <token>
- or ?token=<token>
- or `web serve --token <token>` default token (dev convenience)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


WEB_MANIFEST_VERSION = 3
WEB_MANIFEST_SCHEMA = "mgc.web_manifest.v3"
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg")


def _find_mp3_start(b: bytes) -> Optional[int]:
    i = b.find(b"ID3")
    if i != -1:
        return i
    # MPEG frame sync: 0xFF followed by a byte whose top 3 bits are 1 (0xE0..0xFF)
    for j in range(max(0, len(b) - 1)):
        if b[j] == 0xFF and (b[j + 1] & 0xE0) == 0xE0:
            return j
    return None


def _sanitize_mp3_file_inplace(p: Path) -> None:
    """Best-effort MP3 header cleanup (deterministic)."""
    # 1) Strip any prefix before ID3 or MPEG frame sync.
    try:
        b = p.read_bytes()
        start = _find_mp3_start(b)
        if start is not None and start > 0:
            p.write_bytes(b[start:])
    except Exception:
        pass
    # 2) Use existing project logic if available.
    try:
        from mgc.clean_mp3_headers import clean_mp3_headers  # type: ignore
        clean_mp3_headers(str(p))
    except Exception:
        pass

# ---------------------------
# Utilities
# ---------------------------

def _try_import_stable_json() -> Optional[Any]:
    for name in ("orjson",):
        try:
            return __import__(name)
        except Exception:
            continue
    return None


def _try_import_die() -> Optional[Any]:
    try:
        from mgc.util import die  # type: ignore
        return die
    except Exception:
        return None


def _die(msg: str, code: int = 2) -> None:
    die_fn = _try_import_die()
    if die_fn:
        die_fn(msg, code)
    raise SystemExit(code)


def _stable_json_dumps(obj: Any) -> str:
    orjson = _try_import_stable_json()
    if orjson:
        return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _deterministic_now_iso() -> str:
    fixed = (os.environ.get("MGC_FIXED_TIME") or "").strip()
    if fixed:
        try:
            if fixed.isdigit():
                dt = datetime.fromtimestamp(int(fixed), tz=timezone.utc)
                return dt.isoformat(timespec="seconds")
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(fixed.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
        except Exception:
            pass
    return datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return bool(row)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ensure_portable_relpath(relpath: str) -> None:
    rel = Path(relpath)
    if rel.is_absolute():
        _die(f"unsafe relpath (absolute): {relpath}", 2)
    parts = rel.parts
    if any(p in ("..", "") for p in parts):
        _die(f"unsafe relpath (traversal): {relpath}", 2)


def _tree_hash(root: Path) -> str:
    # IMPORTANT:
    # Exclude web_manifest.json to avoid self-referential hashing.
    items: List[Tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if rel == "web_manifest.json":
            continue
        items.append((rel, _sha256_file(p)))
    payload = _stable_json_dumps(items)
    return _sha256_hex_str(payload)

def _looks_like_audio_path(s: str) -> bool:
    s2 = s.lower().strip()
    return any(s2.endswith(ext) for ext in AUDIO_EXTS)


def _collect_audio_paths_from_json(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_audio_paths_from_json(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_collect_audio_paths_from_json(item))
    elif isinstance(obj, str) and _looks_like_audio_path(obj):
        out.append(obj)
    return out


def _dig_first_str(d: Dict[str, Any], keys: Sequence[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _infer_track_paths_from_obj(track_obj: Dict[str, Any]) -> List[str]:
    for k in ("web_path", "path", "artifact_path", "preview_path", "full_path", "audio_path", "wav", "mp3"):
        v = track_obj.get(k)
        if isinstance(v, str) and _looks_like_audio_path(v):
            return [v]
    return _collect_audio_paths_from_json(track_obj)


def _resolve_input_path(raw: str, *, playlist_dir: Path, repo_root: Path) -> Path:
    s = (raw or "").strip()
    if not s:
        return Path("")
    p = Path(s).expanduser()
    if p.is_absolute():
        return p

    cand1 = playlist_dir / p
    if cand1.exists():
        return cand1

    cand2 = repo_root / p
    if cand2.exists():
        return cand2

    try:
        return Path(s).expanduser().resolve()
    except Exception:
        return Path(s).expanduser()


def _prefer_mp3_path(p: Path) -> Path:
    if p.suffix.lower() == ".wav":
        mp3 = p.with_suffix(".mp3")
        if mp3.exists():
            return mp3
    return p


def _prefer_existing_audio_path(p: Path) -> Path:
    if p.exists():
        return p
    suf = p.suffix.lower()
    if suf == ".mp3":
        wav = p.with_suffix(".wav")
        if wav.exists():
            return wav
    elif suf == ".wav":
        mp3 = p.with_suffix(".mp3")
        if mp3.exists():
            return mp3
    return p


def _list_audio_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    files: List[Path] = []
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


# ---------------------------
# Playlist parsing
# ---------------------------

def _iter_track_dicts(playlist_obj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("tracks", "items", "playlist", "entries"):
        v = playlist_obj.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    yield item


def _collect_entries(playlist_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, t in enumerate(_iter_track_dicts(playlist_obj)):
        track_id = _dig_first_str(t, ("track_id", "id"))
        title = _dig_first_str(t, ("title", "name"))
        raw_path = _dig_first_str(t, ("web_path", "path", "artifact_path", "preview_path", "full_path"))
        entries.append({
            "index": idx,
            "track_id": track_id,
            "title": title,
            "raw_path": raw_path,
            "track_obj": t,
        })
    return entries


def _manifest_generated_at(playlist_obj: Dict[str, Any], deterministic: bool) -> str:
    if deterministic:
        return _deterministic_now_iso()
    for k in ("date", "generated_at", "created_at", "ts"):
        v = playlist_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return _utc_now_iso()


# ---------------------------
# DB track path resolution (library DB)
# ---------------------------

def _resolve_track_paths_from_db(con: sqlite3.Connection, track_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
    if not track_ids:
        return {}
    if not _table_exists(con, "tracks"):
        return {}

    cols = {r["name"] for r in con.execute("PRAGMA table_info(tracks)").fetchall()}
    if not cols or "id" not in cols:
        return {}

    want_full = "full_path" in cols
    want_prev = "preview_path" in cols
    want_path = "path" in cols  # legacy

    sel_cols: List[str] = ["id"]
    if want_full:
        sel_cols.append("full_path")
    if want_prev:
        sel_cols.append("preview_path")
    if (not want_full and not want_prev) and want_path:
        sel_cols.append("path")

    q = f"SELECT {', '.join(sel_cols)} FROM tracks WHERE id IN ({','.join(['?'] * len(track_ids))})"
    rows = con.execute(q, list(track_ids)).fetchall()

    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        tid = str(r["id"])
        d: Dict[str, str] = {}
        if want_full:
            v = r["full_path"]
            if isinstance(v, str) and v.strip():
                d["full_path"] = v.strip()
        if want_prev:
            v = r["preview_path"]
            if isinstance(v, str) and v.strip():
                d["preview_path"] = v.strip()
        if (not want_full and not want_prev) and want_path:
            v = r["path"]
            if isinstance(v, str) and v.strip():
                d["full_path"] = v.strip()
        if d:
            out[tid] = d
    return out


# ---------------------------
# Manifest build + validate
# ---------------------------

def _build_web_manifest(
    *,
    out_dir: Path,
    playlist_obj: Dict[str, Any],
    tracks_payload: List[Dict[str, Any]],
    deterministic: bool,
    bundle_audio: bool,
    marketing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    playlist_sha = _sha256_hex_str(_stable_json_dumps(playlist_obj))
    web_tree = _tree_hash(out_dir)

    out = {
        "schema": WEB_MANIFEST_SCHEMA,
        "version": WEB_MANIFEST_VERSION,
        "generated_at": _manifest_generated_at(playlist_obj, deterministic),
        "bundle_audio": bool(bundle_audio),
        "playlist_sha256": playlist_sha,
        "web_tree_sha256": web_tree,
        "tracks": tracks_payload,
    }
    if marketing:
        out["marketing"] = marketing
    return out


def _validate_web_manifest(out_dir: Path, manifest: Dict[str, Any]) -> None:
    schema = manifest.get("schema")
    version = manifest.get("version")
    if schema != WEB_MANIFEST_SCHEMA:
        _die(f"web_manifest schema mismatch: got {schema!r} expected {WEB_MANIFEST_SCHEMA!r}", 2)
    if int(version or 0) != WEB_MANIFEST_VERSION:
        _die(f"web_manifest version mismatch: got {version!r} expected {WEB_MANIFEST_VERSION!r}", 2)

    bundle_audio = bool(manifest.get("bundle_audio", True))

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        _die("web_manifest invalid: tracks must be a list", 2)

    tracks_dir = out_dir / "tracks"
    if not bundle_audio:
        if tracks_dir.exists():
            for p in tracks_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                    _die(f"web_manifest invalid: bundle_audio=false but found audio file {p.relative_to(out_dir)}", 2)

    for t in tracks:
        if not isinstance(t, dict):
            _die("web_manifest invalid: track entry must be a dict", 2)
        if not t.get("track_id"):
            _die("web_manifest invalid: track missing track_id", 2)

        has_rel = bool(t.get("relpath"))
        has_sha = bool(t.get("sha256"))
        has_bytes = t.get("bytes") is not None

        if bundle_audio:
            if not (has_rel and has_sha and has_bytes):
                _die("web_manifest invalid: bundle_audio=true requires relpath/sha256/bytes for every track", 2)
            relpath = str(t.get("relpath") or "")
            sha = str(t.get("sha256") or "")
            _ensure_portable_relpath(relpath)
            p = (out_dir / relpath).resolve()
            if not p.exists():
                _die(f"web_manifest invalid: missing file {relpath}", 2)
            got = _sha256_file(p)
            if got != sha:
                _die(f"web_manifest invalid: hash mismatch for {relpath} got={got} expected={sha}", 2)
        else:
            if has_rel or has_sha or has_bytes:
                _die("web_manifest invalid: bundle_audio=false must not include relpath/sha256/bytes in track entries", 2)

    got_tree = _tree_hash(out_dir)
    exp_tree = str(manifest.get("web_tree_sha256") or "")
    if exp_tree and got_tree != exp_tree:
        _die(f"web_manifest invalid: web_tree_sha256 mismatch got={got_tree} expected={exp_tree}", 2)


# ---------------------------
# Embedded index.html fallback
# ---------------------------

_EMBEDDED_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="color-scheme" content="dark" />
  <title>Music Generator Company — Player</title>
  <style>
    :root{
      --bg: #000000;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.10);
      --border: rgba(255,255,255,0.14);
      --text: rgba(255,255,255,0.94);
      --muted: rgba(255,255,255,0.62);
      --muted2: rgba(255,255,255,0.45);
      --accent: #7c5cff;
      --accent2: #35d6c7;
      --danger: #ff5c7a;
      --ok: #37d67a;
      --shadow: 0 10px 28px rgba(0,0,0,0.65);
      --radius: 14px;
      --radius2: 18px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }

    *{ box-sizing:border-box; }
    html,body{ height:100%; }
    body{
      margin:0;
      background: var(--bg);
      color:var(--text);
      font-family:var(--sans);
      line-height:1.35;
    }
    a{ color:inherit; text-decoration:none; }
    a:hover{ text-decoration:underline; }

    .wrap{
      max-width: 1300px;
      margin: 0 auto;
      padding: 22px 16px 40px;
    }
    .top{
      display:flex;
      gap:14px;
      align-items:center;
      justify-content:space-between;
      margin-bottom: 14px;
      flex-wrap:wrap;
    }
    .brand{
      display:flex;
      gap:12px;
      align-items:center;
      min-width: 280px;
    }
    .logo{
      width:42px;height:42px;border-radius:12px;
      background: linear-gradient(135deg, rgba(124,92,255,0.95), rgba(53,214,199,0.85));
      box-shadow: var(--shadow);
    }
    .brand h1{
      font-size: 16px;
      margin:0;
      letter-spacing:0.2px;
    }
    .brand .sub{
      font-size: 12px;
      color: var(--muted);
      margin-top: 2px;
    }

    .pillbar{
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      justify-content:flex-end;
    }
    .pill{
      display:inline-flex;
      align-items:center;
      gap:8px;
      padding:7px 10px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
    }
    .pill code{
      font-family:var(--mono);
      font-size: 11px;
      color: var(--text);
    }

    .grid{
      display:grid;
      grid-template-columns: 380px 1fr;
      gap: 14px;
    }
    @media (max-width: 980px){
      .grid{ grid-template-columns: 1fr; }
    }

    .card{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius2);
      box-shadow: var(--shadow);
      overflow:hidden;
    }
    .cardHead{
      padding: 12px 12px 10px;
      border-bottom: 1px solid var(--border);
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap: 10px;
    }
    .cardTitle{
      margin:0;
      font-size: 13px;
      letter-spacing: 0.2px;
    }
    .cardMeta{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .cardBody{
      padding: 12px;
    }

    .row{
      display:flex;
      gap:10px;
      align-items:center;
    }
    .spacer{ flex:1; }

    .btn{
      appearance:none;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      padding: 9px 11px;
      border-radius: 12px;
      font-size: 12px;
      cursor:pointer;
      transition: transform .06s ease, background .12s ease, border-color .12s ease;
      user-select:none;
      display:inline-flex;
      gap: 8px;
      align-items:center;
      justify-content:center;
    }
    .btn:hover{ background: var(--panel2); }
    .btn:active{ transform: translateY(1px); }
    .btn.primary{
      border-color: rgba(124,92,255,0.55);
      background: rgba(124,92,255,0.12);
    }
    .btn.small{ padding: 7px 9px; border-radius: 10px; }

    .input{
      width:100%;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      padding: 10px 11px;
      border-radius: 12px;
      font-size: 13px;
      outline:none;
    }

    .list{
      display:flex;
      flex-direction:column;
      gap:10px;
    }
    .item{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      border-radius: 14px;
      padding: 11px;
      cursor:pointer;
      transition: background .12s ease, transform .06s ease, border-color .12s ease;
    }
    .item:hover{ background: rgba(255,255,255,0.06); }
    .item:active{ transform: translateY(1px); }
    .item.active{
      border-color: rgba(124,92,255,0.65);
      background: rgba(124,92,255,0.10);
    }
    .itemTop{
      display:flex;
      align-items:center;
      gap:10px;
    }
    .badge{
      font-family: var(--mono);
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      white-space:nowrap;
    }
    .itemTitle{
      font-size: 13px;
      margin:0;
      color: var(--text);
      letter-spacing:0.1px;
    }
    .itemSub{
      margin-top: 6px;
      display:flex;
      gap:10px;
      flex-wrap:wrap;
      color: var(--muted);
      font-size: 12px;
    }

    .now{
      display:flex;
      gap:12px;
      align-items:flex-start;
    }
    .cover{
      width: 86px;
      height: 86px;
      border-radius: 16px;
      position: relative;
      overflow: hidden;
      background:
        radial-gradient(60px 60px at 30% 30%, rgba(124,92,255,0.55), transparent 60%),
        radial-gradient(70px 70px at 70% 60%, rgba(53,214,199,0.45), transparent 62%),
        rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      flex: 0 0 auto;
    }
    .cover img{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: none;
    }
    .cover.hasImage img{ display: block; }
    .now h2{
      margin:0;
      font-size: 16px;
      letter-spacing:0.2px;
    }
    .muted{ color: var(--muted); }
    .mono{ font-family: var(--mono); }

    .player{
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.03);
    }
    audio{ width: 100%; }

    .controls{
      display:flex;
      gap: 10px;
      align-items:center;
      flex-wrap:wrap;
      margin-top: 10px;
    }
    .kbd{
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      padding: 3px 6px;
      border-radius: 8px;
    }

    .inspector{
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.03);
    }
    .inspectorHead{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      margin-bottom: 10px;
    }
    .inspectGrid{
      display:grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap:10px;
    }
    @media (max-width: 720px){
      .inspectGrid{ grid-template-columns: 1fr; }
    }
    .inspectItem{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      border-radius: 12px;
      padding: 10px;
      display:flex;
      flex-direction:column;
      gap:6px;
      min-height: 72px;
    }
    .inspectLabel{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--muted2);
    }
    .inspectValue{
      font-size: 12px;
      color: var(--text);
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }
    .inspectActions{
      display:flex;
      gap:8px;
      align-items:center;
    }
    .inspectHint{
      margin-top: 8px;
      font-size: 11px;
      color: var(--muted2);
    }

    .marketing{
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,0.02);
    }
    .marketingHead{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
      margin-bottom: 10px;
    }
    .marketingGrid{
      display:grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap:10px;
    }
    @media (max-width: 720px){
      .marketingGrid{ grid-template-columns: 1fr; }
    }
    .marketingItem{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      border-radius: 12px;
      padding: 10px;
      display:flex;
      flex-direction:column;
      gap:6px;
      min-height: 72px;
    }
    .marketingLabel{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--muted2);
    }
    .marketingValue{
      font-size: 12px;
      color: var(--text);
    }
    .marketingActions{
      display:flex;
      gap:8px;
      align-items:center;
      flex-wrap:wrap;
    }
    .marketingPosts{
      margin-top: 10px;
      display:grid;
      gap:10px;
    }
    .postItem{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      border-radius: 12px;
      padding: 10px;
      display:flex;
      flex-direction:column;
      gap:8px;
    }
    .postTitle{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--muted2);
    }
    .postText{
      font-size: 13px;
      color: var(--text);
      white-space: pre-wrap;
    }

    .toast{
      position: fixed;
      left: 16px;
      bottom: 16px;
      max-width: 560px;
      padding: 12px 12px;
      border-radius: 14px;
      background: rgba(0,0,0,0.84);
      color: rgba(255,255,255,0.92);
      border: 1px solid rgba(255,255,255,0.18);
      box-shadow: var(--shadow);
      display:none;
      z-index: 9999;
    }
    .toast.show{ display:block; }
    .toast .t1{ font-size: 12px; margin:0; }
    .toast .t2{ font-size: 12px; margin: 6px 0 0; color: rgba(255,255,255,0.70); }

    .hr{
      height:1px;
      background: var(--border);
      margin: 12px 0;
    }

    .foot{
      margin-top: 12px;
      color: var(--muted2);
      font-size: 12px;
    }
    .foot code{ font-family: var(--mono); }

    .toggle{
      display:flex;
      align-items:center;
      gap:8px;
      color: var(--muted);
      font-size: 12px;
      user-select:none;
      white-space:nowrap;
    }
    .toggle input{ transform: translateY(1px); }
  </style>
</head>

<body>
  <div class="wrap">
    <div class="top">
      <div class="brand">
        <div class="logo" aria-hidden="true"></div>
        <div>
          <h1>Music Generator Company</h1>
          <div class="sub">All playlists + all tracks • feed-driven web player</div>
        </div>
      </div>
      <div class="pillbar">
        <div class="pill">Feed: <code id="pillFeed">/releases/feed.json</code></div>
        <div class="pill">Mode: <code id="pillMode">auto</code></div>
        <div class="pill">Playlists: <code id="pillPl">0</code></div>
        <div class="pill">Tracks: <code id="pillTr">0</code></div>
      </div>
    </div>

    <div class="grid">
      <!-- LEFT: Playlists -->
      <div class="card">
        <div class="cardHead">
          <div>
            <p class="cardTitle">Playlists</p>
            <div class="cardMeta" id="plMeta">Loading feed…</div>
          </div>
          <div class="row" style="gap:8px;">
            <button class="btn small" id="btnReload" title="Reload feed + playlists">Reload</button>
          </div>
        </div>
        <div class="cardBody">
          <div class="row" style="margin-bottom:10px;">
            <input class="input" id="plSearch" placeholder="Filter playlists (focus, sleep, workout)..." />
          </div>

          <div class="list" id="playlists"></div>

          <div class="hr"></div>

          <div class="row" style="gap:8px; flex-wrap:wrap;">
            <button class="btn small" id="btnCopyLink" title="Copy share link">Copy link</button>
            <button class="btn small" id="btnOpenBundle" title="Open selected playlist bundle">Open bundle</button>
            <div class="spacer"></div>
            <span class="kbd">J/K</span><span class="kbd">Space</span><span class="kbd">N/P</span>
          </div>

          <div class="foot">
            Works on VM (<code>/releases/feed.json</code>) and GitHub Pages (auto-detects repo base path).
          </div>
        </div>
      </div>

      <!-- RIGHT: Player + All tracks -->
      <div class="card">
        <div class="cardHead">
          <div>
            <p class="cardTitle">Now playing</p>
            <div class="cardMeta" id="nowMeta">Loading…</div>
          </div>
          <div class="row" style="gap:8px;">
            <button class="btn small" id="btnPrev" title="Previous track">Prev</button>
            <button class="btn small primary" id="btnPlay" title="Play/Pause">Play</button>
            <button class="btn small" id="btnNext" title="Next track">Next</button>
          </div>
        </div>

        <div class="cardBody">
          <div class="now">
            <div class="cover" id="cover"><img id="coverImg" alt="Cover art"></div>
            <div style="min-width:0;">
              <h2 id="trackTitle">—</h2>
              <div class="muted" id="trackSub" style="margin-top:6px;">—</div>
              <div class="muted mono" id="trackPath" style="margin-top:8px; font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"></div>
            </div>
          </div>

          <div class="player">
            <audio id="audio" controls preload="metadata"></audio>
            <div class="controls">
              <button class="btn small" id="btnSeekBack" title="Back 10s">-10s</button>
              <button class="btn small" id="btnSeekFwd" title="Forward 10s">+10s</button>
              <div class="spacer"></div>

              <label class="toggle" title="Show only audio-looking files (.mp3/.wav/.ogg/.m4a)">
                <input type="checkbox" id="audioOnly" />
                Audio only
              </label>

              <span class="muted" style="font-size:12px;">Volume</span>
              <input id="vol" type="range" min="0" max="1" step="0.01" value="0.85" style="width:160px;">
            </div>
          </div>

          <div class="hr"></div>

          <div class="inspector" id="inspector">
            <div class="inspectorHead">
              <p class="cardTitle">Release Inspector</p>
              <div class="row" style="gap:8px;">
                <button class="btn small" id="btnCopyInspector" title="Copy all inspector fields">Copy all</button>
              </div>
            </div>

            <div class="inspectGrid">
              <div class="inspectItem">
                <div class="inspectLabel">Feed content_sha256</div>
                <div class="inspectValue mono" id="inspectFeedSha">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="feed_sha" data-label="Feed content_sha256">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Playlist context</div>
                <div class="inspectValue mono" id="inspectPlaylist">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="playlist_context" data-label="Playlist context">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Playlist sha256</div>
                <div class="inspectValue mono" id="inspectPlaylistSha">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="playlist_sha" data-label="Playlist sha256">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Web tree sha256</div>
                <div class="inspectValue mono" id="inspectWebTreeSha">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="web_tree_sha" data-label="Web tree sha256">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Manifest URL</div>
                <div class="inspectValue mono" id="inspectManifestUrl">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="manifest_url" data-label="Manifest URL">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Playlist URL</div>
                <div class="inspectValue mono" id="inspectPlaylistUrl">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="playlist_url" data-label="Playlist URL">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Track relpath</div>
                <div class="inspectValue mono" id="inspectTrackRel">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="track_relpath" data-label="Track relpath">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Track sha256</div>
                <div class="inspectValue mono" id="inspectTrackSha">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="track_sha" data-label="Track sha256">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Track bytes</div>
                <div class="inspectValue mono" id="inspectTrackBytes">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="track_bytes" data-label="Track bytes">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Marketing summary</div>
                <div class="inspectValue" id="inspectMarketingSummary">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="marketing_summary" data-label="Marketing summary">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Hashtags</div>
                <div class="inspectValue mono" id="inspectMarketingHashtags">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="marketing_hashtags" data-label="Hashtags">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Marketing cover URL</div>
                <div class="inspectValue mono" id="inspectMarketingCover">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="marketing_cover_url" data-label="Marketing cover URL">Copy</button>
                </div>
              </div>

              <div class="inspectItem">
                <div class="inspectLabel">Marketing media URL</div>
                <div class="inspectValue mono" id="inspectMarketingMedia">—</div>
                <div class="inspectActions">
                  <button class="btn small" data-copy="marketing_media_url" data-label="Marketing media URL">Copy</button>
                </div>
              </div>
            </div>

            <div class="inspectHint">Updates with the current track and selected playlist.</div>
          </div>

          <div class="marketing" id="marketing">
            <div class="marketingHead">
              <p class="cardTitle">Marketing Preview</p>
              <div class="row" style="gap:8px;">
                <button class="btn small" id="btnCopyMarketingAll" title="Copy summary + hashtags + posts">Copy all</button>
              </div>
            </div>

            <div class="marketingGrid">
              <div class="marketingItem">
                <div class="marketingLabel">Summary</div>
                <div class="marketingValue" id="mkSummary">—</div>
                <div class="marketingActions">
                  <button class="btn small" data-mkcopy="summary" data-mklabel="Summary">Copy</button>
                </div>
              </div>

              <div class="marketingItem">
                <div class="marketingLabel">Hashtags</div>
                <div class="marketingValue mono" id="mkHashtags">—</div>
                <div class="marketingActions">
                  <button class="btn small" data-mkcopy="hashtags" data-mklabel="Hashtags">Copy</button>
                </div>
              </div>

              <div class="marketingItem">
                <div class="marketingLabel">Teaser audio</div>
                <div class="marketingValue">
                  <audio id="mkTeaser" controls preload="metadata"></audio>
                  <div class="muted" id="mkTeaserEmpty">No teaser available.</div>
                </div>
                <div class="marketingActions">
                  <button class="btn small" data-mkcopy="teaser_url" data-mklabel="Teaser URL">Copy link</button>
                  <button class="btn small" id="mkTeaserOpen">Open</button>
                </div>
              </div>
            </div>

            <div class="marketingPosts" id="mkPosts"></div>
            <div class="muted" id="mkEmpty">No marketing preview available for this playlist.</div>
          </div>

          <div class="hr"></div>

          <div class="row" style="margin-bottom:10px;">
            <input class="input" id="trSearch" placeholder="Filter tracks (title, playlist, path)..." />
          </div>

          <div class="list" id="tracks"></div>

          <div class="foot" id="bundleFoot"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="toast" id="toast">
    <p class="t1" id="toastT1"></p>
    <p class="t2" id="toastT2"></p>
  </div>

  <script src="bundle_data.js"></script>
  <script>
    "use strict";

    function $(id){ return document.getElementById(id); }

    function toast(t1, t2){
      const el = $("toast");
      $("toastT1").textContent = t1 || "";
      $("toastT2").textContent = t2 || "";
      el.classList.add("show");
      clearTimeout(toast._t);
      toast._t = setTimeout(() => el.classList.remove("show"), 2600);
    }

    function copyText(text, label){
      const value = text ? String(text) : "";
      if (!value){
        toast("Nothing to copy", label ? (label + " is empty.") : "Value is empty.");
        return Promise.resolve(false);
      }
      return navigator.clipboard.writeText(value).then(
        () => { toast("Copied", label || value); return true; },
        () => { toast("Copy failed", "Your browser blocked clipboard access."); return false; }
      );
    }

    function fmtIso(iso){
      if (!iso) return "—";
      try{
        const d = new Date(iso);
        return d.toISOString().replace(".000Z","Z");
      }catch{
        return iso;
      }
    }

    function escapeHtml(s){
      return String(s).replace(/[&<>"']/g, (c) => ({
        "&":"&amp;",
        "<":"&lt;",
        ">":"&gt;",
        "\"":"&quot;",
        "'":"&#039;",
      }[c]));
    }

    function basePrefix(){
      // Directory containing this index.html
      const p = window.location.pathname;
      if (p.endsWith("/index.html")) return p.slice(0, -"/index.html".length);
      if (p.endsWith("/")) return p.slice(0, -1);
      return p;
    }

    function urlJoin(prefix, path){
      // prefix like "" or "/music-generator-company"
      if (!prefix) return path.startsWith("/") ? path : ("/" + path);
      if (!path.startsWith("/")) path = "/" + path;
      return prefix + path;
    }

    function resolveFeedUrl(){
      const prefix = basePrefix();
      return urlJoin(prefix, "/releases/feed.json");
    }

    function repoBaseFromPrefix(prefix){
      const markers = ["/latest/web/", "/releases/"];
      for (const m of markers){
        const idx = prefix.indexOf(m);
        if (idx >= 0) return prefix.slice(0, idx);
      }
      return prefix;
    }

    function feedUrlCandidates(){
      const primary = resolveFeedUrl();
      const prefix = basePrefix();
      const repoBase = repoBaseFromPrefix(prefix);
      const repoFeed = urlJoin(repoBase, "/releases/feed.json");
      const root = "/releases/feed.json";
      const out = [primary];
      if (repoFeed && repoFeed !== primary) out.push(repoFeed);
      if (primary !== root) out.push(root);
      if (window.location.protocol === "file:"){
        const rels = [
          "../releases/feed.json",
          "../../releases/feed.json",
          "../../../releases/feed.json",
          "../../../../releases/feed.json",
          "../../../../../releases/feed.json",
        ];
        rels.forEach((rel) => out.push(urlJoin(prefix, rel)));
      }
      return Array.from(new Set(out));
    }

    async function fetchJson(url){
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
      return await r.json();
    }

    async function fetchText(url){
      try{
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) return null;
        return await r.text();
      }catch (e){
        return null;
      }
    }

    function isAudioPath(p){
      return typeof p === "string" && (p.endsWith(".mp3") || p.endsWith(".wav") || p.endsWith(".ogg") || p.endsWith(".m4a"));
    }

    function playlistTracks(playlist){
      if (!playlist) return [];
      if (Array.isArray(playlist)) return playlist;
      if (Array.isArray(playlist.tracks)) return playlist.tracks;
      if (Array.isArray(playlist.items)) return playlist.items;
      return [];
    }

    function trackDisplay(t, idx){
      const title =
        t.title ||
        t.name ||
        t.track_title ||
        (t.id ? `Track ${String(t.id).slice(0,8)}` : `Track ${idx+1}`);

      const path =
        t.web_path ||
        t.webPath ||
        t.dest ||
        t.path ||
        t.audio ||
        t.audio_path ||
        t.file ||
        "";

      return { title, path, raw: t };
    }

    function shortHash(s){
      if (!s) return "—";
      if (s.length <= 18) return s;
      return s.slice(0, 10) + "..." + s.slice(-6);
    }

    function stripOrigin(u){
      if (!u) return "";
      try{
        const origin = window.location.origin || "";
        if (origin && u.startsWith(origin)) return u.slice(origin.length) || "/";
      }catch{
        return u;
      }
      return u;
    }

    function absoluteUrl(u){
      if (!u) return "";
      try{
        return new URL(u, window.location.href).toString();
      }catch{
        return u;
      }
    }

    function normalizeRelpath(p){
      return String(p || "").replace(/^\.?\//, "");
    }

    function sanitizeRelpath(p){
      let out = normalizeRelpath(p);
      while (out.startsWith("../")){
        out = out.slice(3);
      }
      return out;
    }

    function marketingAssetUrl(bundleBase, rel){
      const cleaned = sanitizeRelpath(rel);
      if (!cleaned) return "";
      if (cleaned.startsWith("marketing/")){
        return withTrailingSlash(bundleBase) + cleaned;
      }
      return withTrailingSlash(bundleBase) + "marketing/" + cleaned;
    }

    function planHashtagsText(plan){
      if (!plan) return "";
      if (plan.hashtags_text) return String(plan.hashtags_text || "").trim();
      if (Array.isArray(plan.hashtags)){
        return plan.hashtags.map(t => "#" + String(t || "").trim()).join(" ").trim();
      }
      return "";
    }

    async function loadMarketingPreview(plan, bundleBase){
      if (!plan) return null;
      const paths = (plan && plan.paths) ? plan.paths : {};

      let summary = String(plan.summary || "").trim();
      const summaryRel = paths.summary || "";
      if (summaryRel){
        const text = await fetchText(marketingAssetUrl(bundleBase, summaryRel));
        if (text) summary = text.trim();
      }

      let hashtags = planHashtagsText(plan);
      const hashtagsRel = paths.hashtags || "";
      if (hashtagsRel){
        const text = await fetchText(marketingAssetUrl(bundleBase, hashtagsRel));
        if (text) hashtags = text.trim();
      }

      const teaserRel = paths.teaser || "";
      const teaserUrl = teaserRel ? marketingAssetUrl(bundleBase, teaserRel) : "";

      const postPaths = Array.isArray(paths.posts) ? paths.posts : [];
      const posts = [];
      if (postPaths.length){
        const items = await Promise.all(
          postPaths.map(async (rel, idx) => {
            const url = marketingAssetUrl(bundleBase, rel);
            const text = await fetchText(url);
            return {
              index: idx + 1,
              url,
              text: text ? text.trim() : "",
            };
          })
        );
        for (const it of items){
          posts.push(it);
        }
      }

      const out = {
        summary,
        hashtags,
        teaser_url: teaserUrl,
        posts,
      };

      if (!summary && !hashtags && !teaserUrl && !posts.length) return null;
      return out;
    }

    function marketingMetaFromPlan(plan, bundleBase){
      if (!plan || typeof plan !== "object") return null;
      const paths = (plan && typeof plan.paths === "object") ? plan.paths : {};
      const coverObj = (plan.cover && typeof plan.cover === "object") ? plan.cover : {};
      const mediaObj = (plan.media && typeof plan.media === "object") ? plan.media : {};

      let summary = String(plan.summary || "").trim();
      let hashtagsText = "";
      if (plan.hashtags_text) hashtagsText = String(plan.hashtags_text || "").trim();
      else if (Array.isArray(plan.hashtags)){
        hashtagsText = plan.hashtags.map(t => "#" + String(t || "").trim()).join(" ").trim();
      }

      let coverUrl = "";
      if (plan.cover_url) coverUrl = String(plan.cover_url || "").trim();
      let coverRel = "";
      if (!coverUrl){
        coverRel = String(coverObj.dst || coverObj.path || "").trim();
        if (!coverRel && paths.cover) coverRel = String(paths.cover || "").trim();
        if (coverRel) coverUrl = marketingAssetUrl(bundleBase, coverRel);
      }

      let mediaUrl = "";
      if (mediaObj.video_url || mediaObj.media_url){
        mediaUrl = String(mediaObj.video_url || mediaObj.media_url || "").trim();
      }
      let mediaRel = "";
      if (!mediaUrl){
        mediaRel = String(mediaObj.video_path || mediaObj.media_path || "").trim();
        if (!mediaRel && paths.media) mediaRel = String(paths.media || "").trim();
        if (mediaRel) mediaUrl = marketingAssetUrl(bundleBase, mediaRel);
      }

      if (!summary && !hashtagsText && !coverUrl && !mediaUrl) return null;
      return {
        summary,
        hashtags_text: hashtagsText,
        cover_url: coverUrl,
        media_url: mediaUrl,
      };
    }

    function embeddedBundle(){
      const b = window.__MGC_BUNDLE__;
      if (!b || typeof b !== "object") return null;
      return b;
    }

    function findManifestTrack(manifest, track){
      if (!manifest || !track) return null;
      const tracks = Array.isArray(manifest.tracks) ? manifest.tracks : [];
      if (!tracks.length) return null;

      const raw = track.raw || {};
      const ids = [raw.track_id, raw.trackId, raw.id, track.track_id, track.id].filter(Boolean);
      for (const id of ids){
        const hit = tracks.find(t => t && t.track_id === id);
        if (hit) return hit;
      }

      const path = track.path || "";
      if (path){
        const n = normalizeRelpath(path);
        const hit = tracks.find(t => t && normalizeRelpath(t.relpath || "") === n);
        if (hit) return hit;
      }

      const src = track.src || "";
      if (src){
        const hit = tracks.find(t => t && typeof t.relpath === "string" && src.endsWith(t.relpath));
        if (hit) return hit;
      }

      const title = track.title || "";
      if (title){
        const hit = tracks.find(t => t && t.title === title);
        if (hit) return hit;
      }

      if (Number.isFinite(track.playlist_idx)){
        const hit = tracks.find(t => t && t.index === track.playlist_idx);
        if (hit) return hit;
      }

      return null;
    }

    function normalizeBundleBase(ctx){
      const prefix = basePrefix();
      const u = (ctx && ctx.url) ? ctx.url : "";
      return urlJoin(prefix, u);
    }

    function withTrailingSlash(u){
      return u.endsWith("/") ? u : (u + "/");
    }

    function computeTrackSrc(bundleBase, path){
      if (!path) return "";
      if (path.startsWith("http://") || path.startsWith("https://")) return path;
      if (path.startsWith("/")) return urlJoin(basePrefix(), path);
      return withTrailingSlash(bundleBase) + path.replace(/^\.?\//, "");
    }

    async function loadBundle(ctx){
      const bundleBase = withTrailingSlash(normalizeBundleBase(ctx));
      const playlistUrl = bundleBase + "playlist.json";
      const manifestUrl = bundleBase + "web_manifest.json";
      const marketingUrl = bundleBase + "marketing/marketing_plan.json";

      const embedded = embeddedBundle();
      let embeddedPreview = null;
      if (embedded && embedded.marketing_preview){
        const preview = embedded.marketing_preview || null;
        let teaserUrl = preview && preview.teaser_path ? marketingAssetUrl(bundleBase, preview.teaser_path) : "";
        let posts = [];
        if (preview && Array.isArray(preview.posts)){
          posts = preview.posts.map((p, i) => ({
            index: p.index || (i + 1),
            url: "",
            text: (p && p.text) ? String(p.text).trim() : "",
          }));
        }
        embeddedPreview = {
          summary: preview.summary || "",
          hashtags: preview.hashtags || "",
          teaser_url: teaserUrl,
          posts,
        };
      }

      if (embedded && (!ctx || !ctx.url)){
        const [playlist, manifest, marketingPlan] = await Promise.all([
          fetchJson(playlistUrl).catch(() => null),
          fetchJson(manifestUrl).catch(() => null),
          fetchJson(marketingUrl).catch(() => null),
        ]);

        const pickedPlaylist = playlist || embedded.playlist || null;
        const pickedPlan = marketingPlan || embedded.marketing_plan || null;
        let marketingPreview = embeddedPreview;
        if (!marketingPreview){
          try{
            marketingPreview = await loadMarketingPreview(pickedPlan, bundleBase);
          }catch (e){
            marketingPreview = null;
          }
        }

        return {
          bundleBase,
          playlistUrl,
          manifestUrl,
          playlist: pickedPlaylist,
          manifest,
          marketingPlan: pickedPlan,
          marketingPreview,
        };
      }

      const [playlist, manifest, marketingPlan] = await Promise.all([
        fetchJson(playlistUrl),
        fetchJson(manifestUrl).catch(() => null),
        fetchJson(marketingUrl).catch(() => null),
      ]);

      let marketingPreview = null;
      try{
        marketingPreview = await loadMarketingPreview(marketingPlan, bundleBase);
      }catch (e){
        marketingPreview = null;
      }

      return { bundleBase, playlistUrl, manifestUrl, playlist, manifest, marketingPlan, marketingPreview };
    }

    function setModeLabel(){
      const p = basePrefix();
      $("pillMode").textContent = p ? "pages" : "vm";
    }

    function setCounts(state){
      $("pillPl").textContent = String(state.playlists.length);
      $("pillTr").textContent = String(state.allTracks.length);
    }

    function currentShareUrl(state){
      const u = new URL(window.location.href);
      if (state && state.selected && state.selected.context){
        u.searchParams.set("playlist", state.selected.context);
      }else{
        u.searchParams.delete("playlist");
      }
      return u.toString();
    }

    function setActivePlaylistItem(name){
      const items = $("playlists").querySelectorAll(".item");
      for (const it of items){
        it.classList.toggle("active", it.dataset.context === name);
      }
    }

    function renderPlaylists(state){
      const list = $("playlists");
      list.innerHTML = "";

      const q = $("plSearch").value.trim().toLowerCase();
      const shown = q
        ? state.playlists.filter(p => (p.context || "").toLowerCase().includes(q))
        : state.playlists;

      // Always show an "All" pseudo-playlist
      const allDiv = document.createElement("div");
      allDiv.className = "item";
      allDiv.dataset.context = "__all__";
      if (!state.selected) allDiv.classList.add("active");
      allDiv.innerHTML = `
        <div class="itemTop">
          <span class="badge">all</span>
          <h3 class="itemTitle">All playlists</h3>
          <span class="spacer"></span>
          <span class="badge">${state.allTracks.length} tracks</span>
        </div>
        <div class="itemSub">
          <span class="mono">Includes every loaded playlist + track</span>
        </div>
      `;
      list.appendChild(allDiv);

      if (!shown.length){
        const div = document.createElement("div");
        div.className = "muted";
        div.style.padding = "6px 2px";
        div.textContent = "No playlists found.";
        list.appendChild(div);
        return;
      }

      for (const p of shown){
        const div = document.createElement("div");
        div.className = "item";
        div.dataset.context = p.context;

        const mtime = fmtIso(p.mtime);
        const tracks = p.tracks ? p.tracks.length : 0;
        const url = p.url || "";

        if (state.selected && state.selected.context === p.context) div.classList.add("active");

        div.innerHTML = `
          <div class="itemTop">
            <span class="badge">playlist</span>
            <h3 class="itemTitle">${escapeHtml(p.context)}</h3>
            <span class="spacer"></span>
            <span class="badge">${tracks} tracks</span>
          </div>
          <div class="itemSub">
            <span>mtime: <span class="mono">${escapeHtml(mtime)}</span></span>
            <span>url: <span class="mono">${escapeHtml(url)}</span></span>
          </div>
        `;
        list.appendChild(div);
      }
    }

    function filteredTracks(state){
      const q = $("trSearch").value.trim().toLowerCase();
      const audioOnly = $("audioOnly").checked;

      let base = state.allTracks;
      if (state.selected){
        base = base.filter(t => t.playlist_context === state.selected.context);
      }

      if (audioOnly){
        base = base.filter(t => isAudioPath(t.path || ""));
      }

      if (!q) return base;

      return base.filter(t => {
        const hay = [
          t.title || "",
          t.playlist_context || "",
          t.path || "",
          t.src || "",
        ].join(" ").toLowerCase();
        return hay.includes(q);
      });
    }

    function renderTracks(state){
      const list = $("tracks");
      list.innerHTML = "";

      const shown = filteredTracks(state);

      if (!shown.length){
        const div = document.createElement("div");
        div.className = "muted";
        div.textContent = "No tracks match your filters.";
        list.appendChild(div);
        return;
      }

      for (let i = 0; i < shown.length; i++){
        const t = shown[i];
        const div = document.createElement("div");
        div.className = "item";
        div.dataset.gidx = String(t.global_idx);

        const active = (state.current && state.current.global_idx === t.global_idx);
        if (active) div.classList.add("active");

        const badge = isAudioPath(t.path || "") ? "audio" : "file";
        div.innerHTML = `
          <div class="itemTop">
            <span class="badge">${badge}</span>
            <h3 class="itemTitle">${escapeHtml(t.title || "—")}</h3>
            <span class="spacer"></span>
            <span class="badge">${escapeHtml(t.playlist_context || "—")}</span>
          </div>
          <div class="itemSub">
            <span class="mono" style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 100%;">${escapeHtml(t.path || "—")}</span>
          </div>
        `;
        list.appendChild(div);
      }
    }

    function setInspectorValue(id, value, opts){
      const el = $(id);
      if (!el) return;
      const has = value !== undefined && value !== null && String(value).length;
      if (!has){
        el.textContent = "—";
        el.title = "";
        return;
      }

      const raw = String(value);
      let display = raw;
      if (opts && opts.shortHash) display = shortHash(raw);
      if (opts && opts.shortUrl) display = stripOrigin(raw);
      if (opts && opts.bytes) display = raw + " bytes";
      el.textContent = display;
      el.title = (display !== raw) ? raw : "";
    }

    function setCoverImage(url){
      const cover = $("cover");
      const img = $("coverImg");
      if (!cover || !img) return;
      if (url){
        cover.classList.add("hasImage");
        img.src = url;
      }else{
        cover.classList.remove("hasImage");
        img.removeAttribute("src");
      }
    }

    function setMarketingValue(id, value){
      const el = $(id);
      if (!el) return;
      const has = value !== undefined && value !== null && String(value).length;
      el.textContent = has ? String(value) : "—";
    }

    function marketingSummaryText(mk){
      if (!mk) return "";
      const lines = [];
      if (mk.summary) lines.push("Summary: " + mk.summary);
      if (mk.hashtags) lines.push("Hashtags: " + mk.hashtags);
      if (mk.teaser_url) lines.push("Teaser: " + mk.teaser_url);
      if (mk.posts && mk.posts.length){
        lines.push("Posts:");
        mk.posts.forEach((p, i) => {
          const txt = p && p.text ? String(p.text).trim() : "";
          if (txt) lines.push((i + 1) + ". " + txt);
        });
      }
      return lines.join("\n");
    }

    function setMarketingPreview(state){
      const pl = state.selected ? state.selected : (state.current ? findPlaylistByName(state, state.current.playlist_context) : null);
      const mk = pl && pl.marketing ? pl.marketing : null;
      state.marketingPreview = mk;

      const empty = $("mkEmpty");
      const postsEl = $("mkPosts");
      const teaser = $("mkTeaser");
      const teaserEmpty = $("mkTeaserEmpty");
      const teaserOpen = $("mkTeaserOpen");

      if (!mk){
        setMarketingValue("mkSummary", "");
        setMarketingValue("mkHashtags", "");
        if (postsEl) postsEl.innerHTML = "";
        if (teaser){
          teaser.removeAttribute("src");
          teaser.load();
          teaser.style.display = "none";
        }
        if (teaserEmpty) teaserEmpty.style.display = "";
        if (teaserOpen) teaserOpen.setAttribute("disabled", "disabled");
        if (empty) empty.style.display = "";
        return;
      }

      if (empty) empty.style.display = "none";
      setMarketingValue("mkSummary", mk.summary || "");
      setMarketingValue("mkHashtags", mk.hashtags || "");

      if (teaser){
        if (mk.teaser_url){
          teaser.src = mk.teaser_url;
          teaser.style.display = "";
          if (teaserEmpty) teaserEmpty.style.display = "none";
          if (teaserOpen) teaserOpen.removeAttribute("disabled");
        }else{
          teaser.removeAttribute("src");
          teaser.load();
          teaser.style.display = "none";
          if (teaserEmpty) teaserEmpty.style.display = "";
          if (teaserOpen) teaserOpen.setAttribute("disabled", "disabled");
        }
      }

      if (postsEl){
        postsEl.innerHTML = "";
        if (mk.posts && mk.posts.length){
          mk.posts.forEach((p, i) => {
            const text = p && p.text ? String(p.text).trim() : "";
            const div = document.createElement("div");
            div.className = "postItem";
            div.innerHTML = `
              <div class="postTitle">Post ${i + 1}</div>
              <div class="postText">${escapeHtml(text || "(missing)")}</div>
              <div class="marketingActions">
                <button class="btn small" data-mkpost="${i}">Copy</button>
              </div>
            `;
            postsEl.appendChild(div);
          });
        }else{
          const div = document.createElement("div");
          div.className = "muted";
          div.textContent = "No posts available.";
          postsEl.appendChild(div);
        }
      }
    }

    function setInspector(state){
      const cur = state.current || null;
      const pl = cur ? findPlaylistByName(state, cur.playlist_context) : state.selected;
      const manifest = pl && pl.manifest ? pl.manifest : null;
      const mTrack = findManifestTrack(manifest, cur);
      const marketing = manifest && manifest.marketing ? manifest.marketing : null;
      const plan = pl && pl.marketingPlan ? pl.marketingPlan : null;
      const planMeta = marketingMetaFromPlan(plan, pl ? pl.bundleBase : "");
      let summaryText = "";
      if (marketing && marketing.summary) summaryText = String(marketing.summary || "").trim();
      if (!summaryText && planMeta && planMeta.summary) summaryText = planMeta.summary;
      let hashtagsText = "";
      if (marketing){
        if (marketing.hashtags_text) hashtagsText = String(marketing.hashtags_text || "").trim();
        else if (Array.isArray(marketing.hashtags)){
          hashtagsText = marketing.hashtags.map(t => "#" + String(t || "").trim()).join(" ").trim();
        }
      }
      if (!hashtagsText && planMeta && planMeta.hashtags_text) hashtagsText = planMeta.hashtags_text;
      let marketingMediaUrl = "";
      if (marketing){
        if (marketing.media_url){
          marketingMediaUrl = String(marketing.media_url || "").trim();
        }else if (marketing.marketing_media_path){
          marketingMediaUrl = computeTrackSrc(pl ? pl.bundleBase : "", String(marketing.marketing_media_path || "").trim());
        }else if (marketing.media_path){
          marketingMediaUrl = computeTrackSrc(pl ? pl.bundleBase : "", "marketing/" + String(marketing.media_path || "").replace(/^\/+/, ""));
        }
      }
      if (!marketingMediaUrl && planMeta && planMeta.media_url) marketingMediaUrl = planMeta.media_url;
      let marketingCoverUrl = "";
      if (marketing){
        if (marketing.cover_url){
          marketingCoverUrl = String(marketing.cover_url || "").trim();
        }else if (marketing.marketing_cover_path){
          marketingCoverUrl = computeTrackSrc(pl ? pl.bundleBase : "", String(marketing.marketing_cover_path || "").trim());
        }else if (marketing.cover_path){
          marketingCoverUrl = computeTrackSrc(pl ? pl.bundleBase : "", "marketing/" + String(marketing.cover_path || "").replace(/^\/+/, ""));
        }
      }
      if (!marketingCoverUrl && planMeta && planMeta.cover_url) marketingCoverUrl = planMeta.cover_url;

      const values = {
        feed_sha: state.feed && state.feed.content_sha256 ? state.feed.content_sha256 : "",
        playlist_context: pl && pl.context ? pl.context : "",
        playlist_sha: manifest && manifest.playlist_sha256 ? manifest.playlist_sha256 : "",
        web_tree_sha: manifest && manifest.web_tree_sha256 ? manifest.web_tree_sha256 : "",
        manifest_url: pl && pl.manifestUrl ? absoluteUrl(pl.manifestUrl) : "",
        playlist_url: pl && pl.playlistUrl ? absoluteUrl(pl.playlistUrl) : "",
        track_relpath: (mTrack && mTrack.relpath) || (cur && cur.path) || "",
        track_sha: (mTrack && mTrack.sha256) || "",
        track_bytes: (mTrack && mTrack.bytes != null) ? String(mTrack.bytes) : "",
        marketing_summary: summaryText,
        marketing_hashtags: hashtagsText,
        marketing_media_url: marketingMediaUrl,
        marketing_cover_url: marketingCoverUrl,
      };

      state.inspector = values;

      setInspectorValue("inspectFeedSha", values.feed_sha, { shortHash: true });
      setInspectorValue("inspectPlaylist", values.playlist_context, {});
      setInspectorValue("inspectPlaylistSha", values.playlist_sha, { shortHash: true });
      setInspectorValue("inspectWebTreeSha", values.web_tree_sha, { shortHash: true });
      setInspectorValue("inspectManifestUrl", values.manifest_url, { shortUrl: true });
      setInspectorValue("inspectPlaylistUrl", values.playlist_url, { shortUrl: true });
      setInspectorValue("inspectTrackRel", values.track_relpath, {});
      setInspectorValue("inspectTrackSha", values.track_sha, { shortHash: true });
      setInspectorValue("inspectTrackBytes", values.track_bytes, { bytes: true });
      setInspectorValue("inspectMarketingSummary", values.marketing_summary, {});
      setInspectorValue("inspectMarketingHashtags", values.marketing_hashtags, {});
      setInspectorValue("inspectMarketingCover", values.marketing_cover_url, { shortUrl: true });
      setInspectorValue("inspectMarketingMedia", values.marketing_media_url, { shortUrl: true });
      setCoverImage(values.marketing_cover_url);
    }

    function inspectorSummary(state){
      const v = state.inspector || {};
      const pairs = [
        ["feed_content_sha256", v.feed_sha],
        ["playlist_context", v.playlist_context],
        ["playlist_sha256", v.playlist_sha],
        ["web_tree_sha256", v.web_tree_sha],
        ["manifest_url", v.manifest_url],
        ["playlist_url", v.playlist_url],
        ["track_relpath", v.track_relpath],
        ["track_sha256", v.track_sha],
        ["track_bytes", v.track_bytes],
        ["marketing_summary", v.marketing_summary],
        ["marketing_hashtags", v.marketing_hashtags],
        ["marketing_cover_url", v.marketing_cover_url],
        ["marketing_media_url", v.marketing_media_url],
      ];
      return pairs.map((p) => p[0] + ": " + (p[1] || "—")).join("\n");
    }

    function setNowPlaying(state){
      const t = state.current || null;
      if (!t){
        $("trackTitle").textContent = "—";
        $("trackSub").textContent = "—";
        $("trackPath").textContent = "";
        $("nowMeta").textContent = "Pick a track from the list.";
        $("bundleFoot").textContent = "";
        setInspector(state);
        return;
      }

      $("trackTitle").textContent = t.title || "—";
      $("trackSub").textContent = `playlist: ${t.playlist_context || "—"}`;
      $("trackPath").textContent = t.src || "";

      const gen = state.feed && state.feed.generated_at ? fmtIso(state.feed.generated_at) : "—";
      const sha = state.feed && state.feed.content_sha256 ? state.feed.content_sha256.slice(0, 16) + "…" : "—";
      $("nowMeta").textContent = `feed generated_at: ${gen} • content_sha256: ${sha}`;

      const footBits = [];
      if (t.bundleBase) footBits.push(`bundle: ${t.bundleBase}`);
      if (t.playlistUrl) footBits.push(`playlist: ${t.playlistUrl}`);
      if (t.manifestUrl) footBits.push(`manifest: ${t.manifestUrl}`);
      $("bundleFoot").textContent = footBits.join(" • ");
      setInspector(state);
      setMarketingPreview(state);
    }

    function setAudioSource(state){
      const a = $("audio");
      const t = state.current || null;
      if (!t || !t.src){
        a.removeAttribute("src");
        a.load();
        $("btnPlay").textContent = "Play";
        return;
      }
      a.src = t.src;
      a.load();
      $("btnPlay").textContent = "Play";
    }

    function playPause(){
      const a = $("audio");
      if (!a.src){
        toast("No track loaded", "Pick a track first.");
        return;
      }
      if (a.paused){
        a.play().catch(() => {});
        $("btnPlay").textContent = "Pause";
      }else{
        a.pause();
        $("btnPlay").textContent = "Play";
      }
    }

    function seekBy(seconds){
      const a = $("audio");
      if (!isFinite(a.duration)) return;
      a.currentTime = Math.max(0, Math.min(a.duration, a.currentTime + seconds));
    }

    function pickPlayable(tracks){
      const idx = tracks.findIndex(t => isAudioPath(t.path || "") && t.src);
      return idx >= 0 ? tracks[idx] : (tracks[0] || null);
    }

    function nextPrev(state, delta){
      const shown = filteredTracks(state);
      if (!shown.length) return;

      const cur = state.current ? shown.findIndex(x => x.global_idx === state.current.global_idx) : -1;
      const j = (cur < 0) ? 0 : (cur + delta + shown.length) % shown.length;

      state.current = shown[j];
      setNowPlaying(state);
      setAudioSource(state);

      renderTracks(state);

      const a = $("audio");
      if (!a.paused){
        a.play().catch(() => {});
        $("btnPlay").textContent = "Pause";
      }
    }

    async function loadAllPlaylists(state){
      const feedUrls = feedUrlCandidates();
      let feed = null;
      let feedUrl = "";
      for (const cand of feedUrls){
        try{
          feed = await fetchJson(cand);
          feedUrl = cand;
          break;
        }catch (e){
          // try next candidate
        }
      }

      const out = [];
      let failed = 0;

      if (feed){
        $("pillFeed").textContent = stripOrigin(feedUrl);
        state.feed = feed;

        const ctxs = (feed && feed.latest && Array.isArray(feed.latest.contexts)) ? feed.latest.contexts : [];
        const webCtxs = ctxs.filter(x => x.kind === "web");

        $("plMeta").textContent = webCtxs.length ? `Found ${webCtxs.length} playlists. Loading bundles…` : "No web playlists found in feed.";

        // Load each bundle with a small concurrency limit to avoid a burst
        const limit = 4;
        const q = webCtxs.slice();

        async function worker(){
          while (q.length){
            const ctx = q.shift();
            try{
              const bundle = await loadBundle(ctx);
              const rawTracks = playlistTracks(bundle.playlist).map((t, i) => {
                const d = trackDisplay(t, i);
                return {
                  title: d.title,
                  path: d.path,
                  raw: d.raw,
                  src: computeTrackSrc(bundle.bundleBase, d.path),
                };
              });

              out.push({
                context: ctx.context,
                mtime: ctx.mtime,
                url: ctx.url,
                track_count: ctx.track_count,
                bundleBase: bundle.bundleBase,
                playlistUrl: bundle.playlistUrl,
                manifestUrl: bundle.manifestUrl,
                manifest: bundle.manifest,
                marketingPlan: bundle.marketingPlan || null,
                marketing: bundle.marketingPreview || null,
                tracks: rawTracks,
              });
            }catch (e){
              failed++;
              console.error("Failed to load playlist:", ctx && ctx.context, e);
            }
          }
        }

        const workers = [];
        for (let i = 0; i < Math.min(limit, webCtxs.length); i++){
          workers.push(worker());
        }
        await Promise.all(workers);

        const msg = failed ? `Loaded ${out.length}/${webCtxs.length} playlists (${failed} failed).` : `Loaded ${out.length} playlists.`;
        $("plMeta").textContent = msg;
      }else{
        $("pillFeed").textContent = "local";
        $("plMeta").textContent = "Feed not available. Loading local bundle…";
        state.feed = null;

        const ctx = { context: "local", url: "" };
        const bundle = await loadBundle(ctx);
        const playlist = bundle.playlist || {};
        const ctxName = playlist.context || playlist.name || "local";
        const rawTracks = playlistTracks(bundle.playlist).map((t, i) => {
          const d = trackDisplay(t, i);
          return {
            title: d.title,
            path: d.path,
            raw: d.raw,
            src: computeTrackSrc(bundle.bundleBase, d.path),
          };
        });

        out.push({
          context: ctxName,
          mtime: (bundle.manifest && bundle.manifest.generated_at) ? bundle.manifest.generated_at : "",
          url: "",
          track_count: rawTracks.length,
          bundleBase: bundle.bundleBase,
          playlistUrl: bundle.playlistUrl,
          manifestUrl: bundle.manifestUrl,
          manifest: bundle.manifest,
          marketingPlan: bundle.marketingPlan || null,
          marketing: bundle.marketingPreview || null,
          tracks: rawTracks,
        });
        $("plMeta").textContent = `Loaded local bundle (${ctxName}).`;
        toast("Local bundle", "Feed not available; loaded playlist.json in this bundle.");
      }

      // Sort playlists (context name)
      out.sort((a,b) => String(a.context).localeCompare(String(b.context)));

      state.playlists = out;

      // Flatten tracks
      const all = [];
      let g = 0;
      for (const p of out){
        for (let i = 0; i < (p.tracks || []).length; i++){
          const t = p.tracks[i];
          all.push({
            global_idx: g++,
            playlist_context: p.context,
            playlist_idx: i,
            title: t.title,
            path: t.path,
            src: t.src,
            bundleBase: p.bundleBase,
            playlistUrl: p.playlistUrl,
            manifestUrl: p.manifestUrl,
          });
        }
      }
      state.allTracks = all;

      setCounts(state);

      if (!state.current){
        const pick = pickPlayable(state.allTracks);
        state.current = pick;
        setNowPlaying(state);
        setAudioSource(state);
      }
    }

    function findPlaylistByName(state, name){
      return state.playlists.find(p => p.context === name) || null;
    }

    function setSelectedPlaylist(state, pl){
      state.selected = pl;
      renderPlaylists(state);
      renderTracks(state);

      if (pl){
        toast("Playlist selected", pl.context);
      }else{
        toast("Playlist selected", "All playlists");
      }

      // If current track isn't in the selected playlist anymore, pick the first playable in view
      const shown = filteredTracks(state);
      if (!shown.length){
        state.current = null;
        setNowPlaying(state);
        setAudioSource(state);
        return;
      }
      if (!state.current || !shown.some(x => state.current && x.global_idx === state.current.global_idx)){
        state.current = pickPlayable(shown);
        setNowPlaying(state);
        setAudioSource(state);
      }else{
        setMarketingPreview(state);
      }
    }

    function movePlaylistSelection(state, delta){
      if (!state.playlists.length) return;

      // When "All" is selected, treat it as index -1; J selects first playlist
      const curName = state.selected ? state.selected.context : null;
      const i = curName ? state.playlists.findIndex(p => p.context === curName) : -1;
      const j = Math.max(-1, Math.min(state.playlists.length - 1, i + delta));

      if (j < 0) setSelectedPlaylist(state, null);
      else setSelectedPlaylist(state, state.playlists[j]);
    }

    async function main(){
      const state = {
        feed: null,
        playlists: [],
        allTracks: [],
        selected: null, // null = All
        current: null,
        inspector: {},
        marketingPreview: null,
      };

      setModeLabel();

      try{
        await loadAllPlaylists(state);
        renderPlaylists(state);
        renderTracks(state);

        // Support ?playlist=focus
        const u = new URL(window.location.href);
        const want = u.searchParams.get("playlist");
        if (want){
          const pl = findPlaylistByName(state, want);
          if (pl) setSelectedPlaylist(state, pl);
        }

        toast("Ready", "Loaded playlists and tracks.");
      }catch (e){
        console.error(e);
        $("plMeta").textContent = "Failed to load feed/playlists.";
        toast("Load failed", String(e && e.message ? e.message : e));
      }

      $("btnReload").addEventListener("click", async () => {
        try{
          $("plMeta").textContent = "Reloading…";
          state.feed = null;
          state.playlists = [];
          state.allTracks = [];
          state.selected = null;
          state.current = null;
          await loadAllPlaylists(state);
          renderPlaylists(state);
          renderTracks(state);
          toast("Reloaded", "Loaded latest playlists and tracks.");
        }catch (e){
          toast("Reload failed", String(e && e.message ? e.message : e));
        }
      });

      $("plSearch").addEventListener("input", () => renderPlaylists(state));
      $("trSearch").addEventListener("input", () => renderTracks(state));
      $("audioOnly").addEventListener("change", () => {
        renderTracks(state);
        // keep current sane
        const shown = filteredTracks(state);
        if (state.current && !shown.some(x => x.global_idx === state.current.global_idx)){
          state.current = pickPlayable(shown);
          setNowPlaying(state);
          setAudioSource(state);
        }
      });

      $("playlists").addEventListener("click", (ev) => {
        const item = ev.target.closest(".item");
        if (!item) return;
        const name = item.dataset.context;

        if (name === "__all__"){
          setSelectedPlaylist(state, null);
          return;
        }
        const pl = findPlaylistByName(state, name);
        if (pl) setSelectedPlaylist(state, pl);
      });

      $("tracks").addEventListener("click", (ev) => {
        const item = ev.target.closest(".item");
        if (!item) return;
        const gidx = parseInt(item.dataset.gidx || "-1", 10);
        if (!Number.isFinite(gidx) || gidx < 0) return;

        const t = state.allTracks.find(x => x.global_idx === gidx);
        if (!t) return;

        state.current = t;
        setNowPlaying(state);
        setAudioSource(state);
        renderTracks(state);
      });

      $("btnPlay").addEventListener("click", () => playPause());
      $("btnPrev").addEventListener("click", () => nextPrev(state, -1));
      $("btnNext").addEventListener("click", () => nextPrev(state, +1));
      $("btnSeekBack").addEventListener("click", () => seekBy(-10));
      $("btnSeekFwd").addEventListener("click", () => seekBy(+10));

      $("vol").addEventListener("input", (ev) => {
        const v = parseFloat(ev.target.value);
        $("audio").volume = Math.max(0, Math.min(1, v));
      });
      $("audio").volume = parseFloat($("vol").value);

      $("audio").addEventListener("play", () => { $("btnPlay").textContent = "Pause"; });
      $("audio").addEventListener("pause", () => { $("btnPlay").textContent = "Play"; });
      $("audio").addEventListener("ended", () => { nextPrev(state, +1); });

      $("btnCopyLink").addEventListener("click", () => {
        const u = currentShareUrl(state);
        copyText(u, "Share link");
      });

      $("btnCopyInspector").addEventListener("click", () => {
        copyText(inspectorSummary(state), "Inspector summary");
      });

      $("inspector").addEventListener("click", (ev) => {
        const btn = ev.target.closest("button[data-copy]");
        if (!btn) return;
        const key = btn.dataset.copy || "";
        const label = btn.dataset.label || key;
        const val = state.inspector && state.inspector[key] ? state.inspector[key] : "";
        copyText(val, label);
      });

      $("btnCopyMarketingAll").addEventListener("click", () => {
        const mk = state.marketingPreview;
        copyText(marketingSummaryText(mk), "Marketing preview");
      });

      $("marketing").addEventListener("click", (ev) => {
        const btn = ev.target.closest("button");
        if (!btn) return;
        const mk = state.marketingPreview;
        if (!mk) return;

        const mkcopy = btn.dataset.mkcopy || "";
        if (mkcopy){
          let val = "";
          if (mkcopy === "summary") val = mk.summary || "";
          if (mkcopy === "hashtags") val = mk.hashtags || "";
          if (mkcopy === "teaser_url") val = mk.teaser_url || "";
          const label = btn.dataset.mklabel || mkcopy;
          copyText(val, label);
          return;
        }

        const mkpost = btn.dataset.mkpost || "";
        if (mkpost){
          const idx = parseInt(mkpost, 10);
          if (Number.isFinite(idx) && mk.posts && mk.posts[idx]){
            const txt = mk.posts[idx].text || "";
            copyText(txt, `Post ${idx + 1}`);
          }
        }
      });

      $("mkTeaserOpen").addEventListener("click", () => {
        const mk = state.marketingPreview;
        if (mk && mk.teaser_url){
          window.open(mk.teaser_url, "_blank", "noopener");
        }
      });

      $("btnOpenBundle").addEventListener("click", () => {
        const pl = state.selected;
        if (!pl){
          toast("No bundle", "Select a playlist first.");
          return;
        }
        const u = normalizeBundleBase({ url: pl.url });
        window.open(u, "_blank", "noopener");
      });

      // Keyboard shortcuts:
      // J/K = move playlist selection (includes "All")
      // Space = play/pause
      // N/P = next/prev track (in the currently filtered track list)
      // Left/Right = seek
      window.addEventListener("keydown", (ev) => {
        const tag = (ev.target && ev.target.tagName) ? ev.target.tagName.toLowerCase() : "";
        const inInput = tag === "input" || tag === "textarea";
        if (inInput) return;

        if (ev.key === " "){
          ev.preventDefault();
          playPause();
          return;
        }
        if (ev.key === "j" || ev.key === "J"){ movePlaylistSelection(state, +1); return; }
        if (ev.key === "k" || ev.key === "K"){ movePlaylistSelection(state, -1); return; }
        if (ev.key === "n" || ev.key === "N"){ nextPrev(state, +1); return; }
        if (ev.key === "p" || ev.key === "P"){ nextPrev(state, -1); return; }
        if (ev.key === "ArrowLeft"){ seekBy(-5); return; }
        if (ev.key === "ArrowRight"){ seekBy(+5); return; }
      });
    }

    main();
  </script>
</body>
</html>
"""


# ---------------------------
# web build
# ---------------------------

def _normalize_playlist_path_arg(p: Path) -> Path:
    """Allow passing a run output directory instead of a playlist.json path."""
    if p.exists() and p.is_dir():
        cand = p / "playlist.json"
        if cand.exists():
            return cand
        cand2 = p / "drop_bundle" / "playlist.json"
        if cand2.exists():
            return cand2
    return p


def _load_marketing_plan_for_playlist(playlist_path: Path) -> Optional[Tuple[Dict[str, Any], Path]]:
    playlist_dir = playlist_path.parent
    candidates = [
        playlist_dir / "marketing" / "marketing_plan.json",
        playlist_dir.parent / "marketing" / "marketing_plan.json",
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            obj = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj, cand.parent
    return None


def _copy_marketing_assets(plan_dir: Optional[Path], out_dir: Path) -> None:
    if not plan_dir or not plan_dir.exists() or not plan_dir.is_dir():
        return
    dest = out_dir / "marketing"
    try:
        if plan_dir.resolve() == dest.resolve():
            return
    except Exception:
        pass
    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(plan_dir, dest)
    except Exception:
        pass


def _marketing_meta_from_plan(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return None

    hashtags = plan.get("hashtags") if isinstance(plan.get("hashtags"), list) else None
    hashtags_text = str(plan.get("hashtags_text") or "").strip()
    if not hashtags_text and isinstance(hashtags, list):
        tags = [str(t).strip() for t in hashtags if str(t).strip()]
        hashtags_text = " ".join([f"#{t}" for t in tags]).strip()

    summary = str(plan.get("summary") or "").strip()

    media_obj = plan.get("media") if isinstance(plan.get("media"), dict) else None
    media_path = ""
    media_url = ""
    if isinstance(media_obj, dict):
        media_path = str(media_obj.get("video_path") or media_obj.get("media_path") or "").strip()
        media_url = str(media_obj.get("video_url") or media_obj.get("media_url") or "").strip()

    marketing_media_path = ""
    if media_path:
        posix_path = Path(media_path).as_posix().lstrip("/")
        marketing_media_path = posix_path if posix_path.startswith("marketing/") else f"marketing/{posix_path}"

    out: Dict[str, Any] = {}
    if summary:
        out["summary"] = summary
    if isinstance(hashtags, list) and hashtags:
        out["hashtags"] = [str(t).strip() for t in hashtags if str(t).strip()]
    if hashtags_text:
        out["hashtags_text"] = hashtags_text
    if media_path:
        out["media_path"] = media_path
    if media_url:
        out["media_url"] = media_url
    if marketing_media_path:
        out["marketing_media_path"] = marketing_media_path

    cover_path = ""
    cover_obj = plan.get("cover") if isinstance(plan.get("cover"), dict) else None
    if isinstance(cover_obj, dict):
        cover_path = str(cover_obj.get("dst") or cover_obj.get("path") or "").strip()
    if not cover_path and isinstance(plan.get("paths"), dict):
        cover_path = str(plan.get("paths", {}).get("cover") or "").strip()

    marketing_cover_path = ""
    if cover_path:
        posix_path = Path(cover_path).as_posix().lstrip("/")
        marketing_cover_path = posix_path if posix_path.startswith("marketing/") else f"marketing/{posix_path}"

    if cover_path:
        out["cover_path"] = cover_path
    if marketing_cover_path:
        out["marketing_cover_path"] = marketing_cover_path

    return out if out else None


def _read_text_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _marketing_preview_from_plan(plan: Dict[str, Any], plan_dir: Path) -> Optional[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return None

    paths = plan.get("paths") if isinstance(plan.get("paths"), dict) else {}

    summary_text = _read_text_optional(plan_dir / str(paths.get("summary") or ""))
    if not summary_text:
        summary_text = str(plan.get("summary") or "").strip()

    hashtags_text = _read_text_optional(plan_dir / str(paths.get("hashtags") or ""))
    if not hashtags_text:
        hashtags_text = str(plan.get("hashtags_text") or "").strip()
    if not hashtags_text and isinstance(plan.get("hashtags"), list):
        tags = [str(t).strip() for t in plan.get("hashtags") if str(t).strip()]
        hashtags_text = " ".join([f"#{t}" for t in tags]).strip()

    teaser_rel = str(paths.get("teaser") or "").strip()
    posts: List[Dict[str, Any]] = []
    post_paths = paths.get("posts") if isinstance(paths.get("posts"), list) else []
    for idx, rel in enumerate(post_paths, start=1):
        text = _read_text_optional(plan_dir / str(rel))
        posts.append({"index": idx, "text": text})

    if not summary_text and not hashtags_text and not teaser_rel and not posts:
        return None

    return {
        "summary": summary_text,
        "hashtags": hashtags_text,
        "teaser_path": teaser_rel,
        "posts": posts,
    }

def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(str(getattr(args, "playlist"))).expanduser().resolve()
    playlist_path = _normalize_playlist_path_arg(playlist_path)
    out_dir = Path(str(getattr(args, "out_dir"))).expanduser().resolve()

    if getattr(args, "clean", False) and out_dir.exists():
        shutil.rmtree(out_dir)

    _safe_mkdir(out_dir)

    if not playlist_path.exists():
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "playlist_missing", "playlist": str(playlist_path)}) + "\n")
        return 2

    repo_root = Path(str(getattr(args, "repo_root", "."))).expanduser().resolve()
    playlist_dir = playlist_path.parent

    playlist_obj = json.loads(playlist_path.read_text(encoding="utf-8"))
    marketing_plan = None
    marketing_plan_dir = None
    marketing_preview = None
    plan_data = _load_marketing_plan_for_playlist(playlist_path)
    if plan_data:
        marketing_plan, marketing_plan_dir = plan_data
    marketing_meta = _marketing_meta_from_plan(marketing_plan) if marketing_plan else None
    if marketing_plan and marketing_plan_dir:
        marketing_preview = _marketing_preview_from_plan(marketing_plan, marketing_plan_dir)
        _copy_marketing_assets(marketing_plan_dir, out_dir)

    prefer_mp3 = bool(getattr(args, "prefer_mp3", False))
    fail_if_empty = bool(getattr(args, "fail_if_empty", False))
    fail_if_none_copied = bool(getattr(args, "fail_if_none_copied", False))
    fail_on_missing = bool(getattr(args, "fail_on_missing", False))
    deterministic = bool(getattr(args, "deterministic", False))

    bundle_audio = not bool(getattr(args, "no_audio", False))

    entries = _collect_entries(playlist_obj)
    if fail_if_empty and not entries:
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "empty_playlist"}) + "\n")
        return 2

    # CI robustness: if playlist has 1 track and drop_bundle/tracks has 1 audio file,
    # and the playlist path doesn't exist, use that file and override effective track_id.
    bundle_tracks_dir = playlist_dir / "tracks"
    bundle_audio_files = _list_audio_files(bundle_tracks_dir)

    db_map: Dict[str, Dict[str, str]] = {}
    db_raw = str(getattr(args, "db", "") or "").strip()
    if not db_raw:
        db_raw = str(os.environ.get("MGC_DB") or "").strip()
    if bundle_audio and db_raw:
        db_path = _resolve_input_path(db_raw, playlist_dir=playlist_dir, repo_root=repo_root)
        if db_path.exists():
            try:
                con = _connect(str(db_path))
                try:
                    ids = [e["track_id"] for e in entries if e.get("track_id")]
                    db_map = _resolve_track_paths_from_db(con, ids)
                finally:
                    con.close()
            except Exception:
                db_map = {}

    tracks_dir = out_dir / "tracks"
    if bundle_audio:
        _safe_mkdir(tracks_dir)
    else:
        if tracks_dir.exists():
            shutil.rmtree(tracks_dir)

    bundled: List[Dict[str, Any]] = []
    copied = 0
    missing = 0

    for e in entries:
        i = int(e.get("index", 0))
        orig_track_id = str(e.get("track_id") or "").strip()
        title = str(e.get("title") or orig_track_id or "").strip()
        raw = str(e.get("raw_path") or "").strip()

        if not bundle_audio:
            bundled.append({"index": i, "track_id": orig_track_id, "title": title})
            continue

        attempted: List[str] = []
        src_path: Optional[Path] = None
        resolved_from: Optional[str] = None
        effective_track_id = orig_track_id

        # 1) DB mapping (full_path -> preview_path)
        if effective_track_id and effective_track_id in db_map:
            d = db_map[effective_track_id]
            for k in ("full_path", "preview_path"):
                if k in d:
                    rp = _resolve_input_path(d[k], playlist_dir=playlist_dir, repo_root=repo_root)
                    if prefer_mp3:
                        rp = _prefer_mp3_path(rp)
                    rp = _prefer_existing_audio_path(rp)
                    attempted.append(str(rp))
                    if rp.exists() and rp.is_file():
                        src_path = rp
                        resolved_from = f"db:{k}"
                        break

        # 2) Playlist-provided path(s)
        if src_path is None:
            candidates: List[str] = []
            if raw:
                candidates.append(raw)
            candidates.extend(_infer_track_paths_from_obj(e.get("track_obj") or {}))
            for c in candidates:
                rp = _resolve_input_path(c, playlist_dir=playlist_dir, repo_root=repo_root)
                if prefer_mp3:
                    rp = _prefer_mp3_path(rp)
                rp = _prefer_existing_audio_path(rp)
                attempted.append(str(rp))
                if rp.exists() and rp.is_file():
                    src_path = rp
                    resolved_from = "playlist"
                    break

        # 3) CI placeholder fallback:
        # If playlist has exactly one entry and drop_bundle/tracks has exactly one audio file,
        # and we still couldn't resolve, use that file and override track_id to filename stem.
        if src_path is None and len(entries) == 1 and len(bundle_audio_files) == 1:
            src_path = bundle_audio_files[0].resolve()
            resolved_from = "bundle_tracks_singleton"
            effective_track_id = src_path.stem
            if not title or title == orig_track_id:
                title = effective_track_id
            attempted.append(str(src_path))

        # Guard: never allow empty effective_track_id; derive from filename if needed.
        # This prevents hard-fail validation in CI when playlist entries have missing/blank track_id.
        if src_path is not None and (not str(effective_track_id or "").strip()):
            effective_track_id = src_path.stem
        if not title:
            title = effective_track_id or orig_track_id

        if src_path is None or (not src_path.exists()) or (not src_path.is_file()):
            missing += 1
            bundled.append({
                "index": i,
                "ok": False,
                "reason": "missing",
                "track_id": orig_track_id,
                "title": title,
                "source": raw,
                "attempted": attempted,
                "resolved_from": resolved_from or None,
            })
            if fail_on_missing:
                sys.stdout.write(_stable_json_dumps({
                    "ok": False,
                    "reason": "missing_track",
                    "index": i,
                    "track_id": orig_track_id,
                    "title": title,
                    "attempted": attempted,
                }) + "\n")
                return 2
            continue

        ext = src_path.suffix.lower()
        dst_name = f"{effective_track_id}{ext}"
        relpath = f"tracks/{dst_name}"
        _ensure_portable_relpath(relpath)
        dst = (out_dir / relpath).resolve()

        try:
            shutil.copy2(src_path, dst)
            sha = _sha256_file(dst)
            size = dst.stat().st_size
            track_obj = e.get("track_obj")
            if isinstance(track_obj, dict):
                track_obj["web_path"] = relpath
            bundled.append({
                "index": i,
                "track_id": effective_track_id,
                "title": title,
                "relpath": relpath,
                "sha256": sha,
                "bytes": size,
                "resolved_from": resolved_from,
                "original_track_id": orig_track_id if orig_track_id != effective_track_id else None,
            })
            copied += 1
        except Exception as ex:
            missing += 1
            bundled.append({
                "index": i,
                "ok": False,
                "reason": "copy_failed",
                "track_id": orig_track_id,
                "title": title,
                "source": raw,
                "attempted": attempted,
                "error": str(ex),
            })
            if fail_on_missing:
                sys.stdout.write(_stable_json_dumps({
                    "ok": False,
                    "reason": "copy_failed",
                    "index": i,
                    "track_id": orig_track_id,
                    "title": title,
                    "error": str(ex),
                }) + "\n")
                return 2

    if bundle_audio and fail_if_none_copied and copied == 0:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "none_copied",
            "missing": [x for x in bundled if isinstance(x, dict) and not x.get("ok", True)],
        }) + "\n")
        return 2

    tracks_payload: List[Dict[str, Any]] = []
    if bundle_audio:
        for t in bundled:
            if isinstance(t, dict) and t.get("relpath") and t.get("sha256"):
                tracks_payload.append({
                    "index": t.get("index"),
                    "track_id": t.get("track_id"),
                    "title": t.get("title"),
                    "relpath": t.get("relpath"),
                    "sha256": t.get("sha256"),
                    "bytes": t.get("bytes"),
                })
    else:
        for t in bundled:
            if isinstance(t, dict):
                tracks_payload.append({
                    "index": t.get("index"),
                    "track_id": t.get("track_id"),
                    "title": t.get("title"),
                })
    # Write playlist snapshot (informational)
    (out_dir / "playlist.json").write_text(_stable_json_dumps(playlist_obj) + "\n", encoding="utf-8")

    # Ensure index.html exists BEFORE computing web_tree_sha256.
    index_from = str(getattr(args, "index_from", "") or "").strip()
    if index_from:
        src = Path(index_from).expanduser().resolve()
        if not src.exists():
            sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "index_from_missing", "path": str(src)}) + "\n")
            return 2
        shutil.copy2(src, out_dir / "index.html")

    if not (out_dir / "index.html").exists():
        for cand in (
            repo_root / "index.html",
            repo_root / "web" / "index.html",
            repo_root / "artifacts" / "player" / "index.html",
            repo_root / "assets" / "web" / "index.html",
        ):
            if cand.exists():
                shutil.copy2(cand, out_dir / "index.html")
                break

    # FINAL fallback: always write an index.html so Pages/contract doesn't fail
    if not (out_dir / "index.html").exists():
        (out_dir / "index.html").write_text(_EMBEDDED_INDEX_HTML, encoding="utf-8")

    bundle_data = {
        "playlist": playlist_obj,
        "marketing_plan": marketing_plan,
        "marketing_preview": marketing_preview,
    }
    (out_dir / "bundle_data.js").write_text(
        "window.__MGC_BUNDLE__ = " + _stable_json_dumps(bundle_data) + ";\n", encoding="utf-8"
    )

    # Build + write manifest LAST (tree hash sees final bundle)
    manifest = _build_web_manifest(
        out_dir=out_dir,
        playlist_obj=playlist_obj,
        tracks_payload=tracks_payload,
        deterministic=deterministic,
        bundle_audio=bundle_audio,
        marketing=marketing_meta,
    )
    (out_dir / "web_manifest.json").write_text(_stable_json_dumps(manifest) + "\n", encoding="utf-8")

    # Validate (emit JSON on failure rather than a silent SystemExit)
    try:
        _validate_web_manifest(out_dir, manifest)
    except SystemExit:
        raise
    except Exception as ex:
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "validate_failed", "error": str(ex)}) + "\n")
        return 2

    sys.stdout.write(_stable_json_dumps({
        "ok": True,
        "out_dir": str(out_dir),
        "bundle_audio": bundle_audio,
        "playlist_sha256": manifest.get("playlist_sha256"),
        "web_tree_sha256": manifest.get("web_tree_sha256"),
        "track_count": len(tracks_payload),
        "copied_count": copied if bundle_audio else 0,
        "missing_count": missing if bundle_audio else 0,
        "missing": [x for x in bundled if isinstance(x, dict) and not x.get("ok", True)],
    }) + "\n")
    return 0

def cmd_web_validate(args: argparse.Namespace) -> int:
    out_dir = Path(str(getattr(args, "out_dir"))).expanduser().resolve()
    manifest_path = out_dir / "web_manifest.json"
    if not manifest_path.exists():
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "manifest_missing", "path": str(manifest_path)}) + "\n")
        return 2
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_web_manifest(out_dir, manifest)
        sys.stdout.write(_stable_json_dumps({"ok": True}) + "\n")
        return 0
    except SystemExit:
        raise
    except Exception as ex:
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "validate_failed", "error": str(ex)}) + "\n")
        return 2


# ---------------------------
# HTTP helpers
# ---------------------------

def _http_send_json(handler: SimpleHTTPRequestHandler, status: int, obj: Dict[str, Any]) -> None:
    payload = _stable_json_dumps(obj).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _http_send_json_head(handler: SimpleHTTPRequestHandler, status: int, obj: Dict[str, Any]) -> None:
    payload = _stable_json_dumps(obj).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()


def _parse_url(handler: SimpleHTTPRequestHandler) -> Tuple[str, Dict[str, List[str]]]:
    from urllib.parse import urlparse, parse_qs
    u = urlparse(handler.path)
    return u.path, parse_qs(u.query)


def _extract_request_token(handler: SimpleHTTPRequestHandler, query: Dict[str, List[str]]) -> Optional[str]:
    auth = handler.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        tok = auth.split(None, 1)[1].strip()
        return tok or None
    qs_tok = (query.get("token") or [None])[0]
    if qs_tok:
        return str(qs_tok).strip() or None
    return None


def _safe_join_under(root: Path, relpath: str) -> Optional[Path]:
    try:
        rel = Path(relpath)
        if rel.is_absolute():
            return None
        cand = (root / rel).resolve()
        if root not in cand.parents and cand != root:
            return None
        return cand
    except Exception:
        return None


# ---------------------------
# Billing access
# ---------------------------

def _billing_resolve_access(con: sqlite3.Connection, token: str) -> Dict[str, Any]:
    token = (token or "").strip()
    if not token:
        return {"ok": False, "reason": "token_empty", "tier": None, "scopes": []}

    now_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if _table_exists(con, "billing_tokens") and _table_exists(con, "billing_entitlements"):
        token_sha = hashlib.sha256(token.encode("utf-8")).hexdigest()

        row = con.execute(
            "SELECT token_sha256, user_id FROM billing_tokens WHERE token_sha256=? LIMIT 1",
            (token_sha,),
        ).fetchone()

        if not row:
            return {"ok": False, "reason": "invalid_token", "tier": None, "scopes": []}

        token_sha_db = str(row["token_sha256"])
        user_id = row["user_id"]

        if _table_exists(con, "billing_token_revocations"):
            revoked = con.execute(
                "SELECT 1 FROM billing_token_revocations WHERE token_sha256=? LIMIT 1",
                (token_sha_db,),
            ).fetchone()
            if revoked:
                return {"ok": False, "reason": "token_revoked", "tier": None, "scopes": []}

        ent = con.execute(
            """
            SELECT tier, starts_ts, ends_ts
            FROM billing_entitlements
            WHERE user_id=?
              AND starts_ts <= ?
              AND (ends_ts IS NULL OR ends_ts > ?)
            ORDER BY starts_ts DESC
            LIMIT 1
            """,
            (user_id, now_ts, now_ts),
        ).fetchone()

        tier = None
        if ent:
            tier = str(ent["tier"] or "").strip().lower() or None

        scopes = ["catalog:full", "stream:full"] if tier == "pro" else ["catalog:recent", "stream:preview"]
        return {"ok": True, "reason": "ok", "tier": tier or "free", "user_id": user_id, "token_sha256": token_sha_db, "scopes": scopes}

    if _table_exists(con, "tokens") and _table_exists(con, "entitlements"):
        row = con.execute(
            "SELECT user_id, revoked_at FROM tokens WHERE token=?",
            (token,),
        ).fetchone()
        if not row:
            return {"ok": False, "reason": "invalid_token", "tier": None, "scopes": []}
        if row["revoked_at"]:
            return {"ok": False, "reason": "token_revoked", "tier": None, "scopes": []}

        user_id = row["user_id"]
        ent = con.execute(
            """
            SELECT tier, starts_ts, ends_ts
            FROM entitlements
            WHERE user_id=?
              AND starts_ts <= ?
              AND (ends_ts IS NULL OR ends_ts > ?)
            ORDER BY starts_ts DESC
            LIMIT 1
            """,
            (user_id, now_ts, now_ts),
        ).fetchone()

        tier = None
        if ent:
            tier = str(ent["tier"] or "").strip().lower() or None

        scopes = ["catalog:full", "stream:full"] if tier == "pro" else ["catalog:recent", "stream:preview"]
        return {"ok": True, "reason": "ok", "tier": tier or "free", "user_id": user_id, "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(), "scopes": scopes}

    return {"ok": False, "reason": "billing_schema_missing", "tier": None, "scopes": []}


# ---------------------------
# Library path resolution for streaming
# ---------------------------

def _resolve_library_audio_path(
    *,
    library_db: Optional[Path],
    track_id: str,
    repo_root: Path,
    prefer_preview: bool,
) -> Tuple[Optional[Path], Optional[str]]:
    if not library_db or not library_db.exists():
        return None, None
    tid = (track_id or "").strip()
    if not tid:
        return None, None

    try:
        con = _connect(str(library_db))
        try:
            cols = {r["name"] for r in con.execute("PRAGMA table_info(tracks)").fetchall()}
            if "id" not in cols:
                return None, None

            full = "full_path" if "full_path" in cols else ("path" if "path" in cols else None)
            prev = "preview_path" if "preview_path" in cols else None

            row = con.execute("SELECT * FROM tracks WHERE id=? LIMIT 1", (tid,)).fetchone()
            if not row:
                return None, None

            candidates: List[Tuple[str, str]] = []
            if prefer_preview and prev:
                v = row[prev]
                if isinstance(v, str) and v.strip():
                    candidates.append((prev, v.strip()))
            if full:
                v = row[full]
                if isinstance(v, str) and v.strip():
                    candidates.append((full, v.strip()))
            if (not prefer_preview) and prev:
                v = row[prev]
                if isinstance(v, str) and v.strip():
                    candidates.append((prev, v.strip()))

            for k, raw in candidates:
                p = Path(raw).expanduser()
                if not p.is_absolute():
                    p = (repo_root / p).resolve()
                else:
                    p = p.resolve()
                if p.exists() and p.is_file():
                    return p, f"library_db:{k}"
        finally:
            con.close()
    except Exception:
        return None, None

    return None, None


# ---------------------------
# Streaming implementations
# ---------------------------

def _stream_file(
    handler: SimpleHTTPRequestHandler,
    *,
    path: Path,
    content_type: str,
    allow_range: bool,
    preview_bytes: Optional[int],
) -> None:
    size = path.stat().st_size

    if preview_bytes is not None:
        to_send = min(int(preview_bytes), int(size))
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(to_send))
        handler.send_header("Accept-Ranges", "none")
        handler.end_headers()
        with path.open("rb") as f:
            handler.wfile.write(f.read(to_send))
        return

    range_header = handler.headers.get("Range", "").strip() if allow_range else ""
    if range_header.lower().startswith("bytes="):
        try:
            spec = range_header.split("=", 1)[1].strip()
            start_s, end_s = (spec.split("-", 1) + [""])[:2]
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else (size - 1)
            if start < 0 or end < start or start >= size:
                raise ValueError("invalid range")
            end = min(end, size - 1)
            length = end - start + 1
            handler.send_response(206)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(length))
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            handler.send_header("Accept-Ranges", "bytes")
            handler.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                handler.wfile.write(f.read(length))
            return
        except Exception:
            pass

    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(size))
    handler.send_header("Accept-Ranges", "bytes" if allow_range else "none")
    handler.end_headers()
    with path.open("rb") as f:
        shutil.copyfileobj(f, handler.wfile)


def _stream_head(
    handler: SimpleHTTPRequestHandler,
    *,
    path: Path,
    content_type: str,
    allow_range: bool,
    preview_bytes: Optional[int],
) -> None:
    size = path.stat().st_size

    if preview_bytes is not None:
        to_send = min(int(preview_bytes), int(size))
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(to_send))
        handler.send_header("Accept-Ranges", "none")
        handler.end_headers()
        return

    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(size))
    handler.send_header("Accept-Ranges", "bytes" if allow_range else "none")
    handler.end_headers()


# ---------------------------
# web serve
# ---------------------------

def _load_web_manifest(directory: Path) -> Optional[Dict[str, Any]]:
    p = directory / "web_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _manifest_tracks(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracks = manifest.get("tracks")
    if isinstance(tracks, list):
        out: List[Dict[str, Any]] = []
        for t in tracks:
            if isinstance(t, dict) and t.get("track_id"):
                out.append(t)
        return out
    return []


def _filter_catalog(tracks: List[Dict[str, Any]], scopes: Sequence[str]) -> List[Dict[str, Any]]:
    if "catalog:full" in scopes:
        return tracks
    if not tracks:
        return []
    tracks2 = sorted(tracks, key=lambda x: int(x.get("index", 0)))
    return tracks2[:1]


def cmd_web_serve(args: argparse.Namespace) -> int:
    directory = Path(str(args.dir)).expanduser().resolve()
    if not directory.exists():
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "dir_not_found", "dir": str(directory)}) + "\n")
        return 2

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))

    repo_root = Path(str(getattr(args, "repo_root", "."))).expanduser().resolve()

    billing_db_str = str(getattr(args, "billing_db", "") or os.environ.get("MGC_BILLING_DB", "") or "")
    billing_db: Optional[Path] = None
    if billing_db_str.strip():
        billing_db = Path(billing_db_str).expanduser().resolve()

    library_db_str = str(getattr(args, "library_db", "") or os.environ.get("MGC_LIBRARY_DB", "") or "")
    library_db: Optional[Path] = None
    if library_db_str.strip():
        library_db = Path(library_db_str).expanduser().resolve()
    else:
        library_db = billing_db

    default_token = getattr(args, "token", None)
    preview_bytes = int(os.environ.get("MGC_WEB_PREVIEW_BYTES", "1000000"))

    manifest = _load_web_manifest(directory)
    if not manifest:
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "manifest_missing", "path": str(directory / "web_manifest.json")}) + "\n")
        return 2

    bundle_audio = bool(manifest.get("bundle_audio", True))
    manifest_tracks = _manifest_tracks(manifest)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, fmt: str, *fmt_args: Any) -> None:
            return

        def do_GET(self) -> None:
            path, query = _parse_url(self)
            if path.startswith("/api/"):
                self._handle_api_get(path, query)
                return
            super().do_GET()

        def do_HEAD(self) -> None:
            path, query = _parse_url(self)
            if path.startswith("/api/"):
                self._handle_api_head(path, query)
                return
            super().do_HEAD()

        def _access_for_request(self, query: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
            tok = _extract_request_token(self, query) or (str(default_token).strip() if default_token else None)
            if not tok:
                return None, None, "missing_token"
            if not billing_db:
                return tok, None, "billing_db_missing"
            con = _connect(str(billing_db))
            try:
                a = _billing_resolve_access(con, tok)
            finally:
                con.close()
            return tok, a, None

        def _resolve_stream_path(self, track_id: str, scopes: Sequence[str]) -> Tuple[Optional[Path], Optional[str], Optional[int]]:
            is_pro = ("stream:full" in scopes)

            # library DB first
            prefer_preview = not is_pro
            p, src = _resolve_library_audio_path(
                library_db=library_db,
                track_id=track_id,
                repo_root=repo_root,
                prefer_preview=prefer_preview,
            )
            if p is not None:
                return p, src, None if is_pro else preview_bytes

            # bundle fallback if bundle_audio=true
            if bundle_audio:
                found = None
                for t in manifest_tracks:
                    if str(t.get("track_id")) == track_id:
                        found = t
                        break
                if found and found.get("relpath"):
                    relpath = str(found.get("relpath") or "")
                    disk_path = _safe_join_under(directory, relpath)
                    if disk_path and disk_path.exists() and disk_path.is_file():
                        return disk_path, "bundle:relpath", None if is_pro else preview_bytes

            return None, None, None

        def _handle_api_get(self, path: str, query: Dict[str, List[str]]) -> None:
            if path == "/api/health":
                _http_send_json(self, 200, {"ok": True})
                return

            tok, a, err = self._access_for_request(query)

            if path == "/api/me":
                if err == "missing_token":
                    _http_send_json(self, 200, {"ok": True, "entitled": False, "tier": None, "scopes": [], "reason": "missing_token"})
                    return
                if err == "billing_db_missing":
                    _http_send_json(self, 500, {"ok": False, "reason": "billing_db_missing"})
                    return
                assert a is not None
                entitled = bool(a.get("ok"))
                _http_send_json(self, 200, {"ok": True, "entitled": entitled, "tier": a.get("tier"), "scopes": a.get("scopes", []), "reason": a.get("reason")})
                return

            if path == "/api/catalog":
                if err == "missing_token":
                    _http_send_json(self, 200, {"ok": True, "tracks": [], "entitled": False})
                    return
                if err == "billing_db_missing":
                    _http_send_json(self, 500, {"ok": False, "reason": "billing_db_missing"})
                    return
                assert a is not None
                if not a.get("ok"):
                    _http_send_json(self, 403, {"ok": False, "reason": str(a.get("reason") or "denied")})
                    return
                scopes = list(a.get("scopes") or [])
                tracks = _filter_catalog(list(manifest_tracks), scopes)

                out_tracks: List[Dict[str, Any]] = []
                for t in tracks:
                    out_tracks.append({
                        "index": t.get("index"),
                        "track_id": t.get("track_id"),
                        "title": t.get("title"),
                        "stream_url": f"/api/stream/{t.get('track_id')}",
                    })
                _http_send_json(self, 200, {"ok": True, "tier": a.get("tier"), "scopes": scopes, "tracks": out_tracks})
                return

            if path.startswith("/api/stream/"):
                track_id = path.split("/", 3)[3] if len(path.split("/", 3)) >= 4 else ""
                track_id = (track_id or "").strip()
                if not track_id:
                    _http_send_json(self, 400, {"ok": False, "reason": "missing_track_id"})
                    return
                if err == "missing_token":
                    _http_send_json(self, 401, {"ok": False, "reason": "missing_token"})
                    return
                if err == "billing_db_missing":
                    _http_send_json(self, 500, {"ok": False, "reason": "billing_db_missing"})
                    return
                assert a is not None
                if not a.get("ok"):
                    _http_send_json(self, 403, {"ok": False, "reason": str(a.get("reason") or "denied")})
                    return

                scopes = list(a.get("scopes") or [])
                p, src, pb = self._resolve_stream_path(track_id, scopes)
                if p is None:
                    _http_send_json(self, 404, {"ok": False, "reason": "file_missing", "track_id": track_id})
                    return

                ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                allow_range = ("stream:full" in scopes) or (str(a.get("tier") or "").lower() == "pro")
                _stream_file(self, path=p, content_type=ctype, allow_range=allow_range, preview_bytes=pb)
                return

            _http_send_json(self, 404, {"ok": False, "reason": "unknown_endpoint", "path": path})

        def _handle_api_head(self, path: str, query: Dict[str, List[str]]) -> None:
            if path == "/api/health":
                _http_send_json_head(self, 200, {"ok": True})
                return

            tok, a, err = self._access_for_request(query)

            if path == "/api/me":
                if err == "missing_token":
                    _http_send_json_head(self, 200, {"ok": True, "entitled": False, "tier": None, "scopes": [], "reason": "missing_token"})
                    return
                if err == "billing_db_missing":
                    _http_send_json_head(self, 500, {"ok": False, "reason": "billing_db_missing"})
                    return
                assert a is not None
                entitled = bool(a.get("ok"))
                _http_send_json_head(self, 200, {"ok": True, "entitled": entitled, "tier": a.get("tier"), "scopes": a.get("scopes", []), "reason": a.get("reason")})
                return

            if path == "/api/catalog":
                if err == "missing_token":
                    _http_send_json_head(self, 200, {"ok": True, "tracks": [], "entitled": False})
                    return
                if err == "billing_db_missing":
                    _http_send_json_head(self, 500, {"ok": False, "reason": "billing_db_missing"})
                    return
                assert a is not None
                if not a.get("ok"):
                    _http_send_json_head(self, 403, {"ok": False, "reason": str(a.get("reason") or "denied")})
                    return
                scopes = list(a.get("scopes") or [])
                tracks = _filter_catalog(list(manifest_tracks), scopes)
                out_tracks: List[Dict[str, Any]] = []
                for t in tracks:
                    out_tracks.append({
                        "index": t.get("index"),
                        "track_id": t.get("track_id"),
                        "title": t.get("title"),
                        "stream_url": f"/api/stream/{t.get('track_id')}",
                    })
                _http_send_json_head(self, 200, {"ok": True, "tier": a.get("tier"), "scopes": scopes, "tracks": out_tracks})
                return

            if path.startswith("/api/stream/"):
                track_id = path.split("/", 3)[3] if len(path.split("/", 3)) >= 4 else ""
                track_id = (track_id or "").strip()
                if not track_id:
                    _http_send_json_head(self, 400, {"ok": False, "reason": "missing_track_id"})
                    return
                if err == "missing_token":
                    _http_send_json_head(self, 401, {"ok": False, "reason": "missing_token"})
                    return
                if err == "billing_db_missing":
                    _http_send_json_head(self, 500, {"ok": False, "reason": "billing_db_missing"})
                    return
                assert a is not None
                if not a.get("ok"):
                    _http_send_json_head(self, 403, {"ok": False, "reason": str(a.get("reason") or "denied")})
                    return

                scopes = list(a.get("scopes") or [])
                p, src, pb = self._resolve_stream_path(track_id, scopes)
                if p is None:
                    _http_send_json_head(self, 404, {"ok": False, "reason": "file_missing", "track_id": track_id})
                    return

                ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                allow_range = ("stream:full" in scopes) or (str(a.get("tier") or "").lower() == "pro")
                _stream_head(self, path=p, content_type=ctype, allow_range=allow_range, preview_bytes=pb)
                return

            _http_send_json_head(self, 404, {"ok": False, "reason": "unknown_endpoint", "path": path})

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), Handler)

    sys.stdout.write(_stable_json_dumps({
        "ok": True,
        "serving": str(directory),
        "url": f"http://{host}:{port}/",
        "bundle_audio": bundle_audio,
        "api": {"me": "/api/me", "catalog": "/api/catalog", "stream": "/api/stream/<track_id>"},
        "billing_db": str(billing_db) if billing_db else None,
        "library_db": str(library_db) if library_db else None,
        "repo_root": str(repo_root),
        "default_token": bool(default_token),
    }) + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


# ---------------------------
# CLI registration
# ---------------------------

def register_web_subcommand(subparsers: argparse._SubParsersAction) -> None:
    web = subparsers.add_parser("web", help="Static web player build/serve")
    ws = web.add_subparsers(dest="web_cmd", required=True)

    build = ws.add_parser("build", help="Build a static web bundle")
    build.add_argument("--playlist", required=True, help="Playlist JSON path OR run output dir")
    build.add_argument("--out-dir", required=True, help="Output directory for the web bundle")
    build.add_argument("--db", default="", help="DB path (optional; used to resolve track_id -> full_path/preview_path when bundling audio)")
    build.add_argument("--repo-root", default=".", help="Repo root used for resolving relative paths (default: .)")
    build.add_argument("--prefer-mp3", action="store_true", help="Prefer .mp3 when a .wav sibling exists")
    build.add_argument("--clean", action="store_true", help="Delete out-dir before building")
    build.add_argument("--fail-if-empty", action="store_true", help="Fail if playlist has zero track entries")
    build.add_argument("--fail-if-none-copied", action="store_true", help="Fail if none of the tracks could be copied (bundle-audio mode only)")
    build.add_argument("--fail-on-missing", action="store_true", help="Fail immediately on the first missing track (bundle-audio mode only)")
    build.add_argument("--deterministic", action="store_true", help="Use deterministic timestamps for contract builds")
    build.add_argument("--no-audio", action="store_true", help="Do not bundle audio files into the web bundle (manifest only)")
    build.add_argument("--index-from", default="", help="Copy index.html from this path into the bundle")
    build.set_defaults(fn=cmd_web_build)

    validate = ws.add_parser("validate", help="Validate an existing web bundle")
    validate.add_argument("--out-dir", required=True, help="Web bundle directory (contains web_manifest.json)")
    validate.set_defaults(fn=cmd_web_validate)

    serve = ws.add_parser("serve", help="Serve a web bundle directory")
    serve.add_argument("--dir", required=True, help="Directory containing index.html + playlist.json + web_manifest.json")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", default=8000, type=int, help="Bind port")
    serve.add_argument("--token", default=None, help="Default token for API calls (dev convenience)")
    serve.add_argument("--repo-root", default=".", help="Repo root for resolving relative library paths (default: .)")
    serve.add_argument("--billing-db", dest="billing_db", default=None, help="DB path for billing/token validation. Uses MGC_BILLING_DB if omitted.")
    serve.add_argument("--library-db", dest="library_db", default=None, help="DB path for streaming library resolution. Uses MGC_LIBRARY_DB, else defaults to billing DB.")
    serve.set_defaults(fn=cmd_web_serve)


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
