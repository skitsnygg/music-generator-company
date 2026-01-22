#!/usr/bin/env python3
"""
src/mgc/web_cli.py

Static web bundle builder + simple dev server.

Web contract hardening:
- Deterministic, contract-grade web output (manifest + assets).
- Manifest includes content hashes and sizes for every bundled track.
- Validation fails loudly on missing assets, hash mismatch, or unsafe path issues.
- Content-based tree hash for entire bundle (paths + sha256), suitable for CI determinism gates.

Runtime gating (dev server):
- Serves static web dir
- Adds tiny /api/* endpoints for entitlement checks + gated streaming:
  - GET /api/health
  - GET /api/me
  - GET /api/catalog
  - GET /api/stream/<track_id>
  - HEAD /api/health
  - HEAD /api/me
  - HEAD /api/catalog
  - HEAD /api/stream/<track_id>   (returns correct headers; no body)

Notes:
- The API relies on a billing DB schema (new or legacy) already supported by this file.
- Tokens can be passed via `Authorization: Bearer <token>` or `?token=<token>` (dev convenience).
- `web serve --token ...` provides a default token if the browser doesn't send one.
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


WEB_MANIFEST_VERSION = 2
WEB_MANIFEST_SCHEMA = "mgc.web_manifest.v2"


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
    """
    Deterministic JSON output (sorted keys, stable separators).
    Uses orjson if available, otherwise json.
    """
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
    return datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat(timespec="seconds")


def _manifest_generated_at(playlist_obj: Dict[str, Any], deterministic: bool) -> str:
    if deterministic:
        return _deterministic_now_iso()
    for k in ("date", "generated_at", "created_at", "ts"):
        v = playlist_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return _utc_now_iso()


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


def _ensure_portable_relpath(relpath: str) -> None:
    rel = Path(relpath)
    if rel.is_absolute():
        _die(f"unsafe relpath (absolute): {relpath}", 2)
    parts = rel.parts
    if any(p in ("..", "") for p in parts):
        _die(f"unsafe relpath (traversal): {relpath}", 2)


def _tree_hash(root: Path) -> str:
    """
    Content-based tree hash: sha256 over sorted (relpath, file_sha256).
    """
    items: List[Tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root)).replace("\\", "/")
            items.append((rel, _sha256_file(p)))
    payload = _stable_json_dumps(items)
    return _sha256_hex_str(payload)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dig_first_str(d: Dict[str, Any], keys: Sequence[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _looks_like_audio_path(s: str) -> bool:
    s2 = s.lower().strip()
    return s2.endswith(".wav") or s2.endswith(".mp3") or s2.endswith(".flac") or s2.endswith(".m4a") or s2.endswith(".aac") or s2.endswith(".ogg")


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


def _infer_track_paths_from_obj(track_obj: Dict[str, Any]) -> List[str]:
    for k in ("web_path", "path", "artifact_path", "preview_path", "full_path", "audio_path", "wav", "mp3"):
        v = track_obj.get(k)
        if isinstance(v, str) and _looks_like_audio_path(v):
            return [v]
    return _collect_audio_paths_from_json(track_obj)


def _prefer_mp3_path(p: Path) -> Path:
    if p.suffix.lower() == ".wav":
        mp3 = p.with_suffix(".mp3")
        if mp3.exists():
            return mp3
    return p


def _resolve_track_paths_from_db(con: sqlite3.Connection, track_ids: Sequence[str]) -> Dict[str, str]:
    """
    Map track_id -> best path column (prefers full_path; falls back to preview_path).
    """
    if not track_ids:
        return {}
    if not _table_exists(con, "tracks"):
        return {}

    cols = {r["name"] for r in con.execute("PRAGMA table_info(tracks)").fetchall()}
    if not cols or "id" not in cols:
        return {}

    candidates: List[str] = []
    for col in ("full_path", "preview_path", "path"):
        if col in cols:
            candidates.append(col)
    if not candidates:
        return {}

    col = candidates[0]
    q = f"SELECT id, {col} AS p FROM tracks WHERE id IN ({','.join(['?'] * len(track_ids))})"
    rows = con.execute(q, list(track_ids)).fetchall()
    out: Dict[str, str] = {}
    for r in rows:
        tid = str(r["id"])
        p = r["p"]
        if isinstance(p, str) and p.strip():
            out[tid] = p.strip()
    return out


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

    # last resort: CWD
    try:
        return Path(s).expanduser().resolve()
    except Exception:
        return Path(s).expanduser()


def _find_track_file(
    *,
    track_id: str,
    playlist_dir: Path,
    repo_root: Path,
    prefer_mp3: bool,
) -> Optional[Path]:
    """
    Last-resort resolver when playlist paths + DB paths don't exist.

    Tries a small set of common layouts (kept deterministic and reasonably fast):
    - <playlist_dir>/tracks/<id>.(wav|mp3)
    - <playlist_dir>/drop_bundle/tracks/<id>.(wav|mp3)
    - <repo_root>/data/tracks/**/<id>.(wav|mp3)   (date-sharded)
    - <repo_root>/artifacts/**/<id>.(wav|mp3)     (CI / runs)
    """
    tid = (track_id or "").strip()
    if not tid:
        return None

    exts = [".mp3", ".wav"] if prefer_mp3 else [".wav", ".mp3"]

    fast: List[Path] = []
    for ext in exts:
        fast.extend([
            playlist_dir / "tracks" / f"{tid}{ext}",
            playlist_dir / "drop_bundle" / "tracks" / f"{tid}{ext}",
            repo_root / "data" / "tracks" / f"{tid}{ext}",
        ])

    for p in fast:
        if p.exists():
            return p.resolve()

    roots = [
        repo_root / "data" / "tracks",
        repo_root / "artifacts",
    ]
    for root in roots:
        if not root.exists():
            continue
        for ext in exts:
            needle = f"{tid}{ext}"
            try:
                for p in root.rglob(needle):
                    if p.is_file():
                        return p.resolve()
            except Exception:
                continue

    return None


def _build_web_manifest(
    *,
    out_dir: Path,
    playlist_obj: Dict[str, Any],
    tracks_payload: List[Dict[str, Any]],
    deterministic: bool,
) -> Dict[str, Any]:
    playlist_sha = _sha256_hex_str(_stable_json_dumps(playlist_obj))
    web_tree = _tree_hash(out_dir)

    return {
        "schema": WEB_MANIFEST_SCHEMA,
        "version": WEB_MANIFEST_VERSION,
        "generated_at": _manifest_generated_at(playlist_obj, deterministic),
        "playlist_sha256": playlist_sha,
        "web_tree_sha256": web_tree,
        "tracks": tracks_payload,
    }


def _validate_web_manifest(out_dir: Path, manifest: Dict[str, Any]) -> None:
    schema = manifest.get("schema")
    version = manifest.get("version")
    if schema != WEB_MANIFEST_SCHEMA:
        _die(f"web_manifest schema mismatch: got {schema!r} expected {WEB_MANIFEST_SCHEMA!r}", 2)
    if int(version or 0) != WEB_MANIFEST_VERSION:
        _die(f"web_manifest version mismatch: got {version!r} expected {WEB_MANIFEST_VERSION!r}", 2)

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        _die("web_manifest invalid: tracks must be a list", 2)

    for t in tracks:
        if not isinstance(t, dict):
            _die("web_manifest invalid: track entry must be a dict", 2)
        relpath = str(t.get("relpath") or "")
        sha = str(t.get("sha256") or "")
        if not relpath or not sha:
            _die("web_manifest invalid: track missing relpath/sha256", 2)
        _ensure_portable_relpath(relpath)
        p = (out_dir / relpath).resolve()
        if not p.exists():
            _die(f"web_manifest invalid: missing file {relpath}", 2)
        got = _sha256_file(p)
        if got != sha:
            _die(f"web_manifest invalid: hash mismatch for {relpath} got={got} expected={sha}", 2)

    got_tree = _tree_hash(out_dir)
    exp_tree = str(manifest.get("web_tree_sha256") or "")
    if exp_tree and got_tree != exp_tree:
        _die(f"web_manifest invalid: web_tree_sha256 mismatch got={got_tree} expected={exp_tree}", 2)


def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(str(getattr(args, "playlist"))).expanduser().resolve()
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

    prefer_mp3 = bool(getattr(args, "prefer_mp3", False))
    fail_if_empty = bool(getattr(args, "fail_if_empty", False))
    fail_if_none_copied = bool(getattr(args, "fail_if_none_copied", False))
    fail_on_missing = bool(getattr(args, "fail_on_missing", False))
    deterministic = bool(getattr(args, "deterministic", False))

    entries = _collect_entries(playlist_obj)
    if fail_if_empty and not entries:
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "empty_playlist"}) + "\n")
        return 2

    db_map: Dict[str, str] = {}
    db_raw = str(getattr(args, "db", "") or "").strip()
    if db_raw:
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
    _safe_mkdir(tracks_dir)

    bundled: List[Dict[str, Any]] = []
    copied = 0
    missing = 0

    for e in entries:
        i = int(e.get("index", 0))
        track_id = str(e.get("track_id") or "").strip()
        title = str(e.get("title") or track_id or "").strip()
        raw = str(e.get("raw_path") or "").strip()

        attempted: List[str] = []
        src_path: Optional[Path] = None
        resolved_from: Optional[str] = None

        # 1) Prefer DB mapping for the track_id
        if track_id and track_id in db_map:
            rp = _resolve_input_path(db_map[track_id], playlist_dir=playlist_dir, repo_root=repo_root)
            if prefer_mp3:
                rp = _prefer_mp3_path(rp)
            attempted.append(str(rp))
            if rp.exists() and rp.is_file():
                src_path = rp
                resolved_from = "db"

        # 2) Fall back to playlist-provided path(s)
        if src_path is None:
            candidates: List[str] = []
            if raw:
                candidates.append(raw)
            candidates.extend(_infer_track_paths_from_obj(e.get("track_obj") or {}))
            for c in candidates:
                rp = _resolve_input_path(c, playlist_dir=playlist_dir, repo_root=repo_root)
                if prefer_mp3:
                    rp = _prefer_mp3_path(rp)
                attempted.append(str(rp))
                if rp.exists() and rp.is_file():
                    src_path = rp
                    resolved_from = "playlist"
                    break

        # 3) Final fallback: search by track_id in common layouts.
        if src_path is None and track_id:
            fb = _find_track_file(track_id=track_id, playlist_dir=playlist_dir, repo_root=repo_root, prefer_mp3=prefer_mp3)
            if fb is not None and fb.exists() and fb.is_file():
                attempted.append(str(fb))
                src_path = fb
                resolved_from = "search"

        if src_path is None or (not src_path.exists()) or (not src_path.is_file()):
            missing += 1
            bundled.append({
                "index": i,
                "ok": False,
                "reason": "missing",
                "track_id": track_id,
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
                    "track_id": track_id,
                    "title": title,
                    "attempted": attempted,
                }) + "\n")
                return 2
            continue

        ext = src_path.suffix.lower()
        if ext not in (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"):
            ext = src_path.suffix.lower() or ".bin"

        dst_name = f"{track_id or f'index_{i}'}{ext}"
        relpath = f"tracks/{dst_name}"
        _ensure_portable_relpath(relpath)
        dst = (out_dir / relpath).resolve()

        try:
            shutil.copy2(src_path, dst)
            sha = _sha256_file(dst)
            size = dst.stat().st_size
            bundled.append({
                "index": i,
                "ok": True,
                "reason": "ok",
                "track_id": track_id,
                "title": title,
                "relpath": relpath,
                "sha256": sha,
                "bytes": size,
                "resolved_from": resolved_from,
            })
            copied += 1
        except Exception as ex:
            missing += 1
            bundled.append({
                "index": i,
                "ok": False,
                "reason": "copy_failed",
                "track_id": track_id,
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
                    "track_id": track_id,
                    "title": title,
                    "error": str(ex),
                }) + "\n")
                return 2

    if fail_if_none_copied and copied == 0:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "none_copied",
            "missing": [x for x in bundled if not x.get("ok")],
        }) + "\n")
        return 2

    tracks_payload: List[Dict[str, Any]] = []
    for t in bundled:
        if not t.get("ok"):
            continue
        tracks_payload.append({
            "index": t.get("index"),
            "track_id": t.get("track_id"),
            "title": t.get("title"),
            "relpath": t.get("relpath"),
            "sha256": t.get("sha256"),
            "bytes": t.get("bytes"),
        })

    manifest = _build_web_manifest(out_dir=out_dir, playlist_obj=playlist_obj, tracks_payload=tracks_payload, deterministic=deterministic)
    (out_dir / "web_manifest.json").write_text(_stable_json_dumps(manifest) + "\n", encoding="utf-8")
    (out_dir / "playlist.json").write_text(_stable_json_dumps(playlist_obj) + "\n", encoding="utf-8")

    # Best-effort copy index.html from common template locations if missing
    if not (out_dir / "index.html").exists():
        for cand in (
            repo_root / "web" / "index.html",
            repo_root / "src" / "mgc" / "web" / "index.html",
            repo_root / "assets" / "web" / "index.html",
        ):
            if cand.exists():
                shutil.copy2(cand, out_dir / "index.html")
                break

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
        "playlist_sha256": manifest.get("playlist_sha256"),
        "web_tree_sha256": manifest.get("web_tree_sha256"),
        "track_count": copied,
        "missing_count": missing,
        "missing": [x for x in bundled if not x.get("ok")],
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


def _http_send_json(handler: SimpleHTTPRequestHandler, status: int, obj: Dict[str, Any]) -> None:
    payload = _stable_json_dumps(obj).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _http_send_json_head(handler: SimpleHTTPRequestHandler, status: int, obj: Dict[str, Any]) -> None:
    """
    HEAD variant: send the same headers as JSON would, but no body.
    """
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


def _billing_resolve_access(con: sqlite3.Connection, token: str) -> Dict[str, Any]:
    token = (token or "").strip()
    if not token:
        return {"ok": False, "reason": "token_empty", "tier": None, "scopes": []}

    def _looks_hex64(s: str) -> bool:
        if len(s) != 64:
            return False
        try:
            int(s, 16)
            return True
        except Exception:
            return False

    now_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if _table_exists(con, "billing_tokens") and _table_exists(con, "billing_entitlements"):
        # Compare sha256(raw_token) to stored token_sha256
        # If caller accidentally passes a 64-hex token, do NOT assume it's a hash; always hash raw input.
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

        if tier == "pro":
            scopes = ["catalog:full", "stream:full"]
        else:
            scopes = ["catalog:recent", "stream:preview"]

        return {
            "ok": True,
            "reason": "ok",
            "tier": tier or "free",
            "user_id": user_id,
            "token_sha256": token_sha_db,
            "scopes": scopes,
        }

    # Legacy schema
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

        return {
            "ok": True,
            "reason": "ok",
            "tier": tier or "free",
            "user_id": user_id,
            "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "scopes": scopes,
        }

    return {"ok": False, "reason": "billing_schema_missing", "tier": None, "scopes": []}


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
            if isinstance(t, dict) and t.get("track_id") and t.get("relpath"):
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
    """
    HEAD variant of streaming: same status + headers, but no body.
    We do not honor Range for HEAD; we return headers for the effective full/preview length.
    """
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
    return


def cmd_web_serve(args: argparse.Namespace) -> int:
    directory = Path(str(args.dir)).expanduser().resolve()
    if not directory.exists():
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "dir_not_found", "dir": str(directory)}) + "\n")
        return 2

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))

    billing_db_str = str(getattr(args, "billing_db", "") or os.environ.get("MGC_BILLING_DB", "") or "")
    billing_db: Optional[Path] = None
    if billing_db_str.strip():
        billing_db = Path(billing_db_str).expanduser().resolve()

    default_token = getattr(args, "token", None)
    preview_bytes = int(os.environ.get("MGC_WEB_PREVIEW_BYTES", "1000000"))

    manifest = _load_web_manifest(directory)
    manifest_tracks = _manifest_tracks(manifest) if manifest else []

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
                    _http_send_json(self, 500, {"ok": False, "reason": "billing_db_missing", "hint": "Pass --billing-db or set MGC_BILLING_DB."})
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
                    _http_send_json(self, 500, {"ok": False, "reason": "billing_db_missing", "hint": "Pass --billing-db or set MGC_BILLING_DB."})
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
                    _http_send_json(self, 500, {"ok": False, "reason": "billing_db_missing", "hint": "Pass --billing-db or set MGC_BILLING_DB."})
                    return
                assert a is not None
                if not a.get("ok"):
                    _http_send_json(self, 403, {"ok": False, "reason": str(a.get("reason") or "denied")})
                    return
                scopes = list(a.get("scopes") or [])

                found = None
                for t in manifest_tracks:
                    if str(t.get("track_id")) == track_id:
                        found = t
                        break
                if not found:
                    _http_send_json(self, 404, {"ok": False, "reason": "track_not_found"})
                    return

                relpath = str(found.get("relpath") or "")
                disk_path = _safe_join_under(directory, relpath)
                if not disk_path or not disk_path.exists():
                    _http_send_json(self, 404, {"ok": False, "reason": "file_missing"})
                    return

                ctype = mimetypes.guess_type(str(disk_path))[0] or "application/octet-stream"
                is_pro = ("stream:full" in scopes) or (str(a.get("tier") or "").lower() == "pro")

                _stream_file(
                    self,
                    path=disk_path,
                    content_type=ctype,
                    allow_range=is_pro,
                    preview_bytes=None if is_pro else preview_bytes,
                )
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
                    _http_send_json_head(self, 500, {"ok": False, "reason": "billing_db_missing", "hint": "Pass --billing-db or set MGC_BILLING_DB."})
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
                    _http_send_json_head(self, 500, {"ok": False, "reason": "billing_db_missing", "hint": "Pass --billing-db or set MGC_BILLING_DB."})
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
                    _http_send_json_head(self, 500, {"ok": False, "reason": "billing_db_missing", "hint": "Pass --billing-db or set MGC_BILLING_DB."})
                    return
                assert a is not None
                if not a.get("ok"):
                    _http_send_json_head(self, 403, {"ok": False, "reason": str(a.get("reason") or "denied")})
                    return
                scopes = list(a.get("scopes") or [])

                found = None
                for t in manifest_tracks:
                    if str(t.get("track_id")) == track_id:
                        found = t
                        break
                if not found:
                    _http_send_json_head(self, 404, {"ok": False, "reason": "track_not_found"})
                    return

                relpath = str(found.get("relpath") or "")
                disk_path = _safe_join_under(directory, relpath)
                if not disk_path or not disk_path.exists():
                    _http_send_json_head(self, 404, {"ok": False, "reason": "file_missing"})
                    return

                ctype = mimetypes.guess_type(str(disk_path))[0] or "application/octet-stream"
                is_pro = ("stream:full" in scopes) or (str(a.get("tier") or "").lower() == "pro")

                _stream_head(
                    self,
                    path=disk_path,
                    content_type=ctype,
                    allow_range=is_pro,
                    preview_bytes=None if is_pro else preview_bytes,
                )
                return

            _http_send_json_head(self, 404, {"ok": False, "reason": "unknown_endpoint", "path": path})

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), Handler)

    sys.stdout.write(_stable_json_dumps({
        "ok": True,
        "serving": str(directory),
        "url": f"http://{host}:{port}/",
        "api": {
            "me": "/api/me",
            "catalog": "/api/catalog",
            "stream": "/api/stream/<track_id>",
        },
        "billing_db": str(billing_db) if billing_db else None,
        "default_token": bool(default_token),
    }) + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def register_web_subcommand(subparsers: argparse._SubParsersAction) -> None:
    web = subparsers.add_parser("web", help="Static web player build/serve")
    ws = web.add_subparsers(dest="web_cmd", required=True)

    build = ws.add_parser("build", help="Build a static web bundle")
    build.add_argument("--playlist", required=True, help="Playlist JSON path")
    build.add_argument("--out-dir", required=True, help="Output directory for the web bundle")
    build.add_argument("--db", default="", help="Main DB path (used to resolve track_id -> full_path/preview_path)")
    build.add_argument("--repo-root", default=".", help="Repo root used for resolving relative paths (default: .)")
    build.add_argument("--prefer-mp3", action="store_true", help="Prefer .mp3 when a .wav sibling exists")
    build.add_argument("--clean", action="store_true", help="Delete out-dir before building")
    build.add_argument("--fail-if-empty", action="store_true", help="Fail if playlist has zero track entries")
    build.add_argument("--fail-if-none-copied", action="store_true", help="Fail if none of the tracks could be copied")
    build.add_argument("--fail-on-missing", action="store_true", help="Fail immediately on the first missing track")
    build.add_argument("--deterministic", action="store_true", help="Use deterministic timestamps for contract builds")
    build.set_defaults(fn=cmd_web_build)

    validate = ws.add_parser("validate", help="Validate an existing web bundle")
    validate.add_argument("--out-dir", required=True, help="Web bundle directory (contains web_manifest.json)")
    validate.set_defaults(fn=cmd_web_validate)

    serve = ws.add_parser("serve", help="Serve a web bundle directory")
    serve.add_argument("--dir", required=True, help="Directory containing index.html + playlist.json + web_manifest.json")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", default=8000, type=int, help="Bind port")
    serve.add_argument("--token", default=None, help="Default token for API calls (dev convenience)")
    serve.add_argument(
        "--billing-db",
        dest="billing_db",
        default=None,
        help="DB path for billing/token validation (recommended). Uses MGC_BILLING_DB.",
    )
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
