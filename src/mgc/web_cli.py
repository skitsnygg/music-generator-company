#!/usr/bin/env python3
"""
src/mgc/web_cli.py

Static web bundle builder + simple dev server.

Web contract hardening:
- Deterministic, contract-grade web output (manifest + assets).
- Manifest includes content hashes and sizes for every bundled track.
- Validation fails loudly on missing assets, hash mismatch, or unsafe path issues.
- Content-based tree hash for entire bundle (paths + sha256), suitable for determinism gates.

Important behaviors for this repo:
- Playlists may contain placeholder track_id/path (e.g. 0000...0001.wav). When the
  playlist track path does not exist, we fall back to:
    1) Resolve real artifact_path from the *main* DB using tracks.track_id
    2) Infer real track paths from sibling drop_evidence.json/manifest.json/contract_report.json
       next to the playlist (common in /tmp/mgc_release output)

Tree hash determinism:
- The bundle tree hash EXCLUDES web_manifest.json to avoid self-referential hashing
  (manifest contains web_tree_sha256, so including it would make the hash unstable).

Registrar:
- mgc.main discovers register_web_subcommand(subparsers) (plus aliases).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Billing access (optional; required only when --token is used)
try:
    from mgc.billing_access import resolve_access, AccessContext  # type: ignore
except Exception as _e:
    resolve_access = None  # type: ignore
    AccessContext = None  # type: ignore
    _BILLING_ACCESS_IMPORT_ERROR = str(_e)
else:
    _BILLING_ACCESS_IMPORT_ERROR = None


# ---------------------------------------------------------------------
# Optional imports (keep web_cli importable in CI/minimal environments)
# ---------------------------------------------------------------------

def _try_import_stable_json() -> Optional[Any]:
    for mod_name, fn_name in (
        ("mgc.util", "stable_json"),
        ("mgc.util", "stable_json_dumps"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            return getattr(mod, fn_name)
        except Exception:
            continue
    return None


def _try_import_die() -> Optional[Any]:
    try:
        mod = __import__("mgc.util", fromlist=["die"])
        return getattr(mod, "die")
    except Exception:
        return None


_STABLE_JSON = _try_import_stable_json()
_DIE = _try_import_die()


def _die(msg: str, code: int = 2) -> None:
    # Prefer project die() if available; otherwise print something.
    if _DIE:
        try:
            _DIE(msg)
        except Exception:
            pass
    else:
        sys.stderr.write(str(msg).rstrip() + "\n")
    se = SystemExit(code)
    # Attach message so callers can record deterministic evidence.
    try:
        setattr(se, "msg", str(msg))
    except Exception:
        pass
    raise se

def _stable_json_dumps(obj: Any) -> str:
    if _STABLE_JSON:
        try:
            s = _STABLE_JSON(obj)
            if isinstance(s, str):
                return s
        except Exception:
            pass
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def _write_billing_evidence(
    out_dir: Path,
    *,
    token: str,
    ok: bool,
    reason: str,
    action: str = "web.build",
    user_id: Optional[str] = None,
    tier: Optional[str] = None,
    entitlements: Optional[Sequence[str]] = None,
) -> None:
    """Write a small, deterministic-ish billing decision evidence file.

    This is intentionally separate from append-only billing receipts (which live under
    billing_cli). Web evidence is a convenient artifact for CI / debugging and can be
    shipped alongside the built web bundle.
    """
    ev_dir = out_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)

    ev = {
        "schema": "mgc.billing_evidence.v1",
        "action": action,
        "token": token,
        "input": {"token": token},
        "decision": {
            "ok": ok,
            "reason": reason,
            "user_id": user_id,
            "tier": tier,
            "entitlements": list(entitlements) if entitlements is not None else None,
        },
    }

    p = ev_dir / "billing_evidence.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump(ev, f, indent=2, sort_keys=True)

def _sha256_hex_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deterministic_now_iso(deterministic: bool) -> str:
    env_ts = (os.environ.get("MGC_DETERMINISTIC_TS") or "").strip()
    if env_ts:
        return env_ts
    if deterministic:
        return "2020-01-01T00:00:00+00:00"
    return _utc_now_iso()

def _manifest_generated_at(playlist_obj: Any, deterministic: bool) -> str:
    """
    Choose a deterministic generated_at whenever possible.

    Priority:
      1) MGC_DETERMINISTIC_TS (handled by _deterministic_now_iso)
      2) If deterministic flag set: fixed timestamp
      3) If playlist has a stable 'ts' field: use that (common in pipeline outputs)
      4) Otherwise: wall clock UTC now
    """
    if deterministic:
        return _deterministic_now_iso(True)
    if isinstance(playlist_obj, dict):
        ts = playlist_obj.get("ts")
        if isinstance(ts, str) and ts.strip():
            # Trust playlist ts to be stable if present.
            return ts.strip()
    return _utc_now_iso()



# ---------------------------------------------------------------------
# Billing gate (token -> require pro entitlement)
# ---------------------------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    p = Path(str(db_path or "")).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass
    if not str(p):
        _die("billing_db path is empty", 2)
    if not p.exists():
        _die(f"billing_db not found: {p}", 2)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _resolve_billing_db_strict(args: argparse.Namespace) -> str:
    raw = getattr(args, "billing_db", None)
    if raw:
        return str(raw).strip()

    env = (os.environ.get("MGC_BILLING_DB") or "").strip()
    if env:
        return env

    db = (getattr(args, "db", "") or "").strip()
    if db:
        return db

    return ""


def _billing_require_pro(con: sqlite3.Connection, token: str) -> None:
    token = (token or "").strip()
    if not token:
        _die("token is empty", 2)

    def _looks_hex64(s: str) -> bool:
        if len(s) != 64:
            return False
        try:
            int(s, 16)
            return True
        except Exception:
            return False

    # New billing schema (preferred)
    if _table_exists(con, "billing_tokens") and _table_exists(con, "billing_entitlements"):
        # Support either a raw token (hashed in DB) or a token_sha256 already.
        candidates: list[str]
        if _looks_hex64(token):
            candidates = [token.lower(), hashlib.sha256(token.encode('utf-8')).hexdigest()]
        else:
            candidates = [hashlib.sha256(token.encode('utf-8')).hexdigest()]

        row = None
        for cand in candidates:
            row = con.execute(
                "SELECT token_sha256, user_id FROM billing_tokens WHERE token_sha256=? LIMIT 1",
                (cand,),
            ).fetchone()
            if row:
                break

        if not row:
            _die("billing denied: invalid token", 2)

        token_sha = row["token_sha256"]
        user_id = row["user_id"]

        # Optional revocations table
        if _table_exists(con, "billing_token_revocations"):
            revoked = con.execute(
                "SELECT 1 FROM billing_token_revocations WHERE token_sha256=? LIMIT 1",
                (token_sha,),
            ).fetchone()
            if revoked:
                _die("billing denied: token revoked", 2)

        now_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
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

        if not ent:
            _die("billing denied: no active entitlement", 2)

        tier = str(ent["tier"] or "").strip().lower()
        if tier != "pro":
            _die(f"billing denied: requires pro (tier={tier!r})", 2)

        return

    # Legacy schema
    if _table_exists(con, "tokens") and _table_exists(con, "entitlements"):
        row = con.execute(
            "SELECT user_id, revoked_at FROM tokens WHERE token=?",
            (token,),
        ).fetchone()
        if not row:
            _die("billing denied: invalid token", 2)
        if row["revoked_at"]:
            _die("billing denied: token revoked", 2)

        user_id = row["user_id"]
        ent = con.execute(
            "SELECT tier, active FROM entitlements WHERE user_id=? ORDER BY rowid DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if not ent:
            _die("billing denied: no entitlements", 2)

        tier = str(ent["tier"] or "").strip().lower()
        active = int(ent["active"] or 0)
        if tier != "pro" or active != 1:
            _die(f"billing denied: requires pro (tier={tier!r}, active={active})", 2)
        return

    _die("billing_db missing required table: tokens (or billing_tokens)", 2)

def _require_pro_if_token(args: argparse.Namespace) -> "AccessContext":
    """If args.token is provided, enforce billing access.

    Policy:
      - token must resolve successfully
      - allow if tier == "pro" OR user has entitlement "web"
    """
    token = getattr(args, "token", None)
    if not token:
        _die("token is empty", 2)

    if resolve_access is None:
        msg = _BILLING_ACCESS_IMPORT_ERROR or "mgc.billing_access unavailable"
        _die(f"billing denied: cannot import billing_access: {msg}", 2)

    db_path = _resolve_billing_db_strict(args)
    if not db_path:
        _die("billing denied: missing billing db path (use --billing-db or MGC_BILLING_DB)", 2)

    ctx = resolve_access(billing_db=str(db_path), token=str(token))
    if not getattr(ctx, "ok", False):
        _die(f"billing denied: {getattr(ctx, 'reason', 'unknown')}", 2)

    # Prefer explicit entitlement gating; allow "pro" as an override to keep older flows working.
    if (getattr(ctx, "tier", None) != "pro") and ("web" not in set(getattr(ctx, "entitlements", set()))):
        _die(f"billing denied: requires entitlement 'web' or pro tier (tier={getattr(ctx, 'tier', None)!r})", 2)

    return ctx



# ---------------------------------------------------------------------
# Playlist parsing + path resolution
# ---------------------------------------------------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dig_first_str(obj: Any, keys: Sequence[str]) -> str:
    if not isinstance(obj, dict):
        return ""
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _looks_like_audio_path(s: str) -> bool:
    s = (s or "").lower()
    return s.endswith(".wav") or s.endswith(".mp3") or s.endswith(".flac") or s.endswith(".m4a")


def _iter_track_dicts(playlist_obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(playlist_obj, dict):
        tracks = playlist_obj.get("tracks")
        if isinstance(tracks, list):
            for t in tracks:
                if isinstance(t, dict):
                    yield t


def _collect_playlist_track_entries(playlist_obj: Any) -> List[Dict[str, Any]]:
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


def _prefer_mp3_path(p: Path) -> Path:
    if not p:
        return p
    if p.suffix.lower() != ".wav":
        return p
    mp3 = p.with_suffix(".mp3")
    if mp3.exists():
        return mp3
    return p


def _resolve_track_paths_from_db(db_path: str, track_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Resolve track IDs -> file paths from the main DB.

    Fixtures schema:
      tracks(track_id TEXT PRIMARY KEY, ..., artifact_path TEXT, ...)

    Returns:
      mapping: {track_id: artifact_path}
      meta: diagnostics
    """
    meta: Dict[str, Any] = {"source": "db", "db_path": str(db_path or ""), "count": 0}
    db_path = (db_path or "").strip()
    if not db_path:
        return {}, meta

    p = Path(db_path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        pass
    meta["db_path"] = str(p)
    if not p.exists():
        return {}, meta

    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    try:
        if not _table_exists(con, "tracks"):
            return {}, meta
        out: Dict[str, str] = {}
        for tid in track_ids:
            tid = (tid or "").strip()
            if not tid or tid in out:
                continue
            row = con.execute(
                "SELECT track_id, artifact_path FROM tracks WHERE track_id=? LIMIT 1",
                (tid,),
            ).fetchone()
            if not row:
                continue
            cand = (row["artifact_path"] or "").strip()
            if cand:
                out[tid] = cand
        meta["count"] = len(out)
        return out, meta
    finally:
        con.close()


def _collect_audio_paths_from_json(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        s = obj.strip()
        if s and _looks_like_audio_path(s):
            out.append(s)
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_collect_audio_paths_from_json(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_audio_paths_from_json(v))

    seen: set[str] = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _infer_track_paths_from_sibling_files(playlist_path: Path) -> List[str]:
    playlist_dir = playlist_path.parent
    candidates = [
        playlist_dir / "drop_evidence.json",
        playlist_dir / "manifest.json",
        playlist_dir / "contract_report.json",
    ]
    for c in candidates:
        if not c.exists():
            continue
        try:
            obj = json.loads(c.read_text(encoding="utf-8"))
        except Exception:
            continue
        paths = _collect_audio_paths_from_json(obj)

        # Prefer paths that actually exist:
        existing: List[str] = []
        for p in paths:
            pp = Path(p)
            if pp.is_absolute():
                if pp.exists():
                    existing.append(str(pp))
            else:
                rel = playlist_dir / pp
                if rel.exists():
                    existing.append(str(rel))
        if existing:
            return existing
    return []


# ---------------------------------------------------------------------
# Web bundle manifest hardening
# ---------------------------------------------------------------------

WEB_MANIFEST_SCHEMA = "mgc.web_manifest.v2"
WEB_MANIFEST_VERSION = 2

_FORBIDDEN_PATH_FRAGMENTS = (
    "/Users/",
    "/private/tmp",
    "file://",
    "\\",
)

# Exclude manifest from content tree hash to avoid self-reference.
_TREE_HASH_EXCLUDE = {"web_manifest.json"}


def _ensure_portable_relpath(rel: str) -> None:
    if not rel or not isinstance(rel, str):
        _die("web manifest contains empty relpath", 2)
    if rel.startswith("/") or "://" in rel:
        _die(f"web manifest contains non-relative path: {rel}", 2)
    for bad in _FORBIDDEN_PATH_FRAGMENTS:
        if bad in rel:
            _die(f"web manifest contains forbidden path fragment: {bad} in {rel}", 2)
    if "\\" in rel:
        _die(f"web manifest contains backslashes: {rel}", 2)


def _tree_hash(root: Path) -> str:
    rows: List[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel in _TREE_HASH_EXCLUDE:
            continue
        rows.append(f"{rel}:{_sha256_file(p)}")
    h = hashlib.sha256()
    for r in rows:
        h.update(r.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class _BundledTrack:
    index: int
    track_id: str
    title: str
    src_path: str
    relpath: str
    sha256: str
    bytes: int


def _build_web_manifest(
    *,
    out_dir: Path,
    playlist_obj: Dict[str, Any],
    bundled_tracks: List[_BundledTrack],
    deterministic: bool,
) -> Dict[str, Any]:
    tracks_payload: List[Dict[str, Any]] = []
    for bt in bundled_tracks:
        _ensure_portable_relpath(bt.relpath)
        tracks_payload.append({
            "index": bt.index,
            "track_id": bt.track_id,
            "title": bt.title,
            "relpath": bt.relpath,
            "sha256": bt.sha256,
            "bytes": bt.bytes,
        })

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


def _validate_web_manifest(manifest: Dict[str, Any], *, out_dir: Path) -> None:
    if not isinstance(manifest, dict):
        _die("web manifest is not an object", 2)
    if manifest.get("schema") != WEB_MANIFEST_SCHEMA:
        _die("web manifest schema mismatch", 2)
    if int(manifest.get("version") or 0) != WEB_MANIFEST_VERSION:
        _die("web manifest version mismatch", 2)

    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        _die("web manifest missing tracks", 2)

    for t in tracks:
        if not isinstance(t, dict):
            _die("web manifest track entry is not an object", 2)
        rel = t.get("relpath")
        _ensure_portable_relpath(rel)
        fpath = out_dir / rel
        if not fpath.exists():
            _die(f"web manifest references missing file: {rel}", 2)

        expected = str(t.get("sha256") or "").strip()
        if not expected:
            _die(f"web manifest missing sha256 for {rel}", 2)
        actual = _sha256_file(fpath)
        if actual != expected:
            _die(f"sha256 mismatch for {rel}: expected {expected} got {actual}", 2)

        expected_bytes = int(t.get("bytes") or 0)
        actual_bytes = int(fpath.stat().st_size)
        if expected_bytes != actual_bytes:
            _die(f"bytes mismatch for {rel}: expected {expected_bytes} got {actual_bytes}", 2)

    expected_tree = str(manifest.get("web_tree_sha256") or "").strip()
    if not expected_tree:
        _die("web manifest missing web_tree_sha256", 2)
    actual_tree = _tree_hash(out_dir)
    if actual_tree != expected_tree:
        _die(f"web_tree_sha256 mismatch: expected {expected_tree} got {actual_tree}", 2)


# ---------------------------------------------------------------------
# HTML (minimal player)
# ---------------------------------------------------------------------

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>MGC Player</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    .track { padding: 12px 0; border-bottom: 1px solid #eee; display: flex; gap: 12px; align-items: center; }
    .title { flex: 1; }
    audio { width: 360px; max-width: 100%; }
    .meta { font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <h1>MGC Player</h1>
  <div id="app"><p>Loading…</p></div>

  <script>
    async function loadJson(path) {
      const r = await fetch(path, { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      return await r.json();
    }

    function el(tag, cls) {
      const e = document.createElement(tag);
      if (cls) e.className = cls;
      return e;
    }

    (async () => {
      const app = document.getElementById("app");
      let playlist;
      try {
        playlist = await loadJson("./playlist.json");
      } catch (e) {
        app.innerHTML = "<p>Failed to load playlist.json</p>";
        return;
      }

      const tracks = (playlist && playlist.tracks) ? playlist.tracks : [];
      if (!Array.isArray(tracks) || tracks.length === 0) {
        app.innerHTML = "<p>No tracks found in playlist.json</p>";
        return;
      }

      app.innerHTML = "";
      for (const t of tracks) {
        const row = el("div", "track");
        const title = el("div", "title");
        title.textContent = t.title || t.track_id || "Track";

        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = t.web_path || t.path || "";

        const meta = el("div", "meta");
        meta.textContent = audio.src;

        row.appendChild(title);
        row.appendChild(audio);
        row.appendChild(meta);
        app.appendChild(row);
      }
    })();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------

def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(str(args.playlist)).expanduser().resolve()
    if not playlist_path.exists():
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "playlist_not_found",
            "playlist": str(playlist_path),
        }) + "\n")
        return 2

    

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    tracks_dir = out_dir / "tracks"
    manifest_path = out_dir / "web_manifest.json"
    out_playlist_path = out_dir / "playlist.json"
    index_path = out_dir / "index.html"

    if bool(getattr(args, "clean", False)) and out_dir.exists():
        shutil.rmtree(out_dir)

    _safe_mkdir(tracks_dir)
    token = getattr(args, "token", None)
    if token:
        try:
            ctx = _require_pro_if_token(args)
            _write_billing_evidence(
                out_dir,
                token=str(token),
                ok=True,
                reason="allow",
                user_id=ctx.user_id,
                tier=ctx.tier,
                entitlements=sorted(ctx.entitlements),
            )
        except BaseException as e:
            reason = getattr(e, "msg", None) or str(e)
            _write_billing_evidence(
                out_dir,
                token=str(token),
                ok=False,
                reason=reason,
            )
            sys.stdout.write(_stable_json_dumps({
                "ok": False,
                "reason": "billing_denied",
                "error": reason,
                "hint": "If using --token, pass --billing-db or set MGC_BILLING_DB.",
            }) + "\n")
            return 2


    prefer_mp3 = bool(getattr(args, "prefer_mp3", False))
    strip_paths = bool(getattr(args, "strip_paths", False))
    fail_if_empty = bool(getattr(args, "fail_if_empty", False))
    fail_if_none_copied = bool(getattr(args, "fail_if_none_copied", False))
    fail_on_missing = bool(getattr(args, "fail_on_missing", False))
    deterministic = bool(getattr(args, "deterministic", False))

    playlist_obj = json.loads(playlist_path.read_text(encoding="utf-8"))
    entries = _collect_playlist_track_entries(playlist_obj)

    if fail_if_empty and not entries:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "empty_playlist",
            "playlist": str(playlist_path),
        }) + "\n")
        return 2

    playlist_dir = playlist_path.parent

    # Determine repo_root (best effort)
    repo_root = Path.cwd().resolve()
    db_path = (getattr(args, "db", "") or "").strip()
    if db_path:
        try:
            dbp = Path(db_path).expanduser().resolve()
            if dbp.parent.name == "data":
                repo_root = dbp.parent.parent
            else:
                repo_root = dbp.parent
        except Exception:
            pass

    # DB mapping: track_id -> artifact_path
    track_ids = [e["track_id"] for e in entries if e.get("track_id")]
    id_to_db_path, db_meta = _resolve_track_paths_from_db(db_path=db_path, track_ids=track_ids)

    # Evidence fallback: if playlist is placeholder-only, infer real audio paths
    inferred_paths = _infer_track_paths_from_sibling_files(playlist_path)

    copied = 0
    missing = 0
    bundled: List[Dict[str, Any]] = []
    bundled_tracks: List[_BundledTrack] = []
    out_tracks_for_playlist: List[Dict[str, Any]] = []

    for e in entries:
        i = int(e["index"])
        track_id = (e.get("track_id") or "").strip()
        title = (e.get("title") or "").strip() or track_id or f"Track {i+1}"
        raw = (e.get("raw_path") or "").strip()

        attempted: List[str] = []
        src_path: Optional[Path] = None
        resolved_from = ""

        # 1) resolve from playlist path if it exists
        if raw:
            rp = _resolve_input_path(raw, playlist_dir=playlist_dir, repo_root=repo_root)
            if prefer_mp3:
                rp = _prefer_mp3_path(rp)
            attempted.append(str(rp))
            if rp.exists() and rp.is_file():
                src_path = rp
                resolved_from = "playlist"

        # 2) fallback to DB artifact_path using track_id
        if (src_path is None or (not src_path.exists())) and track_id and track_id in id_to_db_path:
            db_raw = id_to_db_path[track_id]
            rp = _resolve_input_path(db_raw, playlist_dir=playlist_dir, repo_root=repo_root)
            if prefer_mp3:
                rp = _prefer_mp3_path(rp)
            attempted.append(str(rp))
            if rp.exists() and rp.is_file():
                src_path = rp
                resolved_from = "db"

        # 3) evidence inference fallback (use first inferred path)
        if (src_path is None or (not src_path.exists())) and inferred_paths:
            rp = Path(inferred_paths[0]).expanduser()
            try:
                rp = rp.resolve()
            except Exception:
                pass
            if prefer_mp3:
                rp = _prefer_mp3_path(rp)
            attempted.append(str(rp))
            if rp.exists() and rp.is_file():
                src_path = rp
                resolved_from = "evidence"

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
                    "source": raw,
                    "attempted": attempted,
                    "db_meta": db_meta,
                    "inferred_paths": inferred_paths,
                }) + "\n")
                return 2
            continue

        # Deterministic destination naming:
        placeholder_id = (track_id.startswith("00000000-0000-0000-0000-") and track_id.endswith("000000000001"))
        if track_id and not placeholder_id:
            dest_name = f"{track_id}{src_path.suffix.lower()}"
            out_track_id = track_id
        else:
            dest_name = src_path.name
            out_track_id = src_path.stem

        dest = tracks_dir / dest_name
        shutil.copyfile(str(src_path), str(dest))

        rel = f"tracks/{dest.name}"
        _ensure_portable_relpath(rel)

        sha = _sha256_file(dest)
        size = int(dest.stat().st_size)

        bundled.append({
            "index": i,
            "ok": True,
            "track_id": out_track_id,
            "title": title,
            "source": str(Path("tracks") / src_path.name) if strip_paths else str(src_path),
            "dest": rel,
            "web_path": rel,
            "resolved_from": resolved_from,
            "sha256": sha,
            "bytes": size,
        })

        bundled_tracks.append(_BundledTrack(
            index=i,
            track_id=out_track_id,
            title=title,
            src_path=str(src_path),
            relpath=rel,
            sha256=sha,
            bytes=size,
        ))

        out_tracks_for_playlist.append({
            "title": title,
            "track_id": out_track_id,
            "path": rel,
            "web_path": rel,
            "sha256": sha,
            "bytes": size,
        })

        copied += 1

    if fail_if_none_copied and copied == 0:
        sys.stdout.write(_stable_json_dumps({
            "ok": False,
            "reason": "none_copied",
            "playlist": str(playlist_path),
            "missing": missing,
            "bundled": bundled,
            "db_meta": db_meta,
            "inferred_paths": inferred_paths,
        }) + "\n")
        return 2

    out_playlist = dict(playlist_obj) if isinstance(playlist_obj, dict) else {"tracks": []}
    out_playlist["tracks"] = out_tracks_for_playlist
    out_playlist_path.write_text(_stable_json_dumps(out_playlist), encoding="utf-8")

    index_path.write_text(_INDEX_HTML, encoding="utf-8")

    # Write manifest AFTER other files, but compute tree hash excluding the manifest itself.
    manifest = _build_web_manifest(
        out_dir=out_dir,
        playlist_obj=out_playlist,
        bundled_tracks=bundled_tracks,
        deterministic=deterministic,
    )
    manifest_path.write_text(_stable_json_dumps(manifest), encoding="utf-8")

    # Validate in-place (also checks web_tree_sha256).
    _validate_web_manifest(manifest, out_dir=out_dir)

    sys.stdout.write(_stable_json_dumps({
        "ok": True,
        "out_dir": str(out_dir),
        "index": str(index_path),
        "playlist_out": str(out_playlist_path),
        "manifest": str(manifest_path),
        "copied": copied,
        "missing": missing,
        "bundled": bundled,
        "db_meta": db_meta,
        "inferred_paths": inferred_paths,
        "web_tree_sha256": manifest.get("web_tree_sha256"),
    }) + "\n")
    return 0


def cmd_web_validate(args: argparse.Namespace) -> int:
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    manifest_path = out_dir / "web_manifest.json"
    if not manifest_path.exists():
        _die(f"web_manifest.json missing: {manifest_path}", 2)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_web_manifest(manifest, out_dir=out_dir)

    sys.stdout.write(_stable_json_dumps({
        "ok": True,
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "web_tree_sha256": _tree_hash(out_dir),
    }) + "\n")
    return 0


def cmd_web_serve(args: argparse.Namespace) -> int:
    directory = Path(str(args.dir)).expanduser().resolve()
    if not directory.exists():
        sys.stdout.write(_stable_json_dumps({"ok": False, "reason": "dir_not_found", "dir": str(directory)}) + "\n")
        return 2

    token = getattr(args, "token", None)
    if token:
        try:
            _require_pro_if_token(args)
            _write_billing_evidence(
                Path(args.out_dir),
                token=str(token),
                ok=True,
                reason="allow",
                action="web.build",
            )
        except BaseException as e:
            reason = getattr(e, "msg", None) or str(e)
            _write_billing_evidence(
                Path(args.out_dir),
                token=str(token),
                ok=False,
                reason=str(reason),
                action="web.build",
            )
            sys.stdout.write(_stable_json_dumps({
                "ok": False,
                "reason": "billing_denied",
                "error": str(reason),
                "hint": "If using --token, pass --billing-db or set MGC_BILLING_DB.",
            }) + "\n")
            return 2

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, directory=str(directory), **kw)

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), Handler)

    sys.stdout.write(_stable_json_dumps({
        "ok": True,
        "serving": str(directory),
        "url": f"http://{host}:{port}/",
    }) + "\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


# ---------------------------------------------------------------------
# Registrar + aliases
# ---------------------------------------------------------------------

def register_web_subcommand(subparsers: argparse._SubParsersAction) -> None:
    web = subparsers.add_parser("web", help="Static web player build/serve")
    ws = web.add_subparsers(dest="web_cmd", required=True)

    build = ws.add_parser("build", help="Build a static web bundle")
    build.add_argument("--playlist", required=True, help="Playlist JSON path")
    build.add_argument("--out-dir", required=True, help="Output directory for the web bundle")
    build.add_argument("--db", default="", help="Main DB path (used to resolve track_id -> artifact_path)")
    build.add_argument("--token", default=None, help="Billing access token (requires pro)")
    build.add_argument(
        "--billing-db",
        dest="billing_db",
        default=None,
        help="DB path for billing/token validation (recommended). Uses MGC_BILLING_DB or --db.",
    )
    build.add_argument("--prefer-mp3", action="store_true", help="Prefer .mp3 when a .wav sibling exists")
    build.add_argument("--strip-paths", action="store_true", help="Strip absolute paths in output diagnostics")
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
    serve.add_argument("--dir", required=True, help="Directory containing index.html + playlist.json")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", default=8000, type=int, help="Bind port")
    serve.add_argument("--token", default=None, help="Billing access token (requires pro)")
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
