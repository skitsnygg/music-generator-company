#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import contextlib
import io

@contextlib.contextmanager
def _silence_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    old = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = old


# ---------------------------------------------------------------------------
# Determinism utilities
# ---------------------------------------------------------------------------

def is_deterministic(args: Optional[argparse.Namespace] = None) -> bool:
    if args is not None and getattr(args, "deterministic", False):
        return True
    v = (os.environ.get("MGC_DETERMINISTIC") or os.environ.get("DETERMINISTIC") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def deterministic_now_iso(deterministic: bool) -> str:
    fixed = (os.environ.get("MGC_FIXED_TIME") or "").strip()
    if fixed:
        try:
            if fixed.isdigit():
                dt = datetime.fromtimestamp(int(fixed), tz=timezone.utc)
                return dt.isoformat()
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(fixed.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            if deterministic:
                return "2020-01-01T00:00:00+00:00"

    if deterministic:
        return "2020-01-01T00:00:00+00:00"
    return datetime.now(timezone.utc).isoformat()


def stable_uuid5(*parts: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    return str(uuid.uuid5(namespace, "|".join(parts)))


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text_bytes(path: Path) -> bytes:
    b = path.read_bytes()
    try:
        s = b.decode("utf-8")
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        return s.encode("utf-8")
    except Exception:
        return b


# ---------------------------------------------------------------------------
# DB helpers (schema-tolerant + NOT NULL fillers)
# ---------------------------------------------------------------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def die(msg: str, code: int = 2) -> "NoReturn":
    eprint(msg)
    raise SystemExit(code)


def db_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    col_type: str
    notnull: bool
    dflt_value: Optional[str]


def db_table_info(con: sqlite3.Connection, table: str) -> List[ColumnInfo]:
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return []
    out: List[ColumnInfo] = []
    for r in rows:
        # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
        out.append(
            ColumnInfo(
                name=str(r[1]),
                col_type=str(r[2] or ""),
                notnull=bool(r[3]),
                dflt_value=(r[4] if r[4] is not None else None),
            )
        )
    return out


def db_table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    return [c.name for c in db_table_info(con, table)]


def _pick_first_existing(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _stable_int_from_key(key: str, lo: int, hi: int) -> int:
    if hi <= lo:
        return lo
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return lo + (h % (hi - lo + 1))


def _infer_required_value(col: ColumnInfo, row_data: Dict[str, Any]) -> Any:
    name = col.name.lower()

    ts = (
        row_data.get("ts")
        or row_data.get("occurred_at")
        or row_data.get("created_at")
        or row_data.get("created_ts")
        or row_data.get("time")
    )
    ts_val = ts if isinstance(ts, str) and ts else "2020-01-01T00:00:00+00:00"

    stable_key = (
        str(
            row_data.get("id")
            or row_data.get("track_id")
            or row_data.get("post_id")
            or row_data.get("event_id")
            or row_data.get("drop_id")
            or ""
        )
        + f"|{col.name}"
    )

    if name in ("occurred_at", "created_at", "ts", "timestamp") or name.endswith("_at") or "timestamp" in name:
        return ts_val

    if name == "bpm":
        return _stable_int_from_key(stable_key, 60, 140)
    if name in ("duration_ms", "duration_millis", "duration_milliseconds"):
        return _stable_int_from_key(stable_key, 15_000, 180_000)
    if name in ("duration_s", "duration_sec", "duration_secs", "duration_seconds"):
        return _stable_int_from_key(stable_key, 15, 180)

    if name == "provider":
        return str(row_data.get("provider") or "stub")
    if name in ("kind", "event_kind", "type"):
        return str(row_data.get("kind") or "unknown")
    if name in ("actor", "source"):
        return str(row_data.get("actor") or "system")
    if name in ("status", "state"):
        return str(row_data.get("status") or "draft")
    if name in ("platform", "channel", "destination"):
        return str(row_data.get("platform") or "unknown")
    if name in ("title", "name"):
        return str(row_data.get("title") or row_data.get("name") or "Untitled")
    if name in ("context", "mood"):
        return str(row_data.get("context") or row_data.get("mood") or "focus")

    if name in ("meta_json", "metadata_json", "meta", "metadata"):
        return stable_json_dumps(row_data.get("meta") or {})

    t = (col.col_type or "").upper()
    if "INT" in t or "REAL" in t or "NUM" in t:
        return 0
    if "BLOB" in t:
        return b""
    return ""


def _insert_row(con: sqlite3.Connection, table: str, data: Dict[str, Any]) -> None:
    info = db_table_info(con, table)
    if not info:
        raise sqlite3.OperationalError(f"table {table} does not exist")

    cols_set = {c.name for c in info}
    filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in cols_set}

    for col in info:
        if not col.notnull:
            continue
        if col.name in filtered:
            continue
        if col.dflt_value is not None:
            continue
        filtered[col.name] = _infer_required_value(col, {**data, **filtered})

    if not filtered:
        return

    keys = sorted(filtered.keys())
    placeholders = ",".join(["?"] * len(keys))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(keys)}) VALUES ({placeholders})"
    con.execute(sql, tuple(filtered[k] for k in keys))
    con.commit()


def _update_row(con: sqlite3.Connection, table: str, where_col: str, where_val: Any, patch: Dict[str, Any]) -> None:
    info = db_table_info(con, table)
    if not info:
        raise sqlite3.OperationalError(f"table {table} does not exist")
    cols_set = {c.name for c in info}

    filtered: Dict[str, Any] = {k: v for k, v in patch.items() if k in cols_set}
    if not filtered:
        return

    keys = sorted(filtered.keys())
    set_expr = ", ".join([f"{k} = ?" for k in keys])
    sql = f"UPDATE {table} SET {set_expr} WHERE {where_col} = ?"
    con.execute(sql, tuple(filtered[k] for k in keys) + (where_val,))
    con.commit()


def ensure_tables_minimal(con: sqlite3.Connection) -> None:
    """
    Creates minimal tables *only if missing*. We do not migrate existing schemas.
    """
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor TEXT NOT NULL,
            meta_json TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS marketing_posts (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            platform TEXT NOT NULL,
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            meta_json TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            title TEXT NOT NULL,
            provider TEXT NOT NULL,
            mood TEXT,
            genre TEXT,
            artifact_path TEXT,
            meta_json TEXT NOT NULL
        )
        """
    )
    # Drops: a first-class "release" tying daily run + track + marketing publish batch
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS drops (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            context TEXT NOT NULL,
            seed TEXT NOT NULL,
            run_id TEXT NOT NULL,
            track_id TEXT,
            marketing_batch_id TEXT,
            published_ts TEXT,
            meta_json TEXT NOT NULL
        )
        """
    )
    con.commit()


def _row_first(row: sqlite3.Row, candidates: Sequence[str], default: Any = "") -> Any:
    keys = set(row.keys())
    for c in candidates:
        if c in keys:
            return row[c]
    return default


def _load_json_maybe(s: Any) -> Dict[str, Any]:
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _detect_marketing_content_col(cols: Sequence[str]) -> Optional[str]:
    return _pick_first_existing(
        cols,
        ["content", "body", "text", "draft", "caption", "copy", "post_text", "message"],
    )


def _detect_marketing_meta_col(cols: Sequence[str]) -> Optional[str]:
    return _pick_first_existing(
        cols,
        ["meta_json", "metadata_json", "meta", "metadata", "meta_blob", "payload", "payload_json"],
    )


def _best_text_payload_column(con: sqlite3.Connection, table: str, reserved: Sequence[str]) -> Optional[str]:
    info = db_table_info(con, table)
    if not info:
        return None
    reserved_set = {r.lower() for r in reserved}
    candidates: List[str] = []
    for c in info:
        n = c.name.lower()
        if n in reserved_set:
            continue
        t = (c.col_type or "").upper()
        if "CHAR" in t or "TEXT" in t or "CLOB" in t or t == "":
            candidates.append(c.name)
    if not candidates:
        return None
    priority = ["content", "body", "text", "draft", "caption", "copy", "message", "payload", "data", "json"]
    for p in priority:
        for c in candidates:
            if p in c.lower():
                return c
    return sorted(candidates)[0]


def _extract_inner_content_from_blob(s: str) -> Optional[str]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            v = obj.get("content")
            if isinstance(v, str):
                return v
    except Exception:
        return None
    return None


def _marketing_row_content(con: sqlite3.Connection, row: sqlite3.Row) -> str:
    cols = row.keys()

    # 1) explicit content-like columns
    content_col = _detect_marketing_content_col(cols)
    if content_col:
        v = row[content_col]
        if v is None:
            return ""
        s = str(v)
        inner = _extract_inner_content_from_blob(s)
        return inner if inner is not None else s

    # 2) best available TEXT-ish payload column (schema unknown)
    best_payload = _best_text_payload_column(
        con,
        "marketing_posts",
        reserved=[
            "id", "post_id", "ts", "created_at",
            "platform", "channel", "destination",
            "status", "state",
            "meta_json", "metadata_json", "meta", "metadata",
        ],
    )
    if best_payload and best_payload in set(cols):
        v = row[best_payload]
        if v is not None and str(v).strip():
            s = str(v)
            inner = _extract_inner_content_from_blob(s)
            return inner if inner is not None else s

    # 3) meta-ish columns
    meta_col = _detect_marketing_meta_col(cols)
    if meta_col:
        meta = _load_json_maybe(row[meta_col])
        for k in ("content", "body", "text", "draft", "caption", "copy", "post_text", "message"):
            if k in meta and meta[k] is not None:
                if isinstance(meta[k], dict):
                    return stable_json_dumps(meta[k])
                return str(meta[k])
        for k in ("draft_obj", "payload", "data"):
            if k in meta and meta[k] is not None:
                try:
                    return stable_json_dumps(meta[k])
                except Exception:
                    return str(meta[k])

    return ""


def _marketing_row_meta(con: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
    """
    Best-effort extraction of metadata for a marketing row across arbitrary schemas.
    """
    cols = row.keys()

    # Prefer explicit meta-ish column if it exists
    meta_col = _detect_marketing_meta_col(cols)
    if meta_col:
        meta = _load_json_maybe(row[meta_col])
        if meta:
            return meta

    # Otherwise, attempt to parse the payload blob and see if it's a dict
    best_payload = _best_text_payload_column(
        con,
        "marketing_posts",
        reserved=[
            "id", "post_id", "ts", "created_at",
            "platform", "channel", "destination",
            "status", "state",
        ],
    )
    if best_payload and best_payload in set(cols):
        v = row[best_payload]
        meta = _load_json_maybe(v)
        if meta:
            return meta

    # Finally, try to parse the content string itself if it looks like JSON
    content = _marketing_row_content(con, row)
    meta = _load_json_maybe(content)
    if meta:
        return meta

    return {}


def db_insert_event(con: sqlite3.Connection, *, event_id: str, ts: str, kind: str, actor: str, meta: Dict[str, Any]) -> None:
    _insert_row(
        con,
        "events",
        {
            "id": event_id,
            "event_id": event_id,
            "ts": ts,
            "occurred_at": ts,
            "created_at": ts,
            "kind": kind,
            "actor": actor,
            "meta_json": stable_json_dumps(meta),
            "metadata_json": stable_json_dumps(meta),
            "meta": stable_json_dumps(meta),
            "metadata": stable_json_dumps(meta),
        },
    )

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def db_insert_track(
    con: sqlite3.Connection,
    *,
    track_id: str,
    ts: str,
    title: str,
    provider: str,
    mood: Optional[str],
    genre: Optional[str],
    artifact_path: Optional[str],
    meta: Dict[str, Any],
) -> None:
    cols = db_table_columns(con, "tracks")
    if not cols:
        raise sqlite3.OperationalError("table tracks does not exist")

    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on"])
    path_col = _pick_first_existing(cols, ["artifact_path", "audio_path", "path", "file_path", "uri"])
    bpm_col = _pick_first_existing(cols, ["bpm", "tempo"])

    data: Dict[str, Any] = {
        "id": track_id,
        "track_id": track_id,
        "title": title,
        "name": title,
        "provider": provider,
        "mood": mood,
        "genre": genre,
        "meta_json": stable_json_dumps(meta),
        "metadata_json": stable_json_dumps(meta),
        "meta": stable_json_dumps(meta),
        "metadata": stable_json_dumps(meta),
    }
    if ts_col:
        data[ts_col] = ts
    if path_col and artifact_path is not None:
        data[path_col] = artifact_path
    if bpm_col:
        data[bpm_col] = _stable_int_from_key(f"{track_id}|{title}|{provider}|{bpm_col}", 60, 140)

    _insert_row(con, "tracks", data)


def db_insert_drop(
    con: sqlite3.Connection,
    *,
    drop_id: str,
    ts: str,
    context: str,
    seed: str,
    run_id: str,
    track_id: Optional[str],
    meta: Dict[str, Any],
) -> None:
    cols = db_table_columns(con, "drops")
    if not cols:
        raise sqlite3.OperationalError("table drops does not exist")

    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on"])
    ctx_col = _pick_first_existing(cols, ["context", "mood"])
    seed_col = _pick_first_existing(cols, ["seed"])
    run_col = _pick_first_existing(cols, ["run_id"])
    track_col = _pick_first_existing(cols, ["track_id"])
    meta_col = _pick_first_existing(cols, ["meta_json", "metadata_json", "meta", "metadata"])

    data: Dict[str, Any] = {"id": drop_id, "drop_id": drop_id}
    if ts_col:
        data[ts_col] = ts
    if ctx_col:
        data[ctx_col] = context
    if seed_col:
        data[seed_col] = seed
    if run_col:
        data[run_col] = run_id
    if track_col and track_id is not None:
        data[track_col] = track_id
    if meta_col:
        data[meta_col] = stable_json_dumps(meta)

    # Also add common column names for schema variance
    data.update(
        {
            "context": context,
            "mood": context,
            "seed": seed,
            "run_id": run_id,
            "track_id": track_id,
            "meta_json": stable_json_dumps(meta),
            "metadata_json": stable_json_dumps(meta),
        }
    )

    _insert_row(con, "drops", data)


def db_drop_mark_published(
    con: sqlite3.Connection,
    *,
    run_id: str,
    marketing_batch_id: str,
    published_ts: str,
) -> int:
    cols = db_table_columns(con, "drops")
    if not cols:
        raise sqlite3.OperationalError("table drops does not exist")

    run_col = _pick_first_existing(cols, ["run_id"])
    if not run_col:
        return 0

    batch_col = _pick_first_existing(cols, ["marketing_batch_id", "batch_id"])
    pub_ts_col = _pick_first_existing(cols, ["published_ts", "published_at", "published_time", "ts_published"])

    patch: Dict[str, Any] = {}
    if batch_col:
        patch[batch_col] = marketing_batch_id
    if pub_ts_col:
        patch[pub_ts_col] = published_ts

    if not patch:
        return 0

    # Update all drops with this run_id (should be 1, but keep it tolerant)
    sql = f"UPDATE drops SET {', '.join([f'{k} = ?' for k in sorted(patch.keys())])} WHERE {run_col} = ?"
    params = [patch[k] for k in sorted(patch.keys())] + [run_id]
    cur = con.execute(sql, params)
    con.commit()
    return int(cur.rowcount or 0)


def db_insert_marketing_post(
    con: sqlite3.Connection,
    *,
    post_id: str,
    ts: str,
    platform: str,
    status: str,
    content: str,
    meta: Dict[str, Any],
) -> None:
    cols = db_table_columns(con, "marketing_posts")
    if not cols:
        raise sqlite3.OperationalError("table marketing_posts does not exist")

    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on"])
    platform_col = _pick_first_existing(cols, ["platform", "channel", "destination"])
    status_col = _pick_first_existing(cols, ["status", "state"])

    content_col = _detect_marketing_content_col(cols)
    meta_col = _detect_marketing_meta_col(cols)

    best_payload = None
    if not content_col and not meta_col:
        best_payload = _best_text_payload_column(
            con,
            "marketing_posts",
            reserved=["id", "post_id", "ts", "created_at", "platform", "channel", "destination", "status", "state"],
        )

    meta_to_store = dict(meta)
    if not content_col:
        meta_to_store["content"] = content

    data: Dict[str, Any] = {"id": post_id, "post_id": post_id}
    if ts_col:
        data[ts_col] = ts
    if platform_col:
        data[platform_col] = platform
    if status_col:
        data[status_col] = status

    if content_col:
        data[content_col] = content
    elif meta_col:
        data[meta_col] = stable_json_dumps(meta_to_store)
    elif best_payload:
        data[best_payload] = stable_json_dumps(meta_to_store) if meta_to_store else content

    if meta_col and meta_col not in data:
        data[meta_col] = stable_json_dumps(meta)

    _insert_row(con, "marketing_posts", data)


def db_marketing_posts_pending(con: sqlite3.Connection, *, limit: int = 50) -> List[sqlite3.Row]:
    cols = db_table_columns(con, "marketing_posts")
    if not cols:
        raise sqlite3.OperationalError("table marketing_posts does not exist")

    status_col = _pick_first_existing(cols, ["status", "state"])
    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on", "occurred_at"])
    if not status_col:
        return []

    order = f"{ts_col} ASC, id ASC" if ts_col else "id ASC"
    sql = f"SELECT * FROM marketing_posts WHERE {status_col} = ? ORDER BY {order} LIMIT ?"
    cur = con.execute(sql, ("draft", int(limit)))
    return list(cur.fetchall())


def db_marketing_post_set_status(
    con: sqlite3.Connection,
    *,
    post_id: str,
    status: str,
    ts: str,
    meta_patch: Dict[str, Any],
) -> None:
    cols = db_table_columns(con, "marketing_posts")
    if not cols:
        raise sqlite3.OperationalError("table marketing_posts does not exist")

    status_col = _pick_first_existing(cols, ["status", "state"])
    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on"])
    meta_col = _detect_marketing_meta_col(cols)

    existing_meta: Dict[str, Any] = {}
    if meta_col:
        row = con.execute(f"SELECT {meta_col} FROM marketing_posts WHERE id = ?", (post_id,)).fetchone()
        if row:
            existing_meta = _load_json_maybe(row[0])
    existing_meta.update(meta_patch)

    patch: Dict[str, Any] = {}
    if status_col:
        patch[status_col] = status
    if ts_col:
        patch[ts_col] = ts
    if meta_col:
        patch[meta_col] = stable_json_dumps(existing_meta)

    where_col = "id" if "id" in set(cols) else (_pick_first_existing(cols, ["post_id"]) or "id")
    _update_row(con, "marketing_posts", where_col, post_id, patch)


# ---------------------------------------------------------------------------
# Manifest (deterministic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sha256: str
    size: int


def iter_repo_files(
    repo_root: Path,
    include_globs: Optional[Sequence[str]] = None,
    exclude_dirs: Optional[Sequence[str]] = None,
    exclude_globs: Optional[Sequence[str]] = None,
) -> Iterable[Path]:
    if exclude_dirs is None:
        exclude_dirs = [
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".idea",
            "logs",
            "data",
            "artifacts",
        ]
    if exclude_globs is None:
        exclude_globs = [
            "*.pyc",
            "*.pyo",
            "*.log",
            "*.tmp",
            "*.swp",
            "*.swo",
            "*.sqlite-journal",
            "*.db-journal",
            "*.wav",
            "*.mp3",
            "*.flac",
            "*.ogg",
            "*.sqlite",
            "*.db",
        ]

    if include_globs:
        seen: set[Path] = set()
        for g in include_globs:
            for p in sorted(repo_root.glob(g)):
                if p.is_file():
                    rp = p.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        yield p
        return

    all_files: List[Path] = []
    for p in repo_root.rglob("*"):
        try:
            rel_parts = p.relative_to(repo_root).parts
        except Exception:
            continue
        if any(part in exclude_dirs for part in rel_parts):
            continue
        if p.is_dir():
            continue
        name = p.name
        if any(Path(name).match(pat) for pat in exclude_globs):
            continue
        all_files.append(p)

    all_files.sort(key=lambda x: str(x.relative_to(repo_root)).replace("\\", "/"))
    for p in all_files:
        yield p


def compute_manifest(repo_root: Path, *, include: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    entries: List[ManifestEntry] = []
    for p in iter_repo_files(repo_root, include_globs=include):
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        b = read_text_bytes(p)
        entries.append(ManifestEntry(path=rel, sha256=sha256_hex(b), size=len(b)))

    entries.sort(key=lambda e: e.path)
    manifest_obj = {
        "version": 1,
        "root": str(repo_root.resolve()),
        "entries": [{"path": e.path, "sha256": e.sha256, "size": e.size} for e in entries],
    }
    root_hash_payload = [{"path": e.path, "sha256": e.sha256, "size": e.size} for e in entries]
    manifest_obj["root_tree_sha256"] = sha256_hex(stable_json_dumps(root_hash_payload).encode("utf-8"))
    return manifest_obj


def cmd_run_manifest(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        die(f"repo_root does not exist: {repo_root}")

    include = args.include or None
    manifest = compute_manifest(repo_root, include=include)

    out_path = Path(args.out) if args.out else None
    text = stable_json_dumps(manifest) + "\n"

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)

    if getattr(args, "print_hash", False):
        eprint(manifest["root_tree_sha256"])

    return 0


# ---------------------------------------------------------------------------
# Daily run (deterministic orchestrator)
# ---------------------------------------------------------------------------

def _maybe_call_external_daily_runner(*, db_path: str, context: str, seed: str, deterministic: bool, ts: str) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[str, str]] = [
        ("mgc.daily", "run_daily"),
        ("mgc.music_agent", "run_daily"),
        ("mgc.pipeline", "run_daily"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn(  # type: ignore[misc]
                    db_path=db_path,
                    context=context,
                    seed=seed,
                    deterministic=deterministic,
                    ts=ts,
                )
        except Exception:
            continue
    return None


def _stub_daily_run(*, con: sqlite3.Connection, context: str, seed: str, deterministic: bool, ts: str, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = stable_uuid5("daily_run", context, seed, ts if not deterministic else "fixed")
    drop_id = stable_uuid5("drop", run_id)

    track_id = stable_uuid5("track", context, seed, run_id)
    title = f"{context.title()} Track {seed}"
    provider = "stub"
    mood = context
    genre = "ambient" if context == "focus" else "mixed"

    if deterministic:
        artifact_rel = Path("data") / "tracks" / f"{track_id}.wav"
    else:
        day = ts.split("T", 1)[0]
        artifact_rel = Path("data") / "tracks" / day / f"{track_id}.wav"

    artifact_path = (Path.cwd() / artifact_rel).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"track_id": track_id, "run_id": run_id, "drop_id": drop_id, "context": context, "seed": seed, "ts": ts, "provider": provider}
    artifact_bytes = stable_json_dumps(payload).encode("utf-8")
    artifact_path.write_bytes(artifact_bytes)

    db_insert_track(
        con,
        track_id=track_id,
        ts=ts,
        title=title,
        provider=provider,
        mood=mood,
        genre=genre,
        artifact_path=str(artifact_rel).replace("\\", "/"),
        meta={"run_id": run_id, "drop_id": drop_id, "deterministic": deterministic, "seed": seed, "context": context},
    )

    # Insert the drop *before* marketing so it exists even if later steps fail
    db_insert_drop(
        con,
        drop_id=drop_id,
        ts=ts,
        context=context,
        seed=seed,
        run_id=run_id,
        track_id=track_id,
        meta={"run_id": run_id, "drop_id": drop_id, "track_id": track_id, "deterministic": deterministic, "seed": seed, "context": context},
    )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "drop.created", drop_id),
        ts=ts,
        kind="drop.created",
        actor="system",
        meta={"drop_id": drop_id, "run_id": run_id, "track_id": track_id, "context": context, "seed": seed},
    )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "track.generated", run_id),
        ts=ts,
        kind="track.generated",
        actor="system",
        meta={"run_id": run_id, "drop_id": drop_id, "track_id": track_id, "artifact_path": str(artifact_rel).replace("\\", "/")},
    )

    platforms = ["x", "youtube_shorts", "instagram_reels", "tiktok"]
    post_ids: List[str] = []
    for platform in platforms:
        post_id = stable_uuid5("marketing_post", platform, run_id)
        post_ids.append(post_id)

        content_obj = {
            "platform": platform,
            "hook": f"New {context} track is ready.",
            "cta": "Listen now.",
            "track_id": track_id,
            "run_id": run_id,
            "drop_id": drop_id,
        }
        content = stable_json_dumps(content_obj)

        db_insert_marketing_post(
            con,
            post_id=post_id,
            ts=ts,
            platform=platform,
            status="draft",
            content=content,
            meta={"run_id": run_id, "drop_id": drop_id, "track_id": track_id, "deterministic": deterministic},
        )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "marketing.drafts.created", run_id),
        ts=ts,
        kind="marketing.drafts.created",
        actor="system",
        meta={"run_id": run_id, "drop_id": drop_id, "count": len(post_ids), "post_ids": post_ids},
    )

    evidence = {
        "run_id": run_id,
        "drop_id": drop_id,
        "ts": ts,
        "deterministic": deterministic,
        "context": context,
        "seed": seed,
        "track": {
            "id": track_id,
            "title": title,
            "provider": provider,
            "artifact_path": str(artifact_rel).replace("\\", "/"),
            "sha256": sha256_hex(artifact_bytes),
            "size": len(artifact_bytes),
        },
        "marketing_drafts": [{"id": pid, "platform": p} for pid, p in zip(post_ids, platforms)],
    }
    evidence_path = out_dir / ("daily_evidence.json" if deterministic else f"daily_evidence_{run_id}.json")
    evidence_path.write_text(stable_json_dumps(evidence) + "\n", encoding="utf-8", newline="\n")

    return evidence


def cmd_run_daily(args: argparse.Namespace) -> int:
    deterministic = is_deterministic(args)

    db_path = args.db or os.environ.get("MGC_DB") or "data/db.sqlite"
    context = args.context or os.environ.get("MGC_CONTEXT") or "focus"
    seed = str(args.seed if args.seed is not None else (os.environ.get("MGC_SEED") or "1"))

    ts = deterministic_now_iso(deterministic)
    out_dir = Path(args.out_dir or "data/evidence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe = _maybe_call_external_daily_runner(
        db_path=db_path,
        context=context,
        seed=seed,
        deterministic=deterministic,
        ts=ts,
    )
    if maybe is not None:
        con = db_connect(db_path)
        ensure_tables_minimal(con)
        run_id = str(maybe.get("run_id") or stable_uuid5("daily_run", context, seed, ts if not deterministic else "fixed"))
        drop_id = str(maybe.get("drop_id") or stable_uuid5("drop", run_id))
        track_id = str(maybe.get("track_id") or maybe.get("track", {}).get("id") or "")

        # Ensure a drop exists even with an external runner
        db_insert_drop(
            con,
            drop_id=drop_id,
            ts=ts,
            context=context,
            seed=seed,
            run_id=run_id,
            track_id=track_id if track_id else None,
            meta={"run_id": run_id, "drop_id": drop_id, "track_id": track_id, "deterministic": deterministic, "seed": seed, "context": context, "external_runner": True},
        )
        db_insert_event(
            con,
            event_id=stable_uuid5("event", "daily.external_runner", run_id),
            ts=ts,
            kind="daily.external_runner",
            actor="system",
            meta={"run_id": run_id, "drop_id": drop_id, "module": "external", "deterministic": deterministic},
        )
        sys.stdout.write(stable_json_dumps(maybe) + "\n")
        return 0

    con = db_connect(db_path)
    ensure_tables_minimal(con)
    evidence = _stub_daily_run(con=con, context=context, seed=seed, deterministic=deterministic, ts=ts, out_dir=out_dir)
    sys.stdout.write(stable_json_dumps(evidence) + "\n")
    return 0

def cmd_run_drop(args: argparse.Namespace) -> int:
    """
    Product-level command: daily + publish-marketing + manifest + consolidated evidence.

    Guarantees:
      - stdout emits exactly ONE JSON object (so `json.tool` works)
      - deterministic under --deterministic / MGC_DETERMINISTIC=1 (and MGC_FIXED_TIME)
      - persists evidence/manifest pointers into drops.meta_json
    """
    deterministic = is_deterministic(args)

    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    context = str(args.context or os.environ.get("MGC_CONTEXT") or "focus")
    seed = str(args.seed if args.seed is not None else (os.environ.get("MGC_SEED") or "1"))
    limit = int(getattr(args, "limit", None) or 50)
    dry_run = bool(getattr(args, "dry_run", False))

    ts = deterministic_now_iso(deterministic)
    out_dir = Path(getattr(args, "out_dir", None) or os.environ.get("MGC_EVIDENCE_DIR") or "data/evidence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Silence subcommand stdout so run drop outputs exactly one JSON object.
    with _silence_stdout(True):
        daily_ns = argparse.Namespace(
            db=db_path,
            context=context,
            seed=seed,
            out_dir=str(out_dir),
            deterministic=deterministic,
            json=False,
        )
        cmd_run_daily(daily_ns)

    with _silence_stdout(True):
        pub_ns = argparse.Namespace(
            db=db_path,
            limit=limit,
            dry_run=dry_run,
            deterministic=deterministic,
            json=False,
        )
        cmd_publish_marketing(pub_ns)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    drop_row = con.execute(
        "SELECT * FROM drops WHERE context = ? AND seed = ? ORDER BY ts DESC, id DESC LIMIT 1",
        (context, seed),
    ).fetchone()
    drop_obj: Dict[str, Any] = dict(drop_row) if drop_row else {}

    drop_id = str(drop_obj.get("id") or "")
    out_path = out_dir / ("drop_evidence.json" if deterministic else f"drop_evidence_{drop_id}.json")
    manifest_path = out_dir / ("manifest.json" if deterministic else f"manifest_{drop_id}.json")

    # Run manifest via existing cmd_run_manifest (silenced)
    with _silence_stdout(True):
        manifest_ns = argparse.Namespace(
            repo_root=str(getattr(args, "repo_root", None) or "."),
            out=str(manifest_path),
            print_hash=False,

            # cmd_run_manifest expects these
            include=getattr(args, "include", None),
            exclude=getattr(args, "exclude", None),

            # safe defaults in case cmd_run_manifest reads them
            follow_symlinks=bool(getattr(args, "follow_symlinks", False)),
            max_file_bytes=getattr(args, "max_file_bytes", None),

            # keep signature-compatible if it reads these
            deterministic=deterministic,
            db=db_path,
            json=False,
        )
        cmd_run_manifest(manifest_ns)

    manifest_sha256 = _sha256_file(manifest_path)

    evidence: Dict[str, Any] = {
        "ts": ts,
        "deterministic": deterministic,
        "context": context,
        "seed": seed,
        "drop": {
            "id": drop_obj.get("id"),
            "run_id": drop_obj.get("run_id"),
            "track_id": drop_obj.get("track_id"),
            "marketing_batch_id": drop_obj.get("marketing_batch_id"),
            "published_ts": drop_obj.get("published_ts"),
        },
        "paths": {
            "evidence_path": str(out_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
        },
    }

    # Write consolidated evidence (stable newlines)
    out_path.write_text(stable_json_dumps(evidence) + "\n", encoding="utf-8", newline="\n")

    # Persist pointers in drops.meta_json
    if drop_id:
        existing_meta: Dict[str, Any] = {}
        meta_raw = drop_obj.get("meta_json")
        if isinstance(meta_raw, str) and meta_raw.strip():
            try:
                existing_meta = json.loads(meta_raw)
            except Exception:
                existing_meta = {}

        existing_meta.update(evidence["paths"])

        con.execute(
            "UPDATE drops SET meta_json = ? WHERE id = ?",
            (stable_json_dumps(existing_meta), drop_id),
        )
        con.commit()

    # Emit exactly one JSON object
    sys.stdout.write(stable_json_dumps(evidence) + "\n")
    return 0


# ---------------------------------------------------------------------------
# Marketing publish (deterministic) -- skips empty drafts, updates drops
# ---------------------------------------------------------------------------

def cmd_publish_marketing(args: argparse.Namespace) -> int:
    deterministic = is_deterministic(args)

    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    limit = int(getattr(args, "limit", None) or 50)
    dry_run = bool(getattr(args, "dry_run", False))

    ts = deterministic_now_iso(deterministic)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    pending = db_marketing_posts_pending(con, limit=limit)

    published: List[Dict[str, Any]] = []
    skipped_ids: List[str] = []
    run_ids_touched: List[str] = []

    for row in pending:
        post_id = str(_row_first(row, ["id", "post_id"], default=""))
        platform = str(_row_first(row, ["platform", "channel", "destination"], default="unknown"))
        content = _marketing_row_content(con, row)

        if not content.strip():
            skipped_ids.append(post_id)
            continue

        meta = _marketing_row_meta(con, row)
        run_id = str(meta.get("run_id") or "")
        drop_id = str(meta.get("drop_id") or "")

        publish_id = stable_uuid5("publish", post_id, platform, (ts if not deterministic else "fixed"))

        if not dry_run:
            db_marketing_post_set_status(
                con,
                post_id=post_id,
                status="published",
                ts=ts,
                meta_patch={"published_id": publish_id, "published_ts": ts},
            )

        published.append(
            {
                "post_id": post_id,
                "platform": platform,
                "published_id": publish_id,
                "published_ts": ts,
                "dry_run": dry_run,
                "content": content,
                "run_id": run_id,
                "drop_id": drop_id,
            }
        )

        if run_id:
            run_ids_touched.append(run_id)

    run_ids_touched = sorted(set([r for r in run_ids_touched if r]))

    batch_id = stable_uuid5(
        "marketing_publish_batch",
        (ts if not deterministic else "fixed"),
        str(limit),
        ("dry" if dry_run else "live"),
        str(len(skipped_ids)),
        str(len(published)),
        ("|".join(run_ids_touched) if deterministic else "nondet"),
    )

    drops_updated: Dict[str, int] = {}
    if run_ids_touched and not dry_run:
        for rid in run_ids_touched:
            drops_updated[rid] = db_drop_mark_published(
                con,
                run_id=rid,
                marketing_batch_id=batch_id,
                published_ts=ts,
            )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "marketing.published", batch_id),
        ts=ts,
        kind="marketing.published",
        actor="system",
        meta={
            "batch_id": batch_id,
            "count": len(published),
            "dry_run": dry_run,
            "skipped_empty": len(skipped_ids),
            "run_ids": run_ids_touched,
            "drops_updated": drops_updated,
        },
    )

    out_obj = {
        "batch_id": batch_id,
        "ts": ts,
        "count": len(published),
        "skipped_empty": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "run_ids": run_ids_touched,
        "drops_updated": drops_updated,
        "items": published,
    }
    sys.stdout.write(stable_json_dumps(out_obj) + "\n")
    return 0

# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    run_p = subparsers.add_parser("run", help="Run pipeline steps (daily, publish, manifest)")
    run_p.set_defaults(_mgc_group="run")
    run_sub = run_p.add_subparsers(dest="run_cmd", required=True)

    daily = run_sub.add_parser("daily", help="Run the daily pipeline (deterministic capable)")
    daily.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    daily.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Context/mood (focus/workout/sleep)")
    daily.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"), help="Seed for deterministic behavior")
    daily.add_argument("--out-dir", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"), help="Evidence output directory")
    daily.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (also via MGC_DETERMINISTIC=1)")
    daily.set_defaults(func=cmd_run_daily)

    pub = run_sub.add_parser("publish-marketing", help="Publish pending marketing drafts (draft -> published)")
    pub.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    pub.add_argument("--limit", type=int, default=50, help="Max number of drafts to publish")
    pub.add_argument("--dry-run", action="store_true", help="Do not update DB; just print what would publish")
    pub.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (also via MGC_DETERMINISTIC=1)")
    pub.set_defaults(func=cmd_publish_marketing)

    drop = run_sub.add_parser("drop", help="Run daily + publish-marketing and emit consolidated drop evidence")
    drop.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    drop.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Context/mood")
    drop.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"), help="Seed")
    drop.add_argument("--limit", type=int, default=50, help="Max number of marketing drafts to publish")
    drop.add_argument("--dry-run", action="store_true", help="Do not update DB in publish step")
    drop.add_argument("--out-dir", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"), help="Evidence output directory")
    drop.add_argument("--deterministic", action="store_true", help="Deterministic mode (also via MGC_DETERMINISTIC=1)")
    drop.set_defaults(func=cmd_run_drop)

    man = run_sub.add_parser("manifest", help="Compute deterministic repo manifest (stable file hashing)")
    man.add_argument("--repo-root", default=".", help="Repository root to hash")
    man.add_argument("--include", action="append", default=None, help="Optional glob(s) to include (can repeat)")
    man.add_argument("--out", default=None, help="Write manifest JSON to this path (else stdout)")
    man.add_argument("--print-hash", action="store_true", help="Print root_tree_sha256 to stderr")
    man.set_defaults(func=cmd_run_manifest)


# ---------------------------------------------------------------------------
# Standalone entry (optional)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mgc-run-cli", description="MGC run_cli helpers")
    sub = p.add_subparsers(dest="cmd", required=True)
    register_run_subcommand(sub)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    fn: Optional[Callable[[argparse.Namespace], int]] = getattr(args, "func", None)
    if not callable(fn):
        die("No command selected.")
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
