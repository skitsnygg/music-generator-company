#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import socket
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NoReturn, Optional, Sequence, Tuple


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


@contextlib.contextmanager
def _temp_env(patch: Dict[str, Optional[str]]):
    """
    Temporarily set/unset environment variables.
    patch values:
      - str: set
      - None: unset
    """
    old: Dict[str, Optional[str]] = {}
    try:
        for k, v in patch.items():
            old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
        # epoch seconds
        try:
            if fixed.isdigit():
                dt = datetime.fromtimestamp(int(fixed), tz=timezone.utc)
                return dt.isoformat()
        except Exception:
            pass
        # ISO8601
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


def _parse_iso_utc(s: str, *, fallback: str = "2020-01-01T00:00:00+00:00") -> datetime:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        dt = datetime.fromisoformat(fallback.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


def _iso_add_days(ts_iso: str, days: int) -> str:
    dt = _parse_iso_utc(ts_iso)
    return (dt + timedelta(days=int(days))).astimezone(timezone.utc).isoformat()


def _week_start_date(run_date_yyyy_mm_dd: str) -> str:
    # Monday as week start (ISO style)
    try:
        dt = datetime.strptime(run_date_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    weekday = dt.weekday()  # Monday=0
    ws = dt - timedelta(days=weekday)
    return ws.date().isoformat()


# ---------------------------------------------------------------------------
# Canonical run identity (run_key -> run_id)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunKey:
    run_date: str                 # YYYY-MM-DD
    context: str                  # focus/workout/sleep
    seed: str                     # keep string for schema tolerance
    provider_set_version: str     # e.g. "v1"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_get_or_create_run_id(
    con: sqlite3.Connection,
    *,
    run_key: RunKey,
    ts: str,
    argv: Optional[List[str]] = None,
) -> str:
    """
    Canonical rule: same (run_date, context, seed, provider_set_version) => same run_id.
    Enforced by UNIQUE INDEX in DB. Returns existing run_id or creates one.
    """
    con.execute("PRAGMA foreign_keys = ON;")

    row = con.execute(
        """
        SELECT run_id
          FROM runs
         WHERE run_date = ? AND context = ? AND seed = ? AND provider_set_version = ?
         LIMIT 1
        """,
        (run_key.run_date, run_key.context, run_key.seed, run_key.provider_set_version),
    ).fetchone()
    if row:
        return str(row[0])

    run_id = str(uuid.uuid4())
    created_at = ts if isinstance(ts, str) and ts else _now_iso_utc()
    updated_at = created_at
    argv_json = stable_json_dumps(argv) if argv is not None else None

    try:
        con.execute(
            """
            INSERT INTO runs (
              run_id, run_date, context, seed, provider_set_version,
              created_at, updated_at,
              hostname, argv_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_key.run_date,
                run_key.context,
                run_key.seed,
                run_key.provider_set_version,
                created_at,
                updated_at,
                socket.gethostname(),
                argv_json,
            ),
        )
        con.commit()
        return run_id
    except sqlite3.IntegrityError:
        # Lost a race; read the winner.
        row2 = con.execute(
            """
            SELECT run_id
              FROM runs
             WHERE run_date = ? AND context = ? AND seed = ? AND provider_set_version = ?
             LIMIT 1
            """,
            (run_key.run_date, run_key.context, run_key.seed, run_key.provider_set_version),
        ).fetchone()
        if not row2:
            raise
        return str(row2[0])


# ---------------------------------------------------------------------------
# DB helpers (schema-tolerant + NOT NULL fillers)
# ---------------------------------------------------------------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def die(msg: str, code: int = 2) -> NoReturn:
    eprint(msg)
    raise SystemExit(code)


def db_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def db_table_exists(con: sqlite3.Connection, table: str) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


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
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            run_date TEXT NOT NULL,
            context TEXT NOT NULL,
            seed TEXT NOT NULL,
            provider_set_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            hostname TEXT,
            argv_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_run_key
        ON runs(run_date, context, seed, provider_set_version)
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS run_stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT,
            duration_ms INTEGER,
            error_json TEXT,
            meta_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_run_stages_unique
        ON run_stages(run_id, stage)
        """
    )

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


# ---------------------------------------------------------------------------
# Run stage tracking (resume/observability)
# ---------------------------------------------------------------------------

def _json_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _json_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def db_stage_get(con: sqlite3.Connection, *, run_id: str, stage: str) -> Optional[sqlite3.Row]:
    cols = db_table_columns(con, "run_stages")
    if not cols:
        return None
    return con.execute(
        "SELECT * FROM run_stages WHERE run_id = ? AND stage = ? LIMIT 1",
        (run_id, stage),
    ).fetchone()


def db_stage_upsert(
    con: sqlite3.Connection,
    *,
    run_id: str,
    stage: str,
    status: str,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    duration_ms: Optional[int] = None,
    error: Optional[Dict[str, Any]] = None,
    meta_patch: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_tables_minimal(con)

    existing = db_stage_get(con, run_id=run_id, stage=stage)
    existing_meta: Dict[str, Any] = {}
    if existing is not None:
        raw = existing["meta_json"]
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    existing_meta = parsed
            except Exception:
                existing_meta = {}
    merged_meta = _json_merge(existing_meta, meta_patch or {}) if (existing_meta or meta_patch) else {}

    row_data: Dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
    }
    if started_at is not None:
        row_data["started_at"] = started_at
    if ended_at is not None:
        row_data["ended_at"] = ended_at
    if duration_ms is not None:
        row_data["duration_ms"] = int(duration_ms)
    if error is not None:
        row_data["error_json"] = stable_json_dumps(error)
    if meta_patch is not None or existing_meta:
        row_data["meta_json"] = stable_json_dumps(merged_meta)

    _insert_row(con, "run_stages", row_data)


def stage_is_done(row: Optional[sqlite3.Row]) -> bool:
    if row is None:
        return False
    s = str(row["status"] or "").lower()
    return s in ("ok", "skipped")


@contextlib.contextmanager
def run_stage(
    con: sqlite3.Connection,
    *,
    run_id: str,
    stage: str,
    deterministic: bool,
    allow_resume: bool = True,
    meta: Optional[Dict[str, Any]] = None,
):
    """
    Stage context manager:
      - If allow_resume and stage already ok/skipped: do not run body (NO DB write).
      - Otherwise: mark running, run body, then mark ok; on exception mark error.
    Determinism:
      - duration_ms is normalized to 0 in deterministic mode.
    """
    ensure_tables_minimal(con)

    existing = db_stage_get(con, run_id=run_id, stage=stage)
    if allow_resume and stage_is_done(existing):
        yield False
        return

    started_at = deterministic_now_iso(deterministic)
    t0 = time.perf_counter()

    db_stage_upsert(
        con,
        run_id=run_id,
        stage=stage,
        status="running",
        started_at=started_at,
        meta_patch={"resume": False, **(meta or {})},
    )

    try:
        yield True
    except Exception as e:
        ended_at = deterministic_now_iso(deterministic)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        if deterministic:
            dur_ms = 0
        db_stage_upsert(
            con,
            run_id=run_id,
            stage=stage,
            status="error",
            ended_at=ended_at,
            duration_ms=dur_ms,
            error={"type": type(e).__name__, "message": str(e)},
        )
        raise
    else:
        ended_at = deterministic_now_iso(deterministic)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        if deterministic:
            dur_ms = 0
        db_stage_upsert(
            con,
            run_id=run_id,
            stage=stage,
            status="ok",
            ended_at=ended_at,
            duration_ms=dur_ms,
        )


# ---------------------------------------------------------------------------
# Marketing schema-tolerant helpers
# ---------------------------------------------------------------------------

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

    content_col = _detect_marketing_content_col(cols)
    if content_col:
        v = row[content_col]
        if v is None:
            return ""
        s = str(v)
        inner = _extract_inner_content_from_blob(s)
        return inner if inner is not None else s

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
    cols = row.keys()

    meta_col = _detect_marketing_meta_col(cols)
    if meta_col:
        meta = _load_json_maybe(row[meta_col])
        if meta:
            return meta

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

    content = _marketing_row_content(con, row)
    meta = _load_json_maybe(content)
    if meta:
        return meta

    return {}


# ---------------------------------------------------------------------------
# DB writes for events/tracks/drops/marketing
# ---------------------------------------------------------------------------

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

    def _is_excluded_dir(rel_parts: Tuple[str, ...]) -> bool:
        return any(part in exclude_dirs for part in rel_parts)

    if include_globs:
        seen: set[Path] = set()
        for g in include_globs:
            for p in sorted(repo_root.glob(g)):
                if not p.is_file():
                    continue
                try:
                    rel_parts = p.relative_to(repo_root).parts
                except Exception:
                    continue
                if _is_excluded_dir(rel_parts):
                    continue
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
        if _is_excluded_dir(rel_parts):
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


def compute_manifest(
    repo_root: Path,
    *,
    include: Optional[Sequence[str]] = None,
    exclude_dirs: Optional[Sequence[str]] = None,
    exclude_globs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    entries: List[ManifestEntry] = []
    for p in iter_repo_files(repo_root, include_globs=include, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs):
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

    exclude_dirs = args.exclude_dir or None
    exclude_globs = args.exclude_glob or None

    manifest = compute_manifest(repo_root, include=include, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs)

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

def _maybe_call_external_daily_runner(
    *,
    db_path: str,
    context: str,
    seed: str,
    deterministic: bool,
    ts: str
) -> Optional[Dict[str, Any]]:
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


def _stub_daily_run(
    *,
    con: sqlite3.Connection,
    context: str,
    seed: str,
    deterministic: bool,
    ts: str,
    out_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

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

    payload = {
        "track_id": track_id,
        "run_id": run_id,
        "drop_id": drop_id,
        "context": context,
        "seed": seed,
        "ts": ts,
        "provider": provider,
    }
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

    db_insert_drop(
        con,
        drop_id=drop_id,
        ts=ts,
        context=context,
        seed=seed,
        run_id=run_id,
        track_id=track_id,
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "deterministic": deterministic,
            "seed": seed,
            "context": context,
        },
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
    provider_set_version = str(os.environ.get("MGC_PROVIDER_SET_VERSION") or "v1")

    ts = deterministic_now_iso(deterministic)
    run_date = ts.split("T", 1)[0]

    out_dir = Path(args.out_dir or "data/evidence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    run_id = db_get_or_create_run_id(
        con,
        run_key=RunKey(run_date=run_date, context=context, seed=seed, provider_set_version=provider_set_version),
        ts=ts,
        argv=list(sys.argv),
    )

    maybe = _maybe_call_external_daily_runner(
        db_path=db_path,
        context=context,
        seed=seed,
        deterministic=deterministic,
        ts=ts,
    )
    if maybe is not None:
        external_run_id = str(maybe.get("run_id") or "")
        drop_id = str(maybe.get("drop_id") or stable_uuid5("drop", run_id))
        track_id = str(maybe.get("track_id") or maybe.get("track", {}).get("id") or "")

        db_insert_drop(
            con,
            drop_id=drop_id,
            ts=ts,
            context=context,
            seed=seed,
            run_id=run_id,
            track_id=track_id if track_id else None,
            meta={
                "run_id": run_id,
                "external_run_id": external_run_id,
                "drop_id": drop_id,
                "track_id": track_id,
                "deterministic": deterministic,
                "seed": seed,
                "context": context,
                "external_runner": True,
            },
        )
        db_insert_event(
            con,
            event_id=stable_uuid5("event", "daily.external_runner", run_id),
            ts=ts,
            kind="daily.external_runner",
            actor="system",
            meta={
                "run_id": run_id,
                "external_run_id": external_run_id,
                "drop_id": drop_id,
                "module": "external",
                "deterministic": deterministic,
            },
        )
        out = dict(maybe)
        out.setdefault("run_id", run_id)
        out.setdefault("run_date", run_date)
        out.setdefault("provider_set_version", provider_set_version)
        sys.stdout.write(stable_json_dumps(out) + "\n")
        return 0

    evidence = _stub_daily_run(
        con=con,
        context=context,
        seed=seed,
        deterministic=deterministic,
        ts=ts,
        out_dir=out_dir,
        run_id=run_id,
    )
    sys.stdout.write(stable_json_dumps(evidence) + "\n")
    return 0


# ---------------------------------------------------------------------------
# Marketing publish (deterministic)
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
# Drop (daily + publish + manifest) with stages and resume semantics
# ---------------------------------------------------------------------------

def _drop_latest_for_run(con: sqlite3.Connection, run_id: str) -> Optional[sqlite3.Row]:
    cols = db_table_columns(con, "drops")
    if not cols:
        return None

    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on", "occurred_at"])
    id_col = _pick_first_existing(cols, ["id", "drop_id"]) or "id"

    if ts_col:
        sql = f"SELECT * FROM drops WHERE run_id = ? ORDER BY {ts_col} DESC, {id_col} DESC LIMIT 1"
    else:
        sql = f"SELECT * FROM drops WHERE run_id = ? ORDER BY {id_col} DESC LIMIT 1"

    return con.execute(sql, (run_id,)).fetchone()


def cmd_run_drop(args: argparse.Namespace) -> int:
    """
    Product-level command: daily + publish-marketing + manifest + consolidated evidence.

    Guarantees:
      - stdout emits exactly ONE JSON object (so `json.tool` works)
      - deterministic under --deterministic / MGC_DETERMINISTIC=1 (and MGC_FIXED_TIME)
      - stages recorded into run_stages (daily, publish_marketing, manifest, evidence)
      - resume: skips stages already ok/skipped, unless --no-resume
    """
    deterministic = is_deterministic(args)

    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    context = str(args.context or os.environ.get("MGC_CONTEXT") or "focus")
    seed = str(args.seed if args.seed is not None else (os.environ.get("MGC_SEED") or "1"))
    provider_set_version = str(os.environ.get("MGC_PROVIDER_SET_VERSION") or "v1")

    limit = int(getattr(args, "limit", None) or 50)
    dry_run = bool(getattr(args, "dry_run", False))
    allow_resume = not bool(getattr(args, "no_resume", False))

    ts = deterministic_now_iso(deterministic)
    run_date = ts.split("T", 1)[0]

    out_dir = Path(getattr(args, "out_dir", None) or os.environ.get("MGC_EVIDENCE_DIR") or "data/evidence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    run_id = db_get_or_create_run_id(
        con,
        run_key=RunKey(run_date=run_date, context=context, seed=seed, provider_set_version=provider_set_version),
        ts=ts,
        argv=list(sys.argv),
    )

    # Stage 1: daily
    with run_stage(
        con,
        run_id=run_id,
        stage="daily",
        deterministic=deterministic,
        allow_resume=allow_resume,
        meta={"context": context, "seed": seed},
    ) as should_run:
        if should_run:
            with _silence_stdout(True):
                daily_ns = argparse.Namespace(
                    db=db_path,
                    context=context,
                    seed=seed,
                    out_dir=str(out_dir),
                    deterministic=deterministic,
                )
                cmd_run_daily(daily_ns)

    # Stage 2: publish-marketing
    with run_stage(
        con,
        run_id=run_id,
        stage="publish_marketing",
        deterministic=deterministic,
        allow_resume=allow_resume,
        meta={"limit": limit, "dry_run": dry_run},
    ) as should_run:
        if should_run:
            with _silence_stdout(True):
                pub_ns = argparse.Namespace(
                    db=db_path,
                    limit=limit,
                    dry_run=dry_run,
                    deterministic=deterministic,
                )
                cmd_publish_marketing(pub_ns)

    # Resolve latest drop row for pointers
    drop_row = _drop_latest_for_run(con, run_id)
    drop_obj: Dict[str, Any] = dict(drop_row) if drop_row else {}
    drop_id = str(drop_obj.get("id") or stable_uuid5("drop", run_id))

    out_path = out_dir / ("drop_evidence.json" if deterministic else f"drop_evidence_{drop_id}.json")
    manifest_path = out_dir / ("manifest.json" if deterministic else f"manifest_{drop_id}.json")

    # Stage 3: manifest
    with run_stage(
        con,
        run_id=run_id,
        stage="manifest",
        deterministic=deterministic,
        allow_resume=allow_resume,
        meta={"repo_root": str(getattr(args, "repo_root", None) or ".")},
    ) as should_run:
        if should_run:
            with _silence_stdout(True):
                manifest_ns = argparse.Namespace(
                    repo_root=str(getattr(args, "repo_root", None) or "."),
                    out=str(manifest_path),
                    print_hash=False,
                    include=getattr(args, "include", None),
                    exclude_dir=getattr(args, "exclude_dir", None),
                    exclude_glob=getattr(args, "exclude_glob", None),
                )
                cmd_run_manifest(manifest_ns)

    manifest_sha256 = _sha256_file(manifest_path)

    # Stage 4: evidence write + persist pointers
    with run_stage(
        con,
        run_id=run_id,
        stage="evidence",
        deterministic=deterministic,
        allow_resume=allow_resume,
        meta={"evidence_path": str(out_path), "manifest_path": str(manifest_path)},
    ) as should_run:
        if should_run:
            evidence: Dict[str, Any] = {
                "ts": ts,
                "deterministic": deterministic,
                "run_key": {
                    "run_date": run_date,
                    "context": context,
                    "seed": seed,
                    "provider_set_version": provider_set_version,
                },
                "run_id": run_id,
                "drop": {
                    "id": drop_obj.get("id") or drop_id,
                    "run_id": drop_obj.get("run_id") or run_id,
                    "track_id": drop_obj.get("track_id"),
                    "marketing_batch_id": drop_obj.get("marketing_batch_id"),
                    "published_ts": drop_obj.get("published_ts"),
                },
                "paths": {
                    "evidence_path": str(out_path),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha256,
                },
                "stages": {
                    "allow_resume": allow_resume,
                },
            }

            out_path.write_text(stable_json_dumps(evidence) + "\n", encoding="utf-8", newline="\n")

            # Best-effort persist evidence pointers into drop meta (only if drop exists)
            if drop_obj.get("id"):
                existing_meta: Dict[str, Any] = {}
                meta_raw = drop_obj.get("meta_json")
                if isinstance(meta_raw, str) and meta_raw.strip():
                    try:
                        existing_meta = json.loads(meta_raw)
                        if not isinstance(existing_meta, dict):
                            existing_meta = {}
                    except Exception:
                        existing_meta = {}

                existing_meta.update(evidence["paths"])
                existing_meta.update({"run_id": run_id, "run_key": evidence["run_key"]})

                cols = db_table_columns(con, "drops")
                meta_col = _pick_first_existing(cols, ["meta_json", "metadata_json", "meta", "metadata"])
                id_col = _pick_first_existing(cols, ["id", "drop_id"]) or "id"
                if meta_col:
                    con.execute(
                        f"UPDATE drops SET {meta_col} = ? WHERE {id_col} = ?",
                        (stable_json_dumps(existing_meta), str(drop_obj.get("id"))),
                    )
                    con.commit()

    # Emit exactly one JSON object (re-read evidence file if it exists)
    try:
        sys.stdout.write(out_path.read_text(encoding="utf-8"))
    except Exception:
        sys.stdout.write(
            stable_json_dumps(
                {
                    "ts": ts,
                    "deterministic": deterministic,
                    "run_id": run_id,
                    "drop_id": drop_id,
                    "paths": {
                        "evidence_path": str(out_path),
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": manifest_sha256,
                    },
                }
            )
            + "\n"
        )
    return 0


# ---------------------------------------------------------------------------
# Weekly run (7 dailies + publish + manifest + consolidated evidence)
# ---------------------------------------------------------------------------

def cmd_run_weekly(args: argparse.Namespace) -> int:
    """
    Weekly umbrella command:
      - Runs 7 daily runs (each has its own canonical daily run_id)
      - Publishes marketing once (covers all pending drafts)
      - Writes manifest once
      - Writes weekly evidence JSON
      - Records stages under the weekly umbrella run_id:
          daily_0..daily_6, publish_marketing, manifest, evidence
      - Resume semantics per-stage unless --no-resume
    """
    deterministic = is_deterministic(args)

    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    context = str(args.context or os.environ.get("MGC_CONTEXT") or "focus")
    seed = str(args.seed if args.seed is not None else (os.environ.get("MGC_SEED") or "1"))
    provider_set_version = str(os.environ.get("MGC_PROVIDER_SET_VERSION") or "v1")

    limit = int(getattr(args, "limit", None) or 50)
    dry_run = bool(getattr(args, "dry_run", False))
    allow_resume = not bool(getattr(args, "no_resume", False))

    ts0 = deterministic_now_iso(deterministic)
    run_date0 = ts0.split("T", 1)[0]
    week_start = _week_start_date(run_date0)

    out_dir = Path(getattr(args, "out_dir", None) or os.environ.get("MGC_EVIDENCE_DIR") or "data/evidence").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    # Weekly umbrella run_id keyed by week_start date.
    weekly_run_id = db_get_or_create_run_id(
        con,
        run_key=RunKey(run_date=week_start, context=context, seed=seed, provider_set_version=provider_set_version),
        ts=ts0,
        argv=list(sys.argv),
    )

    daily_results: List[Dict[str, Any]] = []

    # Run 7 dailies as distinct calendar days, starting from week_start.
    # In deterministic mode, we force MGC_FIXED_TIME per day so cmd_run_daily produces distinct run_date/run_id.
    for i in range(7):
        stage_name = f"daily_{i}"
        day_dt = datetime.strptime(week_start, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=i)
        day_ts = day_dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

        with run_stage(
            con,
            run_id=weekly_run_id,
            stage=stage_name,
            deterministic=deterministic,
            allow_resume=allow_resume,
            meta={"context": context, "seed": seed, "day_index": i, "day_ts": day_ts, "week_start": week_start},
        ) as should_run:
            if should_run:
                with _silence_stdout(True):
                    # Ensure deterministic daily uses a distinct day.
                    patch = {"MGC_FIXED_TIME": day_ts} if deterministic else {}
                    with _temp_env(patch):
                        daily_ns = argparse.Namespace(
                            db=db_path,
                            context=context,
                            seed=seed,
                            out_dir=str(out_dir),
                            deterministic=deterministic,
                        )
                        cmd_run_daily(daily_ns)

        # Best-effort: resolve that day's canonical daily run_id and attach pointers.
        # This is deterministic because run_id is canonical for (run_date, context, seed, provider_set_version).
        day_run_id = db_get_or_create_run_id(
            con,
            run_key=RunKey(run_date=day_dt.date().isoformat(), context=context, seed=seed, provider_set_version=provider_set_version),
            ts=(day_ts if deterministic else ts0),
            argv=None,
        )
        drop_row = _drop_latest_for_run(con, day_run_id)
        drop_obj = dict(drop_row) if drop_row else {}
        daily_results.append(
            {
                "day_index": i,
                "run_date": day_dt.date().isoformat(),
                "ts": day_ts,
                "run_id": day_run_id,
                "drop": {
                    "id": drop_obj.get("id"),
                    "track_id": drop_obj.get("track_id"),
                    "marketing_batch_id": drop_obj.get("marketing_batch_id"),
                    "published_ts": drop_obj.get("published_ts"),
                },
            }
        )

    # Publish once
    with run_stage(
        con,
        run_id=weekly_run_id,
        stage="publish_marketing",
        deterministic=deterministic,
        allow_resume=allow_resume,
        meta={"limit": limit, "dry_run": dry_run},
    ) as should_run:
        if should_run:
            with _silence_stdout(True):
                pub_ns = argparse.Namespace(
                    db=db_path,
                    limit=limit,
                    dry_run=dry_run,
                    deterministic=deterministic,
                )
                cmd_publish_marketing(pub_ns)

    # Manifest once
    weekly_evidence_path = out_dir / ("weekly_evidence.json" if deterministic else f"weekly_evidence_{weekly_run_id}.json")
    weekly_manifest_path = out_dir / ("weekly_manifest.json" if deterministic else f"weekly_manifest_{weekly_run_id}.json")

    with run_stage(
        con,
        run_id=weekly_run_id,
        stage="manifest",
        deterministic=deterministic,
        allow_resume=allow_resume,
        meta={"repo_root": str(getattr(args, "repo_root", None) or ".")},
    ) as should_run:
        if should_run:
            with _silence_stdout(True):
                manifest_ns = argparse.Namespace(
                    repo_root=str(getattr(args, "repo_root", None) or "."),
                    out=str(weekly_manifest_path),
                    print_hash=False,
                    include=getattr(args, "include", None),
                    exclude_dir=getattr(args, "exclude_dir", None),
                    exclude_glob=getattr(args, "exclude_glob", None),
                )
                cmd_run_manifest(manifest_ns)

    manifest_sha256 = _sha256_file(weekly_manifest_path)

    # Evidence once
    with run_stage(
        con,
        run_id=weekly_run_id,
        stage="evidence",
        deterministic=deterministic,
        allow_resume=allow_resume,
        meta={"evidence_path": str(weekly_evidence_path), "manifest_path": str(weekly_manifest_path)},
    ) as should_run:
        if should_run:
            evidence: Dict[str, Any] = {
                "ts": ts0,
                "deterministic": deterministic,
                "run_key": {
                    "week_start": week_start,
                    "context": context,
                    "seed": seed,
                    "provider_set_version": provider_set_version,
                },
                "weekly_run_id": weekly_run_id,
                "days": daily_results,
                "paths": {
                    "evidence_path": str(weekly_evidence_path),
                    "manifest_path": str(weekly_manifest_path),
                    "manifest_sha256": manifest_sha256,
                },
                "stages": {"allow_resume": allow_resume},
            }
            weekly_evidence_path.write_text(stable_json_dumps(evidence) + "\n", encoding="utf-8", newline="\n")

    # Emit exactly one JSON object
    try:
        sys.stdout.write(weekly_evidence_path.read_text(encoding="utf-8"))
    except Exception:
        sys.stdout.write(
            stable_json_dumps(
                {
                    "ts": ts0,
                    "deterministic": deterministic,
                    "weekly_run_id": weekly_run_id,
                    "run_key": {"week_start": week_start, "context": context, "seed": seed, "provider_set_version": provider_set_version},
                    "paths": {
                        "evidence_path": str(weekly_evidence_path),
                        "manifest_path": str(weekly_manifest_path),
                        "manifest_sha256": manifest_sha256,
                    },
                }
            )
            + "\n"
        )
    return 0


# ---------------------------------------------------------------------------
# run stage CLI (set/get/list) for resume + debugging
# ---------------------------------------------------------------------------

def cmd_run_stage_set(args: argparse.Namespace) -> int:
    deterministic = is_deterministic(args)
    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    run_id = str(args.run_id).strip()
    stage = str(args.stage).strip()
    status = str(args.status).strip().lower()

    if not run_id:
        die("run_id required")
    if not stage:
        die("stage required")
    if status not in ("pending", "running", "ok", "error", "skipped"):
        die("status must be one of: pending, running, ok, error, skipped")

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    started_at = args.started_at
    ended_at = args.ended_at
    duration_ms = args.duration_ms
    if duration_ms is not None:
        duration_ms = int(duration_ms)
        if deterministic:
            duration_ms = 0

    error_obj: Optional[Dict[str, Any]] = None
    if args.error_json:
        try:
            parsed = json.loads(args.error_json)
            error_obj = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            error_obj = {"raw": args.error_json}

    meta_patch: Optional[Dict[str, Any]] = None
    if args.meta_json:
        try:
            parsed = json.loads(args.meta_json)
            meta_patch = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            meta_patch = {"raw": args.meta_json}

    if started_at is None and status in ("running", "ok", "error", "skipped"):
        started_at = deterministic_now_iso(deterministic)
    if ended_at is None and status in ("ok", "error", "skipped"):
        ended_at = deterministic_now_iso(deterministic)

    db_stage_upsert(
        con,
        run_id=run_id,
        stage=stage,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        error=error_obj,
        meta_patch=meta_patch,
    )

    sys.stdout.write(
        stable_json_dumps(
            {
                "ok": True,
                "run_id": run_id,
                "stage": stage,
                "status": status,
            }
        )
        + "\n"
    )
    return 0


def cmd_run_stage_get(args: argparse.Namespace) -> int:
    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    run_id = str(args.run_id).strip()
    stage = str(args.stage).strip()
    con = db_connect(db_path)
    ensure_tables_minimal(con)
    row = db_stage_get(con, run_id=run_id, stage=stage)
    if row is None:
        sys.stdout.write(stable_json_dumps({"found": False, "run_id": run_id, "stage": stage}) + "\n")
        return 1
    sys.stdout.write(stable_json_dumps({"found": True, "item": dict(row)}) + "\n")
    return 0


def cmd_run_stage_list(args: argparse.Namespace) -> int:
    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    run_id = str(args.run_id).strip()
    con = db_connect(db_path)
    ensure_tables_minimal(con)
    rows = con.execute(
        "SELECT * FROM run_stages WHERE run_id = ? ORDER BY id ASC",
        (run_id,),
    ).fetchall()
    sys.stdout.write(stable_json_dumps({"run_id": run_id, "count": len(rows), "items": [dict(r) for r in rows]}) + "\n")
    return 0


# ---------------------------------------------------------------------------
# run status (NEW)
# ---------------------------------------------------------------------------

def _latest_run_id(con: sqlite3.Connection) -> Optional[str]:
    if not db_table_exists(con, "runs"):
        return None
    cols = db_table_columns(con, "runs")
    if not cols:
        return None
    order_col = _pick_first_existing(cols, ["updated_at", "created_at", "run_date", "run_id"])
    if not order_col:
        order_col = "run_id"
    try:
        row = con.execute(f"SELECT run_id FROM runs ORDER BY {order_col} DESC LIMIT 1").fetchone()
        return str(row["run_id"]) if row else None
    except Exception:
        try:
            row2 = con.execute("SELECT run_id FROM runs LIMIT 1").fetchone()
            return str(row2["run_id"]) if row2 else None
        except Exception:
            return None


def _read_stages_for_run(con: sqlite3.Connection, run_id: str) -> List[Dict[str, Any]]:
    if not db_table_exists(con, "run_stages"):
        return []
    try:
        rows = con.execute(
            "SELECT * FROM run_stages WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _summary_counts(con: sqlite3.Connection) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in ("runs", "run_stages", "drops", "tracks", "marketing_posts", "events"):
        if not db_table_exists(con, t):
            continue
        try:
            out[t] = int(con.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"])
        except Exception:
            continue
    return out


def _list_runs(con: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    if not db_table_exists(con, "runs"):
        return []
    cols = db_table_columns(con, "runs")
    order_col = _pick_first_existing(cols, ["updated_at", "created_at", "run_date", "run_id"]) or "run_id"
    try:
        rows = con.execute(
            f"SELECT * FROM runs ORDER BY {order_col} DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        try:
            rows2 = con.execute("SELECT * FROM runs LIMIT ?", (int(limit),)).fetchall()
            return [dict(r) for r in rows2]
        except Exception:
            return []


def _latest_evidence_files(evidence_dir: Path, limit: int = 5) -> List[Dict[str, Any]]:
    if not evidence_dir.exists() or not evidence_dir.is_dir():
        return []
    candidates: List[Path] = []
    for p in evidence_dir.glob("*.json"):
        if p.is_file():
            candidates.append(p)
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    out: List[Dict[str, Any]] = []
    for p in candidates[: max(0, int(limit))]:
        try:
            out.append(
                {
                    "path": str(p),
                    "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "size": int(p.stat().st_size),
                }
            )
        except Exception:
            out.append({"path": str(p)})
    return out


def cmd_run_status(args: argparse.Namespace) -> int:
    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    evidence_dir = Path(os.environ.get("MGC_EVIDENCE_DIR") or "data/evidence").resolve()
    con = db_connect(db_path)

    # Don't force-create tables for status; just be tolerant.
    run_id = str(args.run_id).strip() if getattr(args, "run_id", None) else ""
    if not run_id:
        run_id = _latest_run_id(con) or ""

    fail_on_error = bool(getattr(args, "fail_on_error", False))

    counts = _summary_counts(con)
    runs = _list_runs(con, limit=int(args.limit))
    stages = _read_stages_for_run(con, run_id) if run_id else []
    evidence_files = _latest_evidence_files(evidence_dir, limit=5)

    payload: Dict[str, Any] = {
        "ok": True,
        "db": db_path,
        "run_id": run_id or None,
        "deterministic_env": {
            "MGC_DETERMINISTIC": os.environ.get("MGC_DETERMINISTIC"),
            "DETERMINISTIC": os.environ.get("DETERMINISTIC"),
            "MGC_FIXED_TIME": os.environ.get("MGC_FIXED_TIME"),
            "MGC_CONTEXT": os.environ.get("MGC_CONTEXT"),
            "MGC_SEED": os.environ.get("MGC_SEED"),
            "MGC_PROVIDER_SET_VERSION": os.environ.get("MGC_PROVIDER_SET_VERSION"),
            "MGC_EVIDENCE_DIR": os.environ.get("MGC_EVIDENCE_DIR"),
        },
        "counts": counts,
        "runs": runs,
        "stages": stages,
        "evidence_dir": str(evidence_dir),
        "recent_evidence_files": evidence_files,
    }

    if args.json:
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        return 0

    print(f"DB: {db_path}")
    print(f"Run: {run_id or '(none)'}")
    if counts:
        print("Counts:")
        for k in sorted(counts.keys()):
            print(f"  {k}: {counts[k]}")
    if stages:
        print("Stages:")
        for s in stages:
            stage = str(s.get("stage") or "")
            status = str(s.get("status") or "")
            started = str(s.get("started_at") or "")
            ended = str(s.get("ended_at") or "")
            print(f"  {stage:18s} {status:10s} {started} {ended}".rstrip())
    if evidence_files:
        print(f"Recent evidence files in {evidence_dir}:")
        for e in evidence_files:
            print(f"  {e.get('mtime','')}  {e.get('size','')}  {e.get('path','')}".rstrip())
    return 0

def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _run_id_latest(con: sqlite3.Connection) -> Optional[str]:
    # Prefer newest by updated_at/created_at if present; else by rowid.
    if not _table_exists(con, "runs"):
        return None
    cols = set(db_table_columns(con, "runs"))
    if "updated_at" in cols:
        row = con.execute("SELECT run_id FROM runs ORDER BY updated_at DESC LIMIT 1").fetchone()
    elif "created_at" in cols:
        row = con.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
    else:
        row = con.execute("SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
    return str(row[0]) if row and row[0] else None


def cmd_run_status(args: argparse.Namespace) -> int:
    """
    Prints ONE JSON object to stdout describing:
      - latest run_id (or requested run_id)
      - run_key fields if available
      - stage statuses (run_stages rows)
      - most recent drop pointers for that run_id (if drops table exists)
    """
    db_path = str(args.db or os.environ.get("MGC_DB") or "data/db.sqlite")
    con = db_connect(db_path)
    ensure_tables_minimal(con)

    run_id = (getattr(args, "run_id", None) or "").strip()
    latest = bool(getattr(args, "latest", False))

    fail_on_error = bool(getattr(args, "fail_on_error", False))

    if not run_id and latest:
        run_id = _run_id_latest(con) or ""

    if not run_id:
        # default: latest
        run_id = _run_id_latest(con) or ""

    out: Dict[str, Any] = {
        "db": db_path,
        "found": False,
        "run_id": run_id or None,
        "run": None,
        "stages": {"count": 0, "items": []},
        "drop": None,
    }

    if not run_id:
        sys.stdout.write(stable_json_dumps(out) + "\n")
        return 1

    # run row
    if _table_exists(con, "runs"):
        cols = set(db_table_columns(con, "runs"))
        if "run_id" in cols:
            row = con.execute("SELECT * FROM runs WHERE run_id = ? LIMIT 1", (run_id,)).fetchone()
            if row is not None:
                out["found"] = True
                out["run"] = dict(row)

    # stages
    if _table_exists(con, "run_stages"):
        rows = con.execute(
            "SELECT * FROM run_stages WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        out["stages"] = {"count": len(rows), "items": [dict(r) for r in rows]}

    # Summary/health
    counts: Dict[str, int] = {}
    for it in out["stages"].get("items", []):
        st = str(it.get("status") or "").lower()
        if not st:
            st = "unknown"
        counts[st] = counts.get(st, 0) + 1

    healthy = (counts.get("error", 0) == 0)
    out["summary"] = {
        "counts": dict(sorted(counts.items())),
        "healthy": healthy,
    }

    # CI mode: fail if any stage is error
    if fail_on_error:
        any_error = False
        for it in out["stages"].get("items", []):
            try:
                st = str(it.get("status") or "").lower()
            except Exception:
                st = ""
            if st == "error":
                any_error = True
                break
        if any_error:
            sys.stdout.write(stable_json_dumps(out) + "\n")
            return 2

    # latest drop for this run_id
    if _table_exists(con, "drops"):
        drop_row = _drop_latest_for_run(con, run_id)
        if drop_row is not None:
            out["drop"] = dict(drop_row)

    sys.stdout.write(stable_json_dumps(out) + "\n")
    return 0 if out["found"] else 1


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    run_p = subparsers.add_parser("run", help="Run pipeline steps (daily, publish, drop, weekly, manifest, stage, status)")
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

    drop = run_sub.add_parser("drop", help="Run daily + publish-marketing + manifest and emit consolidated evidence")
    drop.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    drop.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Context/mood")
    drop.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"), help="Seed")
    drop.add_argument("--limit", type=int, default=50, help="Max number of marketing drafts to publish")
    drop.add_argument("--dry-run", action="store_true", help="Do not update DB in publish step")
    drop.add_argument("--out-dir", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"), help="Evidence output directory")
    drop.add_argument("--repo-root", default=".", help="Repository root to hash for manifest")
    drop.add_argument("--include", action="append", default=None, help="Optional glob(s) to include in manifest (repeatable)")
    drop.add_argument("--exclude-dir", action="append", default=None, help="Directory name(s) to exclude during manifest (repeatable)")
    drop.add_argument("--exclude-glob", action="append", default=None, help="Filename glob(s) to exclude during manifest (repeatable)")
    drop.add_argument("--no-resume", action="store_true", help="Disable resume behavior; always run all stages")
    drop.add_argument("--deterministic", action="store_true", help="Deterministic mode (also via MGC_DETERMINISTIC=1)")
    drop.set_defaults(func=cmd_run_drop)

    weekly = run_sub.add_parser("weekly", help="Run 7 dailies + one publish-marketing + one manifest; emit weekly evidence")
    weekly.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    weekly.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Context/mood")
    weekly.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"), help="Seed")
    weekly.add_argument("--limit", type=int, default=50, help="Max number of marketing drafts to publish")
    weekly.add_argument("--dry-run", action="store_true", help="Do not update DB in publish step")
    weekly.add_argument("--out-dir", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"), help="Evidence output directory")
    weekly.add_argument("--repo-root", default=".", help="Repository root to hash for manifest")
    weekly.add_argument("--include", action="append", default=None, help="Optional glob(s) to include in manifest (repeatable)")
    weekly.add_argument("--exclude-dir", action="append", default=None, help="Directory name(s) to exclude during manifest (repeatable)")
    weekly.add_argument("--exclude-glob", action="append", default=None, help="Filename glob(s) to exclude during manifest (repeatable)")
    weekly.add_argument("--no-resume", action="store_true", help="Disable resume behavior; always run all stages")
    weekly.add_argument("--deterministic", action="store_true", help="Deterministic mode (also via MGC_DETERMINISTIC=1)")
    weekly.set_defaults(func=cmd_run_weekly)

    status = run_sub.add_parser("status", help="Show latest (or specific) run status + stages + drop pointers")
    status.add_argument("--fail-on-error", action="store_true", help="Exit 2 if any stage is error (for CI)")
    status.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    status.add_argument("--run-id", default=None, help="Specific run_id to inspect")
    status.add_argument("--latest", action="store_true", help="Force latest run_id (default if --run-id omitted)")
    status.add_argument("--json", action="store_true", help="(ignored) kept for CLI compatibility; output is always JSON")
    status.set_defaults(func=cmd_run_status)

    man = run_sub.add_parser("manifest", help="Compute deterministic repo manifest (stable file hashing)")
    man.add_argument("--repo-root", default=".", help="Repository root to hash")
    man.add_argument("--include", action="append", default=None, help="Optional glob(s) to include (can repeat)")
    man.add_argument("--exclude-dir", action="append", default=None, help="Directory name(s) to exclude (repeatable)")
    man.add_argument("--exclude-glob", action="append", default=None, help="Filename glob(s) to exclude (repeatable)")
    man.add_argument("--out", default=None, help="Write manifest JSON to this path (else stdout)")
    man.add_argument("--print-hash", action="store_true", help="Print root_tree_sha256 to stderr")
    man.set_defaults(func=cmd_run_manifest)

    stage = run_sub.add_parser("stage", help="Inspect or set run stage state (resume/observability)")
    stage_sub = stage.add_subparsers(dest="stage_cmd", required=True)

    st_set = stage_sub.add_parser("set", help="Upsert a stage row")
    st_set.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    st_set.add_argument("run_id", help="Canonical run_id")
    st_set.add_argument("stage", help="Stage name (e.g. daily, publish_marketing, manifest, evidence)")
    st_set.add_argument("status", help="pending|running|ok|error|skipped")
    st_set.add_argument("--started-at", default=None, help="ISO8601 timestamp")
    st_set.add_argument("--ended-at", default=None, help="ISO8601 timestamp")
    st_set.add_argument("--duration-ms", type=int, default=None, help="Duration in milliseconds (normalized to 0 in deterministic mode)")
    st_set.add_argument("--error-json", default=None, help="JSON string for error details")
    st_set.add_argument("--meta-json", default=None, help="JSON string to merge into meta_json")
    st_set.add_argument("--deterministic", action="store_true", help="Deterministic mode (also via MGC_DETERMINISTIC=1)")
    st_set.set_defaults(func=cmd_run_stage_set)

    st_get = stage_sub.add_parser("get", help="Get a stage row")
    st_get.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    st_get.add_argument("run_id", help="Canonical run_id")
    st_get.add_argument("stage", help="Stage name")
    st_get.set_defaults(func=cmd_run_stage_get)

    st_ls = stage_sub.add_parser("list", help="List all stages for a run_id")
    st_ls.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")
    st_ls.add_argument("run_id", help="Canonical run_id")
    st_ls.set_defaults(func=cmd_run_stage_list)


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
