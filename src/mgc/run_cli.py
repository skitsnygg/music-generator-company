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
import math
import wave
import struct

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NoReturn, Optional, Sequence, Tuple

from mgc.context import build_prompt, get_context_spec
from mgc.providers import GenerateRequest, get_provider


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
# Global argv helpers (fix: honor --db even when passed before "run")
# ---------------------------------------------------------------------------

def _argv_value(flag: str, argv: Optional[Sequence[str]] = None) -> Optional[str]:
    """
    Return the value after `flag` from argv if present (handles --flag=value too).
    Example:
      --db foo.sqlite
      --db=foo.sqlite
    """
    av = list(argv) if argv is not None else list(sys.argv)
    for i, tok in enumerate(av):
        if tok == flag and i + 1 < len(av):
            nxt = av[i + 1]
            if not nxt.startswith("-"):
                return nxt
            return nxt  # allow paths starting with '-' (rare, but)
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _argv_has_flag(flag: str, argv: Optional[Sequence[str]] = None) -> bool:
    av = list(argv) if argv is not None else list(sys.argv)
    return any(tok == flag or tok.startswith(flag + "=") for tok in av)


def resolve_want_json(args: argparse.Namespace) -> bool:
    """
    Global JSON flag resolver.

    We support:
      - mgc.main --json <cmd> ...
      - mgc.main <cmd> ... --json        (if subparser defines it)
      - env: MGC_JSON=1 (optional)
    """
    v = getattr(args, "json", None)
    if isinstance(v, bool):
        return v
    env = (os.environ.get("MGC_JSON") or "").strip().lower()
    return env in ("1", "true", "yes", "on")


def resolve_db_path(args: argparse.Namespace) -> str:
    """
    Global DB flag resolver.

    We support:
      - mgc.main --db PATH <cmd> ...
      - mgc.main <cmd> ... --db PATH     (if subparser defines it)
      - env: MGC_DB=... (preferred) then default data/db.sqlite
    """
    db = getattr(args, "db", None)
    if isinstance(db, str) and db.strip():
        return db.strip()

    env = (os.environ.get("MGC_DB") or "").strip()
    if env:
        return env

    return "data/db.sqlite"



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

def _write_stub_wav(path: Path, seed: int, duration_s: float = 1.5, sample_rate: int = 44100) -> None:
    """
    Write a small deterministic mono PCM16 WAV file.

    Determinism:
    - frequency derived from seed
    - fixed sample rate + duration
    - fixed amplitude
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic frequency in a pleasant range
    base = 220.0
    freq = base + float(seed % 220)  # 220..439 Hz

    nframes = int(duration_s * sample_rate)
    amp = 0.25  # keep conservative to avoid clipping
    two_pi_f = 2.0 * math.pi * freq

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        for i in range(nframes):
            t = i / sample_rate
            sample = amp * math.sin(two_pi_f * t)
            pcm = int(max(-1.0, min(1.0, sample)) * 32767.0)
            wf.writeframes(struct.pack("<h", pcm))


def _write_minimal_manifest(path: Path, fixed_ts_iso: str) -> None:
    """
    Create a minimal manifest.json if one doesn't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": "mgc.manifest.v1",
        "created_ts": fixed_ts_iso,
        "notes": "Auto-created manifest (missing at run drop time).",
    }
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")

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
    """
    Return pending marketing posts with schema drift tolerance.

    Supports schemas where the primary key is either:
      - id
      - post_id
      - marketing_post_id

    Also tolerates created_at / created_ts naming.
    """
    cols = {r["name"] for r in con.execute("PRAGMA table_info(marketing_posts)").fetchall()}

    # pick PK column
    if "id" in cols:
        pk = "id"
    elif "post_id" in cols:
        pk = "post_id"
    elif "marketing_post_id" in cols:
        pk = "marketing_post_id"
    else:
        # fall back: pick the first column that looks like an id
        pk_candidates = [c for c in cols if c.endswith("_id")]
        pk = pk_candidates[0] if pk_candidates else "rowid"

    # pick created timestamp column for ordering
    if "created_at" in cols:
        created_col = "created_at"
    elif "created_ts" in cols:
        created_col = "created_ts"
    elif "ts" in cols:
        created_col = "ts"
    else:
        created_col = pk  # last resort ordering

    # status column name drift
    status_col = "status" if "status" in cols else ("state" if "state" in cols else None)
    if status_col is None:
        # can't filter drafts; return empty rather than crashing CI
        return []

    # select everything, but ensure deterministic ordering and stable limit
    sql = f"""
    SELECT *
    FROM marketing_posts
    WHERE {status_col} = ?
    ORDER BY {created_col} ASC, {pk} ASC
    LIMIT ?
    """

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
    """
    Daily run implementation using provider abstraction.

    Produces:
      - provider artifact under data/tracks/... (repo storage)
      - bundled copy under out_dir/tracks/<track_id>.<ext> (portable drop artifact)
      - out_dir/playlist.json pointing at bundled track (web-build friendly)
      - out_dir/daily_evidence*.json includes bundle paths + sha256

    Supports:
      - stub (deterministic placeholder)
      - riffusion (local server)
      - staged providers (suno, diffsinger)
    """
    import hashlib
    import os
    import shutil

    def _sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _posix(p: Path) -> str:
        return str(p).replace("\\", "/")

    out_dir.mkdir(parents=True, exist_ok=True)

    drop_id = stable_uuid5("drop", run_id)
    track_id = stable_uuid5("track", context, seed, run_id)

    title = f"{context.title()} Track {seed}"

    # Artifact path in repo storage (provider may change extension)
    if deterministic:
        artifact_rel = Path("data") / "tracks" / f"{track_id}"
    else:
        day = ts.split("T", 1)[0]
        artifact_rel = Path("data") / "tracks" / day / f"{track_id}"

    artifact_path = (Path.cwd() / artifact_rel).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    # Build prompt from context
    prompt = build_prompt(context)

    # Pick provider
    provider_name = str(os.environ.get("MGC_PROVIDER") or "stub").strip().lower()
    provider = get_provider(provider_name)

    # Generate
    result = provider.generate(
        GenerateRequest(
            track_id=track_id,
            run_id=run_id,
            context=context,
            seed=seed,
            prompt=prompt,
            deterministic=deterministic,
            ts=ts,
            out_rel=str(artifact_path),
        )
    )

    # Ensure extension matches provider output
    if result.ext:
        ext = result.ext if result.ext.startswith(".") else f".{result.ext}"
        if artifact_path.suffix != ext:
            artifact_rel = artifact_rel.with_suffix(ext)
            artifact_path = (Path.cwd() / artifact_rel).resolve()
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

    # Write provider artifact bytes (repo storage)
    artifact_bytes = result.artifact_bytes or b""
    artifact_path.write_bytes(artifact_bytes)

    # Insert track
    db_insert_track(
        con,
        track_id=track_id,
        ts=ts,
        title=title,
        provider=result.provider,
        mood=context,
        genre=result.meta.get("genre") if isinstance(result.meta, dict) else None,
        artifact_path=_posix(artifact_rel),
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "context": context,
            "seed": seed,
            "deterministic": deterministic,
            **(result.meta or {}),
        },
    )

    # Insert drop
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
            "provider": result.provider,
            "deterministic": deterministic,
            "seed": seed,
            "context": context,
        },
    )

    # Events
    db_insert_event(
        con,
        event_id=stable_uuid5("event", "drop.created", drop_id),
        ts=ts,
        kind="drop.created",
        actor="system",
        meta={"drop_id": drop_id, "run_id": run_id, "track_id": track_id},
    )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "track.generated", run_id),
        ts=ts,
        kind="track.generated",
        actor="system",
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "artifact_path": _posix(artifact_rel),
            "provider": result.provider,
        },
    )

    # Marketing drafts
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

        db_insert_marketing_post(
            con,
            post_id=post_id,
            ts=ts,
            platform=platform,
            status="draft",
            content=stable_json_dumps(content_obj),
            meta={
                "run_id": run_id,
                "drop_id": drop_id,
                "track_id": track_id,
                "provider": result.provider,
                "context": context,
            },
        )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "marketing.drafts.created", run_id),
        ts=ts,
        kind="marketing.drafts.created",
        actor="system",
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "count": len(post_ids),
            "post_ids": post_ids,
        },
    )

    # ---------------------------------------------------------------------
    # Portable bundle outputs (tracks + playlist.json)
    # ---------------------------------------------------------------------
    bundle_tracks_dir = out_dir / "tracks"
    bundle_tracks_dir.mkdir(parents=True, exist_ok=True)

    # Bundle uses track_id + provider extension (same as repo artifact)
    bundled_name = f"{track_id}{artifact_path.suffix}"
    bundled_track_rel = Path("tracks") / bundled_name
    bundled_track_path = (out_dir / bundled_track_rel).resolve()

    # Copy repo artifact into bundle (portable)
    shutil.copy2(str(artifact_path), str(bundled_track_path))

    # Write playlist.json pointing at bundled track (web-build friendly)
    playlist_rel = Path("playlist.json")
    playlist_path = (out_dir / playlist_rel).resolve()

    playlist_obj: Dict[str, Any] = {
        "schema": "mgc.playlist.v1",
        "context": context,
        "ts": ts,
        "drop_id": drop_id,
        "run_id": run_id,
        "deterministic": deterministic,
        "tracks": [
            {
                "track_id": track_id,
                "title": title,
                "provider": result.provider,
                "path": _posix(bundled_track_rel),
            }
        ],
    }
    playlist_path.write_text(stable_json_dumps(playlist_obj), encoding="utf-8")

    # Hashes
    repo_artifact_sha256 = _sha256_file(artifact_path)
    bundled_track_sha256 = _sha256_file(bundled_track_path)
    playlist_sha256 = _sha256_file(playlist_path)

    # Evidence JSON (write a stable filename + an id-scoped filename)
    evidence_obj: Dict[str, Any] = {
        "schema": "mgc.daily_evidence.v1",
        "ts": ts,
        "context": context,
        "seed": seed,
        "deterministic": deterministic,
        "provider": result.provider,
        "ids": {
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "marketing_post_ids": post_ids,
        },
        "paths": {
            "repo_artifact": _posix(artifact_rel),
            "bundle_track": _posix(bundled_track_rel),
            "playlist": _posix(playlist_rel),
        },
        "sha256": {
            "repo_artifact": repo_artifact_sha256,
            "bundle_track": bundled_track_sha256,
            "playlist": playlist_sha256,
        },
    }

    evidence_rel_main = Path("daily_evidence.json")
    evidence_rel_scoped = Path(f"daily_evidence_{drop_id}.json")
    evidence_path_main = (out_dir / evidence_rel_main).resolve()
    evidence_path_scoped = (out_dir / evidence_rel_scoped).resolve()

    evidence_text = stable_json_dumps(evidence_obj)
    evidence_path_main.write_text(evidence_text, encoding="utf-8")
    evidence_path_scoped.write_text(evidence_text, encoding="utf-8")

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "drop.bundle.written", drop_id),
        ts=ts,
        kind="drop.bundle.written",
        actor="system",
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "bundle_track": _posix(bundled_track_rel),
            "playlist": _posix(playlist_rel),
            "evidence": [_posix(evidence_rel_main), _posix(evidence_rel_scoped)],
        },
    )

    return {
        "run_id": run_id,
        "drop_id": drop_id,
        "track_id": track_id,
        "provider": result.provider,
        "context": context,
        "seed": seed,
        "deterministic": deterministic,
        "repo_artifact_path": _posix(artifact_rel),
        "bundle_track_path": _posix(bundled_track_rel),
        "playlist_path": _posix(playlist_rel),
        "evidence_paths": [_posix(evidence_rel_main), _posix(evidence_rel_scoped)],
        "sha256": {
            "repo_artifact": repo_artifact_sha256,
            "bundle_track": bundled_track_sha256,
            "playlist": playlist_sha256,
        },
        "marketing_post_ids": post_ids,
    }

    # ------------------------------------------------------------------
    # Bundle materialization in out_dir
    # ------------------------------------------------------------------
    bundle_tracks_dir = out_dir / "tracks"
    bundle_tracks_dir.mkdir(parents=True, exist_ok=True)

    bundle_track_path = bundle_tracks_dir / f"{track_id}{artifact_path.suffix}"
    shutil.copy2(artifact_path, bundle_track_path)

    bundle_track_sha256 = _sha256_file(bundle_track_path)
    repo_track_sha256 = _sha256_file(artifact_path)

    # Playlist for web build: include MANY common field names (absolute + relative)
    playlist_path = out_dir / "playlist.json"
    rel_bundle_track = str(bundle_track_path.relative_to(out_dir)).replace("\\", "/")

    track_obj = {
        "id": track_id,
        "title": title,

        # Absolute (web build can always resolve)
        "artifact_path": str(bundle_track_path),
        "audio_path": str(bundle_track_path),

        # Relative (portable; relative to out_dir / playlist directory)
        "artifact_rel": rel_bundle_track,
        "audio_rel": rel_bundle_track,

        # Keep existing common keys too
        "path": rel_bundle_track,
        "file": rel_bundle_track,
        "src": rel_bundle_track,
        "abs_path": str(bundle_track_path),

        # Debug / provenance
        "repo_artifact_path": str(artifact_path),
    }

    playlist_obj = {
        "schema": "mgc.playlist.v1",
        "id": f"drop-{str(track_id)[:8]}",
        "name": f"{context}_drop_{seed}",
        "created_ts": ts,
        "context": context,
        "seed": seed,
        "track_count": 1,
        "tracks": [track_obj],
    }

    playlist_path.write_text(
        json.dumps(playlist_obj, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    playlist_sha256 = _sha256_file(playlist_path)

    # Evidence object
    evidence = {
        "run_id": run_id,
        "drop_id": drop_id,
        "ts": ts,
        "deterministic": deterministic,
        "context": context,
        "seed": seed,
        "provider": result.provider,
        "track": {
            "id": track_id,
            "title": title,
            "artifact_path": str(artifact_rel).replace("\\", "/"),
            "size": len(artifact_bytes),
            "mime": result.mime,
            "repo_artifact_sha256": repo_track_sha256,
        },
        "bundle": {
            "out_dir": str(out_dir),
            "tracks_dir": str(bundle_tracks_dir),
            "track_path": str(bundle_track_path),
            "track_rel": rel_bundle_track,
            "track_sha256": bundle_track_sha256,
            "playlist_path": str(playlist_path),
            "playlist_sha256": playlist_sha256,
        },
        "marketing_drafts": [{"id": pid, "platform": p} for pid, p in zip(post_ids, platforms)],
    }

    evidence_path = out_dir / ("daily_evidence.json" if deterministic else f"daily_evidence_{run_id}.json")
    evidence_path.write_text(stable_json_dumps(evidence) + "\n", encoding="utf-8")

    return evidence

def cmd_run_daily(args: argparse.Namespace) -> int:
    deterministic = is_deterministic(args)

    db_path = resolve_db_path(args)
    context = str(getattr(args, "context", None) or os.environ.get("MGC_CONTEXT") or "focus")
    seed = str(getattr(args, "seed", None) if getattr(args, "seed", None) is not None else (os.environ.get("MGC_SEED") or "1"))
    provider_set_version = str(os.environ.get("MGC_PROVIDER_SET_VERSION") or "v1")

    ts = deterministic_now_iso(deterministic)
    run_date = ts.split("T", 1)[0]

    out_dir = Path(getattr(args, "out_dir", None) or "data/evidence").resolve()
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

    db_path = resolve_db_path(args)
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
    Produce one deterministic "drop" bundle under --out-dir.

    This command MUST:
      - ensure manifest.json exists under out_dir
      - run the daily generator (stub/riffusion/...) to materialize audio + DB rows
      - write drop_evidence.json under out_dir, including bundle track + playlist paths
      - build a submission zip at data/submissions/<drop_id>/submission.zip (non-dry-run)
      - write data/submissions/<drop_id>/submission.json (self-describing pointer)
      - print a small JSON summary to stdout
    """
    import os
    import sqlite3
    import tempfile
    import zipfile
    import shutil
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    from typing import Optional

    from mgc.bundle_validate import validate_bundle

    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _safe_mkdir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def _posix_rel(p: Path) -> str:
        return str(p).replace("\\", "/")

    def _zip_add_dir(zf: zipfile.ZipFile, root: Path, arc_root: str) -> None:
        """
        Add directory contents to zip deterministically:
          - stable traversal order (sorted dirs/files)
          - stable archive ordering (sorted arcname)
          - fixed timestamps in ZipInfo for deterministic builds
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
        entries.sort(key=lambda x: x[0])

        fixed_dt = (2020, 1, 1, 0, 0, 0)
        for arcname, file_path in entries:
            data = file_path.read_bytes()
            zi = zipfile.ZipInfo(filename=arcname, date_time=fixed_dt)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, data)

    def _build_readme(evidence_obj: dict) -> str:
        # Use run ts (not current time) so deterministic runs stay deterministic.
        ids = evidence_obj.get("ids") if isinstance(evidence_obj.get("ids"), dict) else {}
        paths = evidence_obj.get("paths") if isinstance(evidence_obj.get("paths"), dict) else {}

        drop_id = ids.get("drop_id", evidence_obj.get("drop_id", ""))
        run_id_local = ids.get("run_id", evidence_obj.get("run_id", ""))
        track_id_local = ids.get("track_id", evidence_obj.get("track_id", ""))
        provider = evidence_obj.get("provider", "")
        context_local = evidence_obj.get("context", evidence_obj.get("context", ""))
        ts_local = evidence_obj.get("ts", "")

        bundle_track = paths.get("bundle_track", paths.get("bundle_track_path", ""))
        playlist = paths.get("playlist", paths.get("playlist_path", "playlist.json"))
        deterministic_local = evidence_obj.get("deterministic", "")

        lines = [
            "# Music Generator Company – Drop Submission",
            "",
            "## Identifiers",
            f"- drop_id: {drop_id}",
            f"- run_id: {run_id_local}",
            f"- track_id: {track_id_local}",
            "",
            "## Run metadata",
            f"- ts: {ts_local}",
            f"- context: {context_local}",
            f"- provider: {provider}",
            f"- deterministic: {deterministic_local}",
            "",
            "## Contents",
            f"- {playlist}: playlist pointing at bundled audio",
            f"- {bundle_track}: bundled audio asset",
            "- daily_evidence.json (or drop bundle evidence): provenance + sha256 hashes",
            "",
            "## How to review",
            "1) Inspect playlist.json (it references the bundled track under tracks/).",
            "2) Confirm hashes match the files in the bundle.",
            "",
            "## Notes",
            f"- Packaged at: {ts_local or _utc_now_iso()}",
        ]
        return "\n".join(lines) + "\n"

    deterministic = bool(getattr(args, "deterministic", False)) or bool(
        (os.environ.get("MGC_DETERMINISTIC") or "").strip().lower() in ("1", "true", "yes")
    )

    context = str(getattr(args, "context", None) or "focus")
    seed = str(getattr(args, "seed", None) or "1")

    allow_resume = not bool(getattr(args, "no_resume", False))
    dry_run = bool(getattr(args, "dry_run", False))

    ts = deterministic_now_iso(deterministic)
    run_date = ts.split("T", 1)[0]

    out_dir = Path(
        getattr(args, "out_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR")
        or "data/evidence"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # DB path
    db_path = (
        getattr(args, "db", None)
        or os.environ.get("MGC_DB")
        or "data/db.sqlite"
    )
    db_path = str(Path(db_path).expanduser())

    # Run id for this drop
    run_id = stable_uuid5("run", "drop", context, seed, run_date, "det" if deterministic else "live")

    # ---------------------------------------------------------------------
    # Ensure manifest exists in out_dir (cmd_run_drop hashes this)
    # ---------------------------------------------------------------------
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        _write_minimal_manifest(manifest_path, ts)
    manifest_sha256 = _sha256_file(manifest_path)

    # ---------------------------------------------------------------------
    # Run the generator path (this is what actually creates audio + DB rows)
    # ---------------------------------------------------------------------
    if dry_run:
        drop_id = stable_uuid5("drop", run_id)
        evidence_obj = {
            "schema": "mgc.drop_evidence.v1",
            "ts": ts,
            "deterministic": deterministic,
            "run_id": run_id,
            "drop_id": drop_id,
            "context": context,
            "seed": seed,
            "stages": {"allow_resume": bool(allow_resume)},
            "paths": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
            },
            "note": "dry_run=true; generator not executed",
        }
    else:
        con = sqlite3.connect(db_path)
        try:
            con.execute("PRAGMA foreign_keys = ON")
            daily = _stub_daily_run(
                con=con,
                context=context,
                seed=seed,
                deterministic=deterministic,
                ts=ts,
                out_dir=out_dir,
                run_id=run_id,
            )
            con.commit()
        finally:
            con.close()

        drop_id = str(daily.get("drop_id") or stable_uuid5("drop", run_id))

        track_id = ""
        try:
            track_id = str((daily.get("track") or {}).get("id") or "")
        except Exception:
            track_id = ""

        evidence_obj = {
            "schema": "mgc.drop_evidence.v1",
            "ts": ts,
            "deterministic": deterministic,
            "run_id": run_id,
            "drop_id": drop_id,
            "context": context,
            "seed": seed,
            "stages": {"allow_resume": bool(allow_resume)},
            "paths": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
            },
            "daily": daily,
        }

        bundle = (daily or {}).get("bundle") if isinstance(daily, dict) else None
        if isinstance(bundle, dict):
            evidence_obj["paths"].update(
                {
                    "bundle_dir": str(bundle.get("out_dir") or str(out_dir)),
                    "bundle_track_path": str(bundle.get("track_path") or ""),
                    "bundle_track_sha256": str(bundle.get("track_sha256") or ""),
                    "playlist_path": str(bundle.get("playlist_path") or (out_dir / "playlist.json")),
                    "playlist_sha256": str(bundle.get("playlist_sha256") or ""),
                }
            )

        if track_id:
            evidence_obj["track_id"] = track_id

    # ---------------------------------------------------------------------
    # Write drop evidence in out_dir (v1)
    # ---------------------------------------------------------------------
    evidence_path = out_dir / "drop_evidence.json"
    evidence_path.write_text(stable_json_dumps(evidence_obj) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------------
    # Build submission zip + submission.json (non-dry-run only)
    # ---------------------------------------------------------------------
    submission_zip_path: Optional[Path] = None
    submission_json_path: Optional[Path] = None

    if not dry_run:
        pths = evidence_obj.get("paths") or {}
        bundle_dir = Path(str(pths.get("bundle_dir") or out_dir)).expanduser().resolve()

        # Validate bundle before packaging
        validate_bundle(bundle_dir)

        # Read daily evidence object if present (for README). Prefer daily_evidence.json if it exists.
        daily_ev_path = bundle_dir / "daily_evidence.json"
        if daily_ev_path.exists():
            try:
                daily_ev_obj = json.loads(daily_ev_path.read_text(encoding="utf-8"))
            except Exception:
                daily_ev_obj = {
                    "ts": ts,
                    "drop_id": evidence_obj.get("drop_id"),
                    "run_id": run_id,
                    "context": context,
                    "deterministic": deterministic,
                }
        else:
            daily_ev_obj = {
                "ts": ts,
                "drop_id": evidence_obj.get("drop_id"),
                "run_id": run_id,
                "context": context,
                "deterministic": deterministic,
            }

        drop_id_for_paths = str(evidence_obj.get("drop_id") or stable_uuid5("drop", run_id))
        submissions_root = (Path("data") / "submissions" / drop_id_for_paths).resolve()
        _safe_mkdir(submissions_root)

        submission_zip_path = (submissions_root / "submission.zip").resolve()
        submission_json_path = (submissions_root / "submission.json").resolve()

        # Stage + zip deterministically
        with tempfile.TemporaryDirectory(prefix="mgc_submission_") as td:
            stage = Path(td).resolve()
            pkg_root = stage / "submission"
            _safe_mkdir(pkg_root)

            # Copy bundle
            drop_bundle_dst = pkg_root / "drop_bundle"
            shutil.copytree(bundle_dir, drop_bundle_dst)

            # Write README.md (stable-ish)
            (pkg_root / "README.md").write_text(_build_readme(daily_ev_obj), encoding="utf-8")

            # Write zip (deterministic ordering + timestamps)
            if submission_zip_path.exists():
                submission_zip_path.unlink()

            with zipfile.ZipFile(str(submission_zip_path), mode="w") as zf:
                fixed_dt = (2020, 1, 1, 0, 0, 0)
                zi = zipfile.ZipInfo(filename="submission/README.md", date_time=fixed_dt)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = 0o644 << 16
                zf.writestr(zi, (pkg_root / "README.md").read_text(encoding="utf-8"))
                _zip_add_dir(zf, drop_bundle_dst, arc_root="submission/drop_bundle")

        # Write submission.json (self-describing pointer file; stable fields)
        submission_obj = {
            "schema": "mgc.submission.v1",
            "drop_id": drop_id_for_paths,
            "run_id": str(run_id),
            "deterministic": bool(deterministic),
            "ts": ts,  # use run ts for determinism
            "submission_zip": "submission.zip",  # relative within submissions_root
        }
        submission_json_path.write_text(stable_json_dumps(submission_obj) + "\n", encoding="utf-8")

        # Record in evidence + rewrite drop_evidence.json so pointer is included
        evidence_obj.setdefault("paths", {})
        evidence_obj["paths"]["submission_dir"] = str(submissions_root)
        evidence_obj["paths"]["submission_zip"] = str(submission_zip_path)
        evidence_obj["paths"]["submission_json"] = str(submission_json_path)
        evidence_path.write_text(stable_json_dumps(evidence_obj) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------------
    # Print stdout summary JSON
    # ---------------------------------------------------------------------
    out = {
        "deterministic": bool(deterministic),
        "drop_id": str(evidence_obj.get("drop_id") or stable_uuid5("drop", run_id)),
        "run_id": run_id,
        "ts": ts,
        "paths": {
            "evidence_path": str(evidence_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
        },
    }

    # bubble up useful bundle paths when available
    try:
        p = evidence_obj.get("paths") or {}
        for k in ("bundle_dir", "bundle_track_path", "bundle_track_sha256", "playlist_path", "playlist_sha256"):
            if k in p and p[k]:
                out["paths"][k] = p[k]
        if submission_zip_path is not None:
            out["paths"]["submission_zip"] = str(submission_zip_path)
        if submission_json_path is not None:
            out["paths"]["submission_json"] = str(submission_json_path)
    except Exception:
        pass

    print(stable_json_dumps(out))
    return 0

def cmd_run_tail(args: argparse.Namespace) -> int:
    evidence_dir = Path(
        getattr(args, "out_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR")
        or "data/evidence"
    ).resolve()

    kind = str(getattr(args, "type", "any") or "any")  # drop | weekly | any
    n = int(getattr(args, "n", 1) or 1)

    if not evidence_dir.exists() or not evidence_dir.is_dir():
        sys.stdout.write(
            stable_json_dumps(
                {
                    "found": False,
                    "reason": "evidence_dir_missing",
                    "path": str(evidence_dir),
                }
            )
            + "\n"
        )
        return 0

    if kind == "drop":
        patterns = ["drop_evidence*.json"]
    elif kind == "weekly":
        patterns = ["weekly_evidence*.json"]
    else:
        patterns = ["drop_evidence*.json", "weekly_evidence*.json"]

    files: List[Path] = []
    for pat in patterns:
        files.extend(evidence_dir.glob(pat))

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        sys.stdout.write(
            stable_json_dumps(
                {
                    "found": False,
                    "reason": "no_evidence_files",
                    "path": str(evidence_dir),
                }
            )
            + "\n"
        )
        return 0

    selected = files[:n]
    items: List[Dict[str, Any]] = []

    for p in selected:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            payload = {"error": str(e)}

        items.append(
            {
                "file": str(p),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                "payload": payload,
            }
        )

    sys.stdout.write(
        stable_json_dumps(
            {
                "found": True,
                "count": len(items),
                "items": items,
            }
        )
        + "\n"
    )
    return 0

# ---------------------------------------------------------------------------
# Weekly run (7 dailies + publish + manifest + consolidated evidence)
# ---------------------------------------------------------------------------

def cmd_run_weekly(args: argparse.Namespace) -> int:
    deterministic = is_deterministic(args)

    db_path = resolve_db_path(args)
    context = str(getattr(args, "context", None) or os.environ.get("MGC_CONTEXT") or "focus")
    seed = str(getattr(args, "seed", None) if getattr(args, "seed", None) is not None else (os.environ.get("MGC_SEED") or "1"))
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

        day_run_id = db_get_or_create_run_id(
            con,
            run_key=RunKey(
                run_date=day_dt.date().isoformat(),
                context=context,
                seed=seed,
                provider_set_version=provider_set_version,
            ),
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

    db_path = args.db
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

    started_at = getattr(args, "started_at", None)
    ended_at = getattr(args, "ended_at", None)
    duration_ms = getattr(args, "duration_ms", None)

    # Normalize duration in deterministic mode
    if duration_ms is not None:
        try:
            duration_ms = int(duration_ms)
        except Exception:
            die("duration_ms must be an integer")
        if deterministic:
            duration_ms = 0

    error_obj: Optional[Dict[str, Any]] = None
    raw_error = getattr(args, "error_json", None)
    if raw_error:
        try:
            parsed = json.loads(raw_error)
            error_obj = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            error_obj = {"raw": str(raw_error)}

    meta_patch: Optional[Dict[str, Any]] = None
    raw_meta = getattr(args, "meta_json", None)
    if raw_meta:
        try:
            parsed = json.loads(raw_meta)
            meta_patch = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            meta_patch = {"raw": str(raw_meta)}

    # Auto-fill timestamps if omitted (keeps UX sane + DB consistent)
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
                "db": db_path,
                "run_id": run_id,
                "stage": stage,
                "status": status,
            }
        )
        + "\n"
    )
    return 0


def cmd_run_stage_get(args: argparse.Namespace) -> int:
    db_path = args.db
    run_id = str(args.run_id).strip()
    stage = str(args.stage).strip()

    if not run_id:
        die("run_id required")
    if not stage:
        die("stage required")

    con = db_connect(db_path)

    # Be tolerant: do not create tables on read-only get; just report not found.
    if not db_table_exists(con, "run_stages"):
        sys.stdout.write(
            stable_json_dumps(
                {
                    "found": False,
                    "db": db_path,
                    "run_id": run_id,
                    "stage": stage,
                    "reason": "run_stages_table_missing",
                }
            )
            + "\n"
        )
        return 1

    row = db_stage_get(con, run_id=run_id, stage=stage)
    if row is None:
        sys.stdout.write(
            stable_json_dumps(
                {
                    "found": False,
                    "db": db_path,
                    "run_id": run_id,
                    "stage": stage,
                }
            )
            + "\n"
        )
        return 1

    sys.stdout.write(
        stable_json_dumps(
            {
                "found": True,
                "db": db_path,
                "run_id": run_id,
                "stage": stage,
                "item": dict(row),
            }
        )
        + "\n"
    )
    return 0


def cmd_run_stage_list(args: argparse.Namespace) -> int:
    db_path = args.db
    run_id = str(args.run_id).strip()

    if not run_id:
        die("run_id required")

    con = db_connect(db_path)

    # Be tolerant: do not create tables on list; just return empty.
    if not db_table_exists(con, "run_stages"):
        sys.stdout.write(
            stable_json_dumps(
                {
                    "db": db_path,
                    "run_id": run_id,
                    "count": 0,
                    "items": [],
                    "reason": "run_stages_table_missing",
                }
            )
            + "\n"
        )
        return 0

    try:
        rows = con.execute(
            "SELECT * FROM run_stages WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
    except sqlite3.Error as e:
        sys.stdout.write(
            stable_json_dumps(
                {
                    "db": db_path,
                    "run_id": run_id,
                    "count": 0,
                    "items": [],
                    "error": {"type": "sqlite3.Error", "message": str(e)},
                }
            )
            + "\n"
        )
        return 1

    sys.stdout.write(
        stable_json_dumps(
            {
                "db": db_path,
                "run_id": run_id,
                "count": len(rows),
                "items": [dict(r) for r in rows],
            }
        )
        + "\n"
    )
    return 0

# ---------------------------------------------------------------------------
# run status
# ---------------------------------------------------------------------------

def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _run_id_latest(con: sqlite3.Connection) -> Optional[str]:
    if not _table_exists(con, "runs"):
        return None
    cols = set(db_table_columns(con, "runs"))
    try:
        if "updated_at" in cols:
            row = con.execute("SELECT run_id FROM runs ORDER BY updated_at DESC LIMIT 1").fetchone()
        elif "created_at" in cols:
            row = con.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        elif "run_date" in cols:
            row = con.execute("SELECT run_id FROM runs ORDER BY run_date DESC LIMIT 1").fetchone()
        else:
            row = con.execute("SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def cmd_run_status(args: argparse.Namespace) -> int:
    """
    Output rules:
      - If global --json is set, emit exactly ONE JSON object.
      - Otherwise emit a short human summary (single line).
    """
    want_json = bool(getattr(args, "json", False))

    db_path = str(getattr(args, "db", None) or os.environ.get("MGC_DB") or "data/db.sqlite")
    con = db_connect(db_path)

    run_id = str(getattr(args, "run_id", None) or "").strip()
    latest = bool(getattr(args, "latest", False))
    fail_on_error = bool(getattr(args, "fail_on_error", False))

    if not run_id and latest:
        run_id = _run_id_latest(con) or ""
    if not run_id:
        run_id = _run_id_latest(con) or ""

    out: Dict[str, Any] = {
        "db": db_path,
        "found": False,
        "run_id": run_id or None,
        "run": None,
        "stages": {"count": 0, "items": []},
        "drop": None,
        "summary": {"counts": {}, "healthy": None},
    }

    # run row
    if run_id and db_table_exists(con, "runs"):
        try:
            row = con.execute("SELECT * FROM runs WHERE run_id = ? LIMIT 1", (run_id,)).fetchone()
            if row is not None:
                out["found"] = True
                out["run"] = dict(row)
        except Exception:
            pass

    # stages
    stage_items: List[Dict[str, Any]] = []
    if run_id and db_table_exists(con, "run_stages"):
        try:
            rows = con.execute(
                "SELECT * FROM run_stages WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
            stage_items = [dict(r) for r in rows]
        except Exception:
            stage_items = []

    out["stages"] = {"count": len(stage_items), "items": stage_items}

    counts: Dict[str, int] = {}
    for it in stage_items:
        st = str(it.get("status") or "").strip().lower() or "unknown"
        counts[st] = counts.get(st, 0) + 1

    healthy = (counts.get("error", 0) == 0)
    out["summary"] = {"counts": dict(sorted(counts.items())), "healthy": healthy}

    # drop pointer
    if run_id and db_table_exists(con, "drops"):
        try:
            drow = _drop_latest_for_run(con, run_id)
            if drow is not None:
                out["drop"] = dict(drow)
        except Exception:
            pass

    # CI fail mode
    if fail_on_error and counts.get("error", 0) > 0:
        if want_json:
            sys.stdout.write(stable_json_dumps(out) + "\n")
        else:
            print(f"run_id={run_id or '(none)'} status=ERROR stages_error={counts.get('error', 0)} db={db_path}")
        return 2

    if want_json:
        sys.stdout.write(stable_json_dumps(out) + "\n")
        return 0 if out["found"] else 1

    # Human single line
    rid = run_id or "(none)"
    err = counts.get("error", 0)
    ok = counts.get("ok", 0)
    running = counts.get("running", 0)
    skipped = counts.get("skipped", 0)

    status = "OK" if err == 0 else "ERROR"
    print(f"run_id={rid} status={status} ok={ok} running={running} skipped={skipped} error={err} db={db_path}")
    return 0 if err == 0 else 2

def cmd_run_open(args: argparse.Namespace) -> int:
    evidence_dir = Path(
        getattr(args, "out_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR")
        or "data/evidence"
    ).resolve()

    kind = str(getattr(args, "type", "any") or "any")  # drop | weekly | any
    n = int(getattr(args, "n", 1) or 1)

    if not evidence_dir.exists() or not evidence_dir.is_dir():
        return 1

    if kind == "drop":
        patterns = ["drop_evidence*.json"]
    elif kind == "weekly":
        patterns = ["weekly_evidence*.json"]
    else:
        patterns = ["drop_evidence*.json", "weekly_evidence*.json"]

    files: List[Path] = []
    for pat in patterns:
        files.extend(evidence_dir.glob(pat))

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return 1

    selected = files[:n]
    for p in selected:
        print(str(p))
        # Also print referenced manifest if present & exists
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                paths = payload.get("paths", {})
                if isinstance(paths, dict):
                    manifest_path = paths.get("manifest_path")
                    if isinstance(manifest_path, str) and manifest_path.strip():
                        mp = Path(manifest_path)
                        if not mp.is_absolute():
                            mp = (evidence_dir / mp).resolve()
                        if mp.exists():
                            print(str(mp))
        except Exception:
            pass

    return 0

def _load_manifest(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    entries = obj.get("entries", []) if isinstance(obj, dict) else []
    by_path = {}
    for e in entries:
        p = e.get("path")
        if p:
            by_path[p] = {"sha256": e.get("sha256"), "size": e.get("size")}
    return {
        "path": str(path),
        "root_tree_sha256": obj.get("root_tree_sha256"),
        "entries": by_path,
    }


def _manifest_entries_map(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    entries = manifest.get("entries", [])
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(entries, list):
        for e in entries:
            if not isinstance(e, dict):
                continue
            p = e.get("path")
            if isinstance(p, str) and p:
                out[p] = e
    return out


def _diff_manifests(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    am = _manifest_entries_map(a)
    bm = _manifest_entries_map(b)

    a_paths = set(am.keys())
    b_paths = set(bm.keys())

    added_paths = sorted(b_paths - a_paths)
    removed_paths = sorted(a_paths - b_paths)
    common_paths = sorted(a_paths & b_paths)

    added: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    modified: List[Dict[str, Any]] = []

    for p in added_paths:
        e = bm[p]
        added.append({"path": p, "sha256": e.get("sha256"), "size": e.get("size")})

    for p in removed_paths:
        e = am[p]
        removed.append({"path": p, "sha256": e.get("sha256"), "size": e.get("size")})

    for p in common_paths:
        ea = am[p]
        eb = bm[p]
        if (ea.get("sha256") != eb.get("sha256")) or (ea.get("size") != eb.get("size")):
            modified.append(
                {
                    "path": p,
                    "a": {"sha256": ea.get("sha256"), "size": ea.get("size")},
                    "b": {"sha256": eb.get("sha256"), "size": eb.get("size")},
                }
            )

    return {"added": added, "removed": removed, "modified": modified}


def _find_manifest_files(evidence_dir: Path, *, type_filter: str = "any") -> List[Path]:
    if not isinstance(evidence_dir, Path):
        evidence_dir = Path(str(evidence_dir))

    if not evidence_dir.exists() or not evidence_dir.is_dir():
        return []

    paths: List[Path] = []
    for p in evidence_dir.glob("*manifest*.json"):
        if not p.is_file():
            continue
        name = p.name
        is_weekly = name.startswith("weekly_manifest")
        is_drop = name.startswith("manifest") and not is_weekly

        if type_filter == "weekly" and not is_weekly:
            continue
        if type_filter == "drop" and not is_drop:
            continue

        paths.append(p)

    def _mtime_key(p: Path) -> Tuple[float, str]:
        try:
            return (p.stat().st_mtime, p.name)
        except Exception:
            return (0.0, p.name)

    paths.sort(key=_mtime_key, reverse=True)
    return paths


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_since_ok_manifest_path(evidence_dir: Path, *, args_for_db: argparse.Namespace) -> Optional[Path]:
    db_path = resolve_db_path(args_for_db)

    candidates: List[Path] = []
    try:
        candidates.extend(sorted(evidence_dir.glob("drop_evidence*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(evidence_dir.glob("weekly_evidence*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
    except Exception:
        candidates = []

    if not candidates:
        return None

    con = None
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row

        for ev_path in candidates:
            ev = _read_json_file(ev_path)
            if not isinstance(ev, dict):
                continue

            run_id = str(ev.get("run_id") or ev.get("weekly_run_id") or "").strip()
            paths = ev.get("paths") if isinstance(ev.get("paths"), dict) else {}
            manifest_raw = str(paths.get("manifest_path") or "").strip()

            if not run_id or not manifest_raw:
                continue

            try:
                row = con.execute(
                    "SELECT COUNT(1) AS n FROM run_stages WHERE run_id = ? AND LOWER(status) = 'error'",
                    (run_id,),
                ).fetchone()
            except sqlite3.Error:
                continue

            n_err = int(row["n"] if row and row["n"] is not None else 0)
            if n_err != 0:
                continue

            mp = Path(manifest_raw)
            if not mp.is_absolute():
                mp = (evidence_dir / mp).resolve()

            if mp.exists() and mp.is_file():
                return mp

        return None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def cmd_run_diff(args: argparse.Namespace) -> int:
    """
    Diff manifests in the evidence dir.

    Output rules:
      - If global --json is set (mgc.main --json ...), emit exactly ONE JSON object.
      - Otherwise print a human summary line:
            +A  -R  ~M  (older=... newer=...)
      - --summary-only: counts only (still JSON if --json is set)
      - --fail-on-changes: exit 2 if there are any non-allowed changes
      - --allow PATH (repeatable): allow specific changed paths when failing on changes
      - --since PATH: compare newest against PATH (PATH is "older")
      - --since-ok: auto-pick an older manifest from the most recent run with no stage errors
    """
    want_json = resolve_want_json(args)

    evidence_dir = Path(
        getattr(args, "out_dir", None)
        or getattr(args, "evidence_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR", "data/evidence")
    ).resolve()

    type_filter = str(getattr(args, "type", "any") or "any").strip().lower()
    fail_on_changes = bool(getattr(args, "fail_on_changes", False))
    summary_only = bool(getattr(args, "summary_only", False))
    since = getattr(args, "since", None)
    since_ok = bool(getattr(args, "since_ok", False))

    allow_list = getattr(args, "allow", None) or []
    allow_set = {str(a).replace("\\", "/") for a in allow_list if str(a).strip()}

    files = _find_manifest_files(evidence_dir, type_filter=type_filter)  # newest-first

    # Choose (older=a_path, newer=b_path)
    since_path: Optional[Path] = None
    if since:
        since_path = Path(str(since)).expanduser().resolve()
        if not since_path.exists():
            sys.stdout.write(
                stable_json_dumps(
                    {"found": False, "reason": "since_not_found", "since": str(since_path)}
                )
                + "\n"
            )
            return 0

        if not files:
            sys.stdout.write(
                stable_json_dumps(
                    {"found": False, "reason": "no_manifests_found", "path": str(evidence_dir)}
                )
                + "\n"
            )
            return 0

        a_path, b_path = since_path, files[0]

    elif since_ok:
        # Must use resolver helper (DB path depends on args/global default/env)
        since_path = _resolve_since_ok_manifest_path(evidence_dir, args_for_db=args)
        if since_path is None:
            sys.stdout.write(
                stable_json_dumps(
                    {"found": False, "reason": "since_ok_not_found", "path": str(evidence_dir)}
                )
                + "\n"
            )
            return 0

        if not files:
            sys.stdout.write(
                stable_json_dumps(
                    {"found": False, "reason": "no_manifests_found", "path": str(evidence_dir)}
                )
                + "\n"
            )
            return 0

        a_path, b_path = since_path, files[0]

    else:
        if len(files) < 2:
            sys.stdout.write(
                stable_json_dumps(
                    {
                        "found": False,
                        "reason": "need_at_least_two_manifests",
                        "count": len(files),
                        "path": str(evidence_dir),
                    }
                )
                + "\n"
            )
            return 0

        a_path, b_path = files[1], files[0]

    a = _load_manifest(a_path)
    b = _load_manifest(b_path)
    diff = _diff_manifests(a, b)

    added = diff.get("added", []) or []
    removed = diff.get("removed", []) or []
    modified = diff.get("modified", []) or []

    # Allow-list applies to modified paths only (keeps CI strict on add/remove)
    modified_paths = [str(x.get("path") or "") for x in modified if isinstance(x, dict)]
    non_allowed_modified = [p for p in modified_paths if p and p not in allow_set]

    summary = {
        "added": len(added),
        "removed": len(removed),
        "modified": len(modified),
    }

    has_any_changes = (summary["added"] + summary["removed"] + summary["modified"]) > 0
    has_blocking_changes = (summary["added"] + summary["removed"] + len(non_allowed_modified)) > 0

    exit_code = 0
    if fail_on_changes and has_any_changes:
        exit_code = 2 if has_blocking_changes else 0

    older_name = a_path.name
    newer_name = b_path.name

    if want_json:
        out: Dict[str, Any] = {
            "found": True,
            "older": str(a_path),
            "newer": str(b_path),
            "older_name": older_name,
            "newer_name": newer_name,
            "type_filter": type_filter,
            "since": str(since_path) if since_path else None,
            "since_ok": since_ok,
            "fail_on_changes": fail_on_changes,
            "allow": sorted(allow_set) if allow_set else [],
            "summary": summary,
            "exit_code": exit_code,
        }

        if not summary_only:
            out["diff"] = diff

        if allow_set:
            out["non_allowed_modified_paths"] = non_allowed_modified

        sys.stdout.write(stable_json_dumps(out) + "\n")
        return exit_code

    print(f"+{summary['added']}  -{summary['removed']}  ~{summary['modified']}  (older={older_name} newer={newer_name})")
    return exit_code

# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    run_p = subparsers.add_parser("run", help="Run pipeline steps (daily, publish, drop, weekly, manifest, stage, status)")
    run_p.set_defaults(_mgc_group="run")
    run_sub = run_p.add_subparsers(dest="run_cmd", required=True)

    open_p = run_sub.add_parser("open", help="Print paths of latest evidence/manifest files (pipe-friendly)")
    open_p.add_argument("--out-dir", default=None, help="Evidence directory (default: data/evidence or MGC_EVIDENCE_DIR)")
    open_p.add_argument("--type", choices=["drop", "weekly", "any"], default="any", help="Evidence type filter")
    open_p.add_argument("--n", type=int, default=1, help="Number of recent files")
    open_p.set_defaults(func=cmd_run_open)

    tail = run_sub.add_parser("tail", help="Show latest evidence file(s)")
    tail.add_argument("--out-dir", default=None, help="Evidence directory (default: data/evidence or MGC_EVIDENCE_DIR)")
    tail.add_argument("--type", choices=["drop", "weekly", "any"], default="any", help="Evidence type filter")
    tail.add_argument("--n", type=int, default=1, help="Number of recent files to show")
    tail.set_defaults(func=cmd_run_tail)

    diff = run_sub.add_parser("diff", help="Diff the two most recent manifests")
    diff.add_argument("--out-dir", default=None, help="Evidence directory (default: data/evidence or MGC_EVIDENCE_DIR)")
    diff.add_argument("--type", choices=["drop", "weekly", "any"], default="any", help="Manifest type filter")
    diff.add_argument("--since", default=None, help="Compare newest manifest against this manifest path (older)")
    diff.add_argument("--since-ok", action="store_true", help="Auto-pick an older manifest from the most recent run with no stage errors")
    diff.add_argument("--summary-only", action="store_true", help="Only output counts in JSON mode (still one-line summary in human mode)")
    diff.add_argument("--fail-on-changes", action="store_true", help="Exit 2 if any non-allowed changes are detected (CI)")
    diff.add_argument("--allow", action="append", default=[], help="Allow specific changed manifest paths (repeatable). Works with --fail-on-changes.")
    diff.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
    diff.set_defaults(func=cmd_run_diff)

    status = run_sub.add_parser("status", help="Show latest (or specific) run status + stages + drop pointers")
    status.add_argument("--fail-on-error", action="store_true", help="Exit 2 if any stage is error (for CI)")
    status.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
    status.add_argument("--run-id", default=None, help="Specific run_id to inspect")
    status.add_argument("--latest", action="store_true", help="Force latest run_id (default if --run-id omitted)")
    status.set_defaults(func=cmd_run_status)

    daily = run_sub.add_parser("daily", help="Run the daily pipeline (deterministic capable)")
    daily.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
    daily.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Context/mood (focus/workout/sleep)")
    daily.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"), help="Seed for deterministic behavior")
    daily.add_argument("--out-dir", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"), help="Evidence output directory")
    daily.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (also via MGC_DETERMINISTIC=1)")
    daily.set_defaults(func=cmd_run_daily)

    pub = run_sub.add_parser("publish-marketing", help="Publish pending marketing drafts (draft -> published)")
    pub.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
    pub.add_argument("--limit", type=int, default=50, help="Max number of drafts to publish")
    pub.add_argument("--dry-run", action="store_true", help="Do not update DB; just print what would publish")
    pub.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (also via MGC_DETERMINISTIC=1)")
    pub.set_defaults(func=cmd_publish_marketing)

    drop = run_sub.add_parser("drop", help="Run daily + publish-marketing + manifest and emit consolidated evidence")
    drop.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
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
    weekly.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
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
    st_set.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
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
    st_get.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
    st_get.add_argument("run_id", help="Canonical run_id")
    st_get.add_argument("stage", help="Stage name")
    st_get.set_defaults(func=cmd_run_stage_get)

    st_ls = stage_sub.add_parser("list", help="List all stages for a run_id")
    st_ls.add_argument("--db", default=None, help="SQLite DB path (optional; also honors global --db and MGC_DB)")
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
