#!/usr/bin/env python3
"""
src/mgc/run_cli.py

Run/pipeline subcommands for MGC.

Goals:
- Import-safe: mgc.main can import this module and build the parser without NameError.
- Deterministic-capable: stable timestamps/IDs when --deterministic or MGC_DETERMINISTIC=1.
- CI-friendly: JSON output support via global --json (mgc.main) or MGC_JSON=1.
- Schema-tolerant DB writes: supports modest drift across environments.
- Provides run subcommands:
    daily, autonomous, publish-marketing, drop, stage, manifest, diff, status

Notes:
- This file is intentionally self-contained. It will *optionally* use richer
  implementations elsewhere (if present), but always falls back to minimal
  working behavior.
"""

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


# -----------------------------------------------------------------------------
# Optional imports (keep import-time safe)
# -----------------------------------------------------------------------------

def _fallback_build_prompt(context: str) -> str:
    return f"Generate a short instrumental track suitable for {context}."

try:
    # Expected project API
    from mgc.context import build_prompt as _build_prompt  # type: ignore
except Exception:
    _build_prompt = _fallback_build_prompt  # type: ignore


@dataclass(frozen=True)
class _ProviderResult:
    provider: str
    ext: str
    artifact_bytes: bytes
    meta: Dict[str, Any]


def _fallback_provider_generate(
    *,
    track_id: str,
    run_id: str,
    context: str,
    seed: str,
    prompt: str,
    deterministic: bool,
    ts: str,
    out_rel: str,
) -> _ProviderResult:
    # Deterministic placeholder audio-ish bytes (not real audio, but stable and non-empty).
    payload = stable_json_dumps(
        {
            "schema": "mgc.stub_audio.v1",
            "track_id": track_id,
            "run_id": run_id,
            "context": context,
            "seed": seed,
            "prompt": prompt,
            "deterministic": deterministic,
            "ts": ts,
            "out_rel": out_rel,
        }
    ).encode("utf-8")
    blob = b"STUBAUDIO\x00" + hashlib.sha256(payload).digest() + payload
    return _ProviderResult(provider="stub", ext=".wav", artifact_bytes=blob, meta={"genre": "ambient"})


# If your repo provides mgc.providers, we’ll use it. Otherwise we fallback.
try:
    from mgc.providers import GenerateRequest, get_provider  # type: ignore
except Exception:
    GenerateRequest = None  # type: ignore
    get_provider = None  # type: ignore


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Global argv helpers (fix: honor --db even when passed before "run")
# -----------------------------------------------------------------------------

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
            return nxt
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
      - env: MGC_JSON=1
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
      - env: MGC_DB=... (preferred) then default data/db.sqlite
    """
    db = getattr(args, "db", None)
    if isinstance(db, str) and db.strip():
        return db.strip()

    db2 = _argv_value("--db")
    if isinstance(db2, str) and db2.strip():
        return db2.strip()

    env = (os.environ.get("MGC_DB") or "").strip()
    if env:
        return env

    return "data/db.sqlite"


# -----------------------------------------------------------------------------
# Determinism utilities
# -----------------------------------------------------------------------------

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


def _week_start_date(run_date_yyyy_mm_dd: str) -> str:
    try:
        dt = datetime.strptime(run_date_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    weekday = dt.weekday()  # Monday=0
    ws = dt - timedelta(days=weekday)
    return ws.date().isoformat()


# -----------------------------------------------------------------------------
# Canonical run identity (run_key -> run_id)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunKey:
    run_date: str
    context: str
    seed: str
    provider_set_version: str


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


# -----------------------------------------------------------------------------
# DB helpers (schema-tolerant + NOT NULL fillers)
# -----------------------------------------------------------------------------

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

    # Prefer explicit IDs if present (prevents FK failures)
    # row_data may include these directly, or inside row_data["meta"].
    meta = row_data.get("meta")
    meta_dict: Dict[str, Any] = meta if isinstance(meta, dict) else {}

    def _pick_id(key: str) -> Optional[str]:
        v = row_data.get(key)
        if v is None:
            v = row_data.get(key.lower())
        if v is None and meta_dict:
            v = meta_dict.get(key) or meta_dict.get(key.lower())
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    if name in ("run_id", "drop_id", "track_id", "marketing_batch_id", "batch_id", "post_id", "id"):
        v = _pick_id(name)
        if v is not None:
            return v
        # If required and missing, do NOT invent a non-existent FK id; return empty string last.
        # The calling insert should generally provide these explicitly when schema requires them.
        return ""

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
        return str(row_data.get("provider") or meta_dict.get("provider") or "stub")
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


# -----------------------------------------------------------------------------
# Run stage tracking (resume/observability)
# -----------------------------------------------------------------------------

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
      - If allow_resume and stage already ok/skipped: do not run body.
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
        meta_patch={"resume": bool(allow_resume), **(meta or {})},
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


# -----------------------------------------------------------------------------
# Marketing schema-tolerant helpers + DB ops
# -----------------------------------------------------------------------------

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
        ["meta_json", "metadata_json", "meta", "metadata", "payload", "payload_json", "data_json"],
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
            "id", "post_id", "marketing_post_id", "ts", "created_at",
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
            "id", "post_id", "marketing_post_id", "ts", "created_at",
            "platform", "channel", "destination",
            "status", "state",
        ],
    )
    if best_payload and best_payload in set(cols):
        v = row[best_payload]
        meta = _load_json_maybe(v)
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

    # IMPORTANT: satisfy FK/NOT NULL columns if your fixtures schema enforces them.
    run_id_col = _pick_first_existing(cols, ["run_id"])
    drop_id_col = _pick_first_existing(cols, ["drop_id"])
    track_id_col = _pick_first_existing(cols, ["track_id"])
    batch_id_col = _pick_first_existing(cols, ["batch_id", "marketing_batch_id"])

    best_payload = None
    if not content_col and not meta_col:
        best_payload = _best_text_payload_column(
            con,
            "marketing_posts",
            reserved=["id", "post_id", "ts", "created_at", "platform", "channel", "destination", "status", "state"],
        )

    meta_to_store = dict(meta or {})
    if not content_col:
        meta_to_store["content"] = content

    data: Dict[str, Any] = {"id": post_id, "post_id": post_id}

    if ts_col:
        data[ts_col] = ts
    if platform_col:
        data[platform_col] = platform
    if status_col:
        data[status_col] = status

    # Satisfy FK columns when present
    if run_id_col and meta_to_store.get("run_id"):
        data[run_id_col] = str(meta_to_store["run_id"])
    if drop_id_col and meta_to_store.get("drop_id"):
        data[drop_id_col] = str(meta_to_store["drop_id"])
    if track_id_col and meta_to_store.get("track_id"):
        data[track_id_col] = str(meta_to_store["track_id"])
    if batch_id_col and meta_to_store.get("batch_id"):
        data[batch_id_col] = str(meta_to_store["batch_id"])

    # Store content/meta in the best-fitting schema
    if content_col:
        data[content_col] = content
    elif meta_col:
        data[meta_col] = stable_json_dumps(meta_to_store)
    elif best_payload:
        data[best_payload] = stable_json_dumps(meta_to_store) if meta_to_store else content

    # Also populate meta col if it exists and we didn't already
    if meta_col and meta_col not in data:
        data[meta_col] = stable_json_dumps(meta_to_store)

    _insert_row(con, "marketing_posts", data)

def db_marketing_posts_pending(con: sqlite3.Connection, *, limit: int = 50) -> List[sqlite3.Row]:
    cols = {r["name"] for r in con.execute("PRAGMA table_info(marketing_posts)").fetchall()}
    if not cols:
        return []

    # pick PK column
    if "id" in cols:
        pk = "id"
    elif "post_id" in cols:
        pk = "post_id"
    elif "marketing_post_id" in cols:
        pk = "marketing_post_id"
    else:
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
        created_col = pk

    status_col = "status" if "status" in cols else ("state" if "state" in cols else None)
    if status_col is None:
        return []

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
        where_col_for_lookup = "id" if "id" in set(cols) else (_pick_first_existing(cols, ["post_id"]) or "id")
        row = con.execute(f"SELECT {meta_col} FROM marketing_posts WHERE {where_col_for_lookup} = ?", (post_id,)).fetchone()
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


# -----------------------------------------------------------------------------
# Manifest (deterministic)
# -----------------------------------------------------------------------------

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
            "data",       # keep repo manifests stable vs generated artifacts
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
    manifest_obj: Dict[str, Any] = {
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
        eprint(str(manifest["root_tree_sha256"]))

    return 0


# -----------------------------------------------------------------------------
# Daily run (deterministic orchestrator)
# -----------------------------------------------------------------------------

def _maybe_call_external_daily_runner(
    *,
    db_path: str,
    context: str,
    seed: str,
    deterministic: bool,
    ts: str,
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


def _provider_generate_bytes(
    *,
    track_id: str,
    run_id: str,
    context: str,
    seed: str,
    prompt: str,
    deterministic: bool,
    ts: str,
    out_rel: str,
) -> _ProviderResult:
    provider_name = str(os.environ.get("MGC_PROVIDER") or "stub").strip().lower()

    if get_provider is None or GenerateRequest is None:
        return _fallback_provider_generate(
            track_id=track_id,
            run_id=run_id,
            context=context,
            seed=seed,
            prompt=prompt,
            deterministic=deterministic,
            ts=ts,
            out_rel=out_rel,
        )

    try:
        provider = get_provider(provider_name)
        req = GenerateRequest(
            track_id=track_id,
            run_id=run_id,
            context=context,
            seed=seed,
            prompt=prompt,
            deterministic=deterministic,
            ts=ts,
            out_rel=str(out_rel),
        )
        result = provider.generate(req)
        ext = result.ext if getattr(result, "ext", None) else ".wav"
        if ext and not str(ext).startswith("."):
            ext = "." + str(ext)
        meta = dict(getattr(result, "meta", {}) or {})
        artifact_bytes = bytes(getattr(result, "artifact_bytes", b"") or b"")
        prov = str(getattr(result, "provider", provider_name) or provider_name)
        return _ProviderResult(provider=prov, ext=str(ext), artifact_bytes=artifact_bytes, meta=meta)
    except Exception:
        return _fallback_provider_generate(
            track_id=track_id,
            run_id=run_id,
            context=context,
            seed=seed,
            prompt=prompt,
            deterministic=deterministic,
            ts=ts,
            out_rel=out_rel,
        )


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
    import shutil

    def _posix(p: Path) -> str:
        return str(p).replace("\\", "/")

    out_dir.mkdir(parents=True, exist_ok=True)

    drop_id = stable_uuid5("drop", run_id)
    track_id = stable_uuid5("track", context, seed, run_id)
    title = f"{context.title()} Track {seed}"

    # Repo artifact path (stable-ish layout)
    if deterministic:
        artifact_rel = Path("data") / "tracks" / f"{track_id}"
    else:
        day = ts.split("T", 1)[0]
        artifact_rel = Path("data") / "tracks" / day / f"{track_id}"

    artifact_path = (Path.cwd() / artifact_rel).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = _build_prompt(context)

    result = _provider_generate_bytes(
        track_id=track_id,
        run_id=run_id,
        context=context,
        seed=seed,
        prompt=prompt,
        deterministic=deterministic,
        ts=ts,
        out_rel=str(artifact_path),
    )

    # Ensure extension matches provider output
    ext = result.ext if result.ext else ".wav"
    if not ext.startswith("."):
        ext = "." + ext
    if artifact_path.suffix != ext:
        artifact_rel = artifact_rel.with_suffix(ext)
        artifact_path = (Path.cwd() / artifact_rel).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_bytes = result.artifact_bytes or b""
    if not artifact_bytes:
        # Always write something non-empty so sha256 is meaningful in CI.
        artifact_bytes = b"STUBAUDIO_EMPTY_FALLBACK"
    artifact_path.write_bytes(artifact_bytes)

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

    # Portable bundle
    bundle_tracks_dir = out_dir / "tracks"
    bundle_tracks_dir.mkdir(parents=True, exist_ok=True)

    bundled_name = f"{track_id}{artifact_path.suffix}"
    bundled_track_rel = Path("tracks") / bundled_name
    bundled_track_path = (out_dir / bundled_track_rel).resolve()
    shutil.copy2(str(artifact_path), str(bundled_track_path))

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
                "path": str(bundled_track_rel).replace("\\", "/"),
            }
        ],
    }
    playlist_path.write_text(stable_json_dumps(playlist_obj) + "\n", encoding="utf-8", newline="\n")

    repo_artifact_sha256 = _sha256_file(artifact_path)
    bundled_track_sha256 = _sha256_file(bundled_track_path)
    playlist_sha256 = _sha256_file(playlist_path)

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
            "repo_artifact": str(artifact_rel).replace("\\", "/"),
            "bundle_track": str(bundled_track_rel).replace("\\", "/"),
            "playlist": str(playlist_rel).replace("\\", "/"),
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

    evidence_text = stable_json_dumps(evidence_obj) + "\n"
    evidence_path_main.write_text(evidence_text, encoding="utf-8", newline="\n")
    evidence_path_scoped.write_text(evidence_text, encoding="utf-8", newline="\n")

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
            "bundle_track": str(bundled_track_rel).replace("\\", "/"),
            "playlist": str(playlist_rel).replace("\\", "/"),
            "evidence": [str(evidence_rel_main), str(evidence_rel_scoped)],
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
        "repo_artifact_path": str(artifact_rel).replace("\\", "/"),
        "bundle_track_path": str(bundled_track_rel).replace("\\", "/"),
        "playlist_path": str(playlist_rel).replace("\\", "/"),
        "evidence_paths": [str(evidence_rel_main), str(evidence_rel_scoped)],
        "sha256": {
            "repo_artifact": repo_artifact_sha256,
            "bundle_track": bundled_track_sha256,
            "playlist": playlist_sha256,
        },
        "marketing_post_ids": post_ids,
    }


def cmd_run_daily(args: argparse.Namespace) -> int:
    deterministic = is_deterministic(args)

    db_path = resolve_db_path(args)
    context = str(getattr(args, "context", None) or os.environ.get("MGC_CONTEXT") or "focus")
    seed = str(getattr(args, "seed", None) if getattr(args, "seed", None) is not None else (os.environ.get("MGC_SEED") or "1"))
    provider_set_version = str(os.environ.get("MGC_PROVIDER_SET_VERSION") or "v1")

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

    # External daily runner hook (optional)
    maybe = _maybe_call_external_daily_runner(
        db_path=db_path,
        context=context,
        seed=seed,
        deterministic=deterministic,
        ts=ts,
    )
    if maybe is not None:
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


# -----------------------------------------------------------------------------
# Marketing publish (deterministic)
# -----------------------------------------------------------------------------

def cmd_publish_marketing(args: argparse.Namespace) -> int:
    deterministic = is_deterministic(args)

    db_path = resolve_db_path(args)
    limit = int(getattr(args, "limit", None) or 50)
    dry_run = bool(getattr(args, "dry_run", False))

    ts = deterministic_now_iso(deterministic)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    pending = db_marketing_posts_pending(con, limit=limit)

    def _first_id(r: sqlite3.Row) -> str:
        v = _row_first(r, ["id", "post_id", "marketing_post_id"], default="")
        return str(v or "")

    def _first_created(r: sqlite3.Row) -> str:
        v = _row_first(r, ["created_at", "created_ts", "ts"], default="")
        return str(v or "")

    pending_sorted = sorted(list(pending), key=lambda r: (_first_created(r), _first_id(r)))

    published: List[Dict[str, Any]] = []
    skipped_ids: List[str] = []
    run_ids_touched: List[str] = []

    for row in pending_sorted:
        meta = _marketing_row_meta(con, row)
        rid = str((meta or {}).get("run_id") or "")
        if rid:
            run_ids_touched.append(rid)
    run_ids_touched = sorted(set([r for r in run_ids_touched if r]))

    batch_id = stable_uuid5(
        "marketing_publish_batch",
        (ts if not deterministic else "fixed"),
        str(limit),
        ("dry" if dry_run else "live"),
        ("|".join(run_ids_touched) if run_ids_touched else "no_runs"),
    )

    for row in pending_sorted:
        post_id = _first_id(row)
        platform = str(_row_first(row, ["platform", "channel", "destination"], default="unknown"))
        content = _marketing_row_content(con, row)

        if not content.strip():
            skipped_ids.append(post_id)
            continue

        meta = _marketing_row_meta(con, row)
        run_id = str((meta or {}).get("run_id") or "")
        drop_id = str((meta or {}).get("drop_id") or "")

        publish_id = stable_uuid5("publish", batch_id, post_id, platform)

        if not dry_run:
            db_marketing_post_set_status(
                con,
                post_id=post_id,
                status="published",
                ts=ts,
                meta_patch={"published_id": publish_id, "published_ts": ts, "batch_id": batch_id},
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


# -----------------------------------------------------------------------------
# Drop (minimal working)
# -----------------------------------------------------------------------------

def _latest_row_by_ts(con: sqlite3.Connection, table: str, ts_candidates: Sequence[str]) -> Optional[sqlite3.Row]:
    if not db_table_exists(con, table):
        return None
    cols = db_table_columns(con, table)
    if not cols:
        return None
    ts_col = _pick_first_existing(cols, list(ts_candidates))
    if not ts_col:
        # fallback: try rowid-ish ordering
        try:
            return con.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 1").fetchone()
        except Exception:
            return None
    try:
        return con.execute(f"SELECT * FROM {table} ORDER BY {ts_col} DESC LIMIT 1").fetchone()
    except Exception:
        return None


def cmd_run_drop(args: argparse.Namespace) -> int:
    """
    Deterministic-safe drop bundler.

    Selection order:
      1) --drop-id (exact match)
      2) --run-id  (drop for that run)
      3) Deterministic mode: most recently INSERTED drop (rowid DESC)
      4) Non-deterministic: latest by timestamp columns
      5) If none exist: run daily to create one
    """
    deterministic = is_deterministic(args)
    db_path = resolve_db_path(args)

    evidence_root = Path(
        getattr(args, "evidence_root", None)
        or os.environ.get("MGC_EVIDENCE_DIR")
        or "data/evidence"
    ).resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    # -------------------------
    # helpers (local + explicit)
    # -------------------------

    def table_has_rowid(table: str) -> bool:
        try:
            con.execute(f"SELECT rowid FROM {table} LIMIT 1")
            return True
        except Exception:
            return False

    def latest_drop_row() -> Optional[sqlite3.Row]:
        cols = set(db_table_columns(con, "drops"))
        if not cols:
            return None

        # Deterministic mode: prefer insertion order
        if deterministic and table_has_rowid("drops"):
            try:
                return con.execute(
                    "SELECT * FROM drops ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
            except Exception:
                pass

        # Non-deterministic fallback: timestamp-based
        ts_cols = [c for c in ("ts", "created_at", "created_ts", "published_ts") if c in cols]
        if ts_cols:
            order = ", ".join(f"{c} DESC" for c in ts_cols)
            return con.execute(
                f"SELECT * FROM drops ORDER BY {order} LIMIT 1"
            ).fetchone()

        # Absolute fallback
        return con.execute("SELECT * FROM drops LIMIT 1").fetchone()

    # -------------------------
    # explicit selection flags
    # -------------------------

    drop_id_arg = getattr(args, "drop_id", None)
    run_id_arg = getattr(args, "run_id", None)

    drop_row: Optional[sqlite3.Row] = None

    if drop_id_arg:
        drop_row = con.execute(
            "SELECT * FROM drops WHERE id = ? OR drop_id = ? LIMIT 1",
            (drop_id_arg, drop_id_arg),
        ).fetchone()

        if not drop_row:
            die(f"Drop not found: {drop_id_arg}")

    elif run_id_arg:
        drop_row = con.execute(
            "SELECT * FROM drops WHERE run_id = ? ORDER BY rowid DESC LIMIT 1",
            (run_id_arg,),
        ).fetchone()

        if not drop_row:
            die(f"No drop found for run_id={run_id_arg}")

    else:
        drop_row = latest_drop_row()

    # -------------------------
    # auto-create via daily
    # -------------------------

    if drop_row is None:
        daily_ns = argparse.Namespace(
            db=db_path,
            context=getattr(args, "context", os.environ.get("MGC_CONTEXT", "focus")),
            seed=str(getattr(args, "seed", os.environ.get("MGC_SEED", "1"))),
            out_dir=str(evidence_root),
            deterministic=deterministic,
            json=getattr(args, "json", False),
        )
        cmd_run_daily(daily_ns)
        drop_row = latest_drop_row()

    if drop_row is None:
        die("Unable to locate or create a drop.")

    # -------------------------
    # normalize fields
    # -------------------------

    drop_id = str(_row_first(drop_row, ["id", "drop_id"], default=""))
    run_id = str(_row_first(drop_row, ["run_id"], default=""))
    track_id = str(_row_first(drop_row, ["track_id"], default=""))
    context = str(_row_first(drop_row, ["context", "mood"], default="focus"))
    seed = str(_row_first(drop_row, ["seed"], default="1"))
    ts = str(
        _row_first(
            drop_row,
            ["ts", "created_at", "created_ts"],
            default=deterministic_now_iso(deterministic),
        )
    )

    # -------------------------
    # bundle payload
    # -------------------------

    bundle = {
        "schema": "mgc.drop_bundle.v1",
        "ts": ts,
        "deterministic": deterministic,
        "drop": {
            "id": drop_id,
            "run_id": run_id,
            "track_id": track_id,
            "context": context,
            "seed": seed,
        },
        "paths": {
            "evidence_root": str(evidence_root).replace("\\", "/"),
        },
    }

    out_path = evidence_root / "drop_bundle.json"
    out_path.write_text(
        stable_json_dumps(bundle) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # -------------------------
    # event log
    # -------------------------

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "drop.bundle.created", drop_id or run_id),
        ts=deterministic_now_iso(deterministic),
        kind="drop.bundle.created",
        actor="system",
        meta={
            "drop_id": drop_id,
            "run_id": run_id,
            "track_id": track_id,
            "path": str(out_path).replace("\\", "/"),
        },
    )

    # -------------------------
    # output
    # -------------------------

    sys.stdout.write(
        stable_json_dumps(
            {
                "ok": True,
                "bundle_path": str(out_path).replace("\\", "/"),
                "drop_id": drop_id,
                "run_id": run_id,
            }
        )
        + "\n"
    )
    return 0

# -----------------------------------------------------------------------------
# Autonomous (minimal but useful)
# -----------------------------------------------------------------------------

def cmd_run_autonomous(args: argparse.Namespace) -> int:
    """
    Minimal autonomous runner used by CI "full-mode" smoke tests.

    Behavior:
    - Creates/gets a run_id for today/context/seed.
    - Executes stages (daily -> publish-marketing -> drop) with run_stages tracking.
    - Respects determinism flags.
    """
    deterministic = is_deterministic(args)
    db_path = resolve_db_path(args)
    context = str(getattr(args, "context", None) or os.environ.get("MGC_CONTEXT") or "focus")
    seed = str(getattr(args, "seed", None) if getattr(args, "seed", None) is not None else (os.environ.get("MGC_SEED") or "1"))
    provider_set_version = str(os.environ.get("MGC_PROVIDER_SET_VERSION") or "v1")

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

    # allow_resume default: True unless --no-resume, but daily in CI often wants no-resume
    no_resume = bool(getattr(args, "no_resume", False))
    allow_resume = not no_resume

    outputs: Dict[str, Any] = {"run_id": run_id, "run_date": run_date, "context": context, "seed": seed, "deterministic": deterministic}

    with run_stage(con, run_id=run_id, stage="daily", deterministic=deterministic, allow_resume=allow_resume):
        daily_ns = argparse.Namespace(
            db=db_path,
            context=context,
            seed=seed,
            out_dir=str(out_dir),
            deterministic=deterministic,
            json=getattr(args, "json", False),
        )
        # cmd_run_daily prints JSON evidence; we also capture the essentials for the final summary.
        # Keep it simple: re-read latest drop/track after it runs.
        cmd_run_daily(daily_ns)

    latest_drop = _latest_row_by_ts(con, "drops", ["ts", "created_at", "created_ts", "published_ts"])
    if latest_drop is not None:
        outputs["drop_id"] = str(_row_first(latest_drop, ["id", "drop_id"], default=""))
        outputs["track_id"] = str(_row_first(latest_drop, ["track_id"], default=""))

    with run_stage(con, run_id=run_id, stage="publish_marketing", deterministic=deterministic, allow_resume=allow_resume):
        pub_ns = argparse.Namespace(
            db=db_path,
            limit=int(getattr(args, "limit", 50) or 50),
            dry_run=bool(getattr(args, "dry_run", False)),
            deterministic=deterministic,
            json=getattr(args, "json", False),
        )
        cmd_publish_marketing(pub_ns)

    with run_stage(con, run_id=run_id, stage="drop", deterministic=deterministic, allow_resume=allow_resume):
        drop_ns = argparse.Namespace(
            db=db_path,
            context=context,
            seed=seed,
            evidence_root=str(out_dir),
            deterministic=deterministic,
            json=getattr(args, "json", False),
        )
        cmd_run_drop(drop_ns)

    sys.stdout.write(stable_json_dumps({"ok": True, **outputs}) + "\n")
    return 0


# -----------------------------------------------------------------------------
# Stage (developer tool): run a named stage with tracking
# -----------------------------------------------------------------------------

def cmd_run_stage(args: argparse.Namespace) -> int:
    """
    Developer tool: run a named stage and record status in run_stages.

    Supported stage names (minimal):
      daily, publish-marketing, drop, manifest
    """
    deterministic = is_deterministic(args)
    db_path = resolve_db_path(args)
    stage_name = str(getattr(args, "name", "") or "").strip()
    if not stage_name:
        die("stage name is required")

    context = str(getattr(args, "context", None) or os.environ.get("MGC_CONTEXT") or "focus")
    seed = str(getattr(args, "seed", None) if getattr(args, "seed", None) is not None else (os.environ.get("MGC_SEED") or "1"))
    provider_set_version = str(os.environ.get("MGC_PROVIDER_SET_VERSION") or "v1")
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

    mapping: Dict[str, Callable[[], int]] = {}

    def _run_daily() -> int:
        ns = argparse.Namespace(db=db_path, context=context, seed=seed, out_dir=str(out_dir), deterministic=deterministic, json=getattr(args, "json", False))
        return int(cmd_run_daily(ns))

    def _run_pub() -> int:
        ns = argparse.Namespace(db=db_path, limit=int(getattr(args, "limit", 50) or 50), dry_run=bool(getattr(args, "dry_run", False)), deterministic=deterministic, json=getattr(args, "json", False))
        return int(cmd_publish_marketing(ns))

    def _run_drop() -> int:
        ns = argparse.Namespace(db=db_path, context=context, seed=seed, evidence_root=str(out_dir), deterministic=deterministic, json=getattr(args, "json", False))
        return int(cmd_run_drop(ns))

    def _run_manifest() -> int:
        ns = argparse.Namespace(repo_root=str(getattr(args, "repo_root", ".") or "."), include=None, exclude_dir=None, exclude_glob=None, out=None, print_hash=False)
        return int(cmd_run_manifest(ns))

    mapping["daily"] = _run_daily
    mapping["publish-marketing"] = _run_pub
    mapping["publish_marketing"] = _run_pub
    mapping["drop"] = _run_drop
    mapping["manifest"] = _run_manifest

    if stage_name not in mapping:
        die(f"Unknown stage: {stage_name}. Supported: {', '.join(sorted(mapping.keys()))}")

    allow_resume = not bool(getattr(args, "no_resume", False))
    with run_stage(con, run_id=run_id, stage=stage_name, deterministic=deterministic, allow_resume=allow_resume):
        rc = mapping[stage_name]()
        if rc != 0:
            die(f"stage {stage_name} failed with code {rc}", code=rc)

    sys.stdout.write(stable_json_dumps({"ok": True, "run_id": run_id, "stage": stage_name}) + "\n")
    return 0


# -----------------------------------------------------------------------------
# Status (snapshot)
# -----------------------------------------------------------------------------

def cmd_run_status(args: argparse.Namespace) -> int:
    db_path = resolve_db_path(args)
    limit = int(getattr(args, "limit", 20) or 20)
    want_json = resolve_want_json(args)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    def _select_recent(table: str, order_cols: Sequence[str]) -> List[Dict[str, Any]]:
        if not db_table_exists(con, table):
            return []
        cols = db_table_columns(con, table)
        if not cols:
            return []
        order_col = _pick_first_existing(cols, list(order_cols)) or cols[0]
        try:
            rows = con.execute(f"SELECT * FROM {table} ORDER BY {order_col} DESC LIMIT ?", (limit,)).fetchall()
        except Exception:
            rows = con.execute(f"SELECT * FROM {table} LIMIT ?", (limit,)).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d: Dict[str, Any] = {}
            for k in r.keys():
                v = r[k]
                if isinstance(v, (bytes, bytearray)):
                    d[k] = f"<{len(v)} bytes>"
                else:
                    d[k] = v
            out.append(d)
        return out

    out_obj = {
        "db": db_path,
        "runs": _select_recent("runs", ["updated_at", "created_at"]),
        "run_stages": _select_recent("run_stages", ["ended_at", "started_at", "id"]),
        "drops": _select_recent("drops", ["published_ts", "ts", "created_at", "created_ts"]),
        "tracks": _select_recent("tracks", ["ts", "created_at", "created_ts"]),
        "marketing_posts": _select_recent("marketing_posts", ["ts", "created_at", "created_ts"]),
        "events": _select_recent("events", ["ts"]),
    }

    if want_json:
        sys.stdout.write(stable_json_dumps(out_obj) + "\n")
        return 0

    # Human-friendly
    def _h(title: str) -> None:
        print(f"\n== {title} ==")

    _h("DB")
    print(db_path)

    _h("Recent runs")
    for r in out_obj["runs"][:limit]:
        print(f"- run_id={r.get('run_id')} run_date={r.get('run_date')} ctx={r.get('context')} seed={r.get('seed')} updated={r.get('updated_at')}")

    _h("Recent stages")
    for s in out_obj["run_stages"][:limit]:
        print(f"- run_id={s.get('run_id')} stage={s.get('stage')} status={s.get('status')} ended={s.get('ended_at')}")

    _h("Recent drops")
    for d in out_obj["drops"][:limit]:
        print(f"- drop_id={d.get('id') or d.get('drop_id')} run_id={d.get('run_id')} track_id={d.get('track_id')} ts={d.get('ts')} published={d.get('published_ts')}")

    return 0


# -----------------------------------------------------------------------------
# Diff (CI gate helper)
# -----------------------------------------------------------------------------

def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _diff_manifests(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a minimal, stable diff summary between two manifest dicts produced by compute_manifest.
    """
    a_hash = str(a.get("root_tree_sha256") or "")
    b_hash = str(b.get("root_tree_sha256") or "")

    a_entries = {e["path"]: e for e in (a.get("entries") or []) if isinstance(e, dict) and "path" in e}
    b_entries = {e["path"]: e for e in (b.get("entries") or []) if isinstance(e, dict) and "path" in e}

    a_paths = set(a_entries.keys())
    b_paths = set(b_entries.keys())

    added = sorted(b_paths - a_paths)
    removed = sorted(a_paths - b_paths)

    changed: List[str] = []
    for p in sorted(a_paths & b_paths):
        ea = a_entries[p]
        eb = b_entries[p]
        if str(ea.get("sha256")) != str(eb.get("sha256")) or int(ea.get("size") or 0) != int(eb.get("size") or 0):
            changed.append(p)

    return {
        "older_hash": a_hash,
        "newer_hash": b_hash,
        "added": added,
        "removed": removed,
        "changed": changed,
        "counts": {"added": len(added), "removed": len(removed), "changed": len(changed)},
    }


def cmd_run_diff(args: argparse.Namespace) -> int:
    """
    Minimal deterministic diff:
    - If --older/--newer are provided, read those manifest files and compare.
    - Otherwise, compare data/evidence/manifest.json vs data/weekly/weekly_manifest.json if present,
      falling back to comparing repo manifest vs manifest.json if present.
    - Outputs JSON when global --json or MGC_JSON=1, otherwise prints a brief summary.
    """
    want_json = resolve_want_json(args)
    fail_on_changes = bool(getattr(args, "fail_on_changes", False))
    summary_only = bool(getattr(args, "summary_only", False))

    repo_root = Path(getattr(args, "repo_root", ".") or ".").resolve()
    older_path = Path(getattr(args, "older", "") or "")
    newer_path = Path(getattr(args, "newer", "") or "")

    # Defaults
    if not str(older_path):
        older_path = Path("data/evidence/manifest.json")
    if not str(newer_path):
        newer_path = Path("data/weekly/weekly_manifest.json")

    older: Optional[Dict[str, Any]] = None
    newer: Optional[Dict[str, Any]] = None

    if older_path and older_path.exists():
        older = _read_json_file(older_path)
    if newer_path and newer_path.exists():
        newer = _read_json_file(newer_path)

    # If newer doesn't exist, compute current repo manifest and compare against older if possible.
    if newer is None:
        newer = compute_manifest(repo_root)

    # If older doesn't exist, treat as "no baseline" (diff empty / ok).
    if older is None:
        out_obj = {
            "ok": True,
            "reason": "missing_older_manifest",
            "older_path": str(older_path).replace("\\", "/"),
            "newer_path": str(newer_path).replace("\\", "/"),
            "newer_hash": str(newer.get("root_tree_sha256") or ""),
            "changes": {"added": [], "removed": [], "changed": [], "counts": {"added": 0, "removed": 0, "changed": 0}},
        }
        if want_json:
            sys.stdout.write(stable_json_dumps(out_obj) + "\n")
        else:
            print("diff: no baseline manifest found; treating as ok")
            print(f"newer_hash={out_obj['newer_hash']}")
        return 0

    diff = _diff_manifests(older, newer)
    changed_total = int(diff["counts"]["added"] + diff["counts"]["removed"] + diff["counts"]["changed"])
    ok = changed_total == 0

    out_obj = {
        "ok": ok,
        "older_path": str(older_path).replace("\\", "/"),
        "newer_path": str(newer_path).replace("\\", "/"),
        "diff": diff if not summary_only else {"counts": diff["counts"], "older_hash": diff["older_hash"], "newer_hash": diff["newer_hash"]},
    }

    if want_json:
        sys.stdout.write(stable_json_dumps(out_obj) + "\n")
    else:
        print(f"+{diff['counts']['added']}  -{diff['counts']['removed']}  ~{diff['counts']['changed']}  (older={Path(out_obj['older_path']).name} newer={Path(out_obj['newer_path']).name})")
        if not summary_only and not ok:
            if diff["added"]:
                print("added:")
                for p in diff["added"][:50]:
                    print(f"  + {p}")
            if diff["removed"]:
                print("removed:")
                for p in diff["removed"][:50]:
                    print(f"  - {p}")
            if diff["changed"]:
                print("changed:")
                for p in diff["changed"][:50]:
                    print(f"  ~ {p}")
        if "older_hash" in diff and "newer_hash" in diff and diff["older_hash"] and diff["newer_hash"]:
            print(f"older_hash={diff['older_hash']}")
            print(f"newer_hash={diff['newer_hash']}")

    if fail_on_changes and not ok:
        return 1
    return 0


# -----------------------------------------------------------------------------
# Argparse wiring (single canonical function; no duplicates)
# -----------------------------------------------------------------------------

def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """
    Wire `mgc run ...` subcommands.

    Contract:
    - This is the canonical run parser wiring.
    - Never reference a handler that isn't defined in this module.
    """
    run_p = subparsers.add_parser(
        "run",
        help="Run pipeline steps (daily, autonomous, publish-marketing, drop, stage, manifest, diff, status)",
    )
    run_p.set_defaults(_mgc_group="run")
    run_sub = run_p.add_subparsers(dest="run_cmd", required=True)

    # daily
    daily = run_sub.add_parser("daily", help="Run the daily pipeline (deterministic capable)")
    daily.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"))
    daily.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"))
    daily.add_argument("--out-dir", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"))
    daily.add_argument("--deterministic", action="store_true")
    daily.set_defaults(func=cmd_run_daily)

    # autonomous
    auto = run_sub.add_parser("autonomous", help="Run autonomous pipeline stages (daily -> publish-marketing -> drop)")
    auto.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"))
    auto.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"))
    auto.add_argument("--out-dir", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"))
    auto.add_argument("--no-resume", action="store_true")
    auto.add_argument("--limit", type=int, default=50, help="publish-marketing limit")
    auto.add_argument("--dry-run", action="store_true", help="publish-marketing dry run")
    auto.add_argument("--deterministic", action="store_true")
    auto.set_defaults(func=cmd_run_autonomous)

    # publish-marketing
    pub = run_sub.add_parser("publish-marketing", help="Publish pending marketing drafts (draft -> published)")
    pub.add_argument("--limit", type=int, default=50)
    pub.add_argument("--dry-run", action="store_true")
    pub.add_argument("--deterministic", action="store_true")
    pub.set_defaults(func=cmd_publish_marketing)

    # drop
    drop = run_sub.add_parser("drop", help="Create a minimal drop bundle from the latest artifacts")
    drop.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"))
    drop.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"))
    drop.add_argument("--evidence-root", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"))
    drop.add_argument("--deterministic", action="store_true")
    drop.set_defaults(func=cmd_run_drop)

    # stage
    stage = run_sub.add_parser("stage", help="Run a named stage with run_stages tracking")
    stage.add_argument("name", help="Stage name (daily, publish-marketing, drop, manifest)")
    stage.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"))
    stage.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"))
    stage.add_argument("--out-dir", default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"))
    stage.add_argument("--no-resume", action="store_true")
    stage.add_argument("--limit", type=int, default=50)
    stage.add_argument("--dry-run", action="store_true")
    stage.add_argument("--repo-root", default=".")
    stage.add_argument("--deterministic", action="store_true")
    stage.set_defaults(func=cmd_run_stage)

    # manifest
    man = run_sub.add_parser("manifest", help="Compute deterministic repo manifest (stable file hashing)")
    man.add_argument("--repo-root", default=".")
    man.add_argument("--include", action="append", default=None)
    man.add_argument("--exclude-dir", action="append", default=None)
    man.add_argument("--exclude-glob", action="append", default=None)
    man.add_argument("--out", default=None)
    man.add_argument("--print-hash", action="store_true")
    man.set_defaults(func=cmd_run_manifest)

    # diff
    diff = run_sub.add_parser("diff", help="Compare manifest files (CI gate helper)")
    diff.add_argument("--repo-root", default=".")
    diff.add_argument("--older", default="", help="Older manifest path (default data/evidence/manifest.json)")
    diff.add_argument("--newer", default="", help="Newer manifest path (default data/weekly/weekly_manifest.json)")
    diff.add_argument("--since-ok", action="store_true", help="Reserved for compatibility; no-op in minimal diff")
    diff.add_argument("--fail-on-changes", action="store_true")
    diff.add_argument("--summary-only", action="store_true")
    diff.set_defaults(func=cmd_run_diff)

    # status
    status = run_sub.add_parser("status", help="Show run/pipeline status snapshot")
    status.add_argument("--limit", type=int, default=20)
    status.set_defaults(func=cmd_run_status)


# -----------------------------------------------------------------------------
# Standalone entrypoint (optional)
# -----------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    # Standalone parser for this module.
    p = argparse.ArgumentParser(prog="mgc-run-cli", description="MGC run_cli helpers")

    # Provide these so this module can be used standalone.
    p.add_argument("--db", default=None, help="SQLite DB path (or set MGC_DB)")
    p.add_argument("--json", action="store_true", help="JSON output where supported (or set MGC_JSON=1)")

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
