# src/mgc/events.py
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _json_default(o: Any) -> Any:
    # Deterministic fallback conversions (no memory addresses)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(list(o))
    if callable(o):
        return getattr(o, "__name__", "<callable>")
    return str(o)


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def new_run_id() -> str:
    return str(uuid.uuid4())


def _table_columns(conn: sqlite3.Connection, table: str) -> Dict[str, str]:
    """
    Returns {column_name: declared_type} for table if it exists, else {}.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not row:
        return {}

    cols: Dict[str, str] = {}
    for r in conn.execute(f"PRAGMA table_info({table})").fetchall():
        cols[str(r["name"])] = str(r["type"] or "")
    return cols


def _create_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
          id TEXT PRIMARY KEY,
          occurred_at TEXT NOT NULL,
          run_id TEXT NOT NULL,
          source TEXT NOT NULL,
          event_type TEXT NOT NULL,
          entity_type TEXT NOT NULL,
          entity_id TEXT,
          payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_type, entity_id)")


def _migrate_events_if_needed(conn: sqlite3.Connection) -> None:
    """
    If an older events schema exists (e.g., missing 'id'), migrate in-place by:
      - renaming old table
      - creating new table
      - copying rows with best-effort column mapping
      - dropping old table
    """
    cols = _table_columns(conn, "events")
    if not cols:
        # no table yet
        _create_events_table(conn)
        return

    if "id" in cols:
        # already new schema
        _create_events_table(conn)
        return

    legacy = "events_legacy"
    # If a prior migration partially happened, clean up conservatively.
    legacy_exists = bool(_table_columns(conn, legacy))
    if legacy_exists:
        # Prefer keeping the current "events" as-is; do nothing further.
        # (User can manually clean if needed.)
        return

    # Rename old -> legacy
    conn.execute(f"ALTER TABLE events RENAME TO {legacy}")

    # Create new
    _create_events_table(conn)

    legacy_cols = _table_columns(conn, legacy)

    # Build SELECT list with fallbacks for missing columns
    # Always generate a new id for legacy rows.
    def sel(col: str, default_sql: str) -> str:
        return col if col in legacy_cols else default_sql

    occurred_at = sel("occurred_at", "''")
    run_id = sel("run_id", "''")
    source = sel("source", "'unknown'")
    event_type = sel("event_type", "'legacy.event'")
    entity_type = sel("entity_type", "'system'")
    entity_id = sel("entity_id", "NULL")

    # Some older versions may have used payload instead of payload_json
    if "payload_json" in legacy_cols:
        payload_json = "payload_json"
    elif "payload" in legacy_cols:
        payload_json = "payload"
    else:
        payload_json = "'{}'"

    # Copy rows
    conn.execute(
        f"""
        INSERT INTO events (id, occurred_at, run_id, source, event_type, entity_type, entity_id, payload_json)
        SELECT
          lower(hex(randomblob(16))) AS id,
          {occurred_at} AS occurred_at,
          {run_id} AS run_id,
          {source} AS source,
          {event_type} AS event_type,
          {entity_type} AS entity_type,
          {entity_id} AS entity_id,
          {payload_json} AS payload_json
        FROM {legacy}
        """
    )

    # Drop legacy table
    conn.execute(f"DROP TABLE {legacy}")


def ensure_events_table(conn: sqlite3.Connection) -> None:
    # Caller controls transaction; still safe to run without one.
    _migrate_events_if_needed(conn)


@dataclass(frozen=True)
class EventContext:
    run_id: str
    source: str  # e.g. "cli"


class EventWriter:
    def __init__(self, conn: sqlite3.Connection, ctx: EventContext):
        self.conn = conn
        self.ctx = ctx
        ensure_events_table(self.conn)

    def emit(
        self,
        event_type: str,
        entity_type: str,
        entity_id: Optional[str],
        payload: Dict[str, Any],
        occurred_at: Optional[str] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        if occurred_at is None:
            occurred_at = ""

        self.conn.execute(
            """
            INSERT INTO events (id, occurred_at, run_id, source, event_type, entity_type, entity_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                occurred_at,
                self.ctx.run_id,
                self.ctx.source,
                event_type,
                entity_type,
                entity_id,
                canonical_json(payload),
            ),
        )
        return event_id
