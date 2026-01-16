#!/usr/bin/env python3
"""
scripts/migrate_db.py

Apply SQL migrations in scripts/migrations/ to a SQLite DB.

Policy:
- Uses schema_migrations table to apply each migration exactly once.
- Migrations are applied in filename-sorted order.
- DB path is provided via MGC_DB. Relative paths are resolved to an absolute path
  so the script can be run from any working directory.

Optional determinism helper:
- If MGC_MIGRATE_NOW is set (ISO timestamp), it is used for applied_at instead of
  the current time. (Useful for deterministic CI fixtures.)
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def utc_now_iso() -> str:
    override = (os.environ.get("MGC_MIGRATE_NOW") or "").strip()
    if override:
        # Accept Z suffix as UTC
        s = override.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError as e:
            die(f"Invalid MGC_MIGRATE_NOW (expected ISO timestamp): {override} ({e})", 2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def list_migrations(migrations_dir: Path) -> List[Path]:
    if not migrations_dir.exists() or not migrations_dir.is_dir():
        die(f"migrations dir missing: {migrations_dir}", 2)
    return sorted([p for p in migrations_dir.glob("*.sql") if p.is_file()])


def ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL
        );
        """
    )


def get_applied(conn: sqlite3.Connection) -> set[str]:
    ensure_schema_migrations(conn)
    rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC").fetchall()
    return {str(r[0]) for r in rows}


def parse_version(path: Path) -> str:
    # version is the filename like 0002_billing.sql
    return path.name


def apply_one(conn: sqlite3.Connection, path: Path) -> None:
    sql = read_sql(path).strip()
    version = parse_version(path)

    # Allow empty migration files (still gets recorded)
    if sql:
        conn.executescript(sql)

    conn.execute(
        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (version, utc_now_iso()),
    )


def main() -> int:
    db_env = (os.environ.get("MGC_DB") or "").strip()
    if not db_env:
        die("MGC_DB not set", 2)

    # This is the key fix: run-from-anywhere safety.
    db_path = Path(db_env).expanduser().resolve()

    if not db_path.exists():
        die(f"DB not found: {db_path}", 2)

    migrations_dir = Path(__file__).resolve().parent / "migrations"
    migrations = list_migrations(migrations_dir)

    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys=ON;")
        ensure_schema_migrations(con)

        applied = get_applied(con)
        to_apply: List[Tuple[str, Path]] = []
        for m in migrations:
            v = parse_version(m)
            if v not in applied:
                to_apply.append((v, m))

        # Apply in a transaction; rollback on failure.
        try:
            con.execute("BEGIN;")
            for _, m in to_apply:
                apply_one(con, m)
            con.commit()
        except Exception:
            con.rollback()
            raise

        print(f"OK: applied {len(to_apply)} migrations to {db_path}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
