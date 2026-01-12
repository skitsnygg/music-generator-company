#!/usr/bin/env python3
# scripts/migrate_db.py

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

def die(msg: str, code: int = 2) -> None:
    raise SystemExit(f"{msg} (code={code})")

def main() -> int:
    db = os.environ.get("MGC_DB")
    if not db:
        die("set MGC_DB to your sqlite path")

    db_path = Path(db)
    if not db_path.exists():
        die(f"DB not found: {db_path}")

    migrations_dir = Path("scripts/migrations")
    if not migrations_dir.exists():
        die(f"Missing migrations dir: {migrations_dir}")

    sql_files = sorted(migrations_dir.glob("*.sql"))
    if not sql_files:
        die(f"No migrations found in {migrations_dir}")

    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        for p in sql_files:
            sql = p.read_text(encoding="utf-8")
            con.executescript(sql)
        con.commit()
    finally:
        con.close()

    print(f"OK: applied {len(sql_files)} migrations to {db_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
