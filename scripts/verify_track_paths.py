#!/usr/bin/env python3
"""
scripts/verify_track_paths.py

Verify that track artifact paths referenced by the DB exist on disk.

This script is used in CI to catch cases where the DB points at audio files that
were not checked into the repo / not produced by rebuild steps.

Important: the repo has evolved, and track primary key column may be either:
- track_id  (current)
- id        (legacy)

This script adapts to both.

Usage:
  python scripts/verify_track_paths.py --db <path> --repo-root <dir> [--limit N] [--allow-missing-zero-byte]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def _table_info(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r["name"]) for r in rows]


def _pick_first(existing: List[str], *candidates: str) -> Optional[str]:
    s = set(existing)
    for c in candidates:
        if c in s:
            return c
    return None


def _fetch_tracks(
    con: sqlite3.Connection, *, limit: int
) -> Tuple[str, str, List[sqlite3.Row]]:
    cols = _table_info(con, "tracks")

    # Primary key column (legacy vs current)
    id_col = _pick_first(cols, "track_id", "id")
    if not id_col:
        raise SystemExit(
            f"tracks table missing expected id column. Columns: {cols}"
        )

    # Path column (current)
    path_col = _pick_first(cols, "artifact_path", "path", "file_path", "filepath")
    if not path_col:
        raise SystemExit(
            f"tracks table missing expected path column. Columns: {cols}"
        )

    # Optional metadata columns we may print if present
    ts_col = _pick_first(cols, "ts", "created_ts", "created_at")
    title_col = _pick_first(cols, "title", "name")
    provider_col = _pick_first(cols, "provider")

    select_cols = [id_col, path_col]
    for c in (ts_col, title_col, provider_col):
        if c and c not in select_cols:
            select_cols.append(c)

    sql = f"""
      SELECT {", ".join(select_cols)}
      FROM tracks
      ORDER BY {id_col} ASC
      LIMIT ?
    """.strip()

    rows = con.execute(sql, (int(limit),)).fetchall()
    return id_col, path_col, rows


def _resolve_artifact(repo_root: Path, artifact_path: str) -> Path:
    p = Path(artifact_path)

    # If DB stores absolute paths, respect them.
    if p.is_absolute():
        return p

    # Common case: paths stored relative to repo root.
    return (repo_root / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--repo-root", required=True, help="Repo root (for resolving relative artifact paths)")
    ap.add_argument("--limit", type=int, default=5000, help="Max tracks to check")
    ap.add_argument(
        "--allow-missing-zero-byte",
        action="store_true",
        help="If a path exists but is 0 bytes, do not fail (default: fail).",
    )
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()

    if not repo_root.exists():
        raise SystemExit(f"repo-root not found: {repo_root}")

    with _connect(db_path) as con:
        # Ensure tracks table exists
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracks' LIMIT 1"
        ).fetchone()
        if row is None:
            print("OK (no tracks table)", file=sys.stderr)
            return 0

        id_col, path_col, tracks = _fetch_tracks(con, limit=args.limit)

    missing: List[str] = []
    zero: List[str] = []

    for r in tracks:
        track_id = str(r[id_col])
        artifact_path = (r[path_col] or "").strip()
        if not artifact_path:
            missing.append(f"{track_id}:<empty_path>")
            continue

        resolved = _resolve_artifact(repo_root, artifact_path)
        if not resolved.exists():
            missing.append(f"{track_id}:{artifact_path}")
            continue

        try:
            size = resolved.stat().st_size
        except OSError:
            missing.append(f"{track_id}:{artifact_path}")
            continue

        if size == 0:
            zero.append(f"{track_id}:{artifact_path}")
            if not args.allow_missing_zero_byte:
                # treated as failure by default
                continue

    if missing or (zero and not args.allow_missing_zero_byte):
        print("ERROR: track artifact verification failed", file=sys.stderr)
        print(f"DB={db_path}", file=sys.stderr)
        print(f"repo_root={repo_root}", file=sys.stderr)
        print(f"tracks_checked={len(tracks)} id_col={id_col} path_col={path_col}", file=sys.stderr)

        if missing:
            print(f"missing_files={len(missing)}", file=sys.stderr)
            for m in missing[:50]:
                print(f"  MISSING {m}", file=sys.stderr)
            if len(missing) > 50:
                print(f"  ... ({len(missing)-50} more)", file=sys.stderr)

        if zero and not args.allow_missing_zero_byte:
            print(f"zero_byte_files={len(zero)}", file=sys.stderr)
            for z in zero[:50]:
                print(f"  ZERO {z}", file=sys.stderr)
            if len(zero) > 50:
                print(f"  ... ({len(zero)-50} more)", file=sys.stderr)

        return 1

    print(f"OK tracks_checked={len(tracks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
