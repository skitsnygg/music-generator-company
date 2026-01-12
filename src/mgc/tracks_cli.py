from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _table_cols(con: sqlite3.Connection, table: str) -> List[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows]


def _has_col(cols: List[str], name: str) -> bool:
    return name in cols


def _q1(con: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> Optional[sqlite3.Row]:
    cur = con.execute(sql, params)
    row = cur.fetchone()
    return row


def _qn(con: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
    cur = con.execute(sql, params)
    return cur.fetchall()


@dataclass(frozen=True)
class TrackStats:
    count: int
    missing_title: int
    missing_duration: int
    negative_duration: int
    non_numeric_duration: int
    min_duration: Optional[float]
    max_duration: Optional[float]
    avg_duration: Optional[float]


def _compute_stats(con: sqlite3.Connection) -> TrackStats:
    cols = _table_cols(con, "tracks")
    if not cols:
        raise ValueError("tracks_table_missing")

    title_expr = None
    if _has_col(cols, "title"):
        title_expr = "title"
    elif _has_col(cols, "name"):
        title_expr = "name"

    if title_expr is None:
        # we can still compute counts, but strict will fail
        title_expr = "NULL"

    duration_expr = "duration_sec" if _has_col(cols, "duration_sec") else "NULL"

    # total count
    row = _q1(con, "SELECT COUNT(*) AS n FROM tracks")
    count = int(row["n"]) if row else 0

    # missing title (NULL or empty)
    row = _q1(
        con,
        f"SELECT SUM(CASE WHEN {title_expr} IS NULL OR TRIM({title_expr})='' THEN 1 ELSE 0 END) AS n FROM tracks",
    )
    missing_title = int(row["n"]) if row and row["n"] is not None else 0

    # duration checks:
    # - missing duration
    row = _q1(
        con,
        f"SELECT SUM(CASE WHEN {duration_expr} IS NULL THEN 1 ELSE 0 END) AS n FROM tracks",
    )
    missing_duration = int(row["n"]) if row and row["n"] is not None else 0

    # - negative duration
    row = _q1(
        con,
        f"SELECT SUM(CASE WHEN {duration_expr} IS NOT NULL AND CAST({duration_expr} AS REAL) < 0 THEN 1 ELSE 0 END) AS n FROM tracks",
    )
    negative_duration = int(row["n"]) if row and row["n"] is not None else 0

    # - non-numeric duration (best-effort): values that cast to real = 0 but original isn't 0-ish.
    # SQLite CAST('abc' AS REAL) => 0.0, so detect "bad" by checking for non-empty string that casts to 0 but isn't '0'/'0.0'
    non_numeric_duration = 0
    try:
        row = _q1(
            con,
            f"""
            SELECT SUM(
              CASE
                WHEN {duration_expr} IS NULL THEN 0
                WHEN typeof({duration_expr}) IN ('integer','real') THEN 0
                WHEN TRIM(CAST({duration_expr} AS TEXT)) IN ('0','0.0','0.00') THEN 0
                WHEN CAST({duration_expr} AS REAL) = 0.0 THEN 1
                ELSE 0
              END
            ) AS n
            FROM tracks
            """,
        )
        non_numeric_duration = int(row["n"]) if row and row["n"] is not None else 0
    except Exception:
        non_numeric_duration = 0

    # numeric aggregates (ignore NULL)
    row = _q1(
        con,
        f"SELECT MIN(CAST({duration_expr} AS REAL)) AS mn, MAX(CAST({duration_expr} AS REAL)) AS mx, AVG(CAST({duration_expr} AS REAL)) AS av FROM tracks WHERE {duration_expr} IS NOT NULL",
    )
    if row and row["mn"] is not None:
        min_duration = float(row["mn"])
        max_duration = float(row["mx"])
        avg_duration = float(row["av"]) if row["av"] is not None else None
    else:
        min_duration = None
        max_duration = None
        avg_duration = None

    return TrackStats(
        count=count,
        missing_title=missing_title,
        missing_duration=missing_duration,
        negative_duration=negative_duration,
        non_numeric_duration=non_numeric_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        avg_duration=avg_duration,
    )


def tracks_stats_cmd(args: argparse.Namespace) -> int:
    con = _connect(args.db)
    try:
        stats = _compute_stats(con)
    finally:
        con.close()

    ok = True
    problems: List[str] = []

    # strict rules
    if args.strict:
        if stats.count <= 0:
            ok = False
            problems.append("no_tracks")
        if stats.missing_title > 0:
            ok = False
            problems.append(f"missing_title={stats.missing_title}")
        if stats.missing_duration > 0:
            ok = False
            problems.append(f"missing_duration={stats.missing_duration}")
        if stats.negative_duration > 0:
            ok = False
            problems.append(f"negative_duration={stats.negative_duration}")
        if stats.non_numeric_duration > 0:
            ok = False
            problems.append(f"non_numeric_duration={stats.non_numeric_duration}")

        if args.min_tracks is not None and stats.count < int(args.min_tracks):
            ok = False
            problems.append(f"min_tracks_expected={int(args.min_tracks)} got={stats.count}")

    out: Dict[str, Any] = {
        "ok": bool(ok),
        "count": stats.count,
        "missing_title": stats.missing_title,
        "missing_duration": stats.missing_duration,
        "negative_duration": stats.negative_duration,
        "non_numeric_duration": stats.non_numeric_duration,
        "min_duration_sec": stats.min_duration,
        "max_duration_sec": stats.max_duration,
        "avg_duration_sec": stats.avg_duration,
        "problems": problems,
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"ok: {str(ok).lower()}")
        print(f"count: {stats.count}")
        print(f"missing_title: {stats.missing_title}")
        print(f"missing_duration: {stats.missing_duration}")
        print(f"negative_duration: {stats.negative_duration}")
        print(f"non_numeric_duration: {stats.non_numeric_duration}")
        print(f"min_duration_sec: {stats.min_duration}")
        print(f"max_duration_sec: {stats.max_duration}")
        print(f"avg_duration_sec: {stats.avg_duration}")
        if problems:
            print(f"problems: {', '.join(problems)}")

    return 0 if ok else 2


def register_tracks_stats_subcommand(subparsers) -> None:
    p = subparsers.add_parser("stats", help="Track library stats")
    p.add_argument("--db", default="data/db.sqlite")
    p.add_argument("--json", action="store_true")
    p.add_argument("--strict", action="store_true", help="Fail if any required fields are missing/invalid")
    p.add_argument("--min-tracks", type=int, default=None, help="Strict: minimum number of tracks required")
    p.set_defaults(func=tracks_stats_cmd)
