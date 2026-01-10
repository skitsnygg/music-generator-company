#!/usr/bin/env python3
"""
CI fixture schema audit

Goal:
- Parse SQL strings in code (starting with db_helpers.py, optionally more files)
- Extract table/column references
- Compare against an SQLite DB file (fixtures/ci_db.sqlite)
- Print missing tables/columns and exit non-zero if anything is missing

This is intentionally heuristic: it catches the common patterns that break CI:
- SELECT ... FROM table [AS alias]
- JOIN table [AS alias]
- INSERT INTO table (col1, col2, ...)
- UPDATE table SET col=...
- WHERE alias.col / table.col
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


SQLITE_INTERNAL_TABLES = {
    "sqlite_master",
    "sqlite_schema",       # newer alias
    "sqlite_temp_master",
    "sqlite_temp_schema",
}


SQL_STRING_RE = re.compile(
    r"""(?:
        (?:^|[^A-Za-z0-9_])          # boundary
        (SELECT|INSERT|UPDATE|DELETE)\s
        .*?
    )""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Very tolerant: finds single-quoted or triple-quoted strings that contain SQL keywords.
POSSIBLE_SQL_LITERAL_RE = re.compile(
    r"""
    (?:r|f|rf|fr)?                  # possible prefixes
    (                               # capture full string literal (roughly)
        (?:
            '''.*?'''               # triple single
          | '([^'\\]|\\.|\\\n)*'    # single
          | "([^"\\]|\\.|\\\n)*"    # double
        )
    )
    """,
    re.VERBOSE | re.DOTALL,
)

FROM_JOIN_RE = re.compile(
    r"""
    \b(?:FROM|JOIN)\s+
    (?P<table>[A-Za-z_][A-Za-z0-9_]*)
    (?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?
    """,
    re.IGNORECASE | re.VERBOSE,
)

INSERT_INTO_RE = re.compile(
    r"""
    \bINSERT\s+INTO\s+
    (?P<table>[A-Za-z_][A-Za-z0-9_]*)
    \s*\((?P<cols>[^)]+)\)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

UPDATE_RE = re.compile(
    r"""
    \bUPDATE\s+
    (?P<table>[A-Za-z_][A-Za-z0-9_]*)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

SET_COL_RE = re.compile(
    r"""
    \bSET\s+(?P<assignments>.+?)
    (?:\bWHERE\b|$)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# captures alias.col patterns (including table.col)
ALIAS_COL_RE = re.compile(r"\b(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.(?P<col>[A-Za-z_][A-Za-z0-9_]*)\b")

# captures bare column lists in SELECT "col1, col2" (weak heuristic; only used when unambiguous)
SELECT_COLS_RE = re.compile(
    r"""
    \bSELECT\s+(?P<select>.+?)
    \bFROM\b
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class Ref:
    table: str
    column: str


def die(msg: str, code: int = 2) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_sql_candidates(py_text: str) -> List[str]:
    out: List[str] = []
    for m in POSSIBLE_SQL_LITERAL_RE.finditer(py_text):
        lit = m.group(1)
        # strip quotes
        if lit.startswith(("'''", '"""')):
            s = lit[3:-3]
        else:
            s = lit[1:-1]
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH)\b", s, flags=re.IGNORECASE):
            out.append(s)
    return out


def infer_table_aliases(sql: str) -> Dict[str, str]:
    """
    Returns alias->table mapping.
    Includes table->table mapping so "table.col" resolves.
    """
    aliases: Dict[str, str] = {}
    for m in FROM_JOIN_RE.finditer(sql):
        table = m.group("table")
        alias = m.group("alias") or table
        aliases[alias] = table
        aliases[table] = table
    return aliases


def infer_required_refs(sql: str) -> Tuple[Set[str], Set[Ref]]:
    """
    Returns (tables, refs) where refs are table+column pairs.
    """
    tables: Set[str] = set()
    refs: Set[Ref] = set()

    # tables/aliases
    aliases = infer_table_aliases(sql)
    tables.update(set(aliases.values()))

    # INSERT INTO table (cols...)
    for m in INSERT_INTO_RE.finditer(sql):
        table = m.group("table")
        tables.add(table)
        cols = [c.strip().strip('"').strip("'") for c in m.group("cols").split(",")]
        for c in cols:
            if c:
                refs.add(Ref(table=table, column=c))

    # UPDATE table SET col=...
    for m in UPDATE_RE.finditer(sql):
        table = m.group("table")
        tables.add(table)
        # Try to parse SET clause (best effort)
        set_m = SET_COL_RE.search(sql)
        if set_m:
            assigns = set_m.group("assignments")
            # split by commas not inside parentheses (simple version)
            parts = [p.strip() for p in assigns.split(",")]
            for p in parts:
                # col = ...
                left = p.split("=", 1)[0].strip()
                # allow alias.col too
                if "." in left:
                    a, c = left.split(".", 1)
                    t = aliases.get(a)
                    if t and c:
                        refs.add(Ref(table=t, column=c.strip()))
                else:
                    if left:
                        refs.add(Ref(table=table, column=left.strip()))

    # alias.col occurrences
    for m in ALIAS_COL_RE.finditer(sql):
        alias = m.group("alias")
        col = m.group("col")
        table = aliases.get(alias)
        if table:
            refs.add(Ref(table=table, column=col))

    # SELECT list bare columns heuristic:
    # Only count bare "col" tokens if there's exactly one table in FROM/JOIN (no ambiguity).
    select_m = SELECT_COLS_RE.search(sql)
    if select_m and len(set(aliases.values())) == 1:
        table = next(iter(set(aliases.values())))
        select_part = select_m.group("select")
        # remove obvious function calls and literals; keep simple identifiers
        # split by commas and take last token in "expr AS alias" patterns
        for piece in select_part.split(","):
            p = piece.strip()
            # skip wildcard and alias.*
            if p == "*" or p.endswith(".*"):
                continue
            # if already qualified, ALIAS_COL_RE handled it
            if "." in p:
                continue
            # handle "col AS something" or "col something"
            p = re.split(r"\bAS\b", p, flags=re.IGNORECASE)[0].strip()
            p = p.split()[-1] if p.split() else ""
            # drop functions "COUNT(" etc.
            if "(" in p or ")" in p:
                continue
            # keep plain identifiers
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p or ""):
                refs.add(Ref(table=table, column=p))

    return tables, refs


def load_db_schema(db_path: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}

        cols_by_table: Dict[str, Set[str]] = {}
        for t in tables:
            cur.execute(f"PRAGMA table_info({t})")
            cols = {r[1] for r in cur.fetchall()}  # r[1] is column name
            cols_by_table[t] = cols
        return tables, cols_by_table
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to fixture sqlite db (e.g. fixtures/ci_db.sqlite)")
    ap.add_argument(
        "--py",
        action="append",
        default=[],
        help="Python file to scan (repeatable). If omitted, defaults to src/mgc/db_helpers.py and src/mgc/events.py when present.",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        die(f"DB not found: {db_path}")

    default_candidates = [
        Path("src/mgc/db_helpers.py"),
        Path("src/mgc/events.py"),
    ]
    py_paths = [Path(p) for p in args.py] if args.py else [p for p in default_candidates if p.exists()]
    if not py_paths:
        die("No python files to scan. Provide --py paths.")

    # collect requirements
    req_tables: Set[str] = set()
    req_refs: Set[Ref] = set()

    for p in py_paths:
        txt = read_text(p)
        sqls = extract_sql_candidates(txt)
        for s in sqls:
            tables, refs = infer_required_refs(s)
            req_tables |= tables
            req_refs |= refs

    # compare against actual schema
    actual_tables, cols_by_table = load_db_schema(db_path)

    missing_tables = sorted(
        t for t in req_tables
        if t not in actual_tables and t not in SQLITE_INTERNAL_TABLES
    )

    missing_cols: Dict[str, Set[str]] = {}

    for r in req_refs:
        if r.table not in actual_tables:
            continue
        if r.column not in cols_by_table.get(r.table, set()):
            missing_cols.setdefault(r.table, set()).add(r.column)

    if not missing_tables and not missing_cols:
        print("OK: fixture DB satisfies inferred schema requirements.")
        print(f"Scanned: {', '.join(str(p) for p in py_paths)}")
        return 0

    print("MISSING SCHEMA ITEMS")
    print("--------------------")
    if missing_tables:
        print("Missing tables:")
        for t in missing_tables:
            print(f"  - {t}")

    if missing_cols:
        print("Missing columns:")
        for t in sorted(missing_cols):
            cols = ", ".join(sorted(missing_cols[t]))
            print(f"  - {t}: {cols}")

    print()
    print("Tip: If you *expect* a table/column but it's not being detected, add more --py files (anything that builds SQL).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
