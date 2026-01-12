from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _db_path(args: argparse.Namespace) -> str:
    # Use global --db if present (mgc.main hoists it), otherwise env, otherwise default.
    return str(getattr(args, "db", None) or os.environ.get("MGC_DB") or "data/db.sqlite")


def db_connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return []
    # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
    return [str(r[1]) for r in rows]


def _pick_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _wants_json(args: argparse.Namespace) -> bool:
    """
    JSON intent is controlled by:
      - global: mgc --json drops list ...
      - per-command: mgc drops list --format json
    We intentionally DO NOT define `--json` on the drops subparsers to avoid
    clobbering the global `args.json` when flags are hoisted.
    """
    if _as_bool(getattr(args, "json", False)):
        return True
    return str(getattr(args, "format", "")).lower() == "json"


def cmd_drops_list(args: argparse.Namespace) -> int:
    path = _db_path(args)
    limit = int(args.limit)
    ctx = (args.context or "").strip()
    seed = (args.seed or "").strip()

    con = db_connect(path)
    try:
        cols = table_columns(con, "drops")
        if not cols:
            sys.stdout.write(stable_json_dumps({"error": "drops table not found", "db": path}) + "\n")
            return 2

        ts_col = _pick_first(cols, ["ts", "created_at", "created_ts", "created", "created_on"])
        ctx_col = _pick_first(cols, ["context", "mood"])
        seed_col = _pick_first(cols, ["seed"])
        id_col = _pick_first(cols, ["id", "drop_id"]) or "id"

        where: List[str] = []
        params: List[Any] = []

        if ctx and ctx_col:
            where.append(f"{ctx_col} = ?")
            params.append(ctx)

        if seed and seed_col:
            where.append(f"{seed_col} = ?")
            params.append(seed)

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        order = f"{ts_col} DESC, {id_col} DESC" if ts_col else f"{id_col} DESC"

        sql = f"SELECT * FROM drops{where_sql} ORDER BY {order} LIMIT ?"
        params.append(limit)

        rows = con.execute(sql, tuple(params)).fetchall()
        items = [_row_to_dict(r) for r in rows]

        if _wants_json(args):
            sys.stdout.write(stable_json_dumps({"count": len(items), "items": items}) + "\n")
            return 0

        # table-ish output
        show_cols = [
            _pick_first(cols, ["id", "drop_id"]) or "id",
            ts_col or "ts",
            ctx_col or "context",
            seed_col or "seed",
            _pick_first(cols, ["run_id"]) or "run_id",
            _pick_first(cols, ["track_id"]) or "track_id",
            _pick_first(cols, ["marketing_batch_id", "batch_id"]) or "marketing_batch_id",
            _pick_first(cols, ["published_ts", "published_at"]) or "published_ts",
        ]

        print(
            "  ".join(
                [
                    c.ljust(36) if c in ("id", "drop_id", "run_id", "track_id", "marketing_batch_id") else c
                    for c in show_cols
                ]
            )
        )
        print("-" * 140)
        for it in items:
            parts: List[str] = []
            for c in show_cols:
                v = it.get(c)
                s = "" if v is None else str(v)
                if c in ("id", "drop_id", "run_id", "track_id", "marketing_batch_id"):
                    parts.append(s.ljust(36))
                else:
                    parts.append(s)
            print("  ".join(parts))

        return 0
    finally:
        con.close()


def cmd_drops_show(args: argparse.Namespace) -> int:
    path = _db_path(args)
    drop_id = str(args.id).strip()

    con = db_connect(path)
    try:
        cols = table_columns(con, "drops")
        if not cols:
            sys.stdout.write(stable_json_dumps({"error": "drops table not found", "db": path}) + "\n")
            return 2

        id_col = _pick_first(cols, ["id", "drop_id"]) or "id"
        row = con.execute(f"SELECT * FROM drops WHERE {id_col} = ? LIMIT 1", (drop_id,)).fetchone()
        if not row:
            sys.stdout.write(stable_json_dumps({"error": "drop not found", "id": drop_id}) + "\n")
            return 1

        obj = _row_to_dict(row)

        if _wants_json(args):
            sys.stdout.write(stable_json_dumps(obj) + "\n")
            return 0

        # pretty output
        for k in sorted(obj.keys()):
            v = obj[k]
            if isinstance(v, str) and k.endswith("_json"):
                try:
                    parsed = json.loads(v) if v.strip() else None
                except Exception:
                    parsed = None
                if isinstance(parsed, (dict, list)):
                    print(f"{k}:")
                    print(json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True))
                    continue
            print(f"{k}: {v}")
        return 0
    finally:
        con.close()


def register_drops_subcommand(subparsers: argparse._SubParsersAction) -> None:
    drops = subparsers.add_parser("drops", help="Inspect drops (daily releases)")
    drops.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"), help="SQLite DB path")

    drops_sub = drops.add_subparsers(dest="drops_cmd", required=True)

    ls = drops_sub.add_parser("list", help="List drops")
    ls.add_argument("--limit", type=int, default=20)
    ls.add_argument("--context", default=None)
    ls.add_argument("--seed", default=None)
    ls.add_argument("--format", choices=["table", "json"], default="table")
    ls.set_defaults(func=cmd_drops_list)

    sh = drops_sub.add_parser("show", help="Show a single drop")
    sh.add_argument("id", help="Drop id")
    sh.add_argument("--format", choices=["pretty", "json"], default="pretty")
    sh.set_defaults(func=cmd_drops_show)
