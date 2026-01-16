#!/usr/bin/env python3
"""
src/mgc/billing_cli.py

Billing (v0): users, API tokens, entitlements.

Design goals:
- SQLite-only, no external deps
- deterministic-friendly (accept --now / --token overrides)
- JSON mode friendly

Schema is created by migrations:
  scripts/migrations/0002_billing.sql
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_now(now_iso: Optional[str]) -> datetime:
    if not now_iso:
        return datetime.now(timezone.utc)
    # Accept "2020-01-01T00:00:00+00:00" and "Z"
    s = now_iso.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DB:
    path: Path

    def connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise FileNotFoundError(f"DB not found: {self.path}")
        con = sqlite3.connect(str(self.path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON;")
        return con


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return row is not None


def _require_tables(con: sqlite3.Connection) -> None:
    missing = []
    for t in ("billing_users", "billing_tokens", "billing_entitlements"):
        if not _table_exists(con, t):
            missing.append(t)
    if missing:
        raise SystemExit(
            "Billing tables missing: "
            + ", ".join(missing)
            + ". Did you run scripts/migrate_db.py (0002_billing.sql)?"
        )


def _user_get(con: sqlite3.Connection, user_id: str) -> Optional[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM billing_users WHERE user_id = ? LIMIT 1", (user_id,)
    ).fetchone()


def _user_upsert(con: sqlite3.Connection, user_id: str, email: Optional[str], created_ts: str) -> None:
    # sqlite upsert
    con.execute(
        """
        INSERT INTO billing_users(user_id, email, created_ts)
        VALUES(?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          email=COALESCE(excluded.email, billing_users.email)
        """,
        (user_id, email, created_ts),
    )


def _entitlement_active_row(
    con: sqlite3.Connection, *, user_id: str, now_ts: str
) -> Optional[sqlite3.Row]:
    # Active if starts_ts <= now and (ends_ts is null or ends_ts > now)
    return con.execute(
        """
        SELECT *
        FROM billing_entitlements
        WHERE user_id = ?
          AND starts_ts <= ?
          AND (ends_ts IS NULL OR ends_ts > ?)
        ORDER BY starts_ts DESC
        LIMIT 1
        """,
        (user_id, now_ts, now_ts),
    ).fetchone()


def cmd_billing_users_add(args: argparse.Namespace) -> int:
    db = DB(Path(args.db))
    created_ts = args.created_ts or utc_now_iso()
    out: Dict[str, Any] = {"ok": True, "cmd": "billing.users.add", "user_id": args.user_id}

    with db.connect() as con:
        _require_tables(con)
        _user_upsert(con, args.user_id, args.email, created_ts)
        con.commit()
        row = _user_get(con, args.user_id)
        out["user"] = dict(row) if row else None

    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK user_id={args.user_id}")
    return 0


def cmd_billing_users_list(args: argparse.Namespace) -> int:
    db = DB(Path(args.db))
    with db.connect() as con:
        _require_tables(con)
        rows = con.execute(
            "SELECT * FROM billing_users ORDER BY created_ts DESC LIMIT ?",
            (int(args.limit),),
        ).fetchall()
    data = [dict(r) for r in rows]
    if args.json:
        print(stable_json({"ok": True, "cmd": "billing.users.list", "users": data}), end="")
    else:
        for u in data:
            print(f"{u.get('user_id')}  {u.get('email','')}".rstrip())
    return 0


def cmd_billing_tokens_mint(args: argparse.Namespace) -> int:
    db = DB(Path(args.db))
    created_ts = args.created_ts or utc_now_iso()

    # Token value may be provided for deterministic tests; otherwise random.
    token_value = args.token or secrets.token_urlsafe(32)
    token_sha = sha256_hex(token_value)

    with db.connect() as con:
        _require_tables(con)

        # Ensure user exists
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, created_ts)

        con.execute(
            """
            INSERT INTO billing_tokens(token_sha256, user_id, created_ts, label)
            VALUES(?, ?, ?, ?)
            """,
            (token_sha, args.user_id, created_ts, args.label),
        )
        con.commit()

    out: Dict[str, Any] = {
        "ok": True,
        "cmd": "billing.tokens.mint",
        "user_id": args.user_id,
        "token_sha256": token_sha,
        "created_ts": created_ts,
        "label": args.label,
    }

    # Only show the raw token when explicitly requested.
    if args.show_token:
        out["token"] = token_value

    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK token_sha256={token_sha} user_id={args.user_id}")
        if args.show_token:
            print(token_value)
    return 0


def cmd_billing_tokens_list(args: argparse.Namespace) -> int:
    db = DB(Path(args.db))
    with db.connect() as con:
        _require_tables(con)
        if args.user_id:
            rows = con.execute(
                """
                SELECT * FROM billing_tokens
                WHERE user_id = ?
                ORDER BY created_ts DESC
                LIMIT ?
                """,
                (args.user_id, int(args.limit)),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM billing_tokens ORDER BY created_ts DESC LIMIT ?",
                (int(args.limit),),
            ).fetchall()
    data = [dict(r) for r in rows]
    if args.json:
        print(stable_json({"ok": True, "cmd": "billing.tokens.list", "tokens": data}), end="")
    else:
        for t in data:
            print(f"{t.get('token_sha256')}  user_id={t.get('user_id')}  {t.get('label','')}".rstrip())
    return 0


def cmd_billing_tokens_revoke(args: argparse.Namespace) -> int:
    db = DB(Path(args.db))
    token_sha = args.token_sha256 or sha256_hex(args.token)
    with db.connect() as con:
        _require_tables(con)
        cur = con.execute("DELETE FROM billing_tokens WHERE token_sha256 = ?", (token_sha,))
        con.commit()
    out = {"ok": True, "cmd": "billing.tokens.revoke", "token_sha256": token_sha, "deleted": cur.rowcount}
    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK deleted={cur.rowcount} token_sha256={token_sha}")
    return 0


def cmd_billing_entitlements_grant(args: argparse.Namespace) -> int:
    db = DB(Path(args.db))
    starts_ts = args.starts_ts or utc_now_iso()

    meta_json = None
    if args.meta_json:
        # validate JSON
        json.loads(args.meta_json)
        meta_json = args.meta_json

    with db.connect() as con:
        _require_tables(con)

        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, utc_now_iso())

        con.execute(
            """
            INSERT INTO billing_entitlements(entitlement_id, user_id, tier, starts_ts, ends_ts, source, meta_json)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (args.entitlement_id, args.user_id, args.tier, starts_ts, args.ends_ts, args.source, meta_json),
        )
        con.commit()

    out = {
        "ok": True,
        "cmd": "billing.entitlements.grant",
        "entitlement_id": args.entitlement_id,
        "user_id": args.user_id,
        "tier": args.tier,
        "starts_ts": starts_ts,
        "ends_ts": args.ends_ts,
        "source": args.source,
    }
    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK entitlement_id={args.entitlement_id} user_id={args.user_id} tier={args.tier}")
    return 0


def cmd_billing_entitlements_list(args: argparse.Namespace) -> int:
    db = DB(Path(args.db))
    with db.connect() as con:
        _require_tables(con)
        if args.user_id:
            rows = con.execute(
                """
                SELECT * FROM billing_entitlements
                WHERE user_id = ?
                ORDER BY starts_ts DESC
                LIMIT ?
                """,
                (args.user_id, int(args.limit)),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM billing_entitlements ORDER BY starts_ts DESC LIMIT ?",
                (int(args.limit),),
            ).fetchall()
    data = [dict(r) for r in rows]
    if args.json:
        print(stable_json({"ok": True, "cmd": "billing.entitlements.list", "entitlements": data}), end="")
    else:
        for e in data:
            print(
                f"{e.get('entitlement_id')}  user_id={e.get('user_id')}  tier={e.get('tier')}  "
                f"starts={e.get('starts_ts')}  ends={e.get('ends_ts') or ''}  source={e.get('source')}"
            )
    return 0


def cmd_billing_check(args: argparse.Namespace) -> int:
    db = DB(Path(args.db))
    now_dt = parse_now(args.now)
    now_ts = now_dt.isoformat(timespec="seconds")

    token_sha = sha256_hex(args.token)

    with db.connect() as con:
        _require_tables(con)

        tok = con.execute(
            "SELECT * FROM billing_tokens WHERE token_sha256 = ? LIMIT 1", (token_sha,)
        ).fetchone()

        if tok is None:
            out = {"ok": False, "cmd": "billing.check", "reason": "invalid_token"}
            if args.json:
                print(stable_json(out), end="")
            else:
                print("DENY invalid_token")
            return 2

        user_id = str(tok["user_id"])
        ent = _entitlement_active_row(con, user_id=user_id, now_ts=now_ts)
        tier = str(ent["tier"]) if ent is not None else "free"

    out = {
        "ok": True,
        "cmd": "billing.check",
        "user_id": user_id,
        "tier": tier,
        "now": now_ts,
        "token_sha256": token_sha,
        "active_entitlement": dict(ent) if ent is not None else None,
    }
    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK user_id={user_id} tier={tier}")
    return 0


def register_billing_subcommand(sub: argparse._SubParsersAction) -> None:
    billing = sub.add_parser("billing", help="Billing tools (users/tokens/entitlements)")
    bs = billing.add_subparsers(dest="billing_cmd", required=True)

    # users
    users = bs.add_parser("users", help="Billing users")
    us = users.add_subparsers(dest="users_cmd", required=True)

    ua = us.add_parser("add", help="Create/update a billing user")
    ua.add_argument("--db", required=True)
    ua.add_argument("user_id")
    ua.add_argument("--email", default=None)
    ua.add_argument("--created-ts", default=None)
    ua.add_argument("--json", action="store_true")
    ua.set_defaults(func=cmd_billing_users_add)

    ul = us.add_parser("list", help="List billing users")
    ul.add_argument("--db", required=True)
    ul.add_argument("--limit", type=int, default=50)
    ul.add_argument("--json", action="store_true")
    ul.set_defaults(func=cmd_billing_users_list)

    # tokens
    tokens = bs.add_parser("tokens", help="API tokens")
    ts = tokens.add_subparsers(dest="tokens_cmd", required=True)

    tm = ts.add_parser("mint", help="Mint a token for a user")
    tm.add_argument("--db", required=True)
    tm.add_argument("user_id")
    tm.add_argument("--email", default=None, help="If user doesn't exist, create with this email")
    tm.add_argument("--label", default=None)
    tm.add_argument("--created-ts", default=None)
    tm.add_argument("--token", default=None, help="Provide token value (deterministic tests). Otherwise random.")
    tm.add_argument("--show-token", action="store_true", help="Include raw token in output")
    tm.add_argument("--json", action="store_true")
    tm.set_defaults(func=cmd_billing_tokens_mint)

    tl = ts.add_parser("list", help="List tokens")
    tl.add_argument("--db", required=True)
    tl.add_argument("--user-id", default=None)
    tl.add_argument("--limit", type=int, default=50)
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(func=cmd_billing_tokens_list)

    tr = ts.add_parser("revoke", help="Revoke a token")
    tr.add_argument("--db", required=True)
    g = tr.add_mutually_exclusive_group(required=True)
    g.add_argument("--token-sha256", default=None)
    g.add_argument("--token", default=None)
    tr.add_argument("--json", action="store_true")
    tr.set_defaults(func=cmd_billing_tokens_revoke)

    # entitlements
    ents = bs.add_parser("entitlements", help="Access entitlements")
    es = ents.add_subparsers(dest="ents_cmd", required=True)

    eg = es.add_parser("grant", help="Grant an entitlement")
    eg.add_argument("--db", required=True)
    eg.add_argument("entitlement_id")
    eg.add_argument("user_id")
    eg.add_argument("tier", choices=["free", "supporter", "pro"])
    eg.add_argument("--starts-ts", default=None)
    eg.add_argument("--ends-ts", default=None)
    eg.add_argument("--source", default="manual")
    eg.add_argument("--meta-json", default=None)
    eg.add_argument("--email", default=None, help="If user doesn't exist, create with this email")
    eg.add_argument("--json", action="store_true")
    eg.set_defaults(func=cmd_billing_entitlements_grant)

    el = es.add_parser("list", help="List entitlements")
    el.add_argument("--db", required=True)
    el.add_argument("--user-id", default=None)
    el.add_argument("--limit", type=int, default=50)
    el.add_argument("--json", action="store_true")
    el.set_defaults(func=cmd_billing_entitlements_list)

    # check
    ck = bs.add_parser("check", help="Validate token and return tier")
    ck.add_argument("--db", required=True)
    ck.add_argument("--token", required=True)
    ck.add_argument("--now", default=None, help="Override current time (ISO).")
    ck.add_argument("--json", action="store_true")
    ck.set_defaults(func=cmd_billing_check)
