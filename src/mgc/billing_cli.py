#!/usr/bin/env python3
"""
src/mgc/billing_cli.py

Billing (v0): users, API tokens, entitlements.

Token UX polish:
- billing whoami
- billing tokens rotate

Important CLI rule:
- Billing commands DO NOT take --db on the subcommand (argparse footgun).
- Billing uses global mgc.main --db, with optional override --billing-db.

Schema created by:
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
    s = now_iso.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


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


def _resolve_db_path(args: argparse.Namespace) -> Path:
    raw = (getattr(args, "billing_db", None) or getattr(args, "db", None) or "").strip()
    if not raw:
        raise SystemExit("DB not set. Pass global --db to mgc.main or use --billing-db.")
    return Path(raw).expanduser().resolve()


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
            + ". Did you run migrations (0002_billing.sql)?"
        )


def _user_get(con: sqlite3.Connection, user_id: str) -> Optional[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM billing_users WHERE user_id = ? LIMIT 1", (user_id,)
    ).fetchone()


def _user_upsert(con: sqlite3.Connection, user_id: str, email: Optional[str], created_ts: str) -> None:
    con.execute(
        """
        INSERT INTO billing_users(user_id, email, created_ts)
        VALUES(?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          email=COALESCE(excluded.email, billing_users.email)
        """,
        (user_id, email, created_ts),
    )


def _entitlement_active_row(con: sqlite3.Connection, *, user_id: str, now_ts: str) -> Optional[sqlite3.Row]:
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


def _token_row_by_sha(con: sqlite3.Connection, token_sha: str) -> Optional[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM billing_tokens WHERE token_sha256 = ? LIMIT 1", (token_sha,)
    ).fetchone()


# -----------------------------
# users
# -----------------------------

def cmd_billing_users_add(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    created_ts = args.created_ts or utc_now_iso()

    with db.connect() as con:
        _require_tables(con)
        _user_upsert(con, args.user_id, args.email, created_ts)
        con.commit()
        row = _user_get(con, args.user_id)

    out: Dict[str, Any] = {"ok": True, "cmd": "billing.users.add", "user_id": args.user_id, "user": dict(row) if row else None}
    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK user_id={args.user_id}")
    return 0


def cmd_billing_users_list(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
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


# -----------------------------
# tokens
# -----------------------------

def cmd_billing_tokens_mint(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    created_ts = args.created_ts or utc_now_iso()

    token_value = args.token or secrets.token_urlsafe(32)
    token_sha = sha256_hex(token_value)

    with db.connect() as con:
        _require_tables(con)
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
    db = DB(_resolve_db_path(args))
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
    db = DB(_resolve_db_path(args))
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


def cmd_billing_tokens_rotate(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    created_ts = args.created_ts or utc_now_iso()

    new_token_value = args.token or secrets.token_urlsafe(32)
    new_token_sha = sha256_hex(new_token_value)

    revoke_sha: Optional[str] = None
    if args.revoke_token_sha256:
        revoke_sha = args.revoke_token_sha256
    elif args.revoke_token:
        revoke_sha = sha256_hex(args.revoke_token)

    with db.connect() as con:
        _require_tables(con)
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, created_ts)

        con.execute(
            """
            INSERT INTO billing_tokens(token_sha256, user_id, created_ts, label)
            VALUES(?, ?, ?, ?)
            """,
            (new_token_sha, args.user_id, created_ts, args.label),
        )

        revoked = 0
        if revoke_sha:
            cur = con.execute("DELETE FROM billing_tokens WHERE token_sha256 = ?", (revoke_sha,))
            revoked = cur.rowcount

        con.commit()

    out: Dict[str, Any] = {
        "ok": True,
        "cmd": "billing.tokens.rotate",
        "user_id": args.user_id,
        "token_sha256": new_token_sha,
        "created_ts": created_ts,
        "label": args.label,
        "revoked_token_sha256": revoke_sha,
        "revoked_count": revoked,
    }
    if args.show_token:
        out["token"] = new_token_value

    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK token_sha256={new_token_sha} user_id={args.user_id} revoked={revoked}")
        if args.show_token:
            print(new_token_value)
    return 0


# -----------------------------
# entitlements
# -----------------------------

def cmd_billing_entitlements_grant(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    starts_ts = args.starts_ts or utc_now_iso()

    meta_json = None
    if args.meta_json:
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


def cmd_billing_entitlements_grant_user(args: argparse.Namespace) -> int:
    """
    Convenience: grant without forcing the caller to supply entitlement_id.

    Usage:
      mgc billing entitlements grant-user <user_id> <tier> [--source manual] [--starts-ts ...] [--ends-ts ...]
    """
    db = DB(_resolve_db_path(args))
    starts_ts = args.starts_ts or utc_now_iso()

    meta_json = None
    if args.meta_json:
        json.loads(args.meta_json)
        meta_json = args.meta_json

    entitlement_id = args.entitlement_id or secrets.token_hex(16)

    with db.connect() as con:
        _require_tables(con)
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, utc_now_iso())

        con.execute(
            """
            INSERT INTO billing_entitlements(entitlement_id, user_id, tier, starts_ts, ends_ts, source, meta_json)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (entitlement_id, args.user_id, args.tier, starts_ts, args.ends_ts, args.source, meta_json),
        )
        con.commit()

    out = {
        "ok": True,
        "cmd": "billing.entitlements.grant-user",
        "entitlement_id": entitlement_id,
        "user_id": args.user_id,
        "tier": args.tier,
        "starts_ts": starts_ts,
        "ends_ts": args.ends_ts,
        "source": args.source,
    }
    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK entitlement_id={entitlement_id} user_id={args.user_id} tier={args.tier}")
    return 0


def cmd_billing_entitlements_revoke(args: argparse.Namespace) -> int:
    """
    Revoke entitlements by setting ends_ts.

    Default behavior:
      - If --entitlement-id is provided: revoke that exact row.
      - Else: revoke ALL active entitlements for user_id (ends_ts is NULL OR ends_ts > now),
              by setting ends_ts = now.
    """
    db = DB(_resolve_db_path(args))
    now_dt = parse_now(args.now)
    now_ts = now_dt.isoformat(timespec="seconds")

    with db.connect() as con:
        _require_tables(con)

        if args.entitlement_id:
            cur = con.execute(
                """
                UPDATE billing_entitlements
                   SET ends_ts = ?
                 WHERE entitlement_id = ?
                """,
                (now_ts, args.entitlement_id),
            )
            revoked = cur.rowcount
        else:
            cur = con.execute(
                """
                UPDATE billing_entitlements
                   SET ends_ts = ?
                 WHERE user_id = ?
                   AND (ends_ts IS NULL OR ends_ts > ?)
                """,
                (now_ts, args.user_id, now_ts),
            )
            revoked = cur.rowcount

        con.commit()

    out = {
        "ok": True,
        "cmd": "billing.entitlements.revoke",
        "user_id": args.user_id,
        "entitlement_id": args.entitlement_id,
        "now": now_ts,
        "revoked": revoked,
    }
    if args.json:
        print(stable_json(out), end="")
    else:
        if args.entitlement_id:
            print(f"OK revoked={revoked} entitlement_id={args.entitlement_id}")
        else:
            print(f"OK revoked={revoked} user_id={args.user_id}")
    return 0


def cmd_billing_entitlements_list(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
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


# -----------------------------
# auth / UX
# -----------------------------

def cmd_billing_check(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    now_dt = parse_now(args.now)
    now_ts = now_dt.isoformat(timespec="seconds")
    token_sha = sha256_hex(args.token)

    with db.connect() as con:
        _require_tables(con)
        tok = _token_row_by_sha(con, token_sha)
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


def cmd_billing_whoami(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    now_dt = parse_now(args.now)
    now_ts = now_dt.isoformat(timespec="seconds")
    token_sha = sha256_hex(args.token)

    with db.connect() as con:
        _require_tables(con)

        tok = _token_row_by_sha(con, token_sha)
        if tok is None:
            out = {"ok": False, "cmd": "billing.whoami", "reason": "invalid_token", "token_sha256": token_sha}
            if args.json:
                print(stable_json(out), end="")
            else:
                print("DENY invalid_token")
            return 2

        user_id = str(tok["user_id"])
        user = _user_get(con, user_id)
        ent = _entitlement_active_row(con, user_id=user_id, now_ts=now_ts)
        tier = str(ent["tier"]) if ent is not None else "free"

    out: Dict[str, Any] = {
        "ok": True,
        "cmd": "billing.whoami",
        "now": now_ts,
        "token_sha256": token_sha,
        "token": {"token_sha256": str(tok["token_sha256"]), "user_id": str(tok["user_id"]), "created_ts": str(tok["created_ts"]), "label": tok["label"]},
        "user": dict(user) if user is not None else None,
        "tier": tier,
        "active_entitlement": dict(ent) if ent is not None else None,
    }
    if args.json:
        print(stable_json(out), end="")
    else:
        email = (dict(user).get("email") if user is not None else "") or ""
        label = tok["label"] or ""
        print(f"OK user_id={user_id} tier={tier} email={email}".rstrip())
        if label:
            print(f"label={label}")
        print(f"token_sha256={token_sha}")
    return 0


# -----------------------------
# CLI registration
# -----------------------------

def register_billing_subcommand(sub: argparse._SubParsersAction) -> None:
    billing = sub.add_parser("billing", help="Billing tools (users/tokens/entitlements)")
    bs = billing.add_subparsers(dest="billing_cmd", required=True)

    def add_billing_db_opt(p: argparse.ArgumentParser) -> None:
        p.add_argument("--billing-db", dest="billing_db", default=None, help="Override billing DB path (optional). Otherwise uses global --db.")

    # users
    users = bs.add_parser("users", help="Billing users")
    us = users.add_subparsers(dest="users_cmd", required=True)

    ua = us.add_parser("add", help="Create/update a billing user")
    add_billing_db_opt(ua)
    ua.add_argument("user_id")
    ua.add_argument("--email", default=None)
    ua.add_argument("--created-ts", default=None)
    ua.add_argument("--json", action="store_true")
    ua.set_defaults(func=cmd_billing_users_add)

    ul = us.add_parser("list", help="List billing users")
    add_billing_db_opt(ul)
    ul.add_argument("--limit", type=int, default=50)
    ul.add_argument("--json", action="store_true")
    ul.set_defaults(func=cmd_billing_users_list)

    # tokens
    tokens = bs.add_parser("tokens", help="API tokens")
    ts = tokens.add_subparsers(dest="tokens_cmd", required=True)

    tm = ts.add_parser("mint", help="Mint a token for a user")
    add_billing_db_opt(tm)
    tm.add_argument("user_id")
    tm.add_argument("--email", default=None)
    tm.add_argument("--label", default=None)
    tm.add_argument("--created-ts", default=None)
    tm.add_argument("--token", default=None)
    tm.add_argument("--show-token", action="store_true")
    tm.add_argument("--json", action="store_true")
    tm.set_defaults(func=cmd_billing_tokens_mint)

    trt = ts.add_parser("rotate", help="Rotate token (mint new, optionally revoke old)")
    add_billing_db_opt(trt)
    trt.add_argument("user_id")
    trt.add_argument("--email", default=None)
    trt.add_argument("--label", default=None)
    trt.add_argument("--created-ts", default=None)
    trt.add_argument("--token", default=None)
    g2 = trt.add_mutually_exclusive_group(required=False)
    g2.add_argument("--revoke-token", default=None)
    g2.add_argument("--revoke-token-sha256", default=None)
    trt.add_argument("--show-token", action="store_true")
    trt.add_argument("--json", action="store_true")
    trt.set_defaults(func=cmd_billing_tokens_rotate)

    tl = ts.add_parser("list", help="List tokens")
    add_billing_db_opt(tl)
    tl.add_argument("--user-id", dest="user_id", default=None)
    tl.add_argument("--limit", type=int, default=50)
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(func=cmd_billing_tokens_list)

    tr = ts.add_parser("revoke", help="Revoke a token")
    add_billing_db_opt(tr)
    g = tr.add_mutually_exclusive_group(required=True)
    g.add_argument("--token-sha256", dest="token_sha256", default=None)
    g.add_argument("--token", default=None)
    tr.add_argument("--json", action="store_true")
    tr.set_defaults(func=cmd_billing_tokens_revoke)

    # entitlements
    ents = bs.add_parser("entitlements", help="Access entitlements")
    es = ents.add_subparsers(dest="ents_cmd", required=True)

    eg = es.add_parser("grant", help="Grant an entitlement (explicit entitlement_id required)")
    add_billing_db_opt(eg)
    eg.add_argument("entitlement_id")
    eg.add_argument("user_id")
    eg.add_argument("tier", choices=["free", "supporter", "pro"])
    eg.add_argument("--starts-ts", default=None)
    eg.add_argument("--ends-ts", default=None)
    eg.add_argument("--source", default="manual")
    eg.add_argument("--meta-json", default=None)
    eg.add_argument("--email", default=None)
    eg.add_argument("--json", action="store_true")
    eg.set_defaults(func=cmd_billing_entitlements_grant)

    eg2 = es.add_parser("grant-user", help="Grant an entitlement (auto-generate entitlement_id)")
    add_billing_db_opt(eg2)
    eg2.add_argument("user_id")
    eg2.add_argument("tier", choices=["free", "supporter", "pro"])
    eg2.add_argument("--entitlement-id", dest="entitlement_id", default=None, help="Optional explicit entitlement_id (otherwise generated).")
    eg2.add_argument("--starts-ts", default=None)
    eg2.add_argument("--ends-ts", default=None)
    eg2.add_argument("--source", default="manual")
    eg2.add_argument("--meta-json", default=None)
    eg2.add_argument("--email", default=None)
    eg2.add_argument("--json", action="store_true")
    eg2.set_defaults(func=cmd_billing_entitlements_grant_user)

    er = es.add_parser("revoke", help="Revoke entitlements by setting ends_ts")
    add_billing_db_opt(er)
    er.add_argument("user_id", help="User whose entitlements will be revoked (unless --entitlement-id is used).")
    er.add_argument("--entitlement-id", dest="entitlement_id", default=None, help="Revoke a specific entitlement_id.")
    er.add_argument("--now", default=None, help="Override current time (ISO).")
    er.add_argument("--json", action="store_true")
    er.set_defaults(func=cmd_billing_entitlements_revoke)

    el = es.add_parser("list", help="List entitlements")
    add_billing_db_opt(el)
    el.add_argument("--user-id", dest="user_id", default=None)
    el.add_argument("--limit", type=int, default=50)
    el.add_argument("--json", action="store_true")
    el.set_defaults(func=cmd_billing_entitlements_list)

    # auth / UX
    ck = bs.add_parser("check", help="Validate token and return tier")
    add_billing_db_opt(ck)
    ck.add_argument("--token", required=True)
    ck.add_argument("--now", default=None)
    ck.add_argument("--json", action="store_true")
    ck.set_defaults(func=cmd_billing_check)

    wa = bs.add_parser("whoami", help="Show who the token is (user/tier/entitlement)")
    add_billing_db_opt(wa)
    wa.add_argument("--token", required=True)
    wa.add_argument("--now", default=None)
    wa.add_argument("--json", action="store_true")
    wa.set_defaults(func=cmd_billing_whoami)


# Compatibility aliases (mgc.main registrar probing)
def register(subparsers: argparse._SubParsersAction) -> None:
    register_billing_subcommand(subparsers)
