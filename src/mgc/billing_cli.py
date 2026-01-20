#!/usr/bin/env python3
"""
src/mgc/billing_cli.py

Billing (v0): users, API tokens, entitlements.

Design goals:
- Single DB (SQLite) shared with the rest of MGC.
- Billing subcommands do NOT take global --db directly (argparse footgun). They use mgc.main --db,
  with optional per-command override --billing-db.
- Deterministic outputs for CI / reproducibility:
  - callers may pass --created-ts and --now
  - callers may pass explicit --token to avoid random generation

Audit / integrity:
- Tokens are NOT hard-deleted anymore.
- Revocation is recorded in billing_token_revocations (tombstones).
- check/whoami treat revoked tokens as invalid.

Schema created by:
  scripts/migrations/0002_billing.sql

Additional table introduced (auto-created if missing):
  billing_token_revocations
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


# -----------------------------
# Exit codes (stable contracts)
# -----------------------------
EXIT_OK = 0
EXIT_INVALID_TOKEN = 2
EXIT_NOT_FOUND = 3
EXIT_CONFLICT = 4
EXIT_BAD_INPUT = 5
EXIT_INTERNAL = 10


# -----------------------------
# Utilities
# -----------------------------
def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_now(now_iso: Optional[str]) -> datetime:
    """Parse ISO-8601 timestamp. Accepts Z. If tz missing, assumes UTC."""
    if not now_iso:
        return datetime.now(timezone.utc)
    s = now_iso.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise SystemExit(f"Invalid --now value (expected ISO-8601): {now_iso}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"Invalid {name}: must be a non-empty string")


def _validate_meta_json(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid --meta-json (must be valid JSON): {e}") from e
    return s


# -----------------------------
# DB wrapper
# -----------------------------
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


def _ensure_revocations_table(con: sqlite3.Connection) -> None:
    """
    Additive schema: keeps existing billing tables untouched, but adds revocation tombstones.

    This is safe to run repeatedly and does not change existing tables.
    """
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_token_revocations (
            token_sha256 TEXT PRIMARY KEY,
            revoked_ts   TEXT NOT NULL,
            reason       TEXT,
            meta_json    TEXT
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_billing_token_revocations_revoked_ts ON billing_token_revocations(revoked_ts)"
    )


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
    _ensure_revocations_table(con)


# -----------------------------
# Queries
# -----------------------------
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


def _token_is_revoked(con: sqlite3.Connection, token_sha: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM billing_token_revocations WHERE token_sha256 = ? LIMIT 1", (token_sha,)
    ).fetchone()
    return row is not None


def _revoke_token(con: sqlite3.Connection, *, token_sha: str, revoked_ts: str, reason: Optional[str], meta_json: Optional[str]) -> bool:
    """
    Idempotent revoke:
    - returns True if we inserted a new revocation
    - returns False if it was already revoked
    """
    try:
        con.execute(
            """
            INSERT INTO billing_token_revocations(token_sha256, revoked_ts, reason, meta_json)
            VALUES(?, ?, ?, ?)
            """,
            (token_sha, revoked_ts, reason, meta_json),
        )
        return True
    except sqlite3.IntegrityError:
        return False


# -----------------------------
# users
# -----------------------------
def cmd_billing_users_add(args: argparse.Namespace) -> int:
    _require_nonempty(args.user_id, "user_id")
    db = DB(_resolve_db_path(args))
    created_ts = (args.created_ts or utc_now_iso()).strip()

    with db.connect() as con:
        _require_tables(con)
        _user_upsert(con, args.user_id, args.email, created_ts)
        con.commit()
        row = _user_get(con, args.user_id)

    out: Dict[str, Any] = {
        "ok": True,
        "cmd": "billing.users.add",
        "user_id": args.user_id,
        "user": dict(row) if row else None,
    }
    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"OK user_id={args.user_id}")
    return EXIT_OK


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
    return EXIT_OK


# -----------------------------
# tokens
# -----------------------------
def cmd_billing_tokens_mint(args: argparse.Namespace) -> int:
    _require_nonempty(args.user_id, "user_id")
    db = DB(_resolve_db_path(args))
    created_ts = (args.created_ts or utc_now_iso()).strip()

    token_value = (args.token or secrets.token_urlsafe(32)).strip()
    _require_nonempty(token_value, "token")
    token_sha = sha256_hex(token_value)

    with db.connect() as con:
        _require_tables(con)
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, created_ts)

        try:
            con.execute(
                """
                INSERT INTO billing_tokens(token_sha256, user_id, created_ts, label)
                VALUES(?, ?, ?, ?)
                """,
                (token_sha, args.user_id, created_ts, args.label),
            )
        except sqlite3.IntegrityError as e:
            # token_sha256 is PK; collision means token already exists
            con.rollback()
            out = {
                "ok": False,
                "cmd": "billing.tokens.mint",
                "error": "conflict",
                "reason": "token_already_exists",
                "token_sha256": token_sha,
            }
            if args.json:
                print(stable_json(out), end="")
            else:
                print(f"DENY token_already_exists token_sha256={token_sha}")
            return EXIT_CONFLICT

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
    return EXIT_OK


def cmd_billing_tokens_list(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    with db.connect() as con:
        _require_tables(con)
        if args.user_id:
            rows = con.execute(
                """
                SELECT t.*,
                       CASE WHEN r.token_sha256 IS NULL THEN 0 ELSE 1 END AS revoked
                FROM billing_tokens t
                LEFT JOIN billing_token_revocations r
                  ON r.token_sha256 = t.token_sha256
                WHERE t.user_id = ?
                ORDER BY t.created_ts DESC
                LIMIT ?
                """,
                (args.user_id, int(args.limit)),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT t.*,
                       CASE WHEN r.token_sha256 IS NULL THEN 0 ELSE 1 END AS revoked
                FROM billing_tokens t
                LEFT JOIN billing_token_revocations r
                  ON r.token_sha256 = t.token_sha256
                ORDER BY t.created_ts DESC
                LIMIT ?
                """,
                (int(args.limit),),
            ).fetchall()

    data = [dict(r) for r in rows]
    if args.json:
        print(stable_json({"ok": True, "cmd": "billing.tokens.list", "tokens": data}), end="")
    else:
        for t in data:
            revoked = " revoked" if int(t.get("revoked") or 0) else ""
            print(
                f"{t.get('token_sha256')}  user_id={t.get('user_id')}  {t.get('label','')}".rstrip()
                + revoked
            )
    return EXIT_OK


def cmd_billing_tokens_revoke(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    revoked_ts = (args.revoked_ts or utc_now_iso()).strip()
    meta_json = _validate_meta_json(args.meta_json)

    token_sha = (args.token_sha256 or sha256_hex(args.token)).strip()
    _require_nonempty(token_sha, "token_sha256")

    with db.connect() as con:
        _require_tables(con)

        # If token never existed, still allow idempotent tombstone (optional behavior),
        # but return NOT_FOUND for clarity.
        existed = _token_row_by_sha(con, token_sha) is not None

        inserted = _revoke_token(
            con,
            token_sha=token_sha,
            revoked_ts=revoked_ts,
            reason=args.reason,
            meta_json=meta_json,
        )
        con.commit()

    out = {
        "ok": True,
        "cmd": "billing.tokens.revoke",
        "token_sha256": token_sha,
        "revoked_ts": revoked_ts,
        "already_revoked": (not inserted),
        "token_existed": existed,
    }
    if args.json:
        print(stable_json(out), end="")
    else:
        msg = "OK"
        if not existed:
            msg += " token_not_found"
        if not inserted:
            msg += " already_revoked"
        print(f"{msg} token_sha256={token_sha}")
    return EXIT_OK if existed else EXIT_NOT_FOUND


def cmd_billing_tokens_rotate(args: argparse.Namespace) -> int:
    _require_nonempty(args.user_id, "user_id")
    db = DB(_resolve_db_path(args))
    created_ts = (args.created_ts or utc_now_iso()).strip()

    new_token_value = (args.token or secrets.token_urlsafe(32)).strip()
    _require_nonempty(new_token_value, "token")
    new_token_sha = sha256_hex(new_token_value)

    revoke_sha: Optional[str] = None
    if args.revoke_token_sha256:
        revoke_sha = args.revoke_token_sha256.strip()
    elif args.revoke_token:
        revoke_sha = sha256_hex(args.revoke_token.strip())

    with db.connect() as con:
        _require_tables(con)

        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, created_ts)

        try:
            con.execute(
                """
                INSERT INTO billing_tokens(token_sha256, user_id, created_ts, label)
                VALUES(?, ?, ?, ?)
                """,
                (new_token_sha, args.user_id, created_ts, args.label),
            )
        except sqlite3.IntegrityError:
            con.rollback()
            out = {
                "ok": False,
                "cmd": "billing.tokens.rotate",
                "error": "conflict",
                "reason": "new_token_already_exists",
                "token_sha256": new_token_sha,
            }
            if args.json:
                print(stable_json(out), end="")
            else:
                print(f"DENY new_token_already_exists token_sha256={new_token_sha}")
            return EXIT_CONFLICT

        revoked_inserted = None
        revoked_existed = None
        if revoke_sha:
            revoked_existed = _token_row_by_sha(con, revoke_sha) is not None
            revoked_inserted = _revoke_token(
                con,
                token_sha=revoke_sha,
                revoked_ts=created_ts,
                reason="rotate",
                meta_json=None,
            )

        con.commit()

    out: Dict[str, Any] = {
        "ok": True,
        "cmd": "billing.tokens.rotate",
        "user_id": args.user_id,
        "token_sha256": new_token_sha,
        "created_ts": created_ts,
        "label": args.label,
        "revoked_token_sha256": revoke_sha,
        "revoked_inserted": revoked_inserted,
        "revoked_token_existed": revoked_existed,
    }
    if args.show_token:
        out["token"] = new_token_value

    if args.json:
        print(stable_json(out), end="")
    else:
        base = f"OK token_sha256={new_token_sha} user_id={args.user_id}"
        if revoke_sha:
            base += f" revoked_token_sha256={revoke_sha}"
        print(base)
        if args.show_token:
            print(new_token_value)
    return EXIT_OK


# -----------------------------
# entitlements
# -----------------------------
def cmd_billing_entitlements_grant(args: argparse.Namespace) -> int:
    _require_nonempty(args.entitlement_id, "entitlement_id")
    _require_nonempty(args.user_id, "user_id")
    db = DB(_resolve_db_path(args))
    starts_ts = (args.starts_ts or utc_now_iso()).strip()
    meta_json = _validate_meta_json(args.meta_json)

    with db.connect() as con:
        _require_tables(con)
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, utc_now_iso())

        try:
            con.execute(
                """
                INSERT INTO billing_entitlements(entitlement_id, user_id, tier, starts_ts, ends_ts, source, meta_json)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (args.entitlement_id, args.user_id, args.tier, starts_ts, args.ends_ts, args.source, meta_json),
            )
        except sqlite3.IntegrityError:
            con.rollback()
            out = {
                "ok": False,
                "cmd": "billing.entitlements.grant",
                "error": "conflict",
                "reason": "entitlement_id_exists",
                "entitlement_id": args.entitlement_id,
            }
            if args.json:
                print(stable_json(out), end="")
            else:
                print(f"DENY entitlement_id_exists entitlement_id={args.entitlement_id}")
            return EXIT_CONFLICT

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
    return EXIT_OK


def cmd_billing_entitlements_grant_user(args: argparse.Namespace) -> int:
    _require_nonempty(args.user_id, "user_id")
    db = DB(_resolve_db_path(args))
    starts_ts = (args.starts_ts or utc_now_iso()).strip()
    meta_json = _validate_meta_json(args.meta_json)

    entitlement_id = (args.entitlement_id or secrets.token_hex(16)).strip()
    _require_nonempty(entitlement_id, "entitlement_id")

    with db.connect() as con:
        _require_tables(con)
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, utc_now_iso())

        try:
            con.execute(
                """
                INSERT INTO billing_entitlements(entitlement_id, user_id, tier, starts_ts, ends_ts, source, meta_json)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (entitlement_id, args.user_id, args.tier, starts_ts, args.ends_ts, args.source, meta_json),
            )
        except sqlite3.IntegrityError:
            con.rollback()
            out = {
                "ok": False,
                "cmd": "billing.entitlements.grant-user",
                "error": "conflict",
                "reason": "entitlement_id_exists",
                "entitlement_id": entitlement_id,
            }
            if args.json:
                print(stable_json(out), end="")
            else:
                print(f"DENY entitlement_id_exists entitlement_id={entitlement_id}")
            return EXIT_CONFLICT

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
    return EXIT_OK


def cmd_billing_entitlements_revoke(args: argparse.Namespace) -> int:
    _require_nonempty(args.user_id, "user_id")
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
    return EXIT_OK


def cmd_billing_entitlements_active(args: argparse.Namespace) -> int:
    _require_nonempty(args.user_id, "user_id")
    db = DB(_resolve_db_path(args))
    now_dt = parse_now(args.now)
    now_ts = now_dt.isoformat(timespec="seconds")

    with db.connect() as con:
        _require_tables(con)
        ent = _entitlement_active_row(con, user_id=args.user_id, now_ts=now_ts)

    tier = str(ent["tier"]) if ent is not None else "free"
    ent_d = dict(ent) if ent is not None else None

    out: Dict[str, Any] = {
        "ok": True,
        "cmd": "billing.entitlements.active",
        "user_id": args.user_id,
        "now": now_ts,
        "tier": tier,
        "active_entitlement": ent_d,
    }
    if args.json:
        print(stable_json(out), end="")
    else:
        if ent_d is None:
            print(f"OK user_id={args.user_id} tier=free active_entitlement=")
        else:
            print(
                f"OK user_id={args.user_id} tier={tier} entitlement_id={ent_d.get('entitlement_id')} "
                f"starts={ent_d.get('starts_ts')} ends={(ent_d.get('ends_ts') or '')} source={ent_d.get('source')}"
            )
    return EXIT_OK


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
    return EXIT_OK


# -----------------------------
# auth / UX
# -----------------------------
def _deny(args: argparse.Namespace, *, cmd: str, reason: str, token_sha256: Optional[str] = None) -> int:
    out: Dict[str, Any] = {"ok": False, "cmd": cmd, "error": "deny", "reason": reason}
    if token_sha256:
        out["token_sha256"] = token_sha256
    if args.json:
        print(stable_json(out), end="")
    else:
        print(f"DENY {reason}")
        if token_sha256:
            print(f"token_sha256={token_sha256}")
    return EXIT_INVALID_TOKEN


def cmd_billing_check(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    now_dt = parse_now(args.now)
    now_ts = now_dt.isoformat(timespec="seconds")

    _require_nonempty(args.token, "token")
    token_sha = sha256_hex(args.token)

    with db.connect() as con:
        _require_tables(con)

        tok = _token_row_by_sha(con, token_sha)
        if tok is None:
            return _deny(args, cmd="billing.check", reason="invalid_token", token_sha256=token_sha)

        if _token_is_revoked(con, token_sha):
            return _deny(args, cmd="billing.check", reason="revoked_token", token_sha256=token_sha)

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
    return EXIT_OK


def cmd_billing_whoami(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    now_dt = parse_now(args.now)
    now_ts = now_dt.isoformat(timespec="seconds")

    _require_nonempty(args.token, "token")
    token_sha = sha256_hex(args.token)

    with db.connect() as con:
        _require_tables(con)

        tok = _token_row_by_sha(con, token_sha)
        if tok is None:
            return _deny(args, cmd="billing.whoami", reason="invalid_token", token_sha256=token_sha)

        if _token_is_revoked(con, token_sha):
            return _deny(args, cmd="billing.whoami", reason="revoked_token", token_sha256=token_sha)

        user_id = str(tok["user_id"])
        user = _user_get(con, user_id)
        ent = _entitlement_active_row(con, user_id=user_id, now_ts=now_ts)
        tier = str(ent["tier"]) if ent is not None else "free"

    out: Dict[str, Any] = {
        "ok": True,
        "cmd": "billing.whoami",
        "now": now_ts,
        "token_sha256": token_sha,
        "token": {
            "token_sha256": str(tok["token_sha256"]),
            "user_id": str(tok["user_id"]),
            "created_ts": str(tok["created_ts"]),
            "label": tok["label"],
        },
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
    return EXIT_OK


# -----------------------------
# CLI registration
# -----------------------------
def register_billing_subcommand(sub: argparse._SubParsersAction) -> None:
    billing = sub.add_parser("billing", help="Billing tools (users/tokens/entitlements)")
    bs = billing.add_subparsers(dest="billing_cmd", required=True)

    def add_billing_db_opt(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--billing-db",
            dest="billing_db",
            default=None,
            help="Override billing DB path (optional). Otherwise uses global --db.",
        )

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
    tm.add_argument("--token", default=None, help="Optional explicit token value (for determinism).")
    tm.add_argument("--show-token", action="store_true")
    tm.add_argument("--json", action="store_true")
    tm.set_defaults(func=cmd_billing_tokens_mint)

    trt = ts.add_parser("rotate", help="Rotate token (mint new, optionally revoke old)")
    add_billing_db_opt(trt)
    trt.add_argument("user_id")
    trt.add_argument("--email", default=None)
    trt.add_argument("--label", default=None)
    trt.add_argument("--created-ts", default=None)
    trt.add_argument("--token", default=None, help="Optional explicit token value (for determinism).")
    g2 = trt.add_mutually_exclusive_group(required=False)
    g2.add_argument("--revoke-token", default=None, help="Old token value to revoke (will be sha256 hashed).")
    g2.add_argument("--revoke-token-sha256", default=None, help="Old token sha256 to revoke.")
    trt.add_argument("--show-token", action="store_true")
    trt.add_argument("--json", action="store_true")
    trt.set_defaults(func=cmd_billing_tokens_rotate)

    tl = ts.add_parser("list", help="List tokens")
    add_billing_db_opt(tl)
    tl.add_argument("--user-id", dest="user_id", default=None)
    tl.add_argument("--limit", type=int, default=50)
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(func=cmd_billing_tokens_list)

    tr = ts.add_parser("revoke", help="Revoke a token (tombstone, no deletion)")
    add_billing_db_opt(tr)
    g = tr.add_mutually_exclusive_group(required=True)
    g.add_argument("--token-sha256", dest="token_sha256", default=None)
    g.add_argument("--token", default=None)
    tr.add_argument("--revoked-ts", default=None, help="Override revocation timestamp (ISO).")
    tr.add_argument("--reason", default="manual", help="Reason for revocation (string).")
    tr.add_argument("--meta-json", default=None, help="Optional JSON metadata.")
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
    eg2.add_argument(
        "--entitlement-id",
        dest="entitlement_id",
        default=None,
        help="Optional explicit entitlement_id (otherwise generated).",
    )
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

    ea = es.add_parser("active", help="Show active entitlement for a user")
    add_billing_db_opt(ea)
    ea.add_argument("user_id")
    ea.add_argument("--now", default=None, help="Override current time (ISO).")
    ea.add_argument("--json", action="store_true")
    ea.set_defaults(func=cmd_billing_entitlements_active)

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


def register(subparsers: argparse._SubParsersAction) -> None:
    register_billing_subcommand(subparsers)
