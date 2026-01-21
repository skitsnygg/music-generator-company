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
import re
import secrets
import sqlite3
import sys
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



# -----------------------------
# Receipts (append-only audit log)
# -----------------------------
RECEIPT_SCHEMA_V1 = "mgc.billing_receipt.v1"


class _BillingConflict(Exception):
    def __init__(self, reason: str, *, detail: Optional[str] = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail

def _resolve_receipts_dir(args: argparse.Namespace, db_path: Path) -> Path:
    """Resolve receipt directory.

    Precedence:
      1) --receipts-dir (billing-level option)
      2) <db_dir>/billing_receipts
    """
    raw = getattr(args, "receipts_dir", None)
    if raw:
        return Path(str(raw)).expanduser().resolve()
    return (db_path.parent / "billing_receipts").resolve()


def _receipt_id_for(action: str, core: Dict[str, Any]) -> str:
    """Deterministic-ish id for pairing attempt/result receipts.

    We hash (action + stable_json(core)). If callers pass deterministic timestamps/inputs,
    this becomes deterministic for CI.
    """
    h = hashlib.sha256()
    h.update(action.encode("utf-8"))
    h.update(b"\n")
    h.update(stable_json(core).encode("utf-8"))
    return h.hexdigest()[:20]


def _write_receipt(receipts_dir: Path, payload: Dict[str, Any]) -> Path:
    """Atomically write a receipt JSON file. Raises on failure."""
    receipts_dir.mkdir(parents=True, exist_ok=True)

    ts = str(payload.get("ts") or utc_now_iso())
    safe_ts = ts.replace(":", "").replace("+", "").replace("Z", "Z")
    phase = str(payload.get("phase") or "event")
    action = str(payload.get("action") or "unknown").replace("/", "_")
    rid = str(payload.get("receipt_id") or "noid")
    fname = f"{safe_ts}_{phase}_{action}_{rid}.json"

    tmp = receipts_dir / (fname + ".tmp")
    final = receipts_dir / fname
    tmp.write_text(stable_json(payload), encoding="utf-8")
    tmp.replace(final)
    return final


def _sanitize_input_for_receipt(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove any secrets from input before writing to disk."""
    out = dict(d)
    for k in ("token", "token_value", "api_token", "auth_token"):
        if k in out:
            out[k] = "[REDACTED]"
    return out


def _mutate_with_receipts(
    args: argparse.Namespace,
    *,
    db: "DB",
    cmd: str,
    action: str,
    now_ts: str,
    core_input: Dict[str, Any],
    mutate_fn,
) -> tuple[int, Dict[str, Any]]:
    """Run a DB mutation with mandatory attempt/result receipts.

    Contract:
      - attempt receipt is written before any mutation
      - result receipt is written before COMMIT (so if receipt write fails, we rollback)

    Returns: (exit_code, payload_for_cli)
    """
    receipts_dir = _resolve_receipts_dir(args, db.path)
    core = _sanitize_input_for_receipt(core_input)
    receipt_id = _receipt_id_for(action, core)

    base: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA_V1,
        "receipt_id": receipt_id,
        "action": action,
        "cmd": cmd,
        "ts": now_ts,
        "db_path": str(db.path),
    }

    # 1) attempt receipt (MUST succeed before any mutation)
    try:
        _write_receipt(receipts_dir, {**base, "phase": "attempt", "input": core})
    except Exception as e:
        out = {"ok": False, "cmd": cmd, "error": "receipt_write_failed", "reason": str(e)}
        return (EXIT_INTERNAL, out)

    con: Optional[sqlite3.Connection] = None
    try:
        with db.connect() as con:
            _require_tables(con)
            con.execute("BEGIN IMMEDIATE")
            result = mutate_fn(con)

            # 2) result receipt before commit
            _write_receipt(receipts_dir, {**base, "phase": "result", "ok": True, "result": result})
            con.commit()

        out = {"ok": True, "cmd": cmd, **result}
        return (EXIT_OK, out)

    except _BillingConflict as e:
        if con is not None:
            try:
                con.rollback()
            except Exception:
                pass
        try:
            _write_receipt(receipts_dir, {**base, "phase": "result", "ok": False, "error": "conflict", "reason": e.reason, "detail": e.detail})
        except Exception:
            pass
        return (EXIT_CONFLICT, {"ok": False, "cmd": cmd, "error": "conflict", "reason": e.reason, "detail": e.detail})

    except sqlite3.IntegrityError as e:
        if con is not None:
            try:
                con.rollback()
            except Exception:
                pass
        try:
            _write_receipt(receipts_dir, {**base, "phase": "result", "ok": False, "error": "integrity_error", "reason": str(e)})
        except Exception:
            pass
        return (EXIT_CONFLICT, {"ok": False, "cmd": cmd, "error": "integrity_error", "reason": str(e)})

    except Exception as e:
        if con is not None:
            try:
                con.rollback()
            except Exception:
                pass
        try:
            _write_receipt(receipts_dir, {**base, "phase": "result", "ok": False, "error": "exception", "reason": str(e)})
        except Exception:
            pass
        return (EXIT_INTERNAL, {"ok": False, "cmd": cmd, "error": "exception", "reason": str(e)})


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

    def _mutate(con: sqlite3.Connection) -> Dict[str, Any]:
        _user_upsert(con, args.user_id, args.email, created_ts)
        row = _user_get(con, args.user_id)
        return {"user_id": args.user_id, "user": dict(row) if row else None}

    code, out = _mutate_with_receipts(
        args,
        db=db,
        cmd="billing.users.add",
        action="users.add",
        now_ts=created_ts,
        core_input={"user_id": args.user_id, "email": args.email, "created_ts": created_ts},
        mutate_fn=_mutate,
    )

    if args.json:
        print(stable_json(out), end="")
    else:
        if out.get("ok"):
            print(f"OK user_id={args.user_id}")
        else:
            print(f"ERROR {out.get('error')}: {out.get('reason')}", file=sys.stderr)
    return code


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

    def _mutate(con: sqlite3.Connection) -> Dict[str, Any]:
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, created_ts)

        # pre-check so we can return a clear reason (and still emit a receipt)
        if _token_row_by_sha(con, token_sha) is not None:
            raise _BillingConflict("token_already_exists", detail=token_sha)

        con.execute(
            """
            INSERT INTO billing_tokens(token_sha256, user_id, created_ts, label)
            VALUES(?, ?, ?, ?)
            """,
            (token_sha, args.user_id, created_ts, args.label),
        )
        return {"token_sha256": token_sha, "user_id": args.user_id, "label": args.label, "created_ts": created_ts}

    code, out = _mutate_with_receipts(
        args,
        db=db,
        cmd="billing.tokens.mint",
        action="tokens.mint",
        now_ts=created_ts,
        core_input={"user_id": args.user_id, "email": args.email, "label": args.label, "created_ts": created_ts, "token_sha256": token_sha},
        mutate_fn=_mutate,
    )

    # include plaintext token only in process output when explicitly requested
    if out.get("ok") and args.show_token:
        out["token"] = token_value

    if args.json:
        print(stable_json(out), end="")
    else:
        if out.get("ok"):
            print(f"OK token_sha256={token_sha} user_id={args.user_id}")
            if args.show_token:
                print(token_value)
        else:
            reason = out.get("reason") or out.get("error")
            print(f"DENY {reason} token_sha256={token_sha}")
    return code


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

    def _mutate(con: sqlite3.Connection) -> Dict[str, Any]:
        existed = _token_row_by_sha(con, token_sha) is not None
        inserted = _revoke_token(
            con,
            token_sha=token_sha,
            revoked_ts=revoked_ts,
            reason=args.reason,
            meta_json=meta_json,
        )
        return {
            "token_sha256": token_sha,
            "revoked_ts": revoked_ts,
            "already_revoked": (not inserted),
            "token_existed": existed,
        }

    code, out = _mutate_with_receipts(
        args,
        db=db,
        cmd="billing.tokens.revoke",
        action="tokens.revoke",
        now_ts=revoked_ts,
        core_input={"token_sha256": token_sha, "revoked_ts": revoked_ts, "reason": args.reason, "meta_json": meta_json},
        mutate_fn=_mutate,
    )

    # Preserve prior behavior: return NOT_FOUND if token never existed (even though we still record a tombstone).
    if out.get("ok") and not out.get("token_existed"):
        code = EXIT_NOT_FOUND

    if args.json:
        print(stable_json(out), end="")
    else:
        if out.get("ok"):
            msg = "OK"
            if not out.get("token_existed"):
                msg += " token_not_found"
            if out.get("already_revoked"):
                msg += " already_revoked"
            print(f"{msg} token_sha256={token_sha}")
        else:
            print(f"ERROR {out.get('error')}: {out.get('reason')}", file=sys.stderr)
    return code



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

    def _mutate(con: sqlite3.Connection) -> Dict[str, Any]:
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, created_ts)

        if _token_row_by_sha(con, new_token_sha) is not None:
            raise _BillingConflict("new_token_already_exists", detail=new_token_sha)

        con.execute(
            """
            INSERT INTO billing_tokens(token_sha256, user_id, created_ts, label)
            VALUES(?, ?, ?, ?)
            """,
            (new_token_sha, args.user_id, created_ts, args.label),
        )

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

        return {
            "user_id": args.user_id,
            "token_sha256": new_token_sha,
            "created_ts": created_ts,
            "label": args.label,
            "revoked_token_sha256": revoke_sha,
            "revoked_inserted": revoked_inserted,
            "revoked_token_existed": revoked_existed,
        }

    code, out = _mutate_with_receipts(
        args,
        db=db,
        cmd="billing.tokens.rotate",
        action="tokens.rotate",
        now_ts=created_ts,
        core_input={
            "user_id": args.user_id,
            "email": args.email,
            "label": args.label,
            "created_ts": created_ts,
            "new_token_sha256": new_token_sha,
            "revoke_token_sha256": revoke_sha,
        },
        mutate_fn=_mutate,
    )

    if out.get("ok") and args.show_token:
        out["token"] = new_token_value

    if args.json:
        print(stable_json(out), end="")
    else:
        if out.get("ok"):
            base = f"OK token_sha256={new_token_sha} user_id={args.user_id}"
            if revoke_sha:
                base += f" revoked_token_sha256={revoke_sha}"
            print(base)
            if args.show_token:
                print(new_token_value)
        else:
            reason = out.get("reason") or out.get("error")
            print(f"DENY {reason} token_sha256={new_token_sha}")
    return code



# -----------------------------
# entitlements
# -----------------------------

def cmd_billing_entitlements_grant(args: argparse.Namespace) -> int:
    _require_nonempty(args.entitlement_id, "entitlement_id")
    _require_nonempty(args.user_id, "user_id")
    db = DB(_resolve_db_path(args))
    starts_ts = (args.starts_ts or utc_now_iso()).strip()
    meta_json = _validate_meta_json(args.meta_json)

    def _mutate(con: sqlite3.Connection) -> Dict[str, Any]:
        if _user_get(con, args.user_id) is None:
            _user_upsert(con, args.user_id, args.email, utc_now_iso())

        # clear conflict reason for receipts + CLI
        row = con.execute(
            "SELECT 1 FROM billing_entitlements WHERE entitlement_id = ? LIMIT 1",
            (args.entitlement_id,),
        ).fetchone()
        if row is not None:
            raise _BillingConflict("entitlement_already_exists", detail=args.entitlement_id)

        con.execute(
            """
            INSERT INTO billing_entitlements(entitlement_id, user_id, tier, starts_ts, ends_ts, source, meta_json)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (args.entitlement_id, args.user_id, args.tier, starts_ts, args.ends_ts, args.source, meta_json),
        )
        return {
            "entitlement_id": args.entitlement_id,
            "user_id": args.user_id,
            "tier": args.tier,
            "starts_ts": starts_ts,
            "ends_ts": args.ends_ts,
            "source": args.source,
        }

    code, out = _mutate_with_receipts(
        args,
        db=db,
        cmd="billing.entitlements.grant",
        action="entitlements.grant",
        now_ts=starts_ts,
        core_input={
            "entitlement_id": args.entitlement_id,
            "user_id": args.user_id,
            "tier": args.tier,
            "starts_ts": starts_ts,
            "ends_ts": args.ends_ts,
            "source": args.source,
            "meta_json": meta_json,
            "email": args.email,
        },
        mutate_fn=_mutate,
    )

    if args.json:
        print(stable_json(out), end="")
    else:
        if out.get("ok"):
            print(f"OK entitlement_id={args.entitlement_id} user_id={args.user_id} tier={args.tier}")
        else:
            reason = out.get("reason") or out.get("error")
            print(f"DENY {reason} entitlement_id={args.entitlement_id}")
    return code


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

    def _mutate(con: sqlite3.Connection) -> Dict[str, Any]:
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

        return {
            "user_id": args.user_id,
            "entitlement_id": args.entitlement_id,
            "revoked": int(revoked or 0),
            "now": now_ts,
        }

    code, out = _mutate_with_receipts(
        args,
        db=db,
        cmd="billing.entitlements.revoke",
        action="entitlements.revoke",
        now_ts=now_ts,
        core_input={"user_id": args.user_id, "entitlement_id": args.entitlement_id, "now": now_ts},
        mutate_fn=_mutate,
    )

    if args.json:
        print(stable_json(out), end="")
    else:
        if out.get("ok"):
            if out.get("revoked", 0) == 0:
                print("OK none")
            else:
                if args.entitlement_id:
                    print(f"OK entitlement_id={args.entitlement_id} revoked")
                else:
                    print(f"OK user_id={args.user_id} revoked={out.get('revoked')}")
        else:
            print(f"ERROR {out.get('error')}: {out.get('reason')}", file=sys.stderr)
    return code




def cmd_billing_entitlements_active(args: argparse.Namespace) -> int:
    _require_nonempty(args.user_id, "user_id")
    db = DB(_resolve_db_path(args))
    now_dt = parse_now(getattr(args, "now", None))
    now_ts = now_dt.isoformat(timespec="seconds")

    with db.connect() as con:
        _require_tables(con)
        row = _entitlement_active_row(con, user_id=args.user_id, now_ts=now_ts)

    if row is None:
        out: Dict[str, Any] = {"ok": True, "user_id": args.user_id, "now": now_ts, "active": False}
    else:
        d = dict(row)
        out = {
            "ok": True,
            "user_id": args.user_id,
            "now": now_ts,
            "active": True,
            "entitlement": {
                "entitlement_id": d.get("entitlement_id"),
                "tier": d.get("tier"),
                "starts_ts": d.get("starts_ts"),
                "ends_ts": d.get("ends_ts"),
                "source": d.get("source"),
                "meta_json": d.get("meta_json"),
            },
        }

    if getattr(args, "json", False):
        print(stable_json(out), end="")
    else:
        if not out.get("active"):
            print(f"NONE user_id={args.user_id} now={now_ts}")
        else:
            e = out["entitlement"]
            print(
                f"ACTIVE user_id={args.user_id} tier={e.get('tier')} "
                f"entitlement_id={e.get('entitlement_id')} starts_ts={e.get('starts_ts')}"
            )
    return 0


def cmd_billing_entitlements_list(args: argparse.Namespace) -> int:
    db = DB(_resolve_db_path(args))
    limit = int(getattr(args, "limit", 50) or 50)
    user_id = getattr(args, "user_id", None)

    with db.connect() as con:
        _require_tables(con)
        if user_id:
            rows = con.execute(
                """
                SELECT *
                  FROM billing_entitlements
                 WHERE user_id = ?
                 ORDER BY starts_ts DESC
                 LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT *
                  FROM billing_entitlements
                 ORDER BY starts_ts DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        items.append(
            {
                "entitlement_id": d.get("entitlement_id"),
                "user_id": d.get("user_id"),
                "tier": d.get("tier"),
                "starts_ts": d.get("starts_ts"),
                "ends_ts": d.get("ends_ts"),
                "source": d.get("source"),
                "meta_json": d.get("meta_json"),
            }
        )

    out: Dict[str, Any] = {"ok": True, "count": len(items), "items": items}
    if user_id:
        out["user_id"] = user_id

    if getattr(args, "json", False):
        print(stable_json(out), end="")
    else:
        if not items:
            if user_id:
                print(f"(no entitlements) user_id={user_id}")
            else:
                print("(no entitlements)")
        else:
            for it in items:
                print(
                    f"{it['entitlement_id']} user_id={it['user_id']} tier={it['tier']} "
                    f"starts_ts={it['starts_ts']} ends_ts={it['ends_ts']}"
                )
    return 0

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
# -------------------

def _looks_like_sha256_hex(s: str) -> bool:
    s = (s or "").strip().lower()
    if len(s) != 64:
        return False
    for ch in s:
        if ch not in "0123456789abcdef":
            return False
    return True


def cmd_billing_validate(args) -> int:
    """
    Validate billing evidence by replaying the pro-access policy against the billing DB.

    Evidence path:
      <out_dir>/evidence/billing_evidence.json

    Policy (current):
      allow iff token exists AND not revoked AND user has an active entitlement with tier == 'pro' at evidence time.
    """
    out_dir = Path(args.out_dir)
    ev_path = out_dir / "evidence" / "billing_evidence.json"
    if not ev_path.exists():
        print(f"[billing.validate] missing evidence: {ev_path}", file=sys.stderr)
        return 2

    with ev_path.open("r", encoding="utf-8") as f:
        evidence = json.load(f)

    checks: list[dict] = []

    schema = evidence.get("schema")
    checks.append({"name": "schema", "ok": schema == "mgc.billing_evidence.v1", "found": schema})

    token = None
    inp = evidence.get("input") or {}
    if isinstance(inp, dict):
        token = inp.get("token")
    if not token:
        token = evidence.get("token")

    checks.append({"name": "input.token", "ok": bool(token)})

    found_ok = None
    if isinstance(evidence.get("decision"), dict):
        found_ok = evidence["decision"].get("ok")

    checks.append({"name": "decision.ok.present", "ok": isinstance(found_ok, bool)})

    if not token or not isinstance(found_ok, bool):
        ok = all(c["ok"] for c in checks)
        if args.json:
            stable_json({"ok": ok, "evidence": str(ev_path), "checks": checks})
        else:
            for c in checks:
                print(f"[{'ok' if c['ok'] else 'FAIL'}] {c['name']}")
            print(f"[billing.validate] {'ok' if ok else 'FAILED'}")
        return 0 if ok else 2

    # Determine evaluation time (prefer evidence created_utc, else now)
    at = evidence.get("created_utc") or evidence.get("created_ts") or evidence.get("ts")
    at_dt = parse_now(at) if at else datetime.now(timezone.utc)
    at_ts = at_dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    # Resolve token_sha256
    token_s = str(token).strip()
    token_sha = token_s.lower() if _looks_like_sha256_hex(token_s) else sha256_hex(token_s)

    billing_db = _resolve_db_path(args)
    with DB(billing_db).connect() as con:
        _require_tables(con)

        tok = con.execute(
            "SELECT token_sha256, user_id FROM billing_tokens WHERE token_sha256=? LIMIT 1",
            (token_sha,),
        ).fetchone()

        token_found = tok is not None
        checks.append({"name": "token.found", "ok": token_found})

        revoked = False
        if token_found and _table_exists(con, "billing_token_revocations"):
            r = con.execute(
                "SELECT 1 FROM billing_token_revocations WHERE token_sha256=? LIMIT 1",
                (token_sha,),
            ).fetchone()
            revoked = r is not None
        checks.append({"name": "token.not_revoked", "ok": token_found and not revoked})

        pro_active = False
        if token_found and not revoked:
            ent = con.execute(
                """
                SELECT tier, starts_ts, ends_ts
                FROM billing_entitlements
                WHERE user_id=?
                  AND starts_ts <= ?
                  AND (ends_ts IS NULL OR ends_ts > ?)
                ORDER BY starts_ts DESC
                LIMIT 1
                """,
                (tok["user_id"], at_ts, at_ts),
            ).fetchone()
            pro_active = bool(ent) and (ent["tier"] == "pro")
        checks.append({"name": "entitlement.pro_active", "ok": pro_active})

        recomputed_ok = token_found and (not revoked) and pro_active
        checks.append({"name": "decision.recompute", "ok": bool(recomputed_ok) == bool(found_ok), "expected": bool(recomputed_ok), "found": bool(found_ok)})

    ok = all(c["ok"] for c in checks)
    if args.json:
        stable_json({"ok": ok, "evidence": str(ev_path), "checks": checks})
    else:
        for c in checks:
            print(f"[{'ok' if c['ok'] else 'FAIL'}] {c['name']}")
        print(f"[billing.validate] {'ok' if ok else 'FAILED'}")
    return 0 if ok else 2


# ----------
def register_billing_subcommand(sub: argparse._SubParsersAction) -> None:
    billing = sub.add_parser("billing", help="Billing tools (users/tokens/entitlements)")
    billing.add_argument(
        "--receipts-dir",
        dest="receipts_dir",
        default=None,
        help="Directory for append-only billing receipts (audit log). Defaults to <db_dir>/billing_receipts. Mutating commands will fail if receipts cannot be written.",
    )
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
    vd = bs.add_parser("validate", help="Validate billing evidence (billing_evidence.json)")
    add_billing_db_opt(vd)
    vd.add_argument("--out-dir", required=True, help="Output directory containing evidence/billing_evidence.json")
    vd.add_argument("--now", default=None, help="Override evaluation time (ISO). Defaults to evidence time or now.")
    vd.add_argument("--json", action="store_true")
    vd.set_defaults(func=cmd_billing_validate)


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
