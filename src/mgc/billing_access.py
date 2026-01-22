#!/usr/bin/env python3
"""
src/mgc/billing_access.py

Canonical, side-effect-free access resolver for MGC.

Contract:
- Read-only: NEVER writes to DB.
- Deterministic: caller may supply `now`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set, Tuple, Sequence
import hashlib
import sqlite3


@dataclass(frozen=True)
class AccessContext:
    ok: bool
    user_id: Optional[str]
    tier: Optional[str]
    entitlements: Set[str]
    reason: Optional[str]


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _tables_present(con: sqlite3.Connection, names: Tuple[str, ...]) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    for n in names:
        row = con.execute(q, (n,)).fetchone()
        if not row:
            return False
    return True


def _row_has(row: sqlite3.Row, key: str) -> bool:
    try:
        return key in row.keys()
    except Exception:
        return False


def _row_val(row: sqlite3.Row, key: str):
    if _row_has(row, key):
        return row[key]
    return None


def _table_columns(con: sqlite3.Connection, table: str) -> Set[str]:
    try:
        return {r["name"] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def _first_present(cols: Set[str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def resolve_access(
    *,
    billing_db: str,
    token: Optional[str],
    now: Optional[datetime] = None,
) -> AccessContext:
    """
    Resolve a token into (user_id, tier, entitlements).

    Your current schema (observed):
      - billing_tokens(token_sha256, user_id, created_ts, label)
      - billing_users(user_id, email, created_ts)
      - billing_entitlements(..., entitlement_id, user_id, tier, ...?)  (column names vary)

    Note:
      - There is currently no tier stored in billing_tokens/billing_users, so tier often returns None.
      - Web gating should rely on entitlements (e.g., "web") rather than tier until tier is modeled.
    """
    if not billing_db:
        return AccessContext(False, None, None, set(), "missing_billing_db")

    dbp = str(billing_db)
    if not Path(dbp).exists():
        return AccessContext(False, None, None, set(), "billing_db_not_found")

    if not token:
        return AccessContext(False, None, None, set(), "missing_token")

    now_dt = now or _utc_now()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    token_hash = _sha256_hex(token)

    try:
        con = _connect(dbp)
    except Exception as e:
        return AccessContext(False, None, None, set(), f"db_open_failed:{e}")

    try:
        if not _tables_present(con, ("billing_tokens",)):
            return AccessContext(False, None, None, set(), "billing_tables_missing")

        # Hard block if explicitly revoked (if table exists)
        if _tables_present(con, ("billing_token_revocations",)):
            r = con.execute(
                "SELECT 1 FROM billing_token_revocations WHERE token_sha256 = ? LIMIT 1",
                (token_hash,),
            ).fetchone()
            if r:
                return AccessContext(False, None, None, set(), "token_revoked")

        tok = con.execute(
            "SELECT * FROM billing_tokens WHERE token_sha256 = ? LIMIT 1",
            (token_hash,),
        ).fetchone()
        if not tok:
            return AccessContext(False, None, None, set(), "token_not_found")

        user_id = _row_val(tok, "user_id")
        if not user_id:
            return AccessContext(False, None, None, set(), "token_missing_user")

        # Tier: not currently modeled in your observed billing_tokens/billing_users schema.
        # Keep this as None unless you later add a tier/plan column somewhere.
        tier = None

        entitlements: Set[str] = set()

        if _tables_present(con, ("billing_entitlements",)):
            ecols = _table_columns(con, "billing_entitlements")

            # Column name varies by schema: support both.
            ent_col = _first_present(ecols, ["entitlement", "entitlement_id"])
            if ent_col:
                # Optional time-window columns (support common variants)
                start_col = _first_present(ecols, ["active_from_ts", "starts_ts", "start_ts"])
                end_col = _first_present(ecols, ["active_to_ts", "ends_ts", "end_ts"])

                if start_col and end_col:
                    iso = now_dt.isoformat()
                    rows = con.execute(
                        f"""
                        SELECT {ent_col} AS entitlement
                        FROM billing_entitlements
                        WHERE user_id = ?
                          AND ({start_col} IS NULL OR {start_col} <= ?)
                          AND ({end_col}   IS NULL OR {end_col}   >  ?)
                        """,
                        (user_id, iso, iso),
                    ).fetchall()
                else:
                    rows = con.execute(
                        f"""
                        SELECT {ent_col} AS entitlement
                        FROM billing_entitlements
                        WHERE user_id = ?
                        """,
                        (user_id,),
                    ).fetchall()

                for r in rows:
                    val = _row_val(r, "entitlement")
                    if val:
                        entitlements.add(str(val))

        return AccessContext(True, str(user_id), tier, entitlements, None)

    except Exception as e:
        return AccessContext(False, None, None, set(), f"resolve_failed:{e}")
    finally:
        try:
            con.close()
        except Exception:
            pass
