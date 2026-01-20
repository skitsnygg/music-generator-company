from __future__ import annotations

"""
src/mgc/agents/billing_agent.py

DB-backed billing identity / entitlement resolver.

Contract:
- Token validity:
  - token must exist in billing_tokens (stored as token_sha256)
  - token must NOT exist in billing_token_revocations
- Tier:
  - tier is derived from the most recent active entitlement at "now"
  - if none, tier = "free"

Design goals:
- Single SQLite DB (shared with rest of repo).
- Safe, additive schema: creates billing_token_revocations if missing.
- Deterministic behavior when "now" is provided.
"""

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# -----------------------------
# Helpers
# -----------------------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_now(now_iso: Optional[str]) -> datetime:
    """Parse ISO-8601 timestamp. Accepts Z. If tz missing, assumes UTC."""
    if not now_iso:
        return datetime.now(timezone.utc)
    s = now_iso.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# -----------------------------
# Types
# -----------------------------
@dataclass(frozen=True)
class BillingIdentity:
    user_id: str
    tier: str
    token_sha256: str
    active_entitlement: Optional[Dict[str, Any]]


class BillingError(RuntimeError):
    pass


# -----------------------------
# Agent
# -----------------------------
class BillingAgent:
    """
    Read-only helper for billing checks.

    Typical usage:
      agent = BillingAgent(db_path=Path("data/db.sqlite"))
      ident = agent.identify(token="...", now="2026-01-20T12:00:00+00:00")
      if ident is None: deny
      if ident.tier == "pro": allow pro
    """

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = Path(db_path).expanduser().resolve()

    def _connect(self) -> sqlite3.Connection:
        if not self._db_path.exists():
            raise BillingError(f"DB not found: {self._db_path}")
        con = sqlite3.connect(str(self._db_path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON;")
        return con

    @staticmethod
    def _table_exists(con: sqlite3.Connection, name: str) -> bool:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _ensure_revocations_table(con: sqlite3.Connection) -> None:
        """
        Additive schema: keeps existing billing tables untouched, but adds revocation tombstones.
        Safe to run repeatedly.
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
            "CREATE INDEX IF NOT EXISTS idx_billing_token_revocations_revoked_ts "
            "ON billing_token_revocations(revoked_ts)"
        )

    def _require_tables(self, con: sqlite3.Connection) -> None:
        missing = []
        for t in ("billing_users", "billing_tokens", "billing_entitlements"):
            if not self._table_exists(con, t):
                missing.append(t)
        if missing:
            raise BillingError("Billing tables missing: " + ", ".join(missing))
        self._ensure_revocations_table(con)

    @staticmethod
    def _token_is_revoked(con: sqlite3.Connection, token_sha: str) -> bool:
        row = con.execute(
            "SELECT 1 FROM billing_token_revocations WHERE token_sha256 = ? LIMIT 1",
            (token_sha,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _entitlement_active_row(
        con: sqlite3.Connection, *, user_id: str, now_ts: str
    ) -> Optional[sqlite3.Row]:
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

    def identify(self, *, token: str, now: Optional[str] = None) -> Optional[BillingIdentity]:
        """
        Return a BillingIdentity for a token, or None if:
        - token missing/blank
        - token not issued
        - token revoked
        """
        if not isinstance(token, str) or not token.strip():
            return None

        now_dt = parse_now(now)
        now_ts = now_dt.isoformat(timespec="seconds")

        token_sha = sha256_hex(token.strip())

        with self._connect() as con:
            self._require_tables(con)

            tok = con.execute(
                "SELECT * FROM billing_tokens WHERE token_sha256 = ? LIMIT 1",
                (token_sha,),
            ).fetchone()
            if tok is None:
                return None

            if self._token_is_revoked(con, token_sha):
                return None

            user_id = str(tok["user_id"])
            ent = self._entitlement_active_row(con, user_id=user_id, now_ts=now_ts)
            tier = str(ent["tier"]) if ent is not None else "free"
            ent_d = dict(ent) if ent is not None else None

        return BillingIdentity(
            user_id=user_id,
            tier=tier,
            token_sha256=token_sha,
            active_entitlement=ent_d,
        )

    def tier_for_token(self, *, token: str, now: Optional[str] = None) -> Optional[str]:
        """Convenience: return tier or None if invalid/revoked."""
        ident = self.identify(token=token, now=now)
        return ident.tier if ident is not None else None

    def is_paid_user(self, *, token: str, now: Optional[str] = None) -> bool:
        """Convenience: supporter/pro => paid."""
        ident = self.identify(token=token, now=now)
        if ident is None:
            return False
        return ident.tier in ("supporter", "pro")
