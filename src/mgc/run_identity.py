# src/mgc/run_identity.py

from __future__ import annotations

import json
import os
import socket
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass(frozen=True)
class RunKey:
    run_date: str                 # YYYY-MM-DD
    context: str                  # focus/workout/sleep
    seed: int
    provider_set_version: str     # "v1" etc.

def _detect_git_meta() -> Dict[str, Optional[str]]:
    # Keep this simple and non-fatal; caller can override
    return {
        "git_sha": os.environ.get("GIT_SHA"),
        "git_branch": os.environ.get("GIT_BRANCH"),
    }

def get_or_create_run_id(
    con: sqlite3.Connection,
    key: RunKey,
    *,
    git_sha: Optional[str] = None,
    git_branch: Optional[str] = None,
    argv: Optional[list[str]] = None,
) -> str:
    """
    Canonical rule: same (run_date, context, seed, provider_set_version) => same run_id.
    Enforced by UNIQUE INDEX in DB. This function returns the existing run_id or creates one.
    """
    con.execute("PRAGMA foreign_keys = ON;")

    row = con.execute(
        """
        SELECT run_id
        FROM runs
        WHERE run_date = ? AND context = ? AND seed = ? AND provider_set_version = ?
        """,
        (key.run_date, key.context, key.seed, key.provider_set_version),
    ).fetchone()

    if row:
        return str(row[0])

    run_id = str(uuid.uuid4())
    ts = now_iso_utc()
    meta = _detect_git_meta()
    gs = git_sha if git_sha is not None else meta.get("git_sha")
    gb = git_branch if git_branch is not None else meta.get("git_branch")

    argv_json = json.dumps(argv, sort_keys=True) if argv is not None else None

    # insert; if another process won the race, read the winner
    try:
        con.execute(
            """
            INSERT INTO runs (
              run_id, run_date, context, seed, provider_set_version,
              created_at, updated_at,
              git_sha, git_branch, hostname, argv_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                key.run_date,
                key.context,
                key.seed,
                key.provider_set_version,
                ts,
                ts,
                gs,
                gb,
                socket.gethostname(),
                argv_json,
            ),
        )
        con.commit()
        return run_id
    except sqlite3.IntegrityError:
        # unique constraint hit; someone else created it
        row2 = con.execute(
            """
            SELECT run_id
            FROM runs
            WHERE run_date = ? AND context = ? AND seed = ? AND provider_set_version = ?
            """,
            (key.run_date, key.context, key.seed, key.provider_set_version),
        ).fetchone()
        if not row2:
            raise
        return str(row2[0])
