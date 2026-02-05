#!/usr/bin/env python3
"""
scripts/migrate_db.py

Apply SQL migrations in scripts/migrations/ to a SQLite DB.

Policy:
- Uses schema_migrations table to apply each migration exactly once.
- Migrations are applied in filename-sorted order.
- DB path is provided via MGC_DB. Relative paths are resolved to an absolute path
  so the script can be run from any working directory.

Optional determinism helper:
- If MGC_MIGRATE_NOW is set (ISO timestamp), it is used for applied_at instead of
  the current time. (Useful for deterministic CI fixtures.)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def utc_now_iso() -> str:
    override = (os.environ.get("MGC_MIGRATE_NOW") or "").strip()
    if override:
        # Accept Z suffix as UTC
        s = override.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError as e:
            die(f"Invalid MGC_MIGRATE_NOW (expected ISO timestamp): {override} ({e})", 2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def list_migrations(migrations_dir: Path) -> List[Path]:
    if not migrations_dir.exists() or not migrations_dir.is_dir():
        die(f"migrations dir missing: {migrations_dir}", 2)
    return sorted([p for p in migrations_dir.glob("*.sql") if p.is_file()])


def ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL
        );
        """
    )


def get_applied(conn: sqlite3.Connection) -> set[str]:
    ensure_schema_migrations(conn)
    rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC").fetchall()
    return {str(r[0]) for r in rows}


def parse_version(path: Path) -> str:
    # version is the filename like 0002_billing.sql
    return path.name


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, name: str) -> List[str]:
    if not _table_exists(conn, name):
        return []
    return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({name})").fetchall()]


def _table_info(conn: sqlite3.Connection, name: str) -> List[sqlite3.Row]:
    if not _table_exists(conn, name):
        return []
    return conn.execute(f"PRAGMA table_info({name})").fetchall()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, name: str, col_type: str) -> bool:
    cols = _table_columns(conn, table)
    if name in cols:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
    return True


def _maybe_load_json(raw: object) -> object | None:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _normalize_fixture_schema(conn: sqlite3.Connection) -> None:
    """
    Bring legacy fixture DBs closer to canonical schema by adding missing tables/columns.
    Safe to run multiple times.
    """
    # tracks: add id/created_at columns for canonical compatibility
    if _table_exists(conn, "tracks"):
        tcols = _table_columns(conn, "tracks")
        if _add_column_if_missing(conn, "tracks", "id", "TEXT"):
            tcols.append("id")
        if "track_id" in tcols:
            conn.execute(
                "UPDATE tracks SET id = track_id WHERE (id IS NULL OR id = '') AND track_id IS NOT NULL"
            )
        if "created_at" not in tcols and "ts" in tcols:
            _add_column_if_missing(conn, "tracks", "created_at", "TEXT")
            tcols.append("created_at")
        if "created_at" in tcols and "ts" in tcols:
            conn.execute(
                "UPDATE tracks SET created_at = ts WHERE (created_at IS NULL OR created_at = '') AND ts IS NOT NULL"
            )
        if "id" in tcols:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tracks_id ON tracks(id)")

    # playlists: add id/created_at columns for canonical compatibility
    if _table_exists(conn, "playlists"):
        pcols = _table_columns(conn, "playlists")
        if _add_column_if_missing(conn, "playlists", "id", "TEXT"):
            pcols.append("id")
        if "playlist_id" in pcols:
            conn.execute(
                "UPDATE playlists SET id = playlist_id WHERE (id IS NULL OR id = '') AND playlist_id IS NOT NULL"
            )
        if "created_at" not in pcols and "ts" in pcols:
            _add_column_if_missing(conn, "playlists", "created_at", "TEXT")
            pcols.append("created_at")
        if "created_at" in pcols and "ts" in pcols:
            conn.execute(
                "UPDATE playlists SET created_at = ts WHERE (created_at IS NULL OR created_at = '') AND ts IS NOT NULL"
            )
        if "id" in pcols:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_playlists_id ON playlists(id)")

    # marketing_posts: add canonical columns + backfill from legacy content/meta
    if not _table_exists(conn, "marketing_posts"):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS marketing_posts (
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              track_id TEXT NOT NULL,
              platform TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              status TEXT NOT NULL,
              FOREIGN KEY(track_id) REFERENCES tracks(id)
            )
            """
        )
    else:
        minfo = _table_info(conn, "marketing_posts")
        mcols = [str(r[1]) for r in minfo]
        ts_notnull = any((str(r[1]) == "ts" and int(r[3] or 0) == 1) for r in minfo)

        if ts_notnull:
            # Rebuild to canonical schema to avoid NOT NULL ts insert failures.
            select_cols = ["id"]
            for c in (
                "created_at",
                "ts",
                "track_id",
                "platform",
                "status",
                "payload_json",
                "content",
                "payload",
                "text",
                "meta",
                "meta_json",
                "metadata_json",
            ):
                if c in mcols:
                    select_cols.append(c)

            rows = conn.execute(f"SELECT {', '.join(select_cols)} FROM marketing_posts").fetchall()

            conn.execute("ALTER TABLE marketing_posts RENAME TO marketing_posts_legacy")
            conn.execute(
                """
                CREATE TABLE marketing_posts (
                  id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  track_id TEXT NOT NULL,
                  platform TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  FOREIGN KEY(track_id) REFERENCES tracks(id)
                )
                """
            )

            for row in rows:
                row_map = {select_cols[i]: row[i] for i in range(len(select_cols))}
                created_at = row_map.get("created_at") or row_map.get("ts") or "1970-01-01T00:00:00Z"
                platform = row_map.get("platform") or "unknown"
                status = row_map.get("status") or "draft"

                payload_json = row_map.get("payload_json")
                content_raw = row_map.get("content") or row_map.get("payload") or row_map.get("text")
                meta_raw = row_map.get("meta") or row_map.get("meta_json") or row_map.get("metadata_json")
                content_obj = _maybe_load_json(content_raw)
                meta_obj = _maybe_load_json(meta_raw)
                if payload_json is None or str(payload_json).strip() == "":
                    payload_obj = {
                        "content": content_obj if content_obj is not None else content_raw,
                        "meta": meta_obj if meta_obj is not None else meta_raw,
                    }
                    payload_json = json.dumps(
                        payload_obj,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )

                track_id = row_map.get("track_id")
                if track_id is None or str(track_id).strip() == "":
                    if isinstance(meta_obj, dict):
                        track_id = meta_obj.get("track_id") or meta_obj.get("trackId")
                    if not track_id and isinstance(content_obj, dict):
                        track_id = content_obj.get("track_id") or content_obj.get("trackId")

                if not track_id:
                    # Skip rows we can't map to a track id.
                    continue

                conn.execute(
                    """
                    INSERT OR IGNORE INTO marketing_posts
                      (id, created_at, track_id, platform, payload_json, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (row_map.get("id"), str(created_at), str(track_id), str(platform), str(payload_json), str(status)),
                )

            conn.execute("DROP TABLE marketing_posts_legacy")
        else:
            _add_column_if_missing(conn, "marketing_posts", "created_at", "TEXT")
            _add_column_if_missing(conn, "marketing_posts", "track_id", "TEXT")
            _add_column_if_missing(conn, "marketing_posts", "payload_json", "TEXT")
            _add_column_if_missing(conn, "marketing_posts", "platform", "TEXT")
            _add_column_if_missing(conn, "marketing_posts", "status", "TEXT")

            mcols = _table_columns(conn, "marketing_posts")
            if "created_at" in mcols and "ts" in mcols:
                conn.execute(
                    "UPDATE marketing_posts SET created_at = ts "
                    "WHERE (created_at IS NULL OR created_at = '') AND ts IS NOT NULL"
                )

            # Build per-row payload_json / track_id when missing
            select_cols = ["id"]
            for c in ("content", "payload", "text", "meta", "meta_json", "metadata_json", "ts", "created_at", "track_id", "payload_json"):
                if c in mcols:
                    select_cols.append(c)
            rows = conn.execute(f"SELECT {', '.join(select_cols)} FROM marketing_posts").fetchall()

            for row in rows:
                row_map = {select_cols[i]: row[i] for i in range(len(select_cols))}
                updates: dict[str, object] = {}

                # payload_json
                if "payload_json" in mcols:
                    existing_payload = row_map.get("payload_json")
                    if existing_payload is None or str(existing_payload).strip() == "":
                        content_raw = row_map.get("content") or row_map.get("payload") or row_map.get("text")
                        meta_raw = row_map.get("meta") or row_map.get("meta_json") or row_map.get("metadata_json")
                        content_obj = _maybe_load_json(content_raw)
                        meta_obj = _maybe_load_json(meta_raw)
                        if content_raw is not None or meta_raw is not None:
                            payload_obj = {
                                "content": content_obj if content_obj is not None else content_raw,
                                "meta": meta_obj if meta_obj is not None else meta_raw,
                            }
                            updates["payload_json"] = json.dumps(
                                payload_obj,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )

                # track_id
                if "track_id" in mcols:
                    existing_track = row_map.get("track_id")
                    if existing_track is None or str(existing_track).strip() == "":
                        content_raw = row_map.get("content") or row_map.get("payload") or row_map.get("text")
                        meta_raw = row_map.get("meta") or row_map.get("meta_json") or row_map.get("metadata_json")
                        content_obj = _maybe_load_json(content_raw)
                        meta_obj = _maybe_load_json(meta_raw)
                        track_id = None
                        if isinstance(meta_obj, dict):
                            track_id = meta_obj.get("track_id") or meta_obj.get("trackId")
                        if not track_id and isinstance(content_obj, dict):
                            track_id = content_obj.get("track_id") or content_obj.get("trackId")
                        if track_id:
                            updates["track_id"] = str(track_id)

                if updates:
                    keys = sorted(updates.keys())
                    set_sql = ", ".join([f"{k} = ?" for k in keys])
                    params = [updates[k] for k in keys] + [row_map["id"]]
                    conn.execute(f"UPDATE marketing_posts SET {set_sql} WHERE id = ?", params)

    # playlist_runs: canonical table (used by analytics)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS playlist_runs (
          id TEXT PRIMARY KEY,
          created_at TEXT NOT NULL,
          playlist_id TEXT NOT NULL,
          seed INTEGER,
          track_ids_json TEXT NOT NULL,
          export_path TEXT,
          notes TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_playlist_runs_playlist_id ON playlist_runs(playlist_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_playlist_runs_created_at ON playlist_runs(created_at)")

    # playlist_items: canonical table + backfill from legacy playlist payloads if available
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS playlist_items (
          playlist_id TEXT NOT NULL,
          track_id TEXT NOT NULL,
          position INTEGER NOT NULL,
          PRIMARY KEY (playlist_id, track_id),
          FOREIGN KEY(track_id) REFERENCES tracks(id),
          FOREIGN KEY(playlist_id) REFERENCES playlists(id)
        )
        """
    )

    if _table_exists(conn, "playlists"):
        pcols = _table_columns(conn, "playlists")
        if "payload" in pcols and "id" in pcols:
            rows = conn.execute("SELECT id, payload FROM playlists WHERE payload IS NOT NULL").fetchall()
            for pid, payload in rows:
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                tracks = obj.get("tracks") if isinstance(obj, dict) else None
                if not isinstance(tracks, list):
                    continue
                for idx, t in enumerate(tracks):
                    if not isinstance(t, dict):
                        continue
                    tid = t.get("id") or t.get("track_id") or t.get("trackId")
                    if not tid:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO playlist_items (playlist_id, track_id, position) VALUES (?, ?, ?)",
                        (pid, str(tid), int(idx) + 1),
                    )


def apply_one(conn: sqlite3.Connection, path: Path) -> None:
    sql = read_sql(path).strip()
    version = parse_version(path)

    # Allow empty migration files (still gets recorded)
    if sql:
        conn.executescript(sql)
    if version == "0004_schema_normalize.sql":
        _normalize_fixture_schema(conn)

    conn.execute(
        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (version, utc_now_iso()),
    )


def main() -> int:
    db_env = (os.environ.get("MGC_DB") or "").strip()
    if not db_env:
        die("MGC_DB not set", 2)

    # This is the key fix: run-from-anywhere safety.
    db_path = Path(db_env).expanduser().resolve()

    if not db_path.exists():
        die(f"DB not found: {db_path}", 2)

    migrations_dir = Path(__file__).resolve().parent / "migrations"
    migrations = list_migrations(migrations_dir)

    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys=ON;")
        ensure_schema_migrations(con)

        applied = get_applied(con)
        to_apply: List[Tuple[str, Path]] = []
        for m in migrations:
            v = parse_version(m)
            if v not in applied:
                to_apply.append((v, m))

        # Apply in a transaction; rollback on failure.
        try:
            con.execute("BEGIN;")
            for _, m in to_apply:
                apply_one(con, m)
            con.commit()
        except Exception:
            con.rollback()
            raise

        print(f"OK: applied {len(to_apply)} migrations to {db_path}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
