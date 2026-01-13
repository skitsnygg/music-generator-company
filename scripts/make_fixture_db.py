#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def now_iso_fixed() -> str:
    # Deterministic timestamp for fixture DB (avoid CI diffs).
    return datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()


def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def main() -> int:
    out_path = Path("fixtures/ci_db.sqlite")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always regenerate from scratch for determinism
    if out_path.exists():
        out_path.unlink()

    con = sqlite3.connect(str(out_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute("PRAGMA journal_mode = DELETE;")  # stable file layout vs WAL
        con.execute("PRAGMA synchronous = FULL;")
        cur = con.cursor()

        # ---------------------------------------------------------------------
        # Schema: match CURRENT code expectations (db_insert_* + run/drop paths)
        # ---------------------------------------------------------------------
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS tracks (
              track_id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              title TEXT,
              provider TEXT,
              mood TEXT,
              genre TEXT,
              artifact_path TEXT,
              meta TEXT
            );

            CREATE TABLE IF NOT EXISTS drops (
              drop_id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              context TEXT,
              seed TEXT,
              run_id TEXT,
              track_id TEXT,
              meta TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              kind TEXT,
              actor TEXT,
              meta TEXT
            );

            CREATE TABLE IF NOT EXISTS marketing_posts (
              post_id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              platform TEXT,
              status TEXT,
              content TEXT,
              meta TEXT
            );

            CREATE TABLE IF NOT EXISTS playlists (
              playlist_id TEXT PRIMARY KEY,
              ts TEXT,
              context TEXT,
              payload TEXT,
              meta TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_tracks_ts ON tracks(ts);
            CREATE INDEX IF NOT EXISTS idx_drops_ts ON drops(ts);
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
            CREATE INDEX IF NOT EXISTS idx_marketing_posts_ts ON marketing_posts(ts);
            CREATE INDEX IF NOT EXISTS idx_playlists_ts ON playlists(ts);
            """
        )

        # ---------------------------------------------------------------------
        # Seed data: minimal deterministic set
        # ---------------------------------------------------------------------
        ts = now_iso_fixed()

        track_id = "00000000-0000-0000-0000-000000000001"
        drop_id = "10000000-0000-0000-0000-000000000001"
        run_id = "20000000-0000-0000-0000-000000000001"

        # Track row (artifact_path is a relative path string; file doesn't need to exist for DB)
        cur.execute(
            """
            INSERT INTO tracks (
              track_id, ts, title, provider, mood, genre, artifact_path, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track_id,
                ts,
                "CI Seed Track",
                "stub",
                "focus",
                "ci_genre",
                "data/tracks/ci_seed_track.wav",
                stable_json(
                    {
                        "fixture": True,
                        "note": "seed row for CI; run/drop will write additional rows",
                    }
                ),
            ),
        )

        # Drop row
        cur.execute(
            """
            INSERT INTO drops (
              drop_id, ts, context, seed, run_id, track_id, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                drop_id,
                ts,
                "focus",
                "ci_seed",
                run_id,
                track_id,
                stable_json(
                    {
                        "fixture": True,
                        "note": "seed row for CI",
                        "provider": "stub",
                    }
                ),
            ),
        )

        # Event row
        cur.execute(
            """
            INSERT INTO events (
              event_id, ts, kind, actor, meta
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "30000000-0000-0000-0000-000000000001",
                ts,
                "fixture.seeded",
                "fixture",
                stable_json({"ok": True, "run_id": run_id, "drop_id": drop_id}),
            ),
        )

        # Marketing post seed
        cur.execute(
            """
            INSERT INTO marketing_posts (
              post_id, ts, platform, status, content, meta
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "40000000-0000-0000-0000-000000000001",
                ts,
                "x",
                "draft",
                stable_json({"hook": "CI seed post", "cta": "Listen now", "track_id": track_id}),
                stable_json({"fixture": True, "run_id": run_id, "drop_id": drop_id}),
            ),
        )

        # Playlist seed (payload is up to your code; keep tiny and deterministic)
        cur.execute(
            """
            INSERT INTO playlists (
              playlist_id, ts, context, payload, meta
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "50000000-0000-0000-0000-000000000001",
                ts,
                "focus",
                stable_json(
                    {
                        "schema": "mgc.playlist.v1",
                        "id": "ci-seed-playlist",
                        "name": "CI Seed Playlist",
                        "created_ts": ts,
                        "context": "focus",
                        "track_count": 1,
                        "tracks": [
                            {
                                "id": track_id,
                                "title": "CI Seed Track",
                                "artifact_path": "data/tracks/ci_seed_track.wav",
                            }
                        ],
                    }
                ),
                stable_json({"fixture": True}),
            ),
        )

        con.commit()

        # Stabilize file size/layout
        try:
            cur.execute("VACUUM;")
            con.commit()
        except Exception:
            pass

    finally:
        con.close()

    print(f"Wrote fixture DB: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
