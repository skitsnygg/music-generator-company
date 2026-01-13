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

        # ------------------------------------------------------------------
        # Schema: match CURRENT run_cli.py expectations
        # ------------------------------------------------------------------
        cur.executescript(
            """
            -- Tracks
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
            CREATE INDEX IF NOT EXISTS idx_tracks_ts ON tracks(ts);

            -- Drops
            CREATE TABLE IF NOT EXISTS drops (
              drop_id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              context TEXT,
              seed TEXT,
              run_id TEXT,
              track_id TEXT,
              meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_drops_ts ON drops(ts);

            -- Events
            CREATE TABLE IF NOT EXISTS events (
              event_id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              kind TEXT,
              actor TEXT,
              meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
            CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);

            -- Marketing posts
            -- IMPORTANT: runtime query expects column 'id'
            CREATE TABLE IF NOT EXISTS marketing_posts (
              id TEXT PRIMARY KEY,
              ts TEXT NOT NULL,
              platform TEXT,
              status TEXT,
              content TEXT,
              meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_marketing_posts_ts ON marketing_posts(ts);
            CREATE INDEX IF NOT EXISTS idx_marketing_posts_status ON marketing_posts(status);

            -- Playlists (minimal)
            CREATE TABLE IF NOT EXISTS playlists (
              playlist_id TEXT PRIMARY KEY,
              ts TEXT,
              context TEXT,
              payload TEXT,
              meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_playlists_ts ON playlists(ts);
            """
        )

        # ------------------------------------------------------------------
        # Seed data: minimal deterministic set
        # ------------------------------------------------------------------
        ts = now_iso_fixed()

        track_id = "00000000-0000-0000-0000-000000000001"
        drop_id = "10000000-0000-0000-0000-000000000001"
        run_id = "20000000-0000-0000-0000-000000000001"
        event_id = "30000000-0000-0000-0000-000000000001"
        post_id = "40000000-0000-0000-0000-000000000001"
        playlist_id = "50000000-0000-0000-0000-000000000001"

        # Track seed row
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
                stable_json({"fixture": True}),
            ),
        )

        # Drop seed row
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
                stable_json({"fixture": True, "provider": "stub"}),
            ),
        )

        # Event seed row
        cur.execute(
            """
            INSERT INTO events (
              event_id, ts, kind, actor, meta
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_id,
                ts,
                "fixture.seeded",
                "fixture",
                stable_json({"ok": True, "run_id": run_id, "drop_id": drop_id}),
            ),
        )

        # Marketing seed row (status=draft so pending query can find it)
        cur.execute(
            """
            INSERT INTO marketing_posts (
              id, ts, platform, status, content, meta
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                post_id,
                ts,
                "x",
                "draft",
                stable_json({"hook": "CI seed post", "cta": "Listen now", "track_id": track_id}),
                stable_json({"fixture": True, "run_id": run_id, "drop_id": drop_id}),
            ),
        )

        # Playlist seed row (payload shape is not critical; keep deterministic)
        cur.execute(
            """
            INSERT INTO playlists (
              playlist_id, ts, context, payload, meta
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                playlist_id,
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
                            {"id": track_id, "title": "CI Seed Track", "artifact_path": "data/tracks/ci_seed_track.wav"}
                        ],
                    }
                ),
                stable_json({"fixture": True}),
            ),
        )

        con.commit()

        # Stabilize file layout/size for determinism
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
