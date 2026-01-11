#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def now_iso_fixed() -> str:
    # Deterministic timestamp for fixture DB (avoid CI diffs).
    # Keep UTC + explicit offset format for readability.
    return datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat()


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

        # ----------------------------
        # schema: minimal but matches code expectations
        # ----------------------------
        cur.executescript(
            """
            -- tracks
            CREATE TABLE tracks (
              id TEXT PRIMARY KEY,
              title TEXT,
              artist TEXT,
              duration_sec INTEGER,
              genre TEXT,
              mood TEXT,

              bpm INTEGER,
              status TEXT,
              created_at TEXT,

              full_path TEXT,
              preview_path TEXT
            );

            -- playlists
            CREATE TABLE playlists (
              id TEXT PRIMARY KEY,
              slug TEXT NOT NULL,
              name TEXT,
              context TEXT,
              created_at TEXT NOT NULL,

              genre TEXT,
              mood TEXT,
              target_minutes INTEGER,
              total_duration_sec INTEGER,

              seed TEXT,
              track_count INTEGER,
              json_path TEXT
            );

            -- playlist_items
            CREATE TABLE playlist_items (
              id TEXT PRIMARY KEY,
              playlist_id TEXT NOT NULL,
              track_id TEXT NOT NULL,
              position INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY (playlist_id) REFERENCES playlists(id),
              FOREIGN KEY (track_id) REFERENCES tracks(id)
            );

            CREATE INDEX idx_playlist_items_playlist_id ON playlist_items(playlist_id);
            CREATE INDEX idx_playlist_items_track_id ON playlist_items(track_id);
            CREATE UNIQUE INDEX ux_playlist_items_playlist_pos ON playlist_items(playlist_id, position);

            -- playlist_runs
            CREATE TABLE playlist_runs (
              id TEXT PRIMARY KEY,
              playlist_id TEXT NOT NULL,
              stamp TEXT NOT NULL,
              created_at TEXT NOT NULL,

              seed TEXT,
              track_ids_json TEXT,
              export_path TEXT,
              notes TEXT,

              FOREIGN KEY (playlist_id) REFERENCES playlists(id)
            );

            CREATE INDEX idx_playlist_runs_playlist_id ON playlist_runs(playlist_id);
            CREATE INDEX idx_playlist_runs_stamp ON playlist_runs(stamp);

            -- events (match events.py expectations: occurred_at + metadata columns)
            CREATE TABLE events (
              id TEXT PRIMARY KEY,

              occurred_at TEXT NOT NULL,
              event_type TEXT NOT NULL,

              entity_type TEXT,
              entity_id TEXT,
              run_id TEXT,
              source TEXT,

              payload_json TEXT
            );

            CREATE INDEX idx_events_occurred_at ON events(occurred_at);
            CREATE INDEX idx_events_event_type ON events(event_type);
            CREATE INDEX idx_events_entity ON events(entity_type, entity_id);
            CREATE INDEX idx_events_run_id ON events(run_id);

            -- marketing_posts (minimal stub table for CI)
            CREATE TABLE marketing_posts (
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              status TEXT,
              title TEXT,
              body TEXT,
              platform TEXT,

              track_id TEXT,
              payload_json TEXT,
              metadata_json TEXT
            );

            CREATE INDEX idx_marketing_posts_created_at ON marketing_posts(created_at);
            CREATE INDEX idx_marketing_posts_status ON marketing_posts(status);
            CREATE INDEX idx_marketing_posts_track_id ON marketing_posts(track_id);
            """
        )

        # ----------------------------
        # seed data: smallest useful set (fully deterministic IDs + timestamps)
        # ----------------------------
        created = now_iso_fixed()

        # Deterministic IDs (avoid uuid4 randomness)
        track_ids: List[str] = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
        ]
        playlist_id = "10000000-0000-0000-0000-000000000001"
        run_id = "20000000-0000-0000-0000-000000000001"
        event_id = "30000000-0000-0000-0000-000000000001"

        durations = [180, 190, 200]

        # 3 tracks
        for i, tid in enumerate(track_ids):
            dur = durations[i]
            cur.execute(
                """
                INSERT INTO tracks (
                  id, title, artist, duration_sec, genre, mood,
                  bpm, status, created_at,
                  full_path, preview_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid,
                    f"CI Track {i+1}",
                    "CI Artist",
                    dur,
                    "ci_genre",
                    "focus",
                    120,
                    "ready",
                    created,
                    f"/ci/full/track_{i+1}.wav",
                    f"/ci/preview/track_{i+1}.mp3",
                ),
            )

        # 1 playlist
        playlist_seed = "ci_seed"
        total_dur = sum(durations)
        cur.execute(
            """
            INSERT INTO playlists (
              id, slug, name, context, created_at,
              genre, mood, target_minutes, total_duration_sec,
              seed, track_count, json_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                playlist_id,
                "focus_radio",
                "Focus Radio (CI)",
                "focus",
                created,
                "ci_genre",
                "focus",
                15,
                total_dur,
                playlist_seed,
                len(track_ids),
                "data/playlists/focus_radio.json",
            ),
        )

        # playlist items (deterministic ids)
        playlist_item_ids = [
            "40000000-0000-0000-0000-000000000001",
            "40000000-0000-0000-0000-000000000002",
            "40000000-0000-0000-0000-000000000003",
        ]
        for pos, (piid, tid) in enumerate(zip(playlist_item_ids, track_ids), start=1):
            cur.execute(
                """
                INSERT INTO playlist_items (
                  id, playlist_id, track_id, position, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (piid, playlist_id, tid, pos, created),
            )

        # 1 playlist run
        cur.execute(
            """
            INSERT INTO playlist_runs (
              id, playlist_id, stamp, created_at,
              seed, track_ids_json, export_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                playlist_id,
                "ci_seed",
                created,
                playlist_seed,
                '["' + '","'.join(track_ids) + '"]',
                "data/playlists/focus_radio.json",
                "seeded by make_fixture_db.py",
            ),
        )

        # 1 event
        cur.execute(
            """
            INSERT INTO events (
              id, occurred_at, event_type, entity_type, entity_id, run_id, source, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                created,
                "fixture.seeded",
                "playlist_run",
                run_id,
                run_id,
                "fixture",
                '{"ok":true}',
            ),
        )

        con.commit()

        # Make DB file more stable (prevents size jitter as pages get allocated differently)
        try:
            cur.execute("VACUUM;")
            con.commit()
        except Exception:
            # If VACUUM isn't allowed for any reason, ignore; determinism is still improved.
            pass
    finally:
        con.close()

    print(f"Wrote fixture DB: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
