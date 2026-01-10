#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    out_path = Path("fixtures/ci_db.sqlite")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always regenerate from scratch for determinism
    if out_path.exists():
        out_path.unlink()

    con = sqlite3.connect(str(out_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
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
        # seed data: smallest useful set
        # ----------------------------
        created = now_iso()

        # 3 tracks
        track_ids: list[str] = []
        durations = []
        for i in range(3):
            tid = str(uuid.uuid4())
            track_ids.append(tid)
            dur = 180 + i * 10
            durations.append(dur)
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
        pid = str(uuid.uuid4())
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
                pid,
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

        # playlist items
        for idx, tid in enumerate(track_ids, start=1):
            cur.execute(
                """
                INSERT INTO playlist_items (
                  id, playlist_id, track_id, position, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), pid, tid, idx, created),
            )

        # 1 playlist run
        run_id = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO playlist_runs (
              id, playlist_id, stamp, created_at,
              seed, track_ids_json, export_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                pid,
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
                str(uuid.uuid4()),
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
    finally:
        con.close()

    print(f"Wrote fixture DB: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
