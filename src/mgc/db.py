from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  title TEXT NOT NULL,
  mood TEXT NOT NULL,
  genre TEXT NOT NULL,
  bpm INTEGER NOT NULL,
  duration_sec REAL NOT NULL,
  full_path TEXT NOT NULL,
  preview_path TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS marketing_posts (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  track_id TEXT NOT NULL,
  platform TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  FOREIGN KEY(track_id) REFERENCES tracks(id)
);


CREATE TABLE IF NOT EXISTS playlists (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  slug TEXT NOT NULL,
  name TEXT NOT NULL,
  context TEXT,
  mood TEXT,
  genre TEXT,
  bpm_min INTEGER,
  bpm_max INTEGER,
  target_minutes INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  track_count INTEGER NOT NULL,
  total_duration_sec INTEGER NOT NULL,
  json_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS playlist_items (
  playlist_id TEXT NOT NULL,
  track_id TEXT NOT NULL,
  position INTEGER NOT NULL,
  PRIMARY KEY (playlist_id, track_id),
  FOREIGN KEY(track_id) REFERENCES tracks(id),
  FOREIGN KEY(playlist_id) REFERENCES playlists(id)
);

"""

class DB:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

        

    def insert_track(self, row: Dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tracks (id, created_at, title, mood, genre, bpm, duration_sec, full_path, preview_path, status)
                VALUES (:id, :created_at, :title, :mood, :genre, :bpm, :duration_sec, :full_path, :preview_path, :status)
                """,
                row,
            )
            conn.commit()

    def insert_post(self, row: Dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO marketing_posts (id, created_at, track_id, platform, payload_json, status)
                VALUES (:id, :created_at, :track_id, :platform, :payload_json, :status)
                """,
                row,
            )
            conn.commit()

    def latest_track(self) -> Optional[dict]:
        with self.connect() as conn:
            cur = conn.execute("SELECT * FROM tracks ORDER BY created_at DESC LIMIT 1")
            r = cur.fetchone()
            return dict(r) if r else None
