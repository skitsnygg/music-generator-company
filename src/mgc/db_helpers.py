# src/mgc/db_helpers.py
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------
# errors
# ----------------------------

class NotFoundError(RuntimeError):
    pass


# ----------------------------
# connection
# ----------------------------

def sqlite_connect(db_path: str) -> sqlite3.Connection:
    """
    Standardized SQLite connection.

    Notes:
      - No commits here (caller controls transactions).
      - WAL + FK on for better safety/throughput.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn
    except Exception as ex:
        raise RuntimeError(f"Failed to open SQLite DB at {db_path}: {ex}") from ex


# ----------------------------
# schema helpers (NO commit)
# ----------------------------

def ensure_playlist_runs_table(conn: sqlite3.Connection) -> None:
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


# ----------------------------
# rows
# ----------------------------

@dataclass(frozen=True)
class PlaylistRow:
    id: str
    created_at: str
    slug: str
    name: str
    context: Optional[str]
    mood: Optional[str]
    genre: Optional[str]
    target_minutes: int
    seed: int
    track_count: int
    total_duration_sec: int
    json_path: str


@dataclass(frozen=True)
class TrackRow:
    id: str
    created_at: str
    title: str
    mood: str
    genre: str
    bpm: int
    duration_sec: float
    full_path: str
    preview_path: str
    status: str


@dataclass(frozen=True)
class PlaylistRunRow:
    id: str
    created_at: str
    playlist_id: str
    seed: Optional[int]
    track_ids: List[str]
    export_path: Optional[str]
    notes: Optional[str]


@dataclass(frozen=True)
class ResolvedRef:
    kind: str  # "run" or "playlist"
    id: str    # run_id or playlist_id
    created_at: str
    playlist_id: str
    seed: Optional[int]
    track_ids: List[str]


# ----------------------------
# small helpers
# ----------------------------

def resolve_json_path(json_path: str) -> Path:
    p = Path(json_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# playlists
# ----------------------------

def db_list_playlists(conn: sqlite3.Connection, limit: int) -> List[PlaylistRow]:
    sql = """
    SELECT
      id, created_at, slug, name, context, mood, genre,
      target_minutes, seed, track_count, total_duration_sec, json_path
    FROM playlists
    ORDER BY created_at DESC
    LIMIT ?
    """
    rows = conn.execute(sql, (int(limit),)).fetchall()
    out: List[PlaylistRow] = []
    for r in rows:
        out.append(
            PlaylistRow(
                id=str(r["id"]),
                created_at=str(r["created_at"]),
                slug=str(r["slug"]),
                name=str(r["name"]),
                context=(str(r["context"]) if r["context"] is not None else None),
                mood=(str(r["mood"]) if r["mood"] is not None else None),
                genre=(str(r["genre"]) if r["genre"] is not None else None),
                target_minutes=_safe_int(r["target_minutes"]),
                seed=_safe_int(r["seed"]),
                track_count=_safe_int(r["track_count"]),
                total_duration_sec=_safe_int(r["total_duration_sec"]),
                json_path=str(r["json_path"]),
            )
        )
    return out


def db_get_playlist(conn: sqlite3.Connection, playlist_id: str) -> PlaylistRow:
    sql = """
    SELECT
      id, created_at, slug, name, context, mood, genre,
      target_minutes, seed, track_count, total_duration_sec, json_path
    FROM playlists
    WHERE id = ?
    LIMIT 1
    """
    r = conn.execute(sql, (playlist_id,)).fetchone()
    if not r:
        raise NotFoundError(f"Playlist not found: {playlist_id}")

    return PlaylistRow(
        id=str(r["id"]),
        created_at=str(r["created_at"]),
        slug=str(r["slug"]),
        name=str(r["name"]),
        context=(str(r["context"]) if r["context"] is not None else None),
        mood=(str(r["mood"]) if r["mood"] is not None else None),
        genre=(str(r["genre"]) if r["genre"] is not None else None),
        target_minutes=_safe_int(r["target_minutes"]),
        seed=_safe_int(r["seed"]),
        track_count=_safe_int(r["track_count"]),
        total_duration_sec=_safe_int(r["total_duration_sec"]),
        json_path=str(r["json_path"]),
    )


def db_list_playlists_by_slug(conn: sqlite3.Connection, slug: str, limit: int) -> List[PlaylistRow]:
    sql = """
    SELECT
      id, created_at, slug, name, context, mood, genre,
      target_minutes, seed, track_count, total_duration_sec, json_path
    FROM playlists
    WHERE slug = ?
    ORDER BY created_at DESC
    LIMIT ?
    """
    rows = conn.execute(sql, (slug, int(limit))).fetchall()
    out: List[PlaylistRow] = []
    for r in rows:
        out.append(
            PlaylistRow(
                id=str(r["id"]),
                created_at=str(r["created_at"]),
                slug=str(r["slug"]),
                name=str(r["name"]),
                context=(str(r["context"]) if r["context"] is not None else None),
                mood=(str(r["mood"]) if r["mood"] is not None else None),
                genre=(str(r["genre"]) if r["genre"] is not None else None),
                target_minutes=_safe_int(r["target_minutes"]),
                seed=_safe_int(r["seed"]),
                track_count=_safe_int(r["track_count"]),
                total_duration_sec=_safe_int(r["total_duration_sec"]),
                json_path=str(r["json_path"]),
            )
        )
    return out


def db_playlist_track_ids(conn: sqlite3.Connection, playlist_id: str) -> List[str]:
    rows = conn.execute(
        """
        SELECT track_id
        FROM playlist_items
        WHERE playlist_id = ?
        ORDER BY position ASC
        """,
        (playlist_id,),
    ).fetchall()
    return [str(r["track_id"]) for r in rows]


def db_build_playlist_json(conn: sqlite3.Connection, pl: PlaylistRow, built_at: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministic builder:
      - DO NOT call now() here.
      - Caller may pass built_at if they want a timestamp.
    """
    sql = """
    SELECT
      pi.position,
      t.id AS track_id,
      t.title,
      t.mood,
      t.genre,
      t.bpm,
      t.duration_sec,
      t.full_path,
      t.preview_path,
      t.status,
      t.created_at
    FROM playlist_items pi
    JOIN tracks t ON t.id = pi.track_id
    WHERE pi.playlist_id = ?
    ORDER BY pi.position ASC
    """
    rows = conn.execute(sql, (pl.id,)).fetchall()

    tracks: List[Dict[str, Any]] = []
    for r in rows:
        tracks.append(
            {
                "id": str(r["track_id"]),
                "title": str(r["title"]),
                "mood": str(r["mood"]),
                "genre": str(r["genre"]),
                "bpm": _safe_int(r["bpm"]),
                "duration_sec": _safe_float(r["duration_sec"]),
                "full_path": str(r["full_path"]),
                "preview_path": str(r["preview_path"]),
                "status": str(r["status"]),
                "created_at": str(r["created_at"]),
                "position": _safe_int(r["position"]),
            }
        )

    obj: Dict[str, Any] = {
        "id": pl.id,
        "created_at": pl.created_at,
        "slug": pl.slug,
        "name": pl.name,
        "context": pl.context,
        "mood": pl.mood,
        "genre": pl.genre,
        "target_minutes": pl.target_minutes,
        "seed": pl.seed,
        "track_count": pl.track_count,
        "total_duration_sec": pl.total_duration_sec,
        "json_path": pl.json_path,
        "tracks": tracks,
    }
    if built_at is not None:
        obj["built_from_db_at"] = built_at
    return obj


# ----------------------------
# tracks
# ----------------------------

def db_list_tracks(
    conn: sqlite3.Connection,
    limit: int,
    mood: Optional[str],
    genre: Optional[str],
    status: Optional[str],
    bpm_min: Optional[int],
    bpm_max: Optional[int],
    q: Optional[str],
) -> List[TrackRow]:
    where: List[str] = []
    params: List[Any] = []

    if mood:
        where.append("mood = ?")
        params.append(mood)
    if genre:
        where.append("genre = ?")
        params.append(genre)
    if status:
        where.append("status = ?")
        params.append(status)
    if bpm_min is not None:
        where.append("bpm >= ?")
        params.append(int(bpm_min))
    if bpm_max is not None:
        where.append("bpm <= ?")
        params.append(int(bpm_max))
    if q:
        where.append("title LIKE ?")
        params.append(f"%{q}%")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
    SELECT id, created_at, title, mood, genre, bpm, duration_sec, full_path, preview_path, status
    FROM tracks
    {where_sql}
    ORDER BY created_at DESC
    LIMIT ?
    """
    params.append(int(limit))

    rows = conn.execute(sql, tuple(params)).fetchall()
    out: List[TrackRow] = []
    for r in rows:
        out.append(
            TrackRow(
                id=str(r["id"]),
                created_at=str(r["created_at"]),
                title=str(r["title"]),
                mood=str(r["mood"]),
                genre=str(r["genre"]),
                bpm=_safe_int(r["bpm"]),
                duration_sec=_safe_float(r["duration_sec"]),
                full_path=str(r["full_path"]),
                preview_path=str(r["preview_path"]),
                status=str(r["status"]),
            )
        )
    return out


def db_get_track(conn: sqlite3.Connection, track_id: str) -> TrackRow:
    sql = """
    SELECT id, created_at, title, mood, genre, bpm, duration_sec, full_path, preview_path, status
    FROM tracks
    WHERE id = ?
    LIMIT 1
    """
    r = conn.execute(sql, (track_id,)).fetchone()
    if not r:
        raise NotFoundError(f"Track not found: {track_id}")

    return TrackRow(
        id=str(r["id"]),
        created_at=str(r["created_at"]),
        title=str(r["title"]),
        mood=str(r["mood"]),
        genre=str(r["genre"]),
        bpm=_safe_int(r["bpm"]),
        duration_sec=_safe_float(r["duration_sec"]),
        full_path=str(r["full_path"]),
        preview_path=str(r["preview_path"]),
        status=str(r["status"]),
    )


def db_tracks_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    total_row = conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()
    total = int(total_row["n"]) if total_row and total_row["n"] is not None else 0

    moods = conn.execute("SELECT mood, COUNT(*) AS n FROM tracks GROUP BY mood ORDER BY n DESC").fetchall()
    genres = conn.execute("SELECT genre, COUNT(*) AS n FROM tracks GROUP BY genre ORDER BY n DESC").fetchall()
    statuses = conn.execute("SELECT status, COUNT(*) AS n FROM tracks GROUP BY status ORDER BY n DESC").fetchall()

    bpm = conn.execute("SELECT MIN(bpm) AS min_bpm, MAX(bpm) AS max_bpm, AVG(bpm) AS avg_bpm FROM tracks").fetchone()
    dur = conn.execute(
        "SELECT MIN(duration_sec) AS min_dur, MAX(duration_sec) AS max_dur, AVG(duration_sec) AS avg_dur FROM tracks"
    ).fetchone()

    return {
        "total_tracks": total,
        "moods": [{"mood": str(r["mood"]), "count": int(r["n"])} for r in moods],
        "genres": [{"genre": str(r["genre"]), "count": int(r["n"])} for r in genres],
        "statuses": [{"status": str(r["status"]), "count": int(r["n"])} for r in statuses],
        "bpm": {
            "min": (_safe_int(bpm["min_bpm"]) if bpm and bpm["min_bpm"] is not None else None),
            "max": (_safe_int(bpm["max_bpm"]) if bpm and bpm["max_bpm"] is not None else None),
            "avg": (_safe_float(bpm["avg_bpm"]) if bpm and bpm["avg_bpm"] is not None else None),
        },
        "duration_sec": {
            "min": (_safe_float(dur["min_dur"]) if dur and dur["min_dur"] is not None else None),
            "max": (_safe_float(dur["max_dur"]) if dur and dur["max_dur"] is not None else None),
            "avg": (_safe_float(dur["avg_dur"]) if dur and dur["avg_dur"] is not None else None),
        },
    }


def db_duration_for_tracks(conn: sqlite3.Connection, track_ids: List[str]) -> float:
    if not track_ids:
        return 0.0

    total = 0.0
    chunk_size = 900
    for i in range(0, len(track_ids), chunk_size):
        chunk = track_ids[i : i + chunk_size]
        qmarks = ",".join(["?"] * len(chunk))
        row = conn.execute(
            f"SELECT SUM(duration_sec) AS s FROM tracks WHERE id IN ({qmarks})",
            tuple(chunk),
        ).fetchone()
        total += float(row["s"] or 0.0)
    return total


# ----------------------------
# playlist runs (NO commit)
# ----------------------------

def db_insert_playlist_run(
    conn: sqlite3.Connection,
    created_at: str,
    playlist_id: str,
    seed: Optional[int],
    track_ids: List[str],
    export_path: Optional[Path],
    notes: Optional[str],
) -> str:
    """
    Insert a playlist run row. No commit here; caller controls transaction.
    """
    ensure_playlist_runs_table(conn)

    run_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO playlist_runs (id, created_at, playlist_id, seed, track_ids_json, export_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            created_at,
            playlist_id,
            int(seed) if seed is not None else None,
            json.dumps([str(x) for x in track_ids], ensure_ascii=False),
            str(export_path) if export_path is not None else None,
            notes,
        ),
    )
    return run_id


def db_list_playlist_runs(conn: sqlite3.Connection, playlist_id: str, limit: int) -> List[PlaylistRunRow]:
    ensure_playlist_runs_table(conn)
    rows = conn.execute(
        """
        SELECT id, created_at, playlist_id, seed, track_ids_json, export_path, notes
        FROM playlist_runs
        WHERE playlist_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (playlist_id, int(limit)),
    ).fetchall()

    out: List[PlaylistRunRow] = []
    for r in rows:
        track_ids: List[str]
        try:
            parsed = json.loads(str(r["track_ids_json"]))
            if isinstance(parsed, list):
                track_ids = [str(x) for x in parsed]
            else:
                track_ids = []
        except Exception:
            track_ids = []

        out.append(
            PlaylistRunRow(
                id=str(r["id"]),
                created_at=str(r["created_at"]),
                playlist_id=str(r["playlist_id"]),
                seed=(int(r["seed"]) if r["seed"] is not None else None),
                track_ids=track_ids,
                export_path=(str(r["export_path"]) if r["export_path"] is not None else None),
                notes=(str(r["notes"]) if r["notes"] is not None else None),
            )
        )
    return out


def db_get_playlist_run(conn: sqlite3.Connection, run_id: str) -> Optional[PlaylistRunRow]:
    ensure_playlist_runs_table(conn)
    r = conn.execute(
        """
        SELECT id, created_at, playlist_id, seed, track_ids_json, export_path, notes
        FROM playlist_runs
        WHERE id = ?
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    if not r:
        return None

    try:
        parsed = json.loads(str(r["track_ids_json"]))
        if isinstance(parsed, list):
            track_ids = [str(x) for x in parsed]
        else:
            track_ids = []
    except Exception:
        track_ids = []

    return PlaylistRunRow(
        id=str(r["id"]),
        created_at=str(r["created_at"]),
        playlist_id=str(r["playlist_id"]),
        seed=(int(r["seed"]) if r["seed"] is not None else None),
        track_ids=track_ids,
        export_path=(str(r["export_path"]) if r["export_path"] is not None else None),
        notes=(str(r["notes"]) if r["notes"] is not None else None),
    )


# ----------------------------
# history/diff helpers
# ----------------------------

def resolve_ref(conn: sqlite3.Connection, ref: str) -> ResolvedRef:
    """
    Accept either:
      - playlist_runs.id (run id)
      - playlists.id (playlist id)
    """
    run = db_get_playlist_run(conn, ref)
    if run is not None:
        return ResolvedRef(
            kind="run",
            id=run.id,
            created_at=run.created_at,
            playlist_id=run.playlist_id,
            seed=run.seed,
            track_ids=run.track_ids,
        )

    pl = db_get_playlist(conn, ref)
    track_ids = db_playlist_track_ids(conn, pl.id)
    return ResolvedRef(
        kind="playlist",
        id=pl.id,
        created_at=pl.created_at,
        playlist_id=pl.id,
        seed=pl.seed,
        track_ids=track_ids,
    )


def compute_diff(a_ids: List[str], b_ids: List[str]) -> Dict[str, Any]:
    a_set = set(a_ids)
    b_set = set(b_ids)

    added = [tid for tid in b_ids if tid not in a_set]
    removed = [tid for tid in a_ids if tid not in b_set]

    a_pos = {tid: i for i, tid in enumerate(a_ids)}
    b_pos = {tid: i for i, tid in enumerate(b_ids)}
    moved: List[Dict[str, Any]] = []
    for tid in b_ids:
        if tid in a_pos and tid in b_pos and a_pos[tid] != b_pos[tid]:
            moved.append({"track_id": tid, "from": a_pos[tid], "to": b_pos[tid]})

    return {
        "added": added,
        "removed": removed,
        "moved": moved,
        "counts": {
            "a": len(a_ids),
            "b": len(b_ids),
            "added": len(added),
            "removed": len(removed),
            "moved": len(moved),
            "same_set": len(a_set & b_set),
        },
    }

# ----------------------------
# events (read-only helpers)
# ----------------------------

def db_list_events(
    conn: sqlite3.Connection,
    limit: int = 50,
    run_id: Optional[str] = None,
    event_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    where = []
    params: List[Any] = []

    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    if event_type:
        where.append("event_type = ?")
        params.append(event_type)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
    SELECT
      id, occurred_at, run_id, source,
      event_type, entity_type, entity_id, payload_json
    FROM events
    {where_sql}
    ORDER BY occurred_at DESC
    LIMIT ?
    """
    params.append(int(limit))

    rows = conn.execute(sql, tuple(params)).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            payload = json.loads(str(r["payload_json"]))
        except Exception:
            payload = {}

        out.append(
            {
                "id": str(r["id"]),
                "occurred_at": str(r["occurred_at"]),
                "run_id": str(r["run_id"]),
                "source": str(r["source"]),
                "event_type": str(r["event_type"]),
                "entity_type": str(r["entity_type"]),
                "entity_id": (str(r["entity_id"]) if r["entity_id"] is not None else None),
                "payload": payload,
            }
        )
    return out
