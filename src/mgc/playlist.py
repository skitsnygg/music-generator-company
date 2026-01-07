from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter



@dataclass(frozen=True)
class Track:
    id: str
    created_at: str
    title: str
    mood: str
    genre: str
    bpm: int
    duration_sec: float
    full_path: str
    preview_path: str


PLAYLIST_PRESETS = {
    "focus":   {"mood": "focus",   "bpm_window": (80, 125),  "name": "Focus Radio"},
    "workout": {"mood": "workout", "bpm_window": (125, 170), "name": "Workout Radio"},
    "sleep":   {"mood": "sleep",   "bpm_window": (50, 90),   "name": "Sleep Radio"},
}


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_tracks(
    conn: sqlite3.Connection,
    *,
    mood: Optional[str] = None,
    genre: Optional[str] = None,
    bpm_min: Optional[int] = None,
    bpm_max: Optional[int] = None,
    limit: int = 400,
) -> List[Track]:
    where = ["status = 'generated'"]
    args: List[Any] = []

    if mood:
        where.append("mood = ?")
        args.append(mood)
    if genre:
        where.append("genre = ?")
        args.append(genre)
    if bpm_min is not None:
        where.append("bpm >= ?")
        args.append(int(bpm_min))
    if bpm_max is not None:
        where.append("bpm <= ?")
        args.append(int(bpm_max))

    sql = f"""
    SELECT id, created_at, title, mood, genre, bpm, duration_sec, full_path, preview_path
    FROM tracks
    WHERE {" AND ".join(where)}
    ORDER BY datetime(created_at) DESC, rowid DESC
    LIMIT ?
    """
    args.append(int(limit))

    rows = conn.execute(sql, args).fetchall()
    return [
        Track(
            id=r["id"],
            created_at=r["created_at"],
            title=r["title"],
            mood=r["mood"],
            genre=r["genre"],
            bpm=int(r["bpm"]),
            duration_sec=float(r["duration_sec"]),
            full_path=r["full_path"],
            preview_path=r["preview_path"],
        )
        for r in rows
    ]


def _shuffle_with_diversity(tracks: List[Track], *, seed: int, avoid_same_genre_run: int = 2) -> List[Track]:
    """Shuffle while trying to avoid long runs of the same genre."""
    rng = random.Random(seed)
    pool = tracks[:]
    rng.shuffle(pool)

    result: List[Track] = []
    recent_genres: List[str] = []

    while pool:
        pick_idx = None
        for i, t in enumerate(pool[: min(len(pool), 20)]):
            if len(recent_genres) < avoid_same_genre_run:
                pick_idx = i
                break
            if not all(g == t.genre for g in recent_genres[-avoid_same_genre_run:]):
                pick_idx = i
                break
        if pick_idx is None:
            pick_idx = 0

        t = pool.pop(pick_idx)
        result.append(t)
        recent_genres.append(t.genre)

    return result



def _fetch_recently_used_track_ids(
    conn: sqlite3.Connection,
    *,
    context: str | None,
    slug: str | None,
    lookback_playlists: int = 3,
) -> set[str]:
    """Return track_ids used in the most recent playlists (dedupe)."""
    where = []
    args: list[object] = []

    if context:
        where.append("context = ?")
        args.append(context)
    elif slug:
        where.append("slug = ?")
        args.append(slug)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(
        f"""
        SELECT id FROM playlists
        {where_sql}
        ORDER BY datetime(created_at) DESC
        LIMIT ?
        """,
        args + [int(lookback_playlists)],
    ).fetchall()

    if not rows:
        return set()

    playlist_ids = [r["id"] for r in rows]
    qmarks = ",".join(["?"] * len(playlist_ids))

    items = conn.execute(
        f"""
        SELECT track_id FROM playlist_items
        WHERE playlist_id IN ({qmarks})
        """,
        playlist_ids,
    ).fetchall()

    return {r["track_id"] for r in items}


def build_playlist(
    *,
    db_path: Path,
    name: str = "Auto Playlist",
    slug: str | None = None,
    context: Optional[str] = None,
    mood: Optional[str] = None,
    genre: Optional[str] = None,
    target_minutes: int = 20,
    bpm_window: Optional[Tuple[int, int]] = None,
    seed: int = 1,
    lookback_playlists: int = 3,
) -> Dict[str, Any]:
    """
    Spotify-like 'radio' playlist builder using metadata only.

    - Filters: mood/genre + bpm window
    - Bias: recency (DB order)
    - Shuffle: diversity constraint
    - Fill: until target duration
    """
    if context and context in PLAYLIST_PRESETS:
        preset = PLAYLIST_PRESETS[context]
        name = preset.get("name", name)
        mood = preset.get("mood", mood)
        bpm_window = preset.get("bpm_window", bpm_window)

    conn = _connect(db_path)
    try:
        bpm_min, bpm_max = (None, None)
        if bpm_window:
            bpm_min, bpm_max = bpm_window

        candidates = _fetch_tracks(
            conn,
            mood=mood,
            genre=genre,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            limit=400,
        )

        if not candidates:
            # fallback: no filters
            candidates = _fetch_tracks(conn, limit=400)

        # Dedupe: avoid tracks used in recent playlists for same context/slug
        recently_used = _fetch_recently_used_track_ids(
            conn,
            context=context,
            slug=slug,
            lookback_playlists=lookback_playlists,
        )
        dedupe = {
            "requested_lookback": int(lookback_playlists),
            "excluded_count": 0,
            "applied": False,
            "reason": None,
        }

        if recently_used:
            filtered = [t for t in candidates if t.id not in recently_used]
            dedupe["excluded_count"] = len(candidates) - len(filtered)

            # Only apply dedupe if we still have enough tracks to build something meaningful.
            # Otherwise repeating is better than returning an empty/boring playlist.
            if len(filtered) >= 10:
                candidates = filtered
                dedupe["applied"] = True
            else:
                dedupe["reason"] = "not_enough_unique_tracks"

        ordered = _shuffle_with_diversity(candidates, seed=seed)

        target_sec = int(target_minutes * 60)
        total = 0
        items: List[Dict[str, Any]] = []

        for t in ordered:
            if total >= target_sec and len(items) >= 3:
                break
            items.append(
                {
                    "track_id": t.id,
                    "title": t.title,
                    "mood": t.mood,
                    "genre": t.genre,
                    "bpm": t.bpm,
                    "duration_sec": t.duration_sec,
                    "preview_path": t.preview_path,
                    "full_path": t.full_path,
                    "created_at": t.created_at,
                }
            )
            total += int(t.duration_sec)

        # Summary stats (helps debug and feels more 'product')
        bpms = [int(x["bpm"]) for x in items if x.get("bpm") is not None]
        avg_bpm = int(round(sum(bpms) / len(bpms))) if bpms else None
        bpm_min_used = min(bpms) if bpms else None
        bpm_max_used = max(bpms) if bpms else None

        mood_counts = Counter([x.get("mood") for x in items if x.get("mood")])
        genre_counts = Counter([x.get("genre") for x in items if x.get("genre")])

        unique_track_count = len({x["track_id"] for x in items})
        duration_minutes = round(total / 60.0, 2)

        return {
            "name": name,
            "filters": {
                "context": context,
                "mood": mood,
                "genre": genre,
                "bpm_window": list(bpm_window) if bpm_window else None,
                "target_minutes": target_minutes,
                "seed": seed,
                "lookback_playlists": lookback_playlists,
            },
            "stats": {
                "track_count": len(items),
                "unique_track_count": unique_track_count,
                "total_duration_sec": total,
                "duration_minutes": duration_minutes,
                "avg_bpm": avg_bpm,
                "bpm_min": bpm_min_used,
                "bpm_max": bpm_max_used,
                "mood_counts": dict(mood_counts),
                "genre_counts": dict(genre_counts),
            },
            "dedupe": dedupe,
            "items": items,
        }
    finally:
        conn.close()
