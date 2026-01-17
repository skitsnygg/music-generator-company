from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    "focus": {"mood": "focus", "bpm_window": (80, 125), "name": "Focus Radio"},
    "workout": {"mood": "workout", "bpm_window": (125, 170), "name": "Workout Radio"},
    "sleep": {"mood": "sleep", "bpm_window": (50, 90), "name": "Sleep Radio"},
}


# -----------------------------------------------------------------------------
# SQLite helpers
# -----------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r["name"]) for r in rows}


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------

def _stable_seed(*parts: str, base_seed: int = 1) -> int:
    """
    Deterministic seed derived from base_seed + parts.
    Returns a stable 32-bit int.
    """
    s = f"{int(base_seed)}:" + ":".join((p or "").strip() for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# -----------------------------------------------------------------------------
# Track fetch (schema compatible)
# -----------------------------------------------------------------------------

def _fetch_tracks(
    conn: sqlite3.Connection,
    mood: str | None = None,
    context: str | None = None,
    genre: str | None = None,
    bpm_min: int | None = None,
    bpm_max: int | None = None,
    limit: int = 400,
    **_ignored: Any,
) -> List[Track]:
    """
    Fetch candidate tracks with backward-compatible column aliasing.

    Newer schema:
      - id, full_path, preview_path, duration_sec, bpm, created_at

    Older schema (fixtures/CI):
      - track_id, artifact_path, ts
      - optional: title, mood, genre, provider, meta/meta_json
    """
    if mood is None:
        mood = context
    mood = (mood or "").strip()
    genre = (genre or "").strip()

    cur = conn.execute("PRAGMA table_info(tracks)")
    cols = {row[1] for row in cur.fetchall()}

    def have(name: str) -> bool:
        return name in cols

    # Required identity + file path (alias if needed)
    if have("id"):
        id_expr = "id"
        id_order = "id ASC"
    elif have("track_id"):
        id_expr = "track_id AS id"
        id_order = "track_id ASC"
    else:
        raise RuntimeError("tracks table missing an id column (need id or track_id)")

    if have("full_path"):
        path_expr = "full_path"
    elif have("artifact_path"):
        path_expr = "artifact_path AS full_path"
    else:
        raise RuntimeError("tracks table missing a path column (need full_path or artifact_path)")

    # Optional columns with safe NULLs
    created_expr = (
        "created_at"
        if have("created_at")
        else ("ts AS created_at" if have("ts") else "NULL AS created_at")
    )
    bpm_expr = "bpm" if have("bpm") else "NULL AS bpm"
    dur_expr = "duration_sec" if have("duration_sec") else "NULL AS duration_sec"
    preview_expr = "preview_path" if have("preview_path") else "NULL AS preview_path"

    title_expr = "title" if have("title") else "NULL AS title"
    mood_expr = "mood" if have("mood") else "NULL AS mood"
    genre_expr = "genre" if have("genre") else "NULL AS genre"

    # WHERE clauses only if columns exist
    where: list[str] = []
    params: list[Any] = []

    if have("mood") and mood:
        where.append("mood = ?")
        params.append(mood)

    if have("genre") and genre:
        where.append("genre = ?")
        params.append(genre)

    if have("bpm") and bpm_min is not None:
        where.append("bpm >= ?")
        params.append(int(bpm_min))

    if have("bpm") and bpm_max is not None:
        where.append("bpm <= ?")
        params.append(int(bpm_max))

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # Deterministic ordering
    order_terms: list[str] = []
    if have("created_at"):
        order_terms.append("created_at DESC")
    elif have("ts"):
        order_terms.append("ts DESC")
    order_terms.append(id_order)
    order_sql = " ORDER BY " + ", ".join(order_terms)

    sql = f"""
        SELECT
            {id_expr},
            {path_expr},
            {preview_expr},
            {dur_expr},
            {bpm_expr},
            {created_expr},
            {title_expr},
            {mood_expr},
            {genre_expr}
        FROM tracks
        {where_sql}
        {order_sql}
        LIMIT ?
    """
    params.append(int(limit))

    cur2 = conn.execute(sql, params)
    rows = cur2.fetchall()

    out: List[Track] = []
    for r in rows:
        tid = str(r["id"]) if r["id"] is not None else ""
        full_path = str(r["full_path"]) if r["full_path"] is not None else ""
        preview_path = str(r["preview_path"]) if r["preview_path"] is not None else ""

        # Defaults for older schema
        created_at = str(r["created_at"]) if r["created_at"] is not None else ""
        title = str(r["title"]) if r["title"] is not None else ""
        mood_val = str(r["mood"]) if r["mood"] is not None else (mood or "")
        genre_val = str(r["genre"]) if r["genre"] is not None else (genre or "")

        bpm_val = int(r["bpm"]) if r["bpm"] is not None else 0
        dur_val = float(r["duration_sec"]) if r["duration_sec"] is not None else 0.0

        # Ensure non-empty id/path
        if not tid or not full_path:
            continue


        # Skip missing files deterministically (prevents stale DB rows from breaking runs)
        p = Path(full_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            continue

        out.append(
            Track(
                id=tid,
                created_at=created_at,
                title=title,
                mood=mood_val,
                genre=genre_val,
                bpm=bpm_val,
                duration_sec=dur_val,
                full_path=full_path,
                preview_path=preview_path,
            )
        )

    return out


def _shuffle_with_diversity(
    tracks: List[Track],
    *,
    seed: int,
    avoid_same_genre_run: int = 2,
) -> List[Track]:
    """
    Shuffle while trying to avoid long runs of the same genre.
    Deterministic given (tracks, seed).
    """
    rng = random.Random(int(seed))

    pool = tracks[:]
    rng.shuffle(pool)

    result: List[Track] = []
    recent_genres: List[str] = []

    while pool:
        pick_idx: Optional[int] = None
        window = pool[: min(len(pool), 20)]

        for i, t in enumerate(window):
            if len(recent_genres) < avoid_same_genre_run:
                pick_idx = i
                break
            last = recent_genres[-avoid_same_genre_run:]
            if not all(g == t.genre for g in last):
                pick_idx = i
                break

        if pick_idx is None:
            pick_idx = 0

        t = pool.pop(pick_idx)
        result.append(t)
        recent_genres.append(t.genre)

    return result


# -----------------------------------------------------------------------------
# Dedupe via recent playlists (optional)
# -----------------------------------------------------------------------------

def _fetch_recently_used_track_ids(
    conn: sqlite3.Connection,
    *,
    context: str | None,
    slug: str | None,
    lookback_playlists: int | None = 3,
) -> Tuple[set[str], Dict[str, Any]]:
    """
    Return (track_ids_used, debug_info).
    If playlist tables aren't available, returns empty set and notes why.
    """
    lookback = int(lookback_playlists or 0)

    debug: Dict[str, Any] = {
        "requested_lookback": lookback,
        "excluded_count": 0,
        "applied": False,
        "reason": None,
    }

    if lookback <= 0:
        debug["reason"] = "lookback_disabled"
        return set(), debug

    if not _table_exists(conn, "playlists") or not _table_exists(conn, "playlist_items"):
        debug["reason"] = "playlist_tables_missing"
        return set(), debug

    pcols = _columns(conn, "playlists")
    icols = _columns(conn, "playlist_items")

    if "id" not in pcols or "playlist_id" not in icols or "track_id" not in icols:
        debug["reason"] = "playlist_schema_missing_columns"
        return set(), debug

    where: List[str] = []
    args: List[Any] = []

    if context and "context" in pcols:
        where.append("context = ?")
        args.append(context)
    elif slug and "slug" in pcols:
        where.append("slug = ?")
        args.append(slug)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # Prefer created_at if present; else fall back to ts; else id
    if "created_at" in pcols:
        order_sql = "ORDER BY created_at DESC, id DESC"
    elif "ts" in pcols:
        order_sql = "ORDER BY ts DESC, id DESC"
    else:
        order_sql = "ORDER BY id DESC"

    rows = conn.execute(
        f"""
        SELECT id FROM playlists
        {where_sql}
        {order_sql}
        LIMIT ?
        """,
        args + [lookback],
    ).fetchall()

    if not rows:
        debug["reason"] = "no_recent_playlists"
        return set(), debug

    playlist_ids = [str(r["id"]) for r in rows]
    qmarks = ",".join(["?"] * len(playlist_ids))

    items = conn.execute(
        f"""
        SELECT track_id FROM playlist_items
        WHERE playlist_id IN ({qmarks})
        """,
        playlist_ids,
    ).fetchall()

    return {str(r["track_id"]) for r in items}, debug


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_playlist(
    *,
    db_path: Path,
    name: str = "Auto Playlist",
    slug: str | None = None,
    context: Optional[str] = None,
    mood: Optional[str] = None,
    genre: Optional[str] = None,
    target_minutes: int | None = 20,
    bpm_window: Optional[Tuple[int, int]] = None,
    # Determinism controls
    base_seed: int = 1,
    period_key: Optional[str] = None,
    lookback_playlists: int | None = 3,
) -> Dict[str, Any]:
    """
    Spotify-like 'radio' playlist builder using metadata only.

    - Filters: mood/genre + bpm window (if schema supports those columns)
    - Bias: recency (DB order)
    - Shuffle: diversity constraint (deterministic)
    - Fill: until target duration (min 3 tracks if possible)

    Returns:
      {
        "tracks": List[Track],     # programmatic use (weekly copies these)
        "items":  List[dict],      # JSON-friendly mirror of tracks
        ...
      }
    """
    if context and context in PLAYLIST_PRESETS:
        preset = PLAYLIST_PRESETS[context]
        name = preset.get("name", name)
        mood = preset.get("mood", mood)
        bpm_window = preset.get("bpm_window", bpm_window)

    # Allow callers to pass None (weekly optional knob)
    if target_minutes is None:
        target_minutes = 20

    bpm_min, bpm_max = (None, None)
    if bpm_window:
        bpm_min, bpm_max = bpm_window

    conn = _connect(db_path)
    try:
        candidates = _fetch_tracks(
            conn,
            mood=mood,
            genre=genre,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            limit=400,
        )

        if not candidates:
            candidates = _fetch_tracks(conn, limit=400)

        recently_used, dedupe = _fetch_recently_used_track_ids(
            conn,
            context=context,
            slug=slug,
            lookback_playlists=lookback_playlists,
        )

        if recently_used:
            filtered = [t for t in candidates if t.id not in recently_used]
            dedupe["excluded_count"] = len(candidates) - len(filtered)

            # Only apply dedupe if still enough material
            if len(filtered) >= 10:
                candidates = filtered
                dedupe["applied"] = True
            else:
                dedupe["reason"] = "not_enough_unique_tracks"

        seed = _stable_seed(
            context or "",
            slug or "",
            period_key or "",
            base_seed=base_seed,
        )

        ordered = _shuffle_with_diversity(candidates, seed=seed)

        target_sec = int(target_minutes * 60)
        total = 0
        chosen: List[Track] = []

        def _dur(t: Track) -> int:
            # Older schema often has duration=0; treat as 0 but still allow selection.
            try:
                return int(float(t.duration_sec))
            except Exception:
                return 0

        for t in ordered:
            if total >= target_sec and len(chosen) >= 3:
                break
            chosen.append(t)
            total += _dur(t)

        # JSON-friendly mirror
        items: List[Dict[str, Any]] = []
        for t in chosen:
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

        # Stats
        bpms = [int(x["bpm"]) for x in items if x.get("bpm") is not None and int(x["bpm"]) > 0]
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
                "target_minutes": int(target_minutes),
                "base_seed": int(base_seed),
                "period_key": period_key,
                "derived_seed": int(seed),
                "lookback_playlists": int(lookback_playlists or 0),
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
            # Programmatic + JSON-friendly outputs:
            "tracks": chosen,
            "items": items,
        }
    finally:
        conn.close()
# -----------------------------
# Thin wrappers for schedules
# -----------------------------

def build_daily_playlist(
    *,
    db_path: Path,
    context: Optional[str] = None,
    period_key: Optional[str] = None,
    base_seed: int = 1,
    target_minutes: int | None = 20,
    lookback_playlists: int | None = 3,
) -> Dict[str, Any]:
    """Build a daily playlist (thin wrapper around build_playlist)."""
    name = "Daily Playlist"
    slug = f"daily-{context or 'mix'}-{period_key or 'latest'}"
    return build_playlist(
        db_path=db_path,
        name=name,
        slug=slug,
        context=context,
        target_minutes=target_minutes,
        base_seed=base_seed,
        period_key=period_key,
        lookback_playlists=lookback_playlists,
    )


def build_weekly_playlist(
    *,
    db_path: Path,
    context: Optional[str] = None,
    period_key: Optional[str] = None,
    base_seed: int = 1,
    target_minutes: int | None = 60,
    lookback_playlists: int | None = 3,
) -> Dict[str, Any]:
    """Build a weekly playlist (thin wrapper around build_playlist)."""
    name = "Weekly Playlist"
    slug = f"weekly-{context or 'mix'}-{period_key or 'latest'}"
    return build_playlist(
        db_path=db_path,
        name=name,
        slug=slug,
        context=context,
        target_minutes=target_minutes,
        base_seed=base_seed,
        period_key=period_key,
        lookback_playlists=lookback_playlists,
    )
