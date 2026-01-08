#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from mgc.logging_setup import setup_logging  # type: ignore
except Exception:
    setup_logging = None  # type: ignore


# ----------------------------
# basic utils
# ----------------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def die(msg: str, code: int = 2) -> "NoReturn":
    eprint(msg)
    raise SystemExit(code)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        die(f"JSON file not found: {path}")
    except json.JSONDecodeError as ex:
        die(f"Invalid JSON in {path}: {ex}")


def write_json_file(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_requests() -> Any:
    try:
        import requests  # type: ignore
        return requests
    except Exception:
        die("Missing dependency: requests. Install with: pip install requests")


# ----------------------------
# export layout
# ----------------------------

def default_export_dir() -> Path:
    return Path(os.environ.get("MGC_PLAYLISTS_DIR", "data/playlists"))


def safe_slug(s: str) -> str:
    s = s.strip()
    out = "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_")).strip("-_")
    return out


def export_filename(playlist_id: str, slug: Optional[str] = None) -> str:
    if slug:
        return f"{safe_slug(slug)}_{playlist_id}.json"
    return f"{playlist_id}.json"


# ----------------------------
# DB access (your schema)
# ----------------------------

def sqlite_connect(db_path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as ex:
        die(f"Failed to open SQLite DB at {db_path}: {ex}")


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
    conn.commit()


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


def db_list_playlists(conn: sqlite3.Connection, limit: int) -> List[PlaylistRow]:
    sql = """
    SELECT
      id, created_at, slug, name, context, mood, genre,
      target_minutes, seed, track_count, total_duration_sec, json_path
    FROM playlists
    ORDER BY created_at DESC
    LIMIT ?
    """
    rows = conn.execute(sql, (limit,)).fetchall()
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
                target_minutes=int(r["target_minutes"]),
                seed=int(r["seed"]),
                track_count=int(r["track_count"]),
                total_duration_sec=int(r["total_duration_sec"]),
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
        die(f"Playlist not found: {playlist_id}")
    return PlaylistRow(
        id=str(r["id"]),
        created_at=str(r["created_at"]),
        slug=str(r["slug"]),
        name=str(r["name"]),
        context=(str(r["context"]) if r["context"] is not None else None),
        mood=(str(r["mood"]) if r["mood"] is not None else None),
        genre=(str(r["genre"]) if r["genre"] is not None else None),
        target_minutes=int(r["target_minutes"]),
        seed=int(r["seed"]),
        track_count=int(r["track_count"]),
        total_duration_sec=int(r["total_duration_sec"]),
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
                target_minutes=int(r["target_minutes"]),
                seed=int(r["seed"]),
                track_count=int(r["track_count"]),
                total_duration_sec=int(r["total_duration_sec"]),
                json_path=str(r["json_path"]),
            )
        )
    return out


def resolve_json_path(json_path: str) -> Path:
    p = Path(json_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


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


def db_build_playlist_json(conn: sqlite3.Connection, pl: PlaylistRow) -> Dict[str, Any]:
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
                "bpm": int(r["bpm"]),
                "duration_sec": float(r["duration_sec"]),
                "full_path": str(r["full_path"]),
                "preview_path": str(r["preview_path"]),
                "status": str(r["status"]),
                "created_at": str(r["created_at"]),
                "position": int(r["position"]),
            }
        )

    return {
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
        "built_from_db_at": now_iso(),
    }


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
                bpm=int(r["bpm"]),
                duration_sec=float(r["duration_sec"]),
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
        die(f"Track not found: {track_id}")
    return TrackRow(
        id=str(r["id"]),
        created_at=str(r["created_at"]),
        title=str(r["title"]),
        mood=str(r["mood"]),
        genre=str(r["genre"]),
        bpm=int(r["bpm"]),
        duration_sec=float(r["duration_sec"]),
        full_path=str(r["full_path"]),
        preview_path=str(r["preview_path"]),
        status=str(r["status"]),
    )


def db_tracks_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) AS n FROM tracks").fetchone()["n"]

    moods = conn.execute("SELECT mood, COUNT(*) AS n FROM tracks GROUP BY mood ORDER BY n DESC").fetchall()
    genres = conn.execute("SELECT genre, COUNT(*) AS n FROM tracks GROUP BY genre ORDER BY n DESC").fetchall()
    statuses = conn.execute("SELECT status, COUNT(*) AS n FROM tracks GROUP BY status ORDER BY n DESC").fetchall()

    bpm = conn.execute("SELECT MIN(bpm) AS min_bpm, MAX(bpm) AS max_bpm, AVG(bpm) AS avg_bpm FROM tracks").fetchone()
    dur = conn.execute("SELECT MIN(duration_sec) AS min_dur, MAX(duration_sec) AS max_dur, AVG(duration_sec) AS avg_dur FROM tracks").fetchone()

    return {
        "total_tracks": int(total),
        "moods": [{"mood": str(r["mood"]), "count": int(r["n"])} for r in moods],
        "genres": [{"genre": str(r["genre"]), "count": int(r["n"])} for r in genres],
        "statuses": [{"status": str(r["status"]), "count": int(r["n"])} for r in statuses],
        "bpm": {
            "min": int(bpm["min_bpm"]) if bpm["min_bpm"] is not None else None,
            "max": int(bpm["max_bpm"]) if bpm["max_bpm"] is not None else None,
            "avg": float(bpm["avg_bpm"]) if bpm["avg_bpm"] is not None else None,
        },
        "duration_sec": {
            "min": float(dur["min_dur"]) if dur["min_dur"] is not None else None,
            "max": float(dur["max_dur"]) if dur["max_dur"] is not None else None,
            "avg": float(dur["avg_dur"]) if dur["avg_dur"] is not None else None,
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
# playlist_runs
# ----------------------------

def db_record_playlist_run(
    conn: sqlite3.Connection,
    playlist_id: str,
    seed: Optional[int],
    track_ids: List[str],
    export_path: Optional[Path],
    notes: Optional[str],
) -> str:
    ensure_playlist_runs_table(conn)
    run_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO playlist_runs (id, created_at, playlist_id, seed, track_ids_json, export_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            now_iso(),
            playlist_id,
            int(seed) if seed is not None else None,
            json.dumps(track_ids, ensure_ascii=False),
            str(export_path) if export_path is not None else None,
            notes,
        ),
    )
    conn.commit()
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
        try:
            track_ids = json.loads(str(r["track_ids_json"]))
            if not isinstance(track_ids, list):
                track_ids = []
            track_ids = [str(x) for x in track_ids]
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
        track_ids = json.loads(str(r["track_ids_json"]))
        if not isinstance(track_ids, list):
            track_ids = []
        track_ids = [str(x) for x in track_ids]
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

@dataclass(frozen=True)
class ResolvedRef:
    kind: str  # "run" or "playlist"
    id: str    # run_id or playlist_id
    created_at: str
    playlist_id: str
    seed: Optional[int]
    track_ids: List[str]


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

    # fallback to playlists table (playlist id)
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
# validation
# ----------------------------

def validate_playlist_json(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "playlist JSON is not an object"
    tracks = obj.get("tracks")
    if not isinstance(tracks, list):
        return False, "missing tracks[] list"
    if len(tracks) == 0:
        return False, "tracks[] is empty"
    return True, "ok"


# ----------------------------
# push targets
# ----------------------------

@dataclass
class PushResult:
    target: str
    ok: bool
    detail: str


def push_local(src_path: Path, dst_dir: Path, overwrite: bool, dry_run: bool) -> PushResult:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src_path.name

    if dst_path.exists() and not overwrite:
        return PushResult("local", True, f"Skipped (exists): {dst_path}")

    if dry_run:
        return PushResult("local", True, f"Dry-run copy to: {dst_path}")

    dst_path.write_bytes(src_path.read_bytes())
    return PushResult("local", True, f"Copied to: {dst_path}")


def push_webhook(src_path: Path, url: str, timeout_s: int, dry_run: bool) -> PushResult:
    requests = ensure_requests()
    payload = read_json_file(src_path)
    headers = {"Content-Type": "application/json", "User-Agent": "mgc-cli/1.0"}

    if dry_run:
        return PushResult("webhook", True, f"Dry-run POST to: {url}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        if 200 <= int(resp.status_code) < 300:
            return PushResult("webhook", True, f"POST {resp.status_code}: {url}")
        return PushResult("webhook", False, f"POST {resp.status_code}: {resp.text[:400]}")
    except Exception as ex:
        return PushResult("webhook", False, f"POST failed: {ex}")


# ----------------------------
# commands
# ----------------------------

def db_schema_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r["name"] for r in cur.fetchall()]
        for t in tables:
            print(t)
            info = conn.execute(f"PRAGMA table_info({t})").fetchall()
            for r in info:
                name = r["name"]
                ctype = r["type"] or ""
                pk = " pk" if r["pk"] else ""
                nn = " notnull" if r["notnull"] else ""
                print(f"  - {name} {ctype}{pk}{nn}")
        return 0
    finally:
        conn.close()


def playlists_list_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        pls = db_list_playlists(conn, limit=args.limit)
    finally:
        conn.close()

    if args.json:
        print(json.dumps([pl.__dict__ for pl in pls], indent=2, ensure_ascii=False))
        return 0

    if not pls:
        print("(no playlists)")
        return 0

    for i, pl in enumerate(pls, 1):
        print(f"{i:>2}. {pl.id}  {pl.slug}  {pl.created_at}  tracks={pl.track_count}  json_path={pl.json_path}")
    return 0


def playlists_reveal_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        pl = db_get_playlist(conn, args.id)

        if args.build:
            obj = db_build_playlist_json(conn, pl)
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            return 0

        jp = resolve_json_path(pl.json_path)
        if jp.exists():
            obj = read_json_file(jp)
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            return 0

        obj = db_build_playlist_json(conn, pl)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        return 0
    finally:
        conn.close()


def export_one_playlist(conn: sqlite3.Connection, pl: PlaylistRow, export_dir: Path, build: bool) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / export_filename(pl.id, pl.slug)

    if build:
        obj = db_build_playlist_json(conn, pl)
        obj.setdefault("exported_at", now_iso())
        write_json_file(out_path, obj)
        return out_path

    src = resolve_json_path(pl.json_path)
    if src.exists():
        obj = read_json_file(src)
        if isinstance(obj, dict):
            obj.setdefault("id", pl.id)
            obj.setdefault("slug", pl.slug)
            obj.setdefault("exported_at", now_iso())
            write_json_file(out_path, obj)
        else:
            out_path.write_bytes(src.read_bytes())
        return out_path

    obj = db_build_playlist_json(conn, pl)
    obj.setdefault("exported_at", now_iso())
    write_json_file(out_path, obj)
    return out_path


def maybe_record_run(
    conn: sqlite3.Connection,
    pl: PlaylistRow,
    export_path: Path,
    record: bool,
    notes: Optional[str],
) -> Optional[str]:
    if not record:
        return None
    ensure_playlist_runs_table(conn)
    track_ids = db_playlist_track_ids(conn, pl.id)
    return db_record_playlist_run(
        conn=conn,
        playlist_id=pl.id,
        seed=pl.seed,
        track_ids=track_ids,
        export_path=export_path,
        notes=notes,
    )


def playlists_export_cmd(args: argparse.Namespace) -> int:
    export_dir = Path(args.out_dir) if args.out_dir else default_export_dir()
    conn = sqlite_connect(args.db)
    try:
        if args.id:
            pl = db_get_playlist(conn, args.id)
            out_path = export_one_playlist(conn, pl, export_dir, build=bool(args.build))
            run_id = maybe_record_run(conn, pl, out_path, record=not args.no_record, notes=args.notes)
            if args.json:
                print(json.dumps({"exported": str(out_path), "run_id": run_id}, indent=2, ensure_ascii=False))
            else:
                print(str(out_path))
                if run_id:
                    print(f"Recorded run: {run_id}")
            return 0

        pls = db_list_playlists(conn, limit=args.limit)
        if not pls:
            die("No playlists found to export.")

        out_paths: List[str] = []
        run_ids: List[str] = []
        for pl in pls:
            out_path = export_one_playlist(conn, pl, export_dir, build=bool(args.build))
            out_paths.append(str(out_path))
            rid = maybe_record_run(conn, pl, out_path, record=not args.no_record, notes=args.notes)
            if rid:
                run_ids.append(rid)

        if args.json:
            print(json.dumps({"exported": out_paths, "run_ids": run_ids}, indent=2, ensure_ascii=False))
        else:
            for p in out_paths:
                print(p)
            if run_ids:
                print("Recorded runs:")
                for rid in run_ids:
                    print(f"  {rid}")
        return 0
    finally:
        conn.close()


def playlists_history_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        ensure_playlist_runs_table(conn)

        # First: explicit runs
        runs = db_list_playlist_runs(conn, args.playlist_id, limit=args.limit)
        if runs:
            if args.json:
                print(json.dumps([{
                    "kind": "run",
                    "id": r.id,
                    "created_at": r.created_at,
                    "playlist_id": r.playlist_id,
                    "seed": r.seed,
                    "track_count": len(r.track_ids),
                    "export_path": r.export_path,
                    "notes": r.notes,
                } for r in runs], indent=2, ensure_ascii=False))
                return 0

            for i, r in enumerate(runs, 1):
                ep = r.export_path or ""
                notes = (r.notes or "").strip()
                notes_s = (notes[:60] + "…") if len(notes) > 60 else notes
                print(f"{i:>2}. RUN {r.id}  {r.created_at}  seed={r.seed}  tracks={len(r.track_ids)}  {ep}  {notes_s}")
            return 0

        # Fallback: implicit runs by slug
        pl = db_get_playlist(conn, args.playlist_id)
        peers = db_list_playlists_by_slug(conn, pl.slug, limit=args.limit)

        if args.json:
            print(json.dumps([{
                "kind": "playlist",
                "id": p.id,
                "created_at": p.created_at,
                "playlist_id": p.id,
                "seed": p.seed,
                "track_count": len(db_playlist_track_ids(conn, p.id)),
                "slug": p.slug,
            } for p in peers], indent=2, ensure_ascii=False))
            return 0

        print("(no recorded runs in playlist_runs; showing implicit history from playlists table by slug)")
        for i, p in enumerate(peers, 1):
            tc = len(db_playlist_track_ids(conn, p.id))
            print(f"{i:>2}. PL  {p.id}  {p.created_at}  seed={p.seed}  tracks={tc}  slug={p.slug}")
        return 0
    finally:
        conn.close()


def playlists_diff_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        ensure_playlist_runs_table(conn)

        a = resolve_ref(conn, args.a)
        b = resolve_ref(conn, args.b)

        diff = compute_diff(a.track_ids, b.track_ids)

        a_dur = db_duration_for_tracks(conn, a.track_ids)
        b_dur = db_duration_for_tracks(conn, b.track_ids)

        if args.json:
            print(json.dumps({
                "a": a.__dict__,
                "b": b.__dict__,
                "duration_sec": {"a": a_dur, "b": b_dur, "delta": (b_dur - a_dur)},
                "diff": diff,
            }, indent=2, ensure_ascii=False))
            return 0

        print(f"A: {a.kind.upper()} {a.id}  created={a.created_at}  seed={a.seed}  tracks={len(a.track_ids)}")
        print(f"B: {b.kind.upper()} {b.id}  created={b.created_at}  seed={b.seed}  tracks={len(b.track_ids)}")
        print(f"Duration: A={a_dur:.1f}s  B={b_dur:.1f}s  Δ={(b_dur - a_dur):+.1f}s")
        print(f"Added:   {len(diff['added'])}")
        print(f"Removed: {len(diff['removed'])}")
        print(f"Moved:   {len(diff['moved'])}")

        if args.verbose:
            if diff["added"]:
                print("\nADDED:")
                for tid in diff["added"]:
                    print(f"  + {tid}")
            if diff["removed"]:
                print("\nREMOVED:")
                for tid in diff["removed"]:
                    print(f"  - {tid}")
            if diff["moved"]:
                print("\nMOVED:")
                for m in diff["moved"][:200]:
                    print(f"  ~ {m['track_id']}  {m['from']} -> {m['to']}")
                if len(diff["moved"]) > 200:
                    print(f"  ... ({len(diff['moved']) - 200} more)")
        return 0
    finally:
        conn.close()


# -------- tracks commands --------

def tracks_list_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        rows = db_list_tracks(
            conn,
            limit=args.limit,
            mood=args.mood,
            genre=args.genre,
            status=args.status,
            bpm_min=args.bpm_min,
            bpm_max=args.bpm_max,
            q=args.q,
        )
    finally:
        conn.close()

    if args.json:
        print(json.dumps([r.__dict__ for r in rows], indent=2, ensure_ascii=False))
        return 0

    if not rows:
        print("(no tracks)")
        return 0

    for i, t in enumerate(rows, 1):
        print(f"{i:>2}. {t.id}  {t.title}  mood={t.mood} genre={t.genre} bpm={t.bpm} dur={t.duration_sec:.1f}s status={t.status}")
    return 0


def tracks_show_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        t = db_get_track(conn, args.id)
    finally:
        conn.close()

    print(json.dumps(t.__dict__, indent=2, ensure_ascii=False))
    return 0


def tracks_stats_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        stats = db_tracks_stats(conn)
    finally:
        conn.close()

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mgc", description="mgc CLI")
    p.add_argument("--env", default=".env", help="Path to .env (default: .env)")
    p.add_argument("--log-level", default=os.environ.get("MGC_LOG_LEVEL", "INFO"))

    sub = p.add_subparsers(dest="cmd", required=True)

    # db
    dbg = sub.add_parser("db", help="DB utilities")
    dbs = dbg.add_subparsers(dest="db_cmd", required=True)

    db_schema = dbs.add_parser("schema", help="Print tables + columns")
    db_schema.add_argument("--db", default="data/db.sqlite")
    db_schema.set_defaults(func=db_schema_cmd)

    # playlists
    pg = sub.add_parser("playlists", help="Playlist operations")
    pgs = pg.add_subparsers(dest="playlists_cmd", required=True)

    pl_list = pgs.add_parser("list", help="List playlists")
    pl_list.add_argument("--db", default="data/db.sqlite")
    pl_list.add_argument("--limit", type=int, default=20)
    pl_list.add_argument("--json", action="store_true")
    pl_list.set_defaults(func=playlists_list_cmd)

    pl_rev = pgs.add_parser("reveal", help="Reveal playlist JSON (json_path) or build from DB")
    pl_rev.add_argument("id")
    pl_rev.add_argument("--db", default="data/db.sqlite")
    pl_rev.add_argument("--build", action="store_true")
    pl_rev.set_defaults(func=playlists_reveal_cmd)

    pl_exp = pgs.add_parser("export", help="Export playlists to JSON files")
    pl_exp.add_argument("--db", default="data/db.sqlite")
    pl_exp.add_argument("--id", default=None)
    pl_exp.add_argument("--limit", type=int, default=20)
    pl_exp.add_argument("--out-dir", default=None)
    pl_exp.add_argument("--json", action="store_true")
    pl_exp.add_argument("--build", action="store_true")
    pl_exp.add_argument("--no-record", action="store_true")
    pl_exp.add_argument("--notes", default=None)
    pl_exp.set_defaults(func=playlists_export_cmd)

    pl_hist = pgs.add_parser("history", help="List recorded runs for a playlist (fallback: playlist rows by slug)")
    pl_hist.add_argument("playlist_id")
    pl_hist.add_argument("--db", default="data/db.sqlite")
    pl_hist.add_argument("--limit", type=int, default=20)
    pl_hist.add_argument("--json", action="store_true")
    pl_hist.set_defaults(func=playlists_history_cmd)

    pl_diff = pgs.add_parser("diff", help="Diff two refs: run IDs or playlist IDs")
    pl_diff.add_argument("a")
    pl_diff.add_argument("b")
    pl_diff.add_argument("--db", default="data/db.sqlite")
    pl_diff.add_argument("--json", action="store_true")
    pl_diff.add_argument("--verbose", action="store_true")
    pl_diff.set_defaults(func=playlists_diff_cmd)

    # tracks
    tg = sub.add_parser("tracks", help="Track library")
    tgs = tg.add_subparsers(dest="tracks_cmd", required=True)

    tl = tgs.add_parser("list", help="List tracks")
    tl.add_argument("--db", default="data/db.sqlite")
    tl.add_argument("--limit", type=int, default=20)
    tl.add_argument("--mood", default=None)
    tl.add_argument("--genre", default=None)
    tl.add_argument("--status", default=None)
    tl.add_argument("--bpm-min", type=int, default=None)
    tl.add_argument("--bpm-max", type=int, default=None)
    tl.add_argument("--q", default=None)
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(func=tracks_list_cmd)

    ts = tgs.add_parser("show", help="Show track details")
    ts.add_argument("id")
    ts.add_argument("--db", default="data/db.sqlite")
    ts.set_defaults(func=tracks_show_cmd)

    tt = tgs.add_parser("stats", help="Track library stats")
    tt.add_argument("--db", default="data/db.sqlite")
    tt.set_defaults(func=tracks_stats_cmd)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    load_dotenv(args.env)

    if setup_logging:
        try:
            setup_logging(level=args.log_level)
        except Exception:
            pass

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
