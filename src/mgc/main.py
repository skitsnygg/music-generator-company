#!/usr/bin/env python3
"""
mgc CLI — Playlists export/push/smoke + Tracks commands (DB-accurate)

Adds:
- tracks list
- tracks show
- tracks stats

Your DB schema (confirmed):
tracks(
  id TEXT pk,
  created_at TEXT notnull,
  title TEXT notnull,
  mood TEXT notnull,
  genre TEXT notnull,
  bpm INTEGER notnull,
  duration_sec REAL notnull,
  full_path TEXT notnull,
  preview_path TEXT notnull,
  status TEXT notnull
)

Existing kept:
- db schema
- playlists list/reveal/export/push
- smoke

Drop-in target: src/mgc/main.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
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


def resolve_json_path(json_path: str) -> Path:
    p = Path(json_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


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

    # mood counts
    moods = conn.execute(
        "SELECT mood, COUNT(*) AS n FROM tracks GROUP BY mood ORDER BY n DESC"
    ).fetchall()
    genres = conn.execute(
        "SELECT genre, COUNT(*) AS n FROM tracks GROUP BY genre ORDER BY n DESC"
    ).fetchall()
    statuses = conn.execute(
        "SELECT status, COUNT(*) AS n FROM tracks GROUP BY status ORDER BY n DESC"
    ).fetchall()

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


def playlists_export_cmd(args: argparse.Namespace) -> int:
    export_dir = Path(args.out_dir) if args.out_dir else default_export_dir()
    conn = sqlite_connect(args.db)
    try:
        if args.id:
            pl = db_get_playlist(conn, args.id)
            out_path = export_one_playlist(conn, pl, export_dir, build=bool(args.build))
            print(str(out_path))
            return 0

        pls = db_list_playlists(conn, limit=args.limit)
        if not pls:
            die("No playlists found to export.")

        out_paths: List[str] = []
        for pl in pls:
            out_path = export_one_playlist(conn, pl, export_dir, build=bool(args.build))
            out_paths.append(str(out_path))

        if args.json:
            print(json.dumps({"exported": out_paths}, indent=2, ensure_ascii=False))
        else:
            for p in out_paths:
                print(p)
        return 0
    finally:
        conn.close()


def playlists_push_cmd(args: argparse.Namespace) -> int:
    export_dir = Path(args.export_dir) if args.export_dir else default_export_dir()

    if args.file:
        src_path = Path(args.file)
        if not src_path.exists():
            die(f"Source file not found: {src_path}")
    elif args.id:
        conn = sqlite_connect(args.db)
        try:
            pl = db_get_playlist(conn, args.id)
            src_path = export_one_playlist(conn, pl, export_dir, build=bool(args.build))
        finally:
            conn.close()
    else:
        die("Provide --id PLAYLIST_ID or --file path.json")

    results: List[PushResult] = []

    if args.local_dir:
        results.append(
            push_local(
                src_path=src_path,
                dst_dir=Path(args.local_dir),
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )
        )

    webhook_url = args.webhook_url or os.environ.get("MGC_WEBHOOK_URL")
    if args.webhook and not webhook_url:
        die("Webhook enabled but no URL provided. Use --webhook-url or set MGC_WEBHOOK_URL.")

    if args.webhook:
        results.append(
            push_webhook(
                src_path=src_path,
                url=str(webhook_url),
                timeout_s=int(args.webhook_timeout),
                dry_run=bool(args.dry_run),
            )
        )

    if not results:
        die("No push targets selected. Use --local-dir and/or --webhook.")

    ok_all = all(r.ok for r in results)
    out = {
        "source": str(src_path),
        "dry_run": bool(args.dry_run),
        "results": [r.__dict__ for r in results],
        "ok": ok_all,
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"Source: {src_path}")
        for r in results:
            status = "OK" if r.ok else "FAIL"
            print(f"- {r.target}: {status} — {r.detail}")

    return 0 if ok_all else 3


def smoke_cmd(args: argparse.Namespace) -> int:
    export_dir = Path(args.export_dir) if args.export_dir else default_export_dir()
    dry_run = not bool(args.no_dry_run)

    conn = sqlite_connect(args.db)
    try:
        pls = db_list_playlists(conn, limit=1)
        if not pls:
            die("Smoke failed: no playlists found in DB.")
        pl = pls[0]
        out_path = export_one_playlist(conn, pl, export_dir, build=bool(args.build))
    finally:
        conn.close()

    obj = read_json_file(out_path)
    ok, msg = validate_playlist_json(obj)
    if not ok:
        die(f"Smoke validation failed: {msg}")

    results: List[PushResult] = []
    if args.local_dir:
        results.append(push_local(out_path, Path(args.local_dir), overwrite=True, dry_run=dry_run))

    webhook_url = args.webhook_url or os.environ.get("MGC_WEBHOOK_URL")
    if args.webhook and not webhook_url:
        die("Smoke webhook enabled but no URL provided. Use --webhook-url or set MGC_WEBHOOK_URL.")

    if args.webhook:
        results.append(push_webhook(out_path, str(webhook_url), timeout_s=int(args.webhook_timeout), dry_run=dry_run))

    out = {
        "exported": str(out_path),
        "validated": True,
        "dry_run": dry_run,
        "push_results": [r.__dict__ for r in results],
        "ok": all(r.ok for r in results) if results else True,
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"Exported: {out_path}")
        print(f"Validated: {msg}")
        if results:
            print(f"Push dry-run: {dry_run}")
            for r in results:
                status = "OK" if r.ok else "FAIL"
                print(f"- {r.target}: {status} — {r.detail}")
        else:
            print("Push: skipped (no targets)")

    return 0 if out["ok"] else 3


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

    pl_rev = pgs.add_parser("reveal", help="Reveal playlist JSON by reading playlists.json_path (or build from DB)")
    pl_rev.add_argument("id")
    pl_rev.add_argument("--db", default="data/db.sqlite")
    pl_rev.add_argument("--build", action="store_true", help="Build JSON from DB joins instead of reading json_path")
    pl_rev.set_defaults(func=playlists_reveal_cmd)

    pl_exp = pgs.add_parser("export", help="Export playlists to JSON files")
    pl_exp.add_argument("--db", default="data/db.sqlite")
    pl_exp.add_argument("--id", default=None, help="Export a specific playlist ID")
    pl_exp.add_argument("--limit", type=int, default=20, help="Export latest N (if --id not set)")
    pl_exp.add_argument("--out-dir", default=None, help="Export directory (default: data/playlists or MGC_PLAYLISTS_DIR)")
    pl_exp.add_argument("--json", action="store_true", help="Output JSON summary")
    pl_exp.add_argument("--build", action="store_true", help="Build JSON from DB joins instead of reading json_path")
    pl_exp.set_defaults(func=playlists_export_cmd)

    pl_push = pgs.add_parser("push", help="Push an exported playlist JSON to targets")
    pl_push.add_argument("--db", default="data/db.sqlite")
    pl_push.add_argument("--id", default=None, help="Playlist ID to export+push")
    pl_push.add_argument("--file", default=None, help="Existing exported playlist JSON file to push")
    pl_push.add_argument("--export-dir", default=None, help="Where to export when using --id (default: data/playlists)")
    pl_push.add_argument("--build", action="store_true", help="Build JSON from DB joins instead of reading json_path")
    pl_push.add_argument("--dry-run", action="store_true")
    pl_push.add_argument("--json", action="store_true")

    pl_push.add_argument("--local-dir", default=None, help="Copy JSON into this directory")
    pl_push.add_argument("--overwrite", action="store_true", help="Overwrite local file if it exists")

    pl_push.add_argument("--webhook", action="store_true", help="Enable webhook push")
    pl_push.add_argument("--webhook-url", default=None, help="Webhook URL (or env MGC_WEBHOOK_URL)")
    pl_push.add_argument("--webhook-timeout", type=int, default=20)
    pl_push.set_defaults(func=playlists_push_cmd)

    # smoke
    sm = sub.add_parser("smoke", help="DB smoke: latest playlist -> export -> validate -> push")
    sm.add_argument("--db", default="data/db.sqlite")
    sm.add_argument("--export-dir", default=None)
    sm.add_argument("--build", action="store_true", help="Build JSON from DB joins instead of reading json_path")
    sm.add_argument("--json", action="store_true")

    sm.add_argument("--local-dir", default=None)
    sm.add_argument("--webhook", action="store_true")
    sm.add_argument("--webhook-url", default=None)
    sm.add_argument("--webhook-timeout", type=int, default=20)
    sm.add_argument("--no-dry-run", action="store_true")
    sm.set_defaults(func=smoke_cmd)

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
    tl.add_argument("--q", default=None, help="Title substring search")
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
