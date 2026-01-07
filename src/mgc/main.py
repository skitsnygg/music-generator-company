from __future__ import annotations
import os
import argparse
import json
import sqlite3
import uuid
from pathlib import Path
from dotenv import load_dotenv

from mgc.db import DB
from mgc.storage import StoragePaths
from mgc.logging_setup import setup_logging

from mgc.agents.music_agent import MusicAgent
from mgc.agents.marketing_agent import MarketingAgent
from mgc.playlist import build_playlist

def run_daily() -> int:
    load_dotenv()

    db_path = Path(os.getenv("MGC_DB_PATH", "./data/db.sqlite"))
    data_dir = Path(os.getenv("MGC_DATA_DIR", "./data"))
    log_dir = Path(os.getenv("MGC_LOG_DIR", "./logs"))

    logger = setup_logging(log_dir)
    logger.info("Starting daily pipeline")

    db = DB(db_path)
    db.init()

    storage = StoragePaths(data_dir)
    storage.ensure()

    music = MusicAgent(storage=storage)
    marketing = MarketingAgent(storage=storage)

    track = music.run_daily()
    db.insert_track(music.to_db_row(track))
    logger.info(f"Generated track: {track.title}")
    logger.info(f"Full: {track.full_path}")
    logger.info(f"Preview: {track.preview_path}")

    posts = marketing.plan_posts(track)
    for p in posts:
        db.insert_post(p)
    logger.info(f"Planned {len(posts)} marketing posts (JSON files in data/posts/)")

    logger.info("Daily pipeline complete ✅")
    return 0


def playlist_cmd(args) -> int:
    load_dotenv()

    db_path = Path(os.getenv("MGC_DB_PATH", "./data/db.sqlite"))
    data_dir = Path(os.getenv("MGC_DATA_DIR", "./data"))

    bpm_window = None
    if args.bpm_min is not None or args.bpm_max is not None:
        bpm_window = (args.bpm_min or 0, args.bpm_max or 999)

    pl = build_playlist(
        db_path=db_path,
        name=args.name,
        slug=args.slug,
        context=args.context,
        mood=args.mood,
        genre=args.genre,
        target_minutes=args.minutes,
        bpm_window=bpm_window,
        seed=args.seed,
        lookback_playlists=args.lookback,
    )

    # Dedupe UX: make it obvious when we had to fall back
    dedupe = (pl.get("dedupe") or {})
    if dedupe:
        if dedupe.get("applied"):
            print(f"dedupe applied (excluded {dedupe.get('excluded_count', 0)} tracks)")
        else:
            reason = dedupe.get("reason")
            excl = dedupe.get("excluded_count", 0)
            if reason == "not_enough_unique_tracks":
                print(f"dedupe skipped: not enough unique tracks after excluding {excl}. Generate more tracks to avoid repeats.")
            else:
                print(f"dedupe skipped (excluded {excl}) reason={reason}")

    out_dir = data_dir / "playlists"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.slug}.json"
    out_path.write_text(json.dumps(pl, indent=2), encoding="utf-8")


    # Persist playlist build (history + dedupe support)
    db = DB(db_path)
    db.init()

    playlist_id = str(uuid.uuid4())
    filters = pl.get("filters", {})
    bpm_window = filters.get("bpm_window") or [None, None]
    bpm_min, bpm_max = bpm_window[0], bpm_window[1]

    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO playlists (
                id, created_at, slug, name, context, mood, genre, bpm_min, bpm_max,
                target_minutes, seed, track_count, total_duration_sec, json_path
            ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                playlist_id,
                args.slug,
                pl["name"],
                filters.get("context"),
                filters.get("mood"),
                filters.get("genre"),
                bpm_min,
                bpm_max,
                int(filters.get("target_minutes") or 0),
                int(filters.get("seed") or 1),
                int(pl["stats"]["track_count"]),
                int(pl["stats"]["total_duration_sec"]),
                str(out_path),
            ),
        )

        for idx, item in enumerate(pl["items"]):
            conn.execute(
                """
                INSERT INTO playlist_items (playlist_id, track_id, position)
                VALUES (?, ?, ?)
                """,
                (playlist_id, item["track_id"], idx),
            )

        conn.commit()


    print(str(out_path))
    return 0



def playlists_list_cmd(args) -> int:
    from pathlib import Path
    from mgc.db import DB

    db_path = Path(args.db)
    db = DB(db_path)
    db.init()

    # Direct SQL (avoids any helper mismatch)
    with db.connect() as conn:
        if args.slug:
            rows = conn.execute(
                """
                SELECT rowid, id, created_at, slug, context, name, track_count, total_duration_sec, json_path
                FROM playlists
                WHERE slug = ?
                ORDER BY datetime(created_at) DESC, rowid DESC
                LIMIT ?
                """,
                (args.slug, int(args.limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT rowid, id, created_at, slug, context, name, track_count, total_duration_sec, json_path
                FROM playlists
                ORDER BY datetime(created_at) DESC, rowid DESC
                LIMIT ?
                """,
                (int(args.limit),),
            ).fetchall()

    print(f"db: {db_path}  playlists_shown: {len(rows)}", flush=True)

    if not rows:
        print("No playlists found.", flush=True)
        return 0

    for r in rows:
        minutes = round((r["total_duration_sec"] or 0) / 60.0, 2)
        print(
            f'{r["created_at"]}  id={r["id"]}  slug={r["slug"]}  context={r["context"]}  tracks={r["track_count"]}  minutes={minutes}',
            flush=True,
        )

    return 0

    db_path = Path(args.db)
    db = DB(db_path)
    db.init()

    rows = db.list_playlists(slug=args.slug, limit=args.limit)
    if not rows:
        print("No playlists found.")
        return 0

    # Simple table-like output
    for r in rows:
        minutes = round((r.get("total_duration_sec") or 0) / 60.0, 2)
        print(f'{r["created_at"]}  id={r["id"]}  slug={r["slug"]}  context={r.get("context")}  tracks={r.get("track_count")}  minutes={minutes}')

    return 0


def playlists_show_cmd(args) -> int:
    from pathlib import Path
    import json
    from mgc.db import DB

    db_path = Path(args.db)
    db = DB(db_path)
    db.init()

    pl = db.get_playlist_with_items(args.id)
    if not pl:
        print("Playlist not found.")
        return 1

    minutes = round((pl.get("total_duration_sec") or 0) / 60.0, 2)
    print(pl.get("name", "Playlist"))
    print(f'created_at: {pl.get("created_at")}')
    print(f'id: {pl.get("id")}')
    print(f'slug: {pl.get("slug")}')
    print(f'context: {pl.get("context")}')
    print(f'tracks: {pl.get("track_count")}  minutes: {minutes}')
    print(f'json_path: {pl.get("json_path")}')
    print("")

    # If the JSON exists, show filters/stats/dedupe (source of truth for analytics)
    try:
        with open(pl.get("json_path"), "r", encoding="utf-8") as f:
            j = json.load(f)
        filters = j.get("filters") or {}
        stats = j.get("stats") or {}
        dedupe = j.get("dedupe") or {}

        print("filters:")
        for k in ["context", "mood", "genre", "bpm_window", "target_minutes", "seed", "lookback_playlists"]:
            if k in filters:
                print(f"  {k}: {filters.get(k)}")

        print("stats:")
        for k in ["track_count", "unique_track_count", "duration_minutes", "avg_bpm", "bpm_min", "bpm_max"]:
            if k in stats:
                print(f"  {k}: {stats.get(k)}")

        if dedupe:
            print("dedupe:")
            for k in ["requested_lookback", "excluded_count", "applied", "reason"]:
                if k in dedupe:
                    print(f"  {k}: {dedupe.get(k)}")

        print("")
    except Exception:
        # Not fatal: still show items from DB
        pass

    print("items:")
    for it in pl["items"]:
        dur = int(it.get("duration_sec") or 0)
        print(f'  {it.get("title")}  bpm={it.get("bpm")}  mood={it.get("mood")}  genre={it.get("genre")}  dur={dur}s  id={it.get("id")}')
    return 0


def playlists_list_cmd_v2(args) -> int:
    from pathlib import Path
    import sqlite3

    db_path = Path(args.db).resolve() if getattr(args, "db", None) else Path("data/db.sqlite").resolve()

    if not db_path.exists():
        print("DB file not found.", flush=True)
        return 1

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        if getattr(args, "slug", None):
            rows = db.execute(
                """
                SELECT rowid, id, created_at, slug, context, name, track_count, total_duration_sec, json_path
                FROM playlists
                WHERE slug = ?
                ORDER BY datetime(created_at) DESC, rowid DESC
                LIMIT ?
                """,
                (args.slug, int(args.limit)),
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT rowid, id, created_at, slug, context, name, track_count, total_duration_sec, json_path
                FROM playlists
                ORDER BY datetime(created_at) DESC, rowid DESC
                LIMIT ?
                """,
                (int(args.limit),),
            ).fetchall()

        print(f"playlists_shown: {len(rows)}", flush=True)

        for r in rows:
            minutes = round((r["total_duration_sec"] or 0) / 60.0, 2)
            print(f'{r["created_at"]}  id={r["id"]}  slug={r["slug"]}  context={r["context"]}  tracks={r["track_count"]}  minutes={minutes}', flush=True)

        return 0
    finally:
        db.close()


def playlists_show_cmd_v2(args) -> int:
    from pathlib import Path
    import sqlite3
    import json

    db_path = Path(args.db).resolve() if getattr(args, "db", None) else Path("data/db.sqlite").resolve()

    if not db_path.exists():
        print("DB file not found.", flush=True)
        return 1

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        pl = db.execute("SELECT rowid, * FROM playlists WHERE id = ?", (args.id,)).fetchone()
        if not pl:
            print("Playlist not found.", flush=True)
            return 1

        pl = dict(pl)
        items = db.execute(
            """
            SELECT pi.position, t.*
            FROM playlist_items pi
            JOIN tracks t ON t.id = pi.track_id
            WHERE pi.playlist_id = ?
            ORDER BY pi.position ASC
            """,
            (args.id,),
        ).fetchall()

        minutes = round((pl.get("total_duration_sec") or 0) / 60.0, 2)
        print(pl.get("name", "Playlist"), flush=True)
        print(f'created_at: {pl.get("created_at")}', flush=True)
        print(f'id: {pl.get("id")}', flush=True)
        print(f'slug: {pl.get("slug")}', flush=True)
        print(f'context: {pl.get("context")}', flush=True)
        print(f'tracks: {pl.get("track_count")}  minutes: {minutes}', flush=True)
        print(f'json_path: {pl.get("json_path")}', flush=True)
        print("", flush=True)

        # Optional: show JSON-derived filters/stats if file exists
        try:
            jp = pl.get("json_path")
            if jp:
                j = json.loads(Path(jp).read_text(encoding="utf-8"))
                filters = j.get("filters") or {}
                stats = j.get("stats") or {}
                dedupe = j.get("dedupe") or {}

                print("filters:", flush=True)
                for k in ["context", "mood", "genre", "bpm_window", "target_minutes", "seed", "lookback_playlists"]:
                    if k in filters:
                        print(f"  {k}: {filters.get(k)}", flush=True)

                print("stats:", flush=True)
                for k in ["track_count", "unique_track_count", "duration_minutes", "avg_bpm", "bpm_min", "bpm_max"]:
                    if k in stats:
                        print(f"  {k}: {stats.get(k)}", flush=True)

                if dedupe:
                    print("dedupe:", flush=True)
                    for k in ["requested_lookback", "excluded_count", "applied", "reason"]:
                        if k in dedupe:
                            print(f"  {k}: {dedupe.get(k)}", flush=True)

                print("", flush=True)
        except Exception:
            pass

        print("items:", flush=True)
        for r in items:
            r = dict(r)
            dur = int(r.get("duration_sec") or 0)
            print(f'  {r.get("title")}  bpm={r.get("bpm")}  mood={r.get("mood")}  genre={r.get("genre")}  dur={dur}s  id={r.get("id")}', flush=True)

        return 0
    finally:
        db.close()



def tracks_list_cmd(args) -> int:
    import sqlite3
    from pathlib import Path

    db_path = Path(args.db).resolve()
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    try:
        where = []
        params = []

        if args.mood:
            where.append("mood = ?")
            params.append(args.mood)
        if args.genre:
            where.append("genre = ?")
            params.append(args.genre)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        rows = db.execute(
            f"""
            SELECT id, created_at, title, mood, genre, bpm, duration_sec
            FROM tracks
            {where_sql}
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            params + [int(args.limit)],
        ).fetchall()

        if not rows:
            print("No tracks found.")
            return 0

        for r in rows:
            dur = int(r["duration_sec"])
            print(f'{r["created_at"]}  id={r["id"]}  "{r["title"]}"  mood={r["mood"]}  genre={r["genre"]}  bpm={r["bpm"]}  dur={dur}s')

        return 0
    finally:
        db.close()


def tracks_show_cmd(args) -> int:
    import sqlite3
    from pathlib import Path

    db_path = Path(args.db).resolve()
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    try:
        r = db.execute("SELECT * FROM tracks WHERE id = ?", (args.id,)).fetchone()
        if not r:
            print("Track not found.")
            return 1

        r = dict(r)
        print(r["title"])
        for k in ["id", "created_at", "mood", "genre", "bpm", "duration_sec", "status"]:
            print(f"{k}: {r.get(k)}")
        print(f'preview_path: {r.get("preview_path")}')
        print(f'full_path: {r.get("full_path")}')
        return 0
    finally:
        db.close()


def tracks_stats_cmd(args) -> int:
    import sqlite3
    from pathlib import Path
    from collections import Counter

    db_path = Path(args.db).resolve()
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    try:
        rows = db.execute("SELECT mood, genre, bpm, duration_sec FROM tracks").fetchall()
        if not rows:
            print("No tracks in library.")
            return 0

        moods = Counter()
        genres = Counter()
        bpms = []
        durations = []

        for r in rows:
            moods[r["mood"]] += 1
            genres[r["genre"]] += 1
            bpms.append(int(r["bpm"]))
            durations.append(int(r["duration_sec"]))

        print(f"total_tracks: {len(rows)}")
        print(f"avg_bpm: {int(sum(bpms) / len(bpms))}")
        print(f"avg_duration_sec: {int(sum(durations) / len(durations))}")

        print("moods:")
        for k, v in moods.most_common():
            print(f"  {k}: {v}")

        print("genres:")
        for k, v in genres.most_common():
            print(f"  {k}: {v}")

        return 0
    finally:
        db.close()

def main() -> int:
    parser = argparse.ArgumentParser(prog="mgc")
    sub = parser.add_subparsers(dest="cmd", required=True)
    # playlists: inspect saved playlist history
    pg = sub.add_parser("playlists", help="Inspect saved playlists")
    pgs = pg.add_subparsers(dest="playlists_cmd")
    # Python argparse behavior differs across versions; enforce requirement manually in main() via handler selection.

    pl_list = pgs.add_parser("list", help="List recent playlists")
    pl_list.add_argument("--db", default="data/db.sqlite", help="Path to SQLite DB")
    pl_list.add_argument("--limit", type=int, default=10, help="Max rows")
    pl_list.add_argument("--slug", default=None, help="Filter by slug")
    pl_list.set_defaults(func=playlists_list_cmd_v2)

    pl_show = pgs.add_parser("show", help="Show playlist details")
    pl_show.add_argument("id", help="Playlist id")
    pl_show.add_argument("--db", default="data/db.sqlite", help="Path to SQLite DB")
    pl_show.set_defaults(func=playlists_show_cmd_v2)

    # tracks: inspect track library
    tg = sub.add_parser("tracks", help="Inspect track library")
    tgs = tg.add_subparsers(dest="tracks_cmd", required=True)

    tl = tgs.add_parser("list", help="List tracks")
    tl.add_argument("--db", default="data/db.sqlite", help="Path to SQLite DB")
    tl.add_argument("--limit", type=int, default=20, help="Max rows")
    tl.add_argument("--mood", default=None, help="Filter mood (exact match)")
    tl.add_argument("--genre", default=None, help="Filter genre (exact match)")
    tl.set_defaults(func=tracks_list_cmd)

    ts = tgs.add_parser("show", help="Show track details")
    ts.add_argument("id", help="Track id")
    ts.add_argument("--db", default="data/db.sqlite", help="Path to SQLite DB")
    ts.set_defaults(func=tracks_show_cmd)

    tt = tgs.add_parser("stats", help="Track library stats")
    tt.add_argument("--db", default="data/db.sqlite", help="Path to SQLite DB")
    tt.set_defaults(func=tracks_stats_cmd)


    sub.add_parser("run-daily", help="Run the daily generation → store → promote pipeline")

    p = sub.add_parser("playlist", help="Build an auto-playlist JSON from existing tracks")
    p.add_argument("--name", default="Auto Playlist", help="Playlist display name")
    p.add_argument("--slug", default="auto_playlist", help="Filename slug (no extension)")
    p.add_argument("--context", default=None, help="Preset context: focus|workout|sleep")
    p.add_argument("--mood", default=None, help="Filter mood (exact match)")
    p.add_argument("--genre", default=None, help="Filter genre (exact match)")
    p.add_argument("--minutes", type=int, default=20, help="Target playlist length in minutes")
    p.add_argument("--bpm-min", dest="bpm_min", type=int, default=None)
    p.add_argument("--bpm-max", dest="bpm_max", type=int, default=None)
    p.add_argument("--seed", type=int, default=1, help="Shuffle seed")
    p.add_argument("--lookback", type=int, default=3, help="Avoid tracks used in last N playlists")
    p.set_defaults(func=playlist_cmd)


    args = parser.parse_args()

    # Special-case: run-daily doesn't use set_defaults(func=...)
    if args.cmd == "run-daily":
        return run_daily()

    # Generic dispatch for all other subcommands
    func = getattr(args, "func", None)
    if callable(func):
        return func(args)

    parser.print_help()
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
