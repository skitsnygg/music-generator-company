from __future__ import annotations
import os
import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="mgc")
    sub = parser.add_subparsers(dest="cmd", required=True)
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
    if args.cmd == "run-daily":
        return run_daily()
    if args.cmd == "playlist":
        return args.func(args)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
