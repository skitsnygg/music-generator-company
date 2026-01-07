from __future__ import annotations
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from mgc.db import DB
from mgc.storage import StoragePaths
from mgc.logging_setup import setup_logging

from mgc.agents.music_agent import MusicAgent
from mgc.agents.marketing_agent import MarketingAgent

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

def main() -> int:
    parser = argparse.ArgumentParser(prog="mgc")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run-daily", help="Run the daily generation → store → promote pipeline")

    args = parser.parse_args()
    if args.cmd == "run-daily":
        return run_daily()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
