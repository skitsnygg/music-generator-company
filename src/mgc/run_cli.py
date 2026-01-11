from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _stable_stamp_default() -> str:
    # Default stamp: UTC date (stable, human-friendly)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class EvidencePaths:
    root: Path

    @property
    def track_json(self) -> Path:
        return self.root / "track.json"

    @property
    def playlist_json(self) -> Path:
        return self.root / "playlist.json"

    @property
    def posts_dir(self) -> Path:
        return self.root / "marketing_posts"

    @property
    def summary_json(self) -> Path:
        return self.root / "run.json"


def _insert_playlist_and_items(
    conn: sqlite3.Connection,
    *,
    playlist_id: str,
    created_at: str,
    slug: str,
    name: str,
    context: str,
    mood: Optional[str],
    genre: Optional[str],
    target_minutes: int,
    seed: int,
    total_duration_sec: int,
    track_ids: List[str],
    json_path: str,
) -> None:
    # Insert playlist (minimal column set; compatible with your fixture schema and DB schema)
    conn.execute(
        """
        INSERT INTO playlists (
          id, created_at, slug, name, context, mood, genre,
          target_minutes, seed, track_count, total_duration_sec, json_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            playlist_id,
            created_at,
            slug,
            name,
            context,
            mood,
            genre,
            int(target_minutes),
            int(seed),
            int(len(track_ids)),
            int(total_duration_sec),
            json_path,
        ),
    )

    # Insert items
    for pos, tid in enumerate(track_ids, start=1):
        conn.execute(
            """
            INSERT INTO playlist_items (playlist_id, track_id, position)
            VALUES (?, ?, ?)
            """,
            (playlist_id, tid, int(pos)),
        )


def cmd_run_daily(args: argparse.Namespace) -> int:
    # Local imports to keep CLI import-time light
    from mgc.storage import StoragePaths
    from mgc.db import DB
    from mgc.agents.music_agent import MusicAgent
    from mgc.agents.marketing_agent import MarketingAgent
    from mgc.playlist import build_playlist

    stamp = (args.stamp or _stable_stamp_default()).strip()
    db_path = Path(args.db)
    data_dir = Path(args.data_dir)
    artifacts_root = Path(args.artifacts_dir) / "runs" / stamp

    evidence = EvidencePaths(artifacts_root)
    _ensure_dir(evidence.root)
    _ensure_dir(evidence.posts_dir)

    # Storage for track audio + post drafts
    storage = StoragePaths(data_dir=data_dir)
    storage.ensure()

    # Init DB schema (CREATE TABLE IF NOT EXISTS)
    db = DB(db_path)
    db.init()

    # 1) Generate a track
    music = MusicAgent(storage=storage)
    track_art = music.run_daily()
    track_row = music.to_db_row(track_art)
    db.insert_track(track_row)

    _write_json(evidence.track_json, track_row)

    # 2) Build a playlist (bonus, but it strengthens the “pipeline” story)
    # Keep this deterministic: seed + lookback are explicit.
    playlist_obj = build_playlist(
        db_path=db_path,
        context=args.context,
        slug=f"{args.context}_radio",
        name=f"{args.context.title()} Radio",
        target_minutes=int(args.target_minutes),
        seed=int(args.seed),
        lookback_playlists=int(args.lookback_playlists),
    )

    # Write playlist JSON to the *data* area (canonical) and evidence bundle.
    playlist_id = str(uuid.uuid4())
    created_at = _now_iso()

    playlist_slug = f"{args.context}_radio"
    playlist_out_dir = data_dir / "playlists"
    _ensure_dir(playlist_out_dir)
    playlist_json_path = playlist_out_dir / f"{playlist_slug}.json"
    _write_json(playlist_json_path, playlist_obj)
    _write_json(evidence.playlist_json, playlist_obj)

    # Insert playlist + items into DB
    track_ids = [it["track_id"] for it in playlist_obj.get("items", []) if it.get("track_id")]
    total_dur = int(playlist_obj.get("stats", {}).get("total_duration_sec", 0) or 0)

    with db.connect() as conn:
        _insert_playlist_and_items(
            conn,
            playlist_id=playlist_id,
            created_at=created_at,
            slug=playlist_slug,
            name=str(playlist_obj.get("name") or f"{args.context.title()} Radio"),
            context=args.context,
            mood=str(playlist_obj.get("filters", {}).get("mood") or "") or None,
            genre=str(playlist_obj.get("filters", {}).get("genre") or "") or None,
            target_minutes=int(args.target_minutes),
            seed=int(args.seed),
            total_duration_sec=total_dur,
            track_ids=track_ids,
            json_path=str(playlist_json_path),
        )
        conn.commit()

    # 3) Plan marketing posts (writes JSON drafts + inserts DB rows)
    marketing = MarketingAgent(storage=storage)
    posts = marketing.plan_posts(track_art)
    for p in posts:
        db.insert_post(p)

    # Also copy post drafts into the evidence bundle (so submission reviewers can find them fast)
    # The MarketingAgent already wrote drafts to storage.posts_dir; we mirror them here.
    try:
        posts_dir = Path(storage.posts_dir)
        for draft in posts_dir.glob(f"{track_art.track_id}_*.json"):
            dst = evidence.posts_dir / draft.name
            dst.write_text(draft.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        # Evidence copy is best-effort; DB rows are the source of truth.
        pass

    # 4) Evidence summary
    run_summary: Dict[str, Any] = {
        "stamp": stamp,
        "created_at": created_at,
        "db": str(db_path),
        "data_dir": str(data_dir),
        "provider": os.getenv("MGC_PROVIDER", "stub"),
        "context": args.context,
        "track": {
            "id": track_art.track_id,
            "title": track_art.title,
            "full_path": track_art.full_path,
            "preview_path": track_art.preview_path,
        },
        "playlist": {
            "id": playlist_id,
            "slug": playlist_slug,
            "json_path": str(playlist_json_path),
            "track_count": len(track_ids),
            "total_duration_sec": total_dur,
        },
        "marketing_posts": {
            "count": len(posts),
            "platforms": sorted({p.get("platform") for p in posts if p.get("platform")}),
        },
        "evidence_dir": str(evidence.root),
    }
    _write_json(evidence.summary_json, run_summary)

    # Human-friendly output
    if args.json:
        print(json.dumps(run_summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(str(evidence.summary_json))

    return 0


def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    run = subparsers.add_parser("run", help="Run the autonomous pipeline (generation → storage → promotion)")
    rs = run.add_subparsers(dest="run_cmd", required=True)

    daily = rs.add_parser("daily", help="Run a daily music drop pipeline")
    daily.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"))
    daily.add_argument("--data-dir", default="data", help="Where tracks/previews/posts/playlists are written")
    daily.add_argument("--artifacts-dir", default="artifacts", help="Where evidence bundles are written")
    daily.add_argument("--stamp", default=None, help="Run stamp (default: UTC date YYYY-MM-DD)")
    daily.add_argument("--context", default="focus", choices=["focus", "workout", "sleep"])
    daily.add_argument("--target-minutes", type=int, default=20)
    daily.add_argument("--seed", type=int, default=1)
    daily.add_argument("--lookback-playlists", type=int, default=3)
    daily.add_argument("--json", action="store_true", help="Print JSON run summary")
    daily.set_defaults(func=cmd_run_daily)

    weekly = rs.add_parser("weekly", help="Alias for daily (you can schedule weekly drops too)")
    # Keep args identical
    for a in daily._actions:  # type: ignore[attr-defined]
        if a.dest in {"help"}:
            continue
    weekly.add_argument("--db", default=os.environ.get("MGC_DB", "data/db.sqlite"))
    weekly.add_argument("--data-dir", default="data")
    weekly.add_argument("--artifacts-dir", default="artifacts")
    weekly.add_argument("--stamp", default=None)
    weekly.add_argument("--context", default="focus", choices=["focus", "workout", "sleep"])
    weekly.add_argument("--target-minutes", type=int, default=20)
    weekly.add_argument("--seed", type=int, default=1)
    weekly.add_argument("--lookback-playlists", type=int, default=3)
    weekly.add_argument("--json", action="store_true")
    weekly.set_defaults(func=cmd_run_daily)
