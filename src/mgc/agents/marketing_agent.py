from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from mgc.agents.music_agent import TrackArtifact


DEFAULT_PLATFORMS = ["x", "youtube_shorts", "instagram_reels", "tiktok"]


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _fixed_now_iso(deterministic: bool) -> str:
    if deterministic:
        fixed = (os.environ.get("MGC_FIXED_TIME") or "2020-01-01T00:00:00Z").strip()
        return fixed.replace("Z", "+00:00")
    return datetime.now(timezone.utc).isoformat()


def _stable_uuid(deterministic: bool, name: str) -> str:
    if deterministic:
        ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
        return str(uuid.uuid5(ns, name))
    return str(uuid.uuid4())


@dataclass(frozen=True)
class StoragePaths:
    """
    Minimal storage interface expected by this agent.

    If your project already has mgc.storage.StoragePaths, you can delete this class
    and import that one instead — but this keeps the agent usable on its own.
    """
    root: str

    @property
    def posts_dir(self) -> str:
        return str(Path(self.root) / "marketing" / "posts")

    def ensure(self) -> None:
        Path(self.posts_dir).mkdir(parents=True, exist_ok=True)


class MarketingAgent:
    """
    Plans marketing posts (JSON payloads) deterministically when requested.

    Determinism rules:
    - In deterministic mode, created_at uses MGC_FIXED_TIME
    - IDs are uuid5 derived from (track_id, platform, created_at)
    """

    def __init__(self, storage: StoragePaths, platforms: Optional[List[str]] = None):
        self.storage = storage
        self.platforms = platforms or DEFAULT_PLATFORMS

    def plan_posts(
        self,
        track: TrackArtifact,
        *,
        deterministic: bool = False,
        created_at: Optional[str] = None,
    ) -> list[Dict]:
        self.storage.ensure()

        det = bool(deterministic) or _env_bool("MGC_DETERMINISTIC", False)
        ts = created_at or _fixed_now_iso(det)

        posts = []
        for platform in self.platforms:
            post_id = _stable_uuid(det, f"post|track={track.track_id}|platform={platform}|ts={ts}")

            payload = {
                "platform": platform,
                "track_id": track.track_id,
                "title": track.title,
                "caption": self._caption(track, platform),
                "hashtags": self._hashtags(track, platform),
                "preview_path": track.preview_path,
                "created_at": ts,
                "status": "planned",
            }

            out_path = Path(self.storage.posts_dir) / f"{track.track_id}_{platform}.json"
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            posts.append(
                {
                    "id": post_id,
                    "created_at": ts,
                    "track_id": track.track_id,
                    "platform": platform,
                    "payload_json": json.dumps(payload, sort_keys=True),
                    "status": "planned",
                }
            )

        # Stable ordering
        posts.sort(key=lambda x: (x["platform"], x["track_id"]))
        return posts

    def _caption(self, track: TrackArtifact, platform: str) -> str:
        # No emojis. Keep it short and reusable.
        if platform == "x":
            return f"Daily AI drop: {track.title} — {track.mood} vibes. Preview attached."
        return f"New AI track: {track.title}. Mood: {track.mood}. Genre: {track.genre}."

    def _hashtags(self, track: TrackArtifact, platform: str) -> list[str]:
        base = ["#AI", "#AIMusic", "#NewMusic", "#Indie"]
        mood = "#Focus" if "focus" in track.mood.lower() else "#Vibes"
        genre = "#Ambient" if "ambient" in track.genre.lower() else "#Electronic"
        tags = base + [mood, genre]
        # Stable ordering
        return sorted(set(tags), key=lambda s: s.lower())
