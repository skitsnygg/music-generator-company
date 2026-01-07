from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from mgc.storage import StoragePaths
from mgc.agents.music_agent import TrackArtifact

DEFAULT_PLATFORMS = ["x", "youtube_shorts", "instagram_reels", "tiktok"]

class MarketingAgent:
    def __init__(self, storage: StoragePaths, platforms: List[str] | None = None):
        self.storage = storage
        self.platforms = platforms or DEFAULT_PLATFORMS

    def plan_posts(self, track: TrackArtifact) -> list[Dict]:
        self.storage.ensure()
        created_at = datetime.now(timezone.utc).isoformat()

        posts = []
        for platform in self.platforms:
            post_id = str(uuid.uuid4())
            payload = {
                "platform": platform,
                "track_id": track.track_id,
                "title": track.title,
                "caption": self._caption(track, platform),
                "hashtags": self._hashtags(track, platform),
                "preview_path": track.preview_path,
                "created_at": created_at,
                "status": "planned",
            }

            out_path = Path(self.storage.posts_dir) / f"{track.track_id}_{platform}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            posts.append({
                "id": post_id,
                "created_at": created_at,
                "track_id": track.track_id,
                "platform": platform,
                "payload_json": json.dumps(payload),
                "status": "planned",
            })

        return posts

    def _caption(self, track: TrackArtifact, platform: str) -> str:
        if platform == "x":
            return f"Daily AI drop: {track.title} — {track.mood} vibes. 🎧 Preview attached."
        return f"New AI track: {track.title} • {track.mood} • {track.genre}. 🎧"

    def _hashtags(self, track: TrackArtifact, platform: str) -> list[str]:
        base = ["#AI", "#AIMusic", "#NewMusic", "#Indie"]
        mood = "#Focus" if "focus" in track.mood.lower() else "#Vibes"
        genre = "#Ambient" if "ambient" in track.genre.lower() else "#Electronic"
        return base + [mood, genre]
