from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from mgc.storage import StoragePaths
from mgc.utils.audio import make_preview_clip
from mgc.providers.generator_stub import GeneratorStub
from mgc.providers.riffusion_provider import RiffusionProvider, RiffusionResult

@dataclass
class TrackArtifact:
    track_id: str
    created_at: str
    title: str
    mood: str
    genre: str
    bpm: int
    duration_sec: float
    full_path: str
    preview_path: str


CONTEXT_PRESETS = {
    "focus": {
        "mood": "focus",
        "genre": "ambient electronic",
        "bpm": 110,
        "prompt": "ambient electronic, minimal, steady rhythm, clean pads, focus music",
    },
    "workout": {
        "mood": "workout",
        "genre": "high-energy electronic",
        "bpm": 140,
        "prompt": "energetic electronic, driving kick, pumping bass, crisp percussion, workout music",
    },
    "sleep": {
        "mood": "sleep",
        "genre": "soft ambient",
        "bpm": 70,
        "prompt": "slow soft ambient, gentle drones, warm pads, calm relaxing sleep music",
    },
}

def pick_context(iso_date: str) -> str:
    # stable rotation by day
    keys = list(CONTEXT_PRESETS.keys())
    n = sum(ord(c) for c in iso_date) % len(keys)
    return keys[n]


class MusicAgent:
    def __init__(self, storage: StoragePaths):
        self.storage = storage

        self.provider = os.getenv("MGC_PROVIDER", "stub").strip().lower()
        self.riffusion_url = os.getenv("RIFFUSION_URL", "http://127.0.0.1:3013/run_inference").strip()

        self.stub = GeneratorStub()
        self.riff = RiffusionProvider(self.riffusion_url)

    def run_daily(self) -> TrackArtifact:
        self.storage.ensure()

        now = datetime.now(timezone.utc)
        created_at = now.isoformat()
        track_id = str(uuid.uuid4())

        ctx_key = pick_context(now.strftime('%Y-%m-%d'))
        ctx = CONTEXT_PRESETS[ctx_key]
        mood = ctx['mood']
        genre = ctx['genre']
        bpm = int(ctx['bpm'])
        title = f"Daily Drop {now.strftime('%Y-%m-%d')} ({ctx_key.title()})"

        # Context preset (rotates daily)

        safe_base = f"{now.strftime('%Y%m%d')}_{track_id[:8]}"

        if self.provider == "riffusion":
            full_path = self.storage.tracks_dir / f"{safe_base}.mp3"
                        prompt = ctx['prompt']
            res: RiffusionResult = self.riff.generate(
                out_mp3=full_path,
                title=title,
                mood=mood,
                genre=genre,
                bpm=bpm,
                prompt=prompt,
            )
            duration_sec = res.duration_sec
        else:
            full_path = self.storage.tracks_dir / f"{safe_base}.wav"
            res2 = self.stub.generate(
                out_wav=full_path,
                title=title,
                mood=mood,
                genre=genre,
                bpm=bpm,
                duration_sec=60.0,
            )
            duration_sec = res2.duration_sec

        preview_path = self.storage.previews_dir / f"{safe_base}_preview.mp3"
        make_preview_clip(Path(full_path), preview_path, start_sec=5.0, duration_sec=20.0)

        return TrackArtifact(
            track_id=track_id,
            created_at=created_at,
            title=title,
            mood=mood,
            genre=genre,
            bpm=bpm,
            duration_sec=duration_sec,
            full_path=str(full_path),
            preview_path=str(preview_path),
        )

    def to_db_row(self, art: TrackArtifact) -> Dict:
        return {
            "id": art.track_id,
            "created_at": art.created_at,
            "title": art.title,
            "mood": art.mood,
            "genre": art.genre,
            "bpm": art.bpm,
            "duration_sec": art.duration_sec,
            "full_path": art.full_path,
            "preview_path": art.preview_path,
            "status": "generated",
        }
