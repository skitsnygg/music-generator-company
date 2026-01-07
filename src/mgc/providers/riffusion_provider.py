from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import requests

@dataclass
class RiffusionResult:
    duration_sec: float
    bpm: int
    title: str
    mood: str
    genre: str
    full_path: str  # mp3 written by provider

class RiffusionProvider:
    """
    Calls a locally running Riffusion server:
      python -m riffusion.server --host 127.0.0.1 --port 3013

    Endpoint returns base64 encoded MP3 audio in JSON. :contentReference[oaicite:1]{index=1}
    """

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def generate(
        self,
        *,
        out_mp3: Path,
        title: str,
        mood: str,
        genre: str,
        bpm: int,
        prompt: str,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance: float | None = None,
        denoising: float | None = None,
        seed_image_id: str = "og_beat",
    ) -> RiffusionResult:
        out_mp3.parent.mkdir(parents=True, exist_ok=True)

        seed = int(seed if seed is not None else (uuid.uuid4().int % 2_000_000_000))
        steps = int(num_inference_steps or int(os.getenv("RIFFUSION_STEPS", "25")))
        guidance = float(guidance if guidance is not None else float(os.getenv("RIFFUSION_GUIDANCE", "7.0")))
        denoising = float(denoising if denoising is not None else float(os.getenv("RIFFUSION_DENOISE", "0.75")))

        payload: Dict[str, Any] = {
            "alpha": 1.0,
            "num_inference_steps": steps,
            "seed_image_id": seed_image_id,
            "start": {
                "prompt": prompt,
                "seed": seed,
                "denoising": denoising,
                "guidance": guidance,
            },
            "end": {
                "prompt": prompt,
                "seed": seed,
                "denoising": denoising,
                "guidance": guidance,
            },
        }

        r = requests.post(self.server_url, json=payload, timeout=300, allow_redirects=True)

        r.raise_for_status()
        data = r.json()

        if "audio" not in data:
            raise RuntimeError(f"Riffusion response missing 'audio': {json.dumps(data)[:500]}")

        audio_bytes = base64.b64decode(data["audio"])
        out_mp3.write_bytes(audio_bytes)

        # Riffusion returns a short clip; we’ll treat it as the “full track” for now.
        return RiffusionResult(
            duration_sec=float(data.get('duration_s', 30.0)),
            bpm=bpm,
            title=title,
            mood=mood,
            genre=genre,
            full_path=str(out_mp3),
        )
