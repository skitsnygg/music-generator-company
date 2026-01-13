from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

from mgc.context import get_context_spec
from mgc.providers.base import GenerateRequest, GenerateResult

# IMPORTANT:
# Update this import path to wherever your existing RiffusionProvider lives.
# Example possibilities:
#   from mgc.riffusion_provider import RiffusionProvider
#   from mgc.providers.riffusion_provider import RiffusionProvider
from mgc.riffusion_provider import RiffusionProvider  # <- change if needed


def _stable_int_from_key(key: str, lo: int, hi: int) -> int:
    if hi <= lo:
        return lo
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return lo + (h % (hi - lo + 1))


class RiffusionAdapter:
    name = "riffusion"

    def generate(self, req: GenerateRequest) -> GenerateResult:
        spec = get_context_spec(req.context)

        # Deterministic seed if requested; otherwise let riffusion provider randomize
        seed_int = None
        if req.deterministic:
            seed_int = _stable_int_from_key(f"{req.track_id}|{req.seed}|{req.context}", 1, 2_000_000_000)

        bpm = _stable_int_from_key(f"{req.track_id}|bpm", spec.bpm_min, spec.bpm_max) if req.deterministic else spec.bpm_max

        title = f"{spec.name.title()} Track {req.seed}"
        mood = spec.mood
        genre = spec.genre

        server_url = (os.environ.get("MGC_RIFFUSION_URL") or "http://127.0.0.1:3013").strip()
        prov = RiffusionProvider(server_url=server_url)

        # out_rel points at where run_cli intends to write the artifact.
        # We'll honor that location and force .mp3.
        out_path = Path(req.out_rel)
        if out_path.suffix.lower() != ".mp3":
            out_path = out_path.with_suffix(".mp3")

        res = prov.generate(
            out_mp3=out_path,
            title=title,
            mood=mood,
            genre=genre,
            bpm=int(bpm),
            prompt=req.prompt,
            seed=seed_int,
        )

        mp3_bytes = Path(res.full_path).read_bytes()

        meta: Dict[str, Any] = {
            "provider": self.name,
            "riffusion_url": server_url,
            "duration_sec": res.duration_sec,
            "bpm": res.bpm,
            "title": res.title,
            "mood": res.mood,
            "genre": res.genre,
        }

        return GenerateResult(
            provider=self.name,
            artifact_bytes=mp3_bytes,
            mime="audio/mpeg",
            ext=".mp3",
            meta=meta,
        )
