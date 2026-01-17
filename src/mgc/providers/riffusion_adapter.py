from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from mgc.context import get_context_spec
from mgc.providers.base import GenerateRequest, GenerateResult, ProviderError, sha256_bytes
from mgc.providers.riffusion_provider import RiffusionProvider


def _stable_int_from_key(key: str, lo: int, hi: int) -> int:
    if hi <= lo:
        return lo
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return lo + (h % (hi - lo + 1))


class RiffusionAdapter:
    name = "riffusion"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        if req is None:
            req = GenerateRequest(
                track_id=str(kwargs.get("track_id") or "track"),
                context=str(kwargs.get("context") or "focus"),
                seed=int(kwargs.get("seed") or 1),
                deterministic=bool(kwargs.get("deterministic") or False),
                schedule=str(kwargs.get("schedule") or ""),
                period_key=str(kwargs.get("period_key") or ""),
                out_dir=str(kwargs.get("out_dir") or ""),
                out_rel=str(kwargs.get("out_rel") or ""),
                run_id=kwargs.get("run_id"),
                prompt=str(kwargs.get("prompt") or ""),
                ts=str(kwargs.get("ts") or kwargs.get("now_iso") or ""),
            )

        spec = get_context_spec(req.context)

        # Deterministic seed if requested; otherwise let provider randomize
        seed_int = None
        if req.deterministic:
            seed_int = _stable_int_from_key(f"{req.track_id}|{req.seed}|{req.context}", 1, 2_000_000_000)

        bpm = _stable_int_from_key(f"{req.track_id}|bpm", spec.bpm_min, spec.bpm_max) if req.deterministic else spec.bpm_max

        title = f"{spec.name.title()} Track {req.seed}"
        mood = spec.mood
        genre = spec.genre

        server_url = (os.environ.get("MGC_RIFFUSION_URL") or "http://127.0.0.1:3013").strip()
        prov = RiffusionProvider(server_url=server_url)

        prompt = req.prompt or spec.prompt

        # Use a temp file; pipeline will write bytes to final location
        with tempfile.TemporaryDirectory(prefix="mgc_riffusion_") as td:
            out_mp3 = Path(td) / f"{req.track_id}.mp3"
            res = prov.generate(
                out_mp3=out_mp3,
                title=title,
                mood=mood,
                genre=genre,
                bpm=int(bpm),
                prompt=prompt,
                seed=seed_int,
            )

            mp3_bytes = out_mp3.read_bytes()

        meta: Dict[str, Any] = {
            "provider": self.name,
            "riffusion_url": server_url,
            "duration_sec": res.duration_sec,
            "bpm": res.bpm,
            "title": res.title,
            "mood": res.mood,
            "genre": res.genre,
            "prompt": prompt,
            "bytes": len(mp3_bytes),
            "sha256": sha256_bytes(mp3_bytes),
        }

        return GenerateResult(provider=self.name, artifact_bytes=mp3_bytes, mime="audio/mpeg", ext=".mp3", meta=meta)
