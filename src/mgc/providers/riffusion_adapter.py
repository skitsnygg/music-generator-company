from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from mgc.context import get_context_spec
from mgc.providers.base import GenerateRequest, ProviderError, sha256_bytes
from mgc.providers.riffusion_provider import RiffusionProvider


def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    u = u.rstrip("/")
    if "/run_inference" not in u:
        return u + "/run_inference/"
    return u.rstrip("/") + "/"


def _stable_int_from_key(key: str, lo: int, hi: int) -> int:
    import hashlib

    if hi <= lo:
        return lo
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return lo + (h % (hi - lo + 1))


class RiffusionAdapter:
    """
    Provider adapter that returns a dict (NOT GenerateResult) to avoid signature/kwarg
    filtering dropping ext/mime.

    Required output keys for MusicAgent:
      - provider, track_id
      - artifact_bytes (mp3 bytes) OR artifact_path
      - ext (".mp3") and mime ("audio/mpeg") so MusicAgent materializes correctly
      - sha256 (optional; MusicAgent can compute if missing)
      - meta (dict)
    """

    name = "riffusion"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> Dict[str, Any]:
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
        seed_int: Optional[int] = None
        if req.deterministic:
            seed_int = _stable_int_from_key(f"{req.track_id}|{req.seed}|{req.context}", 1, 2_000_000_000)

        bpm = (
            _stable_int_from_key(f"{req.track_id}|bpm", spec.bpm_min, spec.bpm_max)
            if req.deterministic
            else spec.bpm_max
        )

        title = f"{spec.name.title()} Track {req.seed}"
        mood = spec.mood
        genre = spec.genre

        server_url = _normalize_url(
            os.environ.get("MGC_RIFFUSION_URL") or "http://127.0.0.1:3013/run_inference/"
        )
        prov = RiffusionProvider(server_url=server_url)

        prompt = req.prompt or spec.prompt

        # Pull optional overrides from env (so you can tune without code changes)
        steps_env = os.environ.get("RIFFUSION_STEPS")
        guidance_env = os.environ.get("RIFFUSION_GUIDANCE")
        denoise_env = os.environ.get("RIFFUSION_DENOISE")
        timeout_env = os.environ.get("RIFFUSION_TIMEOUT")

        num_steps = int(steps_env) if (steps_env or "").strip() else None
        guidance = float(guidance_env) if (guidance_env or "").strip() else None
        denoise = float(denoise_env) if (denoise_env or "").strip() else None
        timeout_s = int(timeout_env) if (timeout_env or "").strip() else None

        mp3_bytes: bytes
        preview_bytes: Optional[bytes] = None

        # Use a temp dir for provider I/O; we return bytes to MusicAgent.
        with tempfile.TemporaryDirectory(prefix="mgc_riffusion_") as td:
            out_mp3 = Path(td) / f"{req.track_id}.mp3"
            out_preview = Path(td) / f"{req.track_id}.jpg"

            try:
                res = prov.generate(
                    out_mp3=out_mp3,
                    out_preview_jpg=out_preview,
                    title=title,
                    mood=mood,
                    genre=genre,
                    bpm=int(bpm),
                    prompt=prompt,
                    seed=seed_int,
                    num_inference_steps=num_steps,
                    guidance=guidance,
                    denoising=denoise,
                    timeout_s=timeout_s,
                )
            except Exception as e:
                raise ProviderError(f"riffusion.generate failed: {e}") from e

            mp3_bytes = out_mp3.read_bytes()
            if out_preview.exists() and out_preview.stat().st_size > 0:
                preview_bytes = out_preview.read_bytes()

        meta: Dict[str, Any] = {
            "provider": self.name,
            "riffusion_url": server_url,
            "duration_sec": float(res.duration_sec),
            "bpm": int(res.bpm),
            "title": res.title,
            "mood": res.mood,
            "genre": res.genre,
            "prompt": prompt,
            "deterministic": bool(req.deterministic),
            "seed": int(req.seed),
            "context": str(req.context),
            "schedule": str(req.schedule),
            "period_key": str(req.period_key),
            # IMPORTANT: carry ext/mime into meta as well
            "ext": ".mp3",
            "mime": "audio/mpeg",
            "sha256": sha256_bytes(mp3_bytes),
            "bytes": len(mp3_bytes),
        }

        out: Dict[str, Any] = {
            "provider": self.name,
            "track_id": str(req.track_id),
            "artifact_bytes": mp3_bytes,
            "ext": ".mp3",
            "mime": "audio/mpeg",
            "sha256": meta["sha256"],
            "title": res.title,
            "mood": res.mood,
            "genre": res.genre,
            "meta": meta,
        }

        # Optional: preview bytes (MusicAgent will materialize and set preview_path)
        if preview_bytes:
            out["preview_bytes"] = preview_bytes
            out["preview_ext"] = ".jpg"
            out["preview_mime"] = "image/jpeg"
            # Also keep hashes in meta (JSON-safe)
            out["meta"] = dict(meta)
            out["meta"].update(
                {
                    "preview_sha256": sha256_bytes(preview_bytes),
                    "preview_bytes": len(preview_bytes),
                    "preview_ext": ".jpg",
                    "preview_mime": "image/jpeg",
                }
            )

        return out
