from __future__ import annotations

import inspect
import os
import shutil
import subprocess
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


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return (v if v is not None else default).strip()


def _has_kw(fn: Any, key: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return key in sig.parameters
    except Exception:
        return False


def _which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def _encode_mp3_with_lame(wav_path: Path, mp3_path: Path, *, quality: str) -> Dict[str, Any]:
    """
    Deterministic MP3 encoding via LAME.

    quality:
      - "v0": VBR V0 (recommended)
      - "320": CBR 320kbps
    """
    lame = _which("lame")
    if not lame:
        raise ProviderError("MGC_MP3_QUALITY requested LAME encoding but 'lame' was not found on PATH.")

    q = (quality or "").strip().lower()
    if q not in ("v0", "320"):
        raise ProviderError(f"Unsupported MGC_MP3_QUALITY='{quality}'. Use 'v0' or '320' or 'server'.")

    cmd = [
        lame,
        "--noreplaygain",
        "--nohist",
        "--silent",
    ]

    if q == "v0":
        cmd += ["-V0"]
    else:
        cmd += ["--cbr", "-b", "320"]

    cmd += [str(wav_path), str(mp3_path)]
    subprocess.run(cmd, check=True)

    # keep meta small + path-free (determinism / privacy)
    return {
        "mp3_encoder": "lame",
        "mp3_quality": q,
        "mp3_flags": " ".join(cmd[1:7]),
    }


class RiffusionAdapter:
    """
    Adapter returns a dict so downstream kwarg filtering doesn't drop ext/mime.

    We prefer: server WAV -> LAME encode -> MP3 bytes.
    Fallback: server MP3 bytes (quality="server").
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

        server_url = _normalize_url(_env_str("MGC_RIFFUSION_URL", "http://127.0.0.1:3013/run_inference/"))
        prov = RiffusionProvider(server_url=server_url)

        prompt = req.prompt or spec.prompt

        # optional overrides
        steps_env = _env_str("RIFFUSION_STEPS") or _env_str("MGC_RIFFUSION_STEPS")
        guidance_env = _env_str("RIFFUSION_GUIDANCE") or _env_str("MGC_RIFFUSION_GUIDANCE")
        denoise_env = _env_str("RIFFUSION_DENOISE") or _env_str("MGC_RIFFUSION_DENOISE")
        timeout_env = _env_str("RIFFUSION_TIMEOUT") or _env_str("MGC_RIFFUSION_TIMEOUT")

        num_steps = int(steps_env) if steps_env else None
        guidance = float(guidance_env) if guidance_env else None
        denoise = float(denoise_env) if denoise_env else None
        timeout_s = int(timeout_env) if timeout_env else None

        # MP3 quality selection:
        #   server: trust server MP3
        #   v0: LAME V0 from server WAV (preferred)
        #   320: LAME 320 from server WAV
        mp3_quality = (_env_str("MGC_MP3_QUALITY", "server") or "server").lower()
        want_lame = mp3_quality in ("v0", "320")

        mp3_bytes: bytes
        preview_bytes: Optional[bytes] = None
        enc_meta: Dict[str, Any] = {}

        with tempfile.TemporaryDirectory(prefix="mgc_riffusion_") as td:
            td_path = Path(td)
            out_preview = td_path / f"{req.track_id}.jpg"
            out_mp3 = td_path / f"{req.track_id}.mp3"
            out_wav = td_path / f"{req.track_id}.wav"

            # Prefer WAV path if provider supports it and we want LAME encoding.
            used_wav = False

            try:
                if want_lame and _has_kw(prov.generate, "out_wav"):
                    prov.generate(
                        out_wav=out_wav,
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
                    if out_wav.exists() and out_wav.stat().st_size > 0:
                        used_wav = True
                else:
                    prov.generate(
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
            except TypeError:
                # Signature mismatch (older provider): retry with MP3-only
                prov.generate(
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

            if out_preview.exists() and out_preview.stat().st_size > 0:
                preview_bytes = out_preview.read_bytes()

            if used_wav:
                # encode with LAME
                try:
                    enc_meta = _encode_mp3_with_lame(out_wav, out_mp3, quality=mp3_quality)
                except Exception as e:
                    raise ProviderError(f"riffusion mp3 re-encode failed: {e}") from e
            else:
                # fallback: server MP3 must exist
                if want_lame:
                    # We wanted LAME but couldn't get WAV; record that for debugging.
                    enc_meta = {"mp3_encoder": "server", "mp3_quality": "server", "mp3_note": "no_wav_available"}

            if not out_mp3.exists() or out_mp3.stat().st_size <= 0:
                raise ProviderError("riffusion did not produce an mp3 artifact (missing or empty).")

            mp3_bytes = out_mp3.read_bytes()

        meta: Dict[str, Any] = {
            "provider": self.name,
            "riffusion_url": server_url,
            "bpm": int(bpm),
            "title": title,
            "mood": mood,
            "genre": genre,
            "prompt": prompt,
            "deterministic": bool(req.deterministic),
            "seed": int(req.seed),
            "context": str(req.context),
            "schedule": str(req.schedule),
            "period_key": str(req.period_key),
            "ext": ".mp3",
            "mime": "audio/mpeg",
            "sha256": sha256_bytes(mp3_bytes),
            "bytes": len(mp3_bytes),
            "mp3_quality": mp3_quality,
        }
        if enc_meta:
            meta.update(enc_meta)

        out: Dict[str, Any] = {
            "provider": self.name,
            "track_id": str(req.track_id),
            "artifact_bytes": mp3_bytes,
            "ext": ".mp3",
            "mime": "audio/mpeg",
            "sha256": meta["sha256"],
            "title": meta["title"],
            "mood": meta["mood"],
            "genre": meta["genre"],
            "meta": meta,
        }

        if preview_bytes:
            out["preview_bytes"] = preview_bytes
            out["preview_ext"] = ".jpg"
            out["preview_mime"] = "image/jpeg"
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
