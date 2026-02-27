from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import requests


class ProviderError(RuntimeError):
    pass


def _env_int(name: str, default: int = 0) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _normalize_run_inference_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return "http://127.0.0.1:3013/run_inference/"

    if "/run_inference" not in url:
        return url.rstrip("/") + "/run_inference/"

    if url.endswith("/run_inference"):
        return url + "/"
    return url


def _timeouts(timeout_s: Optional[float]) -> Tuple[float, float]:
    connect_default = float(_env_int("MGC_RIFFUSION_CONNECT_TIMEOUT", 2))
    read_default = float(_env_int("MGC_RIFFUSION_READ_TIMEOUT", 120))
    if timeout_s is None:
        return (connect_default, read_default)
    # Adapter timeout should behave like read timeout; keep connect fast.
    return (connect_default, float(timeout_s))


def _retries() -> int:
    return max(0, _env_int("MGC_RIFFUSION_RETRIES", 1))


def _retry_sleep_s() -> float:
    return float(_env_float("MGC_RIFFUSION_RETRY_SLEEP", 0.25))


def _b64_strip_prefix(s: str) -> str:
    # Accept "data:audio/mpeg;base64,...."
    if s.lower().startswith("data:") and "," in s:
        return s.split(",", 1)[1]
    return s


def _extract_audio_bytes(obj: Dict[str, Any]) -> bytes:
    candidates = [
        "audio",
        "mp3",
        "audio_mp3",
        "audio_base64",
        "wav",
        "audio_wav",
    ]

    used_key = None
    b64 = None
    for k in candidates:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            used_key = k
            b64 = v.strip()
            break

    if b64 is None:
        raise ProviderError(
            "riffusion server response missing audio data (expected one of: "
            + ", ".join(repr(k) for k in candidates)
            + ")"
        )

    try:
        raw = base64.b64decode(_b64_strip_prefix(b64), validate=False)
    except Exception as e:
        raise ProviderError(f"riffusion server returned non-base64 audio in field {used_key!r}: {e}") from e

    if not raw:
        raise ProviderError(f"riffusion server returned empty audio bytes in field {used_key!r}")

    return raw


class RiffusionProvider:
    """
    HTTP client for a riffusion.server instance.

    Adapter calls: RiffusionProvider(server_url=...)
    Server expects the classic InferenceInput JSON shape:
      - start/end PromptInput objects
      - alpha
      - num_inference_steps
      - seed_image_id
      - (optional) mask_image_id

    IMPORTANT: Do not send extra keys (e.g. bpm) — some server variants 500 on unexpected fields.
    """

    def __init__(
        self,
        *,
        server_url: Optional[str] = None,
        url: Optional[str] = None,  # allow alt name
        steps: int = 50,
        guidance: float = 7.0,
        denoising: float = 0.75,
        timeout_s: Optional[float] = None,
    ):
        u = server_url or url or ""
        self.url = _normalize_run_inference_url(u)

        # Allow env overrides if adapter doesn’t pass them
        steps_env = os.environ.get("RIFFUSION_STEPS") or os.environ.get("MGC_RIFFUSION_STEPS")
        guidance_env = os.environ.get("RIFFUSION_GUIDANCE") or os.environ.get("MGC_RIFFUSION_GUIDANCE")
        denoise_env = os.environ.get("RIFFUSION_DENOISE") or os.environ.get("MGC_RIFFUSION_DENOISE")
        timeout_env = os.environ.get("RIFFUSION_TIMEOUT") or os.environ.get("MGC_RIFFUSION_TIMEOUT")

        if steps_env:
            try:
                steps = int(float(steps_env))
            except ValueError:
                pass
        if guidance_env:
            try:
                guidance = float(guidance_env)
            except ValueError:
                pass
        if denoise_env:
            try:
                denoising = float(denoise_env)
            except ValueError:
                pass
        if timeout_env and timeout_s is None:
            try:
                timeout_s = float(timeout_env)
            except ValueError:
                pass

        self.steps = int(steps)
        self.guidance = float(guidance)
        self.denoising = float(denoising)
        self.timeout_s = timeout_s

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        t = _timeouts(self.timeout_s)
        retries = _retries()
        sleep_s = _retry_sleep_s()
        last_err: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                r = requests.post(
                    self.url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=t,
                )
                if r.status_code != 200:
                    snippet = (r.text or "").strip()
                    if len(snippet) > 400:
                        snippet = snippet[:400] + "…"
                    raise ProviderError(f"riffusion server returned {r.status_code}: {snippet}")

                try:
                    obj = r.json()
                except Exception:
                    raise ProviderError("riffusion server returned non-JSON response")

                if not isinstance(obj, dict):
                    raise ProviderError("riffusion server returned unexpected JSON shape (expected object)")
                return obj

            except Exception as e:
                last_err = e
                if attempt < retries:
                    time.sleep(sleep_s)
                    continue
                break

        raise ProviderError(f"riffusion server request failed after {retries + 1} attempts: {last_err}") from last_err

    def generate(self, *, prompt: str, seed: int, out_mp3_path: str, negative_prompt: Optional[str] = None) -> Dict[str, Any]:
        seed_image_id = os.environ.get("MGC_RIFFUSION_SEED_IMAGE_ID") or "og_beat"

        payload: Dict[str, Any] = {
            "alpha": 0.0,
            "num_inference_steps": self.steps,
            "seed_image_id": seed_image_id,
            "start": {
                "prompt": prompt,
                "seed": int(seed),
                "denoising": self.denoising,
                "guidance": self.guidance,
            },
            "end": {
                "prompt": prompt,
                "seed": int(seed),
                "denoising": self.denoising,
                "guidance": self.guidance,
            },
        }

        if negative_prompt:
            payload["start"]["negative_prompt"] = negative_prompt
            payload["end"]["negative_prompt"] = negative_prompt

        obj = self._post_json(payload)
        audio_bytes = _extract_audio_bytes(obj)

        os.makedirs(os.path.dirname(out_mp3_path) or ".", exist_ok=True)
        with open(out_mp3_path, "wb") as f:
            f.write(audio_bytes)

        if os.path.getsize(out_mp3_path) <= 0:
            raise ProviderError("riffusion did not produce an mp3 artifact (missing or empty).")

        meta: Dict[str, Any] = {
            "riffusion_url": self.url,
            "seed_image_id": seed_image_id,
            "num_inference_steps": self.steps,
            "guidance": self.guidance,
            "denoising": self.denoising,
        }
        if isinstance(obj.get("duration_s"), (int, float)):
            meta["duration_s"] = float(obj["duration_s"])
        if isinstance(obj.get("image"), str) and obj["image"]:
            meta["image"] = obj["image"]

        return meta
