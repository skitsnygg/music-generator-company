from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


class ProviderError(RuntimeError):
    pass


def _env_int(name: str, default: int = 0) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _normalize_base_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return "http://127.0.0.1:3013"
    u = u.rstrip("/")
    if u.endswith("/run_inference"):
        u = u[: -len("/run_inference")]
        u = u.rstrip("/")
    return u or "http://127.0.0.1:3013"


def _timeouts(timeout_s: Optional[float]) -> Tuple[float, float]:
    # split connect vs read so connect fails fast but generation can take longer
    connect_default = float(_env_int("MGC_RIFFUSION_CONNECT_TIMEOUT", 2))
    read_default = float(_env_int("MGC_RIFFUSION_READ_TIMEOUT", 120))
    if timeout_s is None:
        return (connect_default, read_default)
    # treat adapter timeout as read timeout; keep connect default
    return (connect_default, float(timeout_s))


def _retries() -> int:
    return max(0, _env_int("MGC_RIFFUSION_RETRIES", 1))


def _retry_sleep_s() -> float:
    return float(_env_float("MGC_RIFFUSION_RETRY_SLEEP", 0.25))


def _b64_strip_prefix(s: str) -> str:
    # Accept data:audio/mpeg;base64,... etc.
    if "," in s and s[:50].lower().startswith("data:"):
        return s.split(",", 1)[1]
    return s


def _extract_audio_bytes(obj: Dict[str, Any]) -> bytes:
    b64 = obj.get("audio")
    if not isinstance(b64, str) or not b64.strip():
        raise ProviderError("riffusion server response missing audio data in 'audio'")

    try:
        raw = base64.b64decode(_b64_strip_prefix(b64.strip()), validate=False)
    except Exception as e:
        raise ProviderError(f"riffusion server returned non-base64 audio in field 'audio': {e}") from e

    if not raw:
        raise ProviderError("riffusion server returned empty audio bytes in field 'audio'")

    return raw


@dataclass(frozen=True)
class RiffusionProviderConfig:
    url: str
    steps: int
    guidance: float
    denoising: float
    timeout_s: Optional[float] = None


class RiffusionProvider:
    """
    Provider that posts InferenceInput to a riffusion server and writes MP3 output.

    This provider expects the InferenceInput schema:
      - start/end: PromptInput(prompt, seed, negative_prompt?, denoising, guidance)
      - alpha: float
      - num_inference_steps: int
      - seed_image_id: str
      - mask_image_id: optional str

    The server returns JSON with base64 MP3 audio under key "audio".
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        server_url: Optional[str] = None,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        denoising: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ):
        raw = url if url not in (None, "") else server_url
        self.base_url = _normalize_base_url(raw or "")
        self.steps = int(steps) if steps is not None else None
        self.guidance = float(guidance) if guidance is not None else None
        self.denoising = float(denoising) if denoising is not None else None
        self.timeout_s = timeout_s

    def _run_inference_url(self) -> str:
        return f"{self.base_url}/run_inference/"

    def _post_json(self, payload: Dict[str, Any], *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        t = _timeouts(self.timeout_s if timeout_s is None else timeout_s)
        retries = _retries()
        sleep_s = _retry_sleep_s()
        last_err: Optional[Exception] = None
        url = self._run_inference_url()

        for attempt in range(retries + 1):
            try:
                r = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=t,
                )
                if r.status_code != 200:
                    body = (r.text or "").strip()
                    raise ProviderError(f"riffusion server returned {r.status_code}: {body}")

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

    def generate(
        self,
        *,
        prompt: str,
        seed: Optional[int] = None,
        out_mp3_path: Optional[str] = None,
        out_mp3: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        alpha: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed_image_id: Optional[str] = None,
        mask_image_id: Optional[str] = None,
        denoising: Optional[float] = None,
        guidance: Optional[float] = None,
        timeout_s: Optional[float] = None,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        if out_mp3_path is None:
            out_mp3_path = out_mp3
        if not out_mp3_path:
            raise ProviderError("riffusion provider requires out_mp3_path (or out_mp3)")

        steps_val = int(num_inference_steps) if num_inference_steps is not None else int(self.steps or 50)
        guidance_val = float(guidance) if guidance is not None else float(self.guidance or 7.0)
        denoise_val = float(denoising) if denoising is not None else float(self.denoising or 0.75)
        alpha_val = float(alpha) if alpha is not None else 0.0

        seed_val = int(seed) if seed is not None else 0

        seed_image_id_val = seed_image_id or os.environ.get("MGC_RIFFUSION_SEED_IMAGE_ID") or "og_beat"

        start: Dict[str, Any] = {
            "prompt": prompt,
            "seed": seed_val,
            "denoising": denoise_val,
            "guidance": guidance_val,
        }
        end: Dict[str, Any] = {
            "prompt": prompt,
            "seed": seed_val,
            "denoising": denoise_val,
            "guidance": guidance_val,
        }
        if negative_prompt:
            start["negative_prompt"] = negative_prompt
            end["negative_prompt"] = negative_prompt

        payload: Dict[str, Any] = {
            "start": start,
            "end": end,
            "alpha": alpha_val,
            "num_inference_steps": steps_val,
            "seed_image_id": seed_image_id_val,
        }
        if mask_image_id:
            payload["mask_image_id"] = mask_image_id

        obj = self._post_json(payload, timeout_s=timeout_s)
        audio_bytes = _extract_audio_bytes(obj)

        out_mp3_path = os.fspath(out_mp3_path)
        os.makedirs(os.path.dirname(out_mp3_path) or ".", exist_ok=True)
        with open(out_mp3_path, "wb") as f:
            f.write(audio_bytes)

        if os.path.getsize(out_mp3_path) <= 0:
            raise ProviderError("riffusion did not produce an mp3 artifact (missing or empty).")

        meta: Dict[str, Any] = {
            "riffusion_base_url": self.base_url,
            "riffusion_url": self._run_inference_url(),
            "seed_image_id": seed_image_id_val,
            "mask_image_id": mask_image_id,
            "num_inference_steps": steps_val,
            "guidance": guidance_val,
            "denoising": denoise_val,
            "alpha": alpha_val,
        }
        if isinstance(obj.get("duration_s"), (int, float)):
            meta["duration_s"] = float(obj["duration_s"])
        if isinstance(obj.get("image"), str) and obj["image"]:
            meta["image"] = obj["image"]

        return meta
