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


def _normalize_run_inference_url(url: str) -> str:
    url = url.strip()
    if not url:
        return "http://127.0.0.1:3013/run_inference/"

    # Accept either ".../run_inference" or ".../run_inference/" or base URL.
    if "/run_inference" not in url:
        url = url.rstrip("/") + "/run_inference/"
    else:
        # If it ends in /run_inference (no trailing slash), add slash (server accepts both but keep consistent)
        if url.endswith("/run_inference"):
            url = url + "/"
        elif url.endswith("/run_inference/"):
            pass
        else:
            # e.g. ".../run_inference/whatever" – keep it, user knows what they’re doing
            pass

    return url


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
    """
    Many riffusion server variants return audio in different keys.
    Prefer "audio", then fall back to common alternatives.
    """
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


@dataclass(frozen=True)
class RiffusionProviderConfig:
    url: str
    steps: int
    guidance: float
    denoising: float
    timeout_s: Optional[float] = None


class RiffusionProvider:
    """
    Minimal result surface expected by mgc.providers.riffusion_adapter.RiffusionAdapter

    This provider speaks to a riffusion.server that expects the classic InferenceInput JSON:

      {
        "start": {"prompt": "...", "seed": 1, "denoising": 0.4, "guidance": 5.0, "negative_prompt": null},
        "end":   {"prompt": "...", "seed": 1, "denoising": 0.4, "guidance": 5.0, "negative_prompt": null},
        "alpha": 0.0,
        "num_inference_steps": 10,
        "seed_image_id": "og_beat",
        "mask_image_id": null
      }

    IMPORTANT: Do not send extra keys like "bpm" — some servers will 500 on unexpected fields.
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
        if url is None:
            url = server_url or ""
        self.url = _normalize_run_inference_url(url)
        self.steps = int(steps) if steps is not None else None
        self.guidance = float(guidance) if guidance is not None else None
        self.denoising = float(denoising) if denoising is not None else None
        self.timeout_s = timeout_s

    def _post_json(self, payload: Dict[str, Any], *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        t = _timeouts(self.timeout_s if timeout_s is None else timeout_s)
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

    def generate(
        self,
        *,
        prompt: str,
        seed: Optional[int] = None,
        out_mp3_path: Optional[str] = None,
        out_mp3: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance: Optional[float] = None,
        denoising: Optional[float] = None,
        timeout_s: Optional[float] = None,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        # Back-compat with older adapter call sites.
        if out_mp3_path is None:
            out_mp3_path = out_mp3
        if not out_mp3_path:
            raise ProviderError("riffusion provider requires out_mp3_path (or out_mp3)")

        steps_val = num_inference_steps if num_inference_steps is not None else self.steps
        guidance_val = guidance if guidance is not None else self.guidance
        denoise_val = denoising if denoising is not None else self.denoising

        # Seed image id can be overridden externally by setting MGC_RIFFUSION_SEED_IMAGE_ID.
        seed_image_id = os.environ.get("MGC_RIFFUSION_SEED_IMAGE_ID") or "og_beat"

        # Classic riffusion interpolation form: start/end PromptInput + alpha.
        start: Dict[str, Any] = {"prompt": prompt}
        end: Dict[str, Any] = {"prompt": prompt}

        if seed is not None:
            start["seed"] = int(seed)
            end["seed"] = int(seed)
        if guidance_val is not None:
            start["guidance"] = float(guidance_val)
            end["guidance"] = float(guidance_val)
        if denoise_val is not None:
            start["denoising"] = float(denoise_val)
            end["denoising"] = float(denoise_val)

        payload: Dict[str, Any] = {
            "alpha": 0.0,
            "seed_image_id": seed_image_id,
            "start": start,
            "end": end,
        }
        if steps_val is not None:
            payload["num_inference_steps"] = int(steps_val)

        if negative_prompt:
            payload["start"]["negative_prompt"] = negative_prompt
            payload["end"]["negative_prompt"] = negative_prompt

        obj = self._post_json(payload, timeout_s=timeout_s)
        audio_bytes = _extract_audio_bytes(obj)

        # Write mp3 artifact
        out_mp3_path = os.fspath(out_mp3_path)
        os.makedirs(os.path.dirname(out_mp3_path) or ".", exist_ok=True)
        with open(out_mp3_path, "wb") as f:
            f.write(audio_bytes)

        if os.path.getsize(out_mp3_path) <= 0:
            raise ProviderError("riffusion did not produce an mp3 artifact (missing or empty).")

        # Optional metadata surface
        meta: Dict[str, Any] = {
            "riffusion_url": self.url,
            "seed_image_id": seed_image_id,
            "num_inference_steps": steps_val,
            "guidance": guidance_val,
            "denoising": denoise_val,
        }
        # Pass through duration if server provides it
        if isinstance(obj.get("duration_s"), (int, float)):
            meta["duration_s"] = float(obj["duration_s"])

        # Pass through preview image if present
        if isinstance(obj.get("image"), str) and obj["image"]:
            meta["image"] = obj["image"]

        return meta
