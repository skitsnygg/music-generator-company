from __future__ import annotations

import base64
import os
from typing import Optional, Dict, Any

import requests

from mgc.providers.base import ProviderError


class RiffusionProvider:
    def __init__(self, server_url: str):
        if not server_url:
            raise ProviderError("riffusion provider requires server_url")
        self.server_url = server_url.rstrip("/")

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
        # Optional riffusion inference params
        alpha: Optional[float] = None,
        seed_image_id: Optional[str] = None,
        mask_image_id: Optional[str] = None,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        # Compatibility: allow either kwarg name
        if out_mp3_path is None:
            out_mp3_path = out_mp3
        if not out_mp3_path:
            raise ProviderError("riffusion provider requires out_mp3_path (or out_mp3)")

        if seed is None:
            seed = 0

        # PromptInput schema (see riffusion/datatypes.py)
        start: Dict[str, Any] = {
            "prompt": prompt,
            "seed": int(seed),
            "denoising": float(denoising if denoising is not None else 0.75),
            "guidance": float(guidance if guidance is not None else 7.0),
        }
        if negative_prompt is not None:
            start["negative_prompt"] = negative_prompt

        # For non-interpolated generation, use same for end
        end = dict(start)

        # InferenceInput schema: requires alpha
        payload: Dict[str, Any] = {
            "start": start,
            "end": end,
            "alpha": float(alpha if alpha is not None else 0.0),
            "num_inference_steps": int(num_inference_steps if num_inference_steps is not None else 50),
            "seed_image_id": seed_image_id if seed_image_id is not None else "og_beat",
            "mask_image_id": mask_image_id,
        }

        # Remove explicit None keys to keep payload clean
        if payload.get("mask_image_id") is None:
            payload.pop("mask_image_id", None)

        # Trailing slash matters (Flask redirects otherwise)
        url = f"{self.server_url}/run_inference/"

        try:
            resp = requests.post(url, json=payload, timeout=timeout_s or 300)
        except Exception as e:
            raise ProviderError(f"riffusion request failed: {e}") from e

        if resp.status_code != 200:
            raise ProviderError(f"riffusion returned {resp.status_code}: {resp.text}")

        # Server returns JSON InferenceOutput with base64 audio
        try:
            data = resp.json()
        except Exception as e:
            raise ProviderError(f"riffusion returned non-JSON response: {e}") from e

        audio_b64 = data.get("audio")
        if not audio_b64 or not isinstance(audio_b64, str):
            raise ProviderError("riffusion response missing 'audio' (base64 mp3)")

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            raise ProviderError(f"failed to decode riffusion audio base64: {e}") from e

        os.makedirs(os.path.dirname(out_mp3_path) or ".", exist_ok=True)
        with open(out_mp3_path, "wb") as f:
            f.write(audio_bytes)

        if os.path.getsize(out_mp3_path) <= 0:
            raise ProviderError("riffusion wrote empty mp3")

        return {
            "path": out_mp3_path,
            "bytes": len(audio_bytes),
            "duration_s": data.get("duration_s"),
            "image_b64": data.get("image"),
        }
