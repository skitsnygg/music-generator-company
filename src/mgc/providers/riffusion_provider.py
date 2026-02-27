from __future__ import annotations

import os
import json
import requests
from typing import Optional, Dict, Any

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
        **_ignored: Any,
    ) -> Dict[str, Any]:

        if out_mp3_path is None:
            out_mp3_path = out_mp3

        if not out_mp3_path:
            raise ProviderError(
                "riffusion provider requires out_mp3_path (or out_mp3)"
            )

        payload: Dict[str, Any] = {
            "prompt": prompt,
        }

        if seed is not None:
            payload["seed"] = seed
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        if num_inference_steps is not None:
            payload["num_inference_steps"] = num_inference_steps
        if guidance is not None:
            payload["guidance"] = guidance
        if denoising is not None:
            payload["denoising"] = denoising

        url = f"{self.server_url}/run_inference/"

        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=timeout_s or 300,
            )
        except Exception as e:
            raise ProviderError(f"riffusion request failed: {e}") from e

        if resp.status_code != 200:
            raise ProviderError(
                f"riffusion returned {resp.status_code}: {resp.text}"
            )

        content_type = resp.headers.get("content-type", "")

        os.makedirs(os.path.dirname(out_mp3_path) or ".", exist_ok=True)

        if "application/json" in content_type:
            data = resp.json()
            if "audio_bytes" not in data:
                raise ProviderError("riffusion response missing audio_bytes")
            audio_bytes = bytes(data["audio_bytes"])
        else:
            audio_bytes = resp.content

        with open(out_mp3_path, "wb") as f:
            f.write(audio_bytes)

        if os.path.getsize(out_mp3_path) <= 0:
            raise ProviderError("riffusion wrote empty mp3")

        return {
            "path": out_mp3_path,
            "bytes": len(audio_bytes),
        }
