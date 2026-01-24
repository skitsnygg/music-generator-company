from __future__ import annotations

import base64
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class RiffusionResult:
    duration_sec: float
    bpm: int
    title: str
    mood: str
    genre: str
    full_path: str
    preview_path: Optional[str]


def _strip_data_uri_base64(s: str) -> Tuple[str, Optional[str]]:
    s = (s or "").strip()
    if s.startswith("data:") and "," in s:
        header, payload = s.split(",", 1)
        mime = None
        if header.startswith("data:"):
            mime = header[5:].split(";", 1)[0].strip() or None
        return payload.strip(), mime
    return s, None


def _normalize_server_url(server_url: str) -> str:
    u = (server_url or "").strip()
    if not u:
        return ""
    u = u.rstrip("/")
    if u.endswith("/run_inference"):
        return u + "/"
    return u + "/run_inference/"


def _parse_json_lenient(resp: requests.Response) -> Dict[str, Any]:
    """
    Some riffusion servers return JSON but label it as text/html.
    Parse JSON regardless of content-type; fail only if it isn't JSON.
    """
    try:
        obj = resp.json()
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj2 = json.loads(resp.text or "")
        if isinstance(obj2, dict):
            return obj2
    except Exception as e:
        ct = (resp.headers.get("content-type") or "").lower()
        body = (resp.text or "")[:1200].replace("\n", "\\n")
        raise RuntimeError(
            f"Riffusion returned non-JSON (status={resp.status_code}, content-type={ct}): {body}"
        ) from e

    # If it parsed but isn't a dict, still treat as error (we need keys like 'audio')
    ct = (resp.headers.get("content-type") or "").lower()
    body = (resp.text or "")[:1200].replace("\n", "\\n")
    raise RuntimeError(
        f"Riffusion returned JSON but not an object (status={resp.status_code}, content-type={ct}): {body}"
    )


class RiffusionProvider:
    def __init__(self, server_url: str):
        self.server_url = _normalize_server_url(server_url)

    def generate(
        self,
        *,
        out_mp3: Path,
        out_preview_jpg: Optional[Path] = None,
        title: str,
        mood: str,
        genre: str,
        bpm: int,
        prompt: str,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance: Optional[float] = None,
        denoising: Optional[float] = None,
        seed_image_id: str = "og_beat",
        timeout_s: Optional[int] = None,
        smoke: bool = False,
    ) -> RiffusionResult:
        if not self.server_url:
            raise RuntimeError(
                "Riffusion server_url is empty. Set MGC_RIFFUSION_URL, e.g. "
                "export MGC_RIFFUSION_URL='http://127.0.0.1:3013/run_inference/'"
            )

        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        if out_preview_jpg is not None:
            out_preview_jpg.parent.mkdir(parents=True, exist_ok=True)

        seed_int = int(seed if seed is not None else (uuid.uuid4().int % 2_000_000_000))
        steps = int(num_inference_steps or int(os.getenv("RIFFUSION_STEPS", "25")))
        guidance_f = float(guidance if guidance is not None else float(os.getenv("RIFFUSION_GUIDANCE", "7.0")))
        denoising_f = float(denoising if denoising is not None else float(os.getenv("RIFFUSION_DENOISE", "0.75")))
        timeout = int(timeout_s or int(os.getenv("RIFFUSION_TIMEOUT", "300")))

        payload: Dict[str, Any] = {
            "alpha": 1.0,
            "num_inference_steps": steps,
            "seed_image_id": seed_image_id,
            "start": {
                "prompt": prompt,
                "seed": seed_int,
                "denoising": denoising_f,
                "guidance": guidance_f,
            },
            "end": {
                "prompt": prompt,
                "seed": seed_int,
                "denoising": denoising_f,
                "guidance": guidance_f,
            },
        }

        print(
            f"[riffusion] POST {self.server_url} steps={steps} guidance={guidance_f} "
            f"denoise={denoising_f} seed={seed_int} seed_image_id={seed_image_id} timeout={timeout}s smoke={smoke}"
        )

        r = requests.post(
            self.server_url,
            json=payload,
            timeout=timeout,
            allow_redirects=False,
        )

        # Non-2xx: raise with useful body snippet
        if not (200 <= r.status_code < 300):
            ct = (r.headers.get("content-type") or "").lower()
            body = (r.text or "")[:1200].replace("\n", "\\n")
            raise RuntimeError(f"Riffusion HTTP {r.status_code} (content-type={ct}): {body}")

        data = _parse_json_lenient(r)

        if "audio" not in data:
            raise RuntimeError(f"Riffusion response missing 'audio': {json.dumps(data)[:1200]}")

        audio_b64, _audio_mime = _strip_data_uri_base64(str(data["audio"]))
        out_mp3.write_bytes(base64.b64decode(audio_b64))

        preview_path: Optional[str] = None
        if out_preview_jpg is not None and data.get("image"):
            img_b64, _img_mime = _strip_data_uri_base64(str(data["image"]))
            out_preview_jpg.write_bytes(base64.b64decode(img_b64))
            preview_path = str(out_preview_jpg)

        return RiffusionResult(
            duration_sec=float(data.get("duration_s", 30.0)),
            bpm=bpm,
            title=title,
            mood=mood,
            genre=genre,
            full_path=str(out_mp3),
            preview_path=preview_path,
        )
