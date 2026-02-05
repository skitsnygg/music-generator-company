from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from mgc.providers.base import ProviderError


@dataclass(frozen=True)
class RiffusionGenerateResult:
    """
    Minimal result surface expected by mgc.providers.riffusion_adapter.RiffusionAdapter
    """

    title: str
    mood: str
    genre: str
    bpm: int
    duration_sec: float


def _env_int(name: str, default: int) -> int:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _timeout_tuple(timeout_s: Optional[int]) -> Tuple[float, float]:
    """
    Requests timeout tuple: (connect_timeout, read_timeout)

    Defaults are intentionally "fail fast" for local inference servers.
    Override with:
      - MGC_RIFFUSION_CONNECT_TIMEOUT (seconds)
      - MGC_RIFFUSION_READ_TIMEOUT (seconds)
    Adapter may pass timeout_s via RIFFUSION_TIMEOUT; we treat that as read timeout.
    """
    connect_default = float(_env_int("MGC_RIFFUSION_CONNECT_TIMEOUT", 2))
    read_default = float(_env_int("MGC_RIFFUSION_READ_TIMEOUT", 30))

    if timeout_s is None:
        return (connect_default, read_default)

    # Treat timeout_s as the read timeout; clamp connect timeout to something reasonable.
    read_t = float(max(1, int(timeout_s)))
    connect_t = min(connect_default, read_t)
    return (connect_t, read_t)


def _num_retries() -> int:
    # Keep deterministic: fixed retry count, no jitter/backoff by default.
    return max(0, _env_int("MGC_RIFFUSION_RETRIES", 1))


def _post_json_with_retry(url: str, payload: Dict[str, Any], *, timeout: Tuple[float, float]) -> requests.Response:
    retries = _num_retries()
    last_err: Optional[BaseException] = None

    # Deterministic retry timing: constant sleep (or none).
    retry_sleep_s = float(_env_float("MGC_RIFFUSION_RETRY_SLEEP", 0.25))

    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout, allow_redirects=True)
            return r
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last_err = e
            if attempt >= retries:
                break
            if retry_sleep_s > 0:
                time.sleep(retry_sleep_s)

    raise ProviderError(f"riffusion server request failed after {retries + 1} attempts: {last_err}") from last_err


def _find_mp3_start(b: bytes) -> Optional[int]:
    """
    Find the start of MP3 data within a byte blob.

    Some riffusion servers occasionally return junk/prefix bytes before the ID3 tag
    or the first MPEG frame sync. We strip deterministically to the first plausible
    MP3 header so we never write garbage headers into .mp3 files.
    """
    i = b.find(b"ID3")
    if i != -1:
        return i

    # MPEG frame sync: 0xFF followed by a byte whose top 3 bits are 1 (0xE0..0xFF)
    for j in range(max(0, len(b) - 1)):
        if b[j] == 0xFF and (b[j + 1] & 0xE0) == 0xE0:
            return j
    return None


def _clean_mp3_bytes(b: bytes) -> Tuple[bytes, int]:
    """
    Returns (clean_bytes, stripped_prefix_len).

    If no recognizable MP3 header exists anywhere, returns original bytes with 0.
    Deterministic: pure function of input bytes.
    """
    start = _find_mp3_start(b)
    if start is None or start == 0:
        return b, 0
    return b[start:], start


def _looks_like_mp3(b: bytes) -> bool:
    return b[:3] == b"ID3" or (len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0)


class RiffusionProvider:
    """
    HTTP client for a local riffusion.server instance.

    Notes:
      - CI-safe: uses explicit connect/read timeouts; no long hangs.
      - Deterministic-friendly: retries are bounded and constant timing (no jitter).
      - Does NOT attempt automatic fallback to other providers; caller decides.
    """

    def __init__(self, server_url: str) -> None:
        self.server_url = (server_url or "").strip().rstrip("/")

    def _build_payload(
        self,
        *,
        prompt: str,
        seed: Optional[int],
        num_inference_steps: Optional[int],
        guidance: Optional[float],
        denoising: Optional[float],
        bpm: int,
    ) -> Dict[str, Any]:
        # Server expects the classic riffusion JSON shape.
        # Keep it minimal and stable.
        if seed is None:
            # Nondeterministic mode: still avoid Python's random module (global state).
            seed = secrets.randbelow(2_000_000_000) + 1

        start = {
            "prompt": prompt,
            "seed": int(seed),
            "denoising": float(denoising) if denoising is not None else 0.6,
            "guidance": float(guidance) if guidance is not None else 6.0,
        }
        end = {
            "prompt": prompt,
            "seed": int(seed),
            "denoising": float(denoising) if denoising is not None else 0.6,
            "guidance": float(guidance) if guidance is not None else 6.0,
        }

        payload: Dict[str, Any] = {
            "alpha": 1.0,
            "num_inference_steps": int(num_inference_steps) if num_inference_steps is not None else 50,
            # Seed image id can be overridden externally by setting MGC_RIFFUSION_SEED_IMAGE_ID.
            "seed_image_id": (os.environ.get("MGC_RIFFUSION_SEED_IMAGE_ID") or "og_beat"),
            "start": start,
            "end": end,
        }

        # Some server variants accept bpm; harmless if ignored.
        payload["bpm"] = int(bpm)
        return payload

    def generate(
        self,
        *,
        out_mp3: Path,
        out_wav: Optional[Path] = None,
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
        timeout_s: Optional[int] = None,
    ) -> RiffusionGenerateResult:
        out_mp3 = Path(out_mp3)
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        if out_preview_jpg is not None:
            Path(out_preview_jpg).parent.mkdir(parents=True, exist_ok=True)
        if out_wav is not None:
            Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

        timeout = _timeout_tuple(timeout_s)
        payload = self._build_payload(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            denoising=denoising,
            bpm=bpm,
        )

        url = self.server_url
        if not url.endswith("/run_inference"):
            # Accept both ".../run_inference/" and ".../run_inference"
            if not url.endswith("/run_inference/"):
                url = url.rstrip("/") + "/run_inference"

        r = _post_json_with_retry(url if url.endswith("/") else url + "/", payload, timeout=timeout)

        if r.status_code != 200:
            snippet = (r.text or "").strip().replace("\n", " ")[:200]
            raise ProviderError(f"riffusion server returned {r.status_code}: {snippet}")

        try:
            obj = r.json()
        except Exception:
            raise ProviderError("riffusion server returned non-JSON response")

        if not isinstance(obj, dict):
            raise ProviderError("riffusion server returned unexpected JSON shape (expected object)")

        # Many riffusion server variants return audio in different keys.
        candidates = [
            "mp3_base64",
            "audio_mp3_base64",
            "audio_base64",
            "wav_base64",
            "mp3",
            "audio_mp3",
            "audio",
            "wav",
        ]

        b64: Optional[str] = None
        used_key: Optional[str] = None
        for k in candidates:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                b64 = v
                used_key = k
                break

        if b64 is None:
            keys = ",".join(sorted(obj.keys()))
            raise ProviderError(
                "riffusion server response missing audio data (expected one of: "
                "mp3_base64/audio_base64/audio). keys=" + keys
            )

        import base64

        try:
            b64s = b64.strip()
            if b64s.startswith("data:"):
                parts = b64s.split(",", 1)
                if len(parts) == 2:
                    b64s = parts[1]
            audio_bytes = base64.b64decode(b64s, validate=False)
        except Exception as e:  # pragma: no cover
            raise ProviderError(f"riffusion server returned non-base64 audio in field {used_key!r}: {e}") from e

        # Detect audio container quickly to avoid writing wav bytes to an .mp3 path.
        is_wav = audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"

        # Some servers may prepend junk bytes before the real MP3 header.
        cleaned_mp3, stripped = _clean_mp3_bytes(audio_bytes)
        if stripped and _looks_like_mp3(cleaned_mp3):
            audio_bytes = cleaned_mp3

        is_mp3 = _looks_like_mp3(audio_bytes)

        if is_wav and out_wav is not None:
            out_wav.write_bytes(audio_bytes)
        elif is_mp3:
            out_mp3.write_bytes(audio_bytes)
        elif is_wav and out_wav is None:
            raise ProviderError(
                "riffusion server returned WAV bytes but caller did not provide out_wav; "
                "set provider to return mp3, or update adapter to pass out_wav"
            )
        else:
            raise ProviderError(
                f"riffusion server returned unknown audio bytes (first12={audio_bytes[:12].hex()}); "
                "cannot write safely (no MP3 header found)"
            )

        # Preview is optional. If server provides a base64 image, save it.
        if out_preview_jpg is not None:
            b64img = obj.get("image_base64") or obj.get("preview_base64") or None
            if isinstance(b64img, str) and b64img.strip():
                try:
                    b64i = b64img.strip()
                    if b64i.startswith("data:"):
                        parts = b64i.split(",", 1)
                        if len(parts) == 2:
                            b64i = parts[1]
                    Path(out_preview_jpg).write_bytes(base64.b64decode(b64i, validate=False))
                except Exception:
                    # Best-effort; preview is non-critical.
                    pass

        # Duration: server may provide; otherwise unknown.
        dur = 0.0
        d = obj.get("duration_sec") or obj.get("duration") or 0.0
        try:
            dur = float(d)
        except Exception:
            dur = 0.0

        return RiffusionGenerateResult(
            title=str(title),
            mood=str(mood),
            genre=str(genre),
            bpm=int(bpm),
            duration_sec=float(dur),
        )
