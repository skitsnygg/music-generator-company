#!/usr/bin/env python3
"""
src/mgc/providers/stub.py

Deterministic stub audio provider.

Contract (as used by src/mgc/run_cli.py):
- get_provider("stub") returns an object with:
    generate(out_dir, track_id, context, seed, deterministic, now_iso, schedule, period_key) -> dict
- Returned dict must include:
    artifact_path (str), sha256 (str or None), track_id (str), provider (str),
    meta (dict), genre (str), mood (str), title (str)

Determinism:
- In deterministic mode, WAV bytes are fully determined by:
    seed, context, schedule, period_key, track_id
  (track_id is stable in deterministic mode in run_cli.py; still safe to include)
- No system time, random, host, pid, etc. is used in waveform generation.
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ProviderError(RuntimeError):
    pass


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash_u64(s: str) -> int:
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)


def _deterministic_freq_hz(
    *,
    seed: int,
    context: str,
    schedule: str,
    period_key: str,
    track_id: str,
    min_hz: float = 220.0,
    max_hz: float = 880.0,
) -> float:
    """
    Map a stable hash -> float in [min_hz, max_hz], rounded to 3 decimals.
    """
    if max_hz <= min_hz:
        raise ValueError("max_hz must be > min_hz")

    key = f"mgc.stub.freq|seed={seed}|context={context}|schedule={schedule}|period_key={period_key}|track_id={track_id}"
    u = _stable_hash_u64(key)

    # Use a large modulus to get a stable fractional [0,1)
    frac = (u % (10**12)) / float(10**12)
    hz = min_hz + (max_hz - min_hz) * frac
    return float(f"{hz:.3f}")


def _deterministic_phase(
    *,
    seed: int,
    context: str,
    schedule: str,
    period_key: str,
    track_id: str,
) -> float:
    key = f"mgc.stub.phase|seed={seed}|context={context}|schedule={schedule}|period_key={period_key}|track_id={track_id}"
    u = _stable_hash_u64(key)
    frac = (u % (10**12)) / float(10**12)
    return (2.0 * math.pi) * frac


def _render_wav_bytes(
    *,
    freq_hz: float,
    phase: float,
    seconds: float = 2.0,
    sample_rate: int = 44100,
    channels: int = 1,
    amplitude: float = 0.20,
) -> bytes:
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    if not (0.0 < amplitude <= 1.0):
        raise ValueError("amplitude must be in (0, 1]")

    # Deterministic frame count
    n_frames = int(round(seconds * sample_rate))
    n_frames = max(1, n_frames)

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)

        max_i16 = 32767
        for i in range(n_frames):
            t = i / sample_rate
            v = math.sin((2.0 * math.pi * freq_hz * t) + phase)
            s = int(round(max_i16 * amplitude * v))
            s = max(-32768, min(32767, s))

            frame = struct.pack("<h", s)
            if channels == 2:
                frame = frame + frame
            wf.writeframesraw(frame)

        wf.writeframes(b"")

    return buf.getvalue()


@dataclass
class StubProvider:
    """
    Provider used for CI + deterministic tests.

    The `generate()` signature matches what run_cli.py calls.
    """
    name: str = "stub"

    def generate(
        self,
        *,
        out_dir: str | Path,
        track_id: str,
        context: str,
        seed: int,
        deterministic: bool,
        now_iso: str,
        schedule: str,
        period_key: str,
        seconds: float = 2.0,
        sample_rate: int = 44100,
        channels: int = 1,
        amplitude: float = 0.20,
    ) -> Dict[str, Any]:
        try:
            out_dir = Path(out_dir)
            tracks_dir = out_dir / "tracks"
            tracks_dir.mkdir(parents=True, exist_ok=True)

            # Stable filename: derived from track_id (already stable in deterministic mode)
            wav_path = tracks_dir / f"{track_id}.wav"

            freq_hz = _deterministic_freq_hz(
                seed=int(seed),
                context=str(context),
                schedule=str(schedule),
                period_key=str(period_key),
                track_id=str(track_id),
            )
            phase = _deterministic_phase(
                seed=int(seed),
                context=str(context),
                schedule=str(schedule),
                period_key=str(period_key),
                track_id=str(track_id),
            )

            wav_bytes = _render_wav_bytes(
                freq_hz=freq_hz,
                phase=phase,
                seconds=float(seconds),
                sample_rate=int(sample_rate),
                channels=int(channels),
                amplitude=float(amplitude),
            )
            wav_path.write_bytes(wav_bytes)

            sha = _sha256_bytes(wav_bytes)

            meta = {
                "context": str(context),
                "deterministic": bool(deterministic),
                "schedule": str(schedule),
                "period_key": str(period_key),
                "seed": int(seed),
                "freq_hz": float(f"{freq_hz:.3f}"),
                "seconds": float(f"{float(seconds):.3f}"),
                "sample_rate": int(sample_rate),
                "channels": int(channels),
                "amplitude": float(f"{float(amplitude):.3f}"),
                "note": "deterministic sine wave",
            }

            return {
                "provider": self.name,
                "track_id": str(track_id),
                "artifact_path": str(wav_path),
                "sha256": sha,
                "meta": meta,
                "genre": "stub",
                "mood": str(context),
                "title": "CI Seed Track" if str(schedule) == "daily" else f"{str(schedule).title()} Track",
            }
        except Exception as e:
            raise ProviderError(str(e)) from e


def build_provider(**_kwargs: Any) -> StubProvider:
    return StubProvider()


def get_provider(**_kwargs: Any) -> StubProvider:
    return StubProvider()


# Back-compat alias some codebases use
Provider = StubProvider
