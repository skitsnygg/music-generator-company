from __future__ import annotations

import hashlib
import math
import struct
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional

from .base import GenerateRequest, GenerateResult, ProviderError


def _stable_hash_u64(s: str) -> int:
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)


def _deterministic_freq_hz(*, seed: int, context: str, schedule: str, period_key: str, track_id: str, min_hz: float = 220.0, max_hz: float = 880.0) -> float:
    key = f"mgc.stub.freq|seed={seed}|context={context}|schedule={schedule}|period_key={period_key}|track_id={track_id}"
    u = _stable_hash_u64(key)
    frac = (u % (10**12)) / float(10**12)
    hz = min_hz + (max_hz - min_hz) * frac
    return float(f"{hz:.3f}")


def _deterministic_phase(*, seed: int, context: str, schedule: str, period_key: str, track_id: str) -> float:
    key = f"mgc.stub.phase|seed={seed}|context={context}|schedule={schedule}|period_key={period_key}|track_id={track_id}"
    u = _stable_hash_u64(key)
    frac = (u % (10**12)) / float(10**12)
    return (2.0 * math.pi) * frac


def _render_wav_bytes(*, freq_hz: float, phase: float, seconds: float = 2.0, sample_rate: int = 44100, channels: int = 1, amplitude: float = 0.20) -> bytes:
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if channels not in (1, 2):
        raise ValueError("channels must be 1 or 2")
    if not (0.0 < amplitude <= 1.0):
        raise ValueError("amplitude must be in (0, 1]")

    n_frames = int(round(seconds * sample_rate))
    n_frames = max(1, n_frames)

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
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
    name: str = "stub"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        """Deterministic sine-wave stub.

        Supports BOTH:
        - generate(req)
        - generate(**kwargs) (legacy)
        """
        try:
            if req is None:
                # Legacy kwargs path
                req = GenerateRequest(
                    track_id=str(kwargs.get("track_id") or "track"),
                    context=str(kwargs.get("context") or "focus"),
                    seed=int(kwargs.get("seed") or 1),
                    run_id=kwargs.get("run_id"),
                    prompt=str(kwargs.get("prompt") or ""),
                    deterministic=bool(kwargs.get("deterministic") or False),
                    ts=str(kwargs.get("ts") or kwargs.get("now_iso") or ""),
                    schedule=str(kwargs.get("schedule") or ""),
                    period_key=str(kwargs.get("period_key") or ""),
                    out_rel=str(kwargs.get("out_rel") or ""),
                    out_dir=str(kwargs.get("out_dir") or ""),
                )

            schedule = req.schedule or str(kwargs.get("schedule") or "")
            period_key = req.period_key or str(kwargs.get("period_key") or "")

            seconds = float(kwargs.get("seconds", 2.0))
            sample_rate = int(kwargs.get("sample_rate", 44100))
            channels = int(kwargs.get("channels", 1))
            amplitude = float(kwargs.get("amplitude", 0.20))

            freq_hz = _deterministic_freq_hz(
                seed=int(req.seed),
                context=str(req.context),
                schedule=str(schedule),
                period_key=str(period_key),
                track_id=str(req.track_id),
            )
            phase = _deterministic_phase(
                seed=int(req.seed),
                context=str(req.context),
                schedule=str(schedule),
                period_key=str(period_key),
                track_id=str(req.track_id),
            )

            wav_bytes = _render_wav_bytes(
                freq_hz=freq_hz,
                phase=phase,
                seconds=seconds,
                sample_rate=sample_rate,
                channels=channels,
                amplitude=amplitude,
            )

            meta: Dict[str, Any] = {
                "context": str(req.context),
                "deterministic": bool(req.deterministic),
                "schedule": str(schedule),
                "period_key": str(period_key),
                "seed": int(req.seed),
                "freq_hz": float(f"{freq_hz:.3f}"),
                "seconds": float(f"{seconds:.3f}"),
                "sample_rate": int(sample_rate),
                "channels": int(channels),
                "amplitude": float(f"{amplitude:.3f}"),
                "note": "deterministic sine wave",
                "genre": "stub",
            }

            return GenerateResult(
                provider=self.name,
                artifact_bytes=wav_bytes,
                ext=".wav",
                mime="audio/wav",
                meta=meta,
            )
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(str(e)) from e


# Back-compat aliases some code paths might use
Provider = StubProvider


def build_provider(**_kwargs: Any) -> StubProvider:
    return StubProvider()


def get_provider(**_kwargs: Any) -> StubProvider:
    return StubProvider()
