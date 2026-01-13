from __future__ import annotations

import io
import math
import struct
import wave
from typing import Any, Dict

from .base import GenerateRequest, GenerateResult


def _stub_wav_bytes(seed: int, duration_s: float = 1.5, sample_rate: int = 44100) -> bytes:
    """
    Deterministic mono PCM16 WAV, generated fully in-memory.
    """
    freq = 220.0 + float(seed % 220)  # 220..439 Hz
    nframes = int(duration_s * sample_rate)
    amp = 0.25
    two_pi_f = 2.0 * math.pi * freq

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(nframes):
            t = i / sample_rate
            sample = amp * math.sin(two_pi_f * t)
            pcm = int(max(-1.0, min(1.0, sample)) * 32767.0)
            wf.writeframes(struct.pack("<h", pcm))
    return buf.getvalue()


class StubProvider:
    """
    Back-compat stub provider (some code paths may import this module name).
    Produces real WAV audio, not JSON.
    """

    name = "stub"

    def generate(self, req: GenerateRequest) -> GenerateResult:
        try:
            seed_int = int(str(req.seed))
        except Exception:
            seed_int = 1

        wav = _stub_wav_bytes(seed_int)

        meta: Dict[str, Any] = {
            "genre": "stub",
            "note": "deterministic sine wave",
            "seed_int": seed_int,
        }

        return GenerateResult(
            provider=self.name,
            ext=".wav",
            mime="audio/wav",
            artifact_bytes=wav,
            meta=meta,
        )
