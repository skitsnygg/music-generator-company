from __future__ import annotations
import math
import wave
import struct
from dataclasses import dataclass
from pathlib import Path

@dataclass
class StubGenResult:
    duration_sec: float
    bpm: int
    title: str
    mood: str
    genre: str

class GeneratorStub:
    """
    Deterministic "music" generator for MVP: creates a simple layered tone WAV so the pipeline is real.
    Swap this provider later with Suno/Diff-Singer/Riffusion without changing agents.
    """
    def generate(self, *, out_wav: Path, title: str, mood: str, genre: str, bpm: int, duration_sec: float = 60.0) -> StubGenResult:
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = 44100
        n_channels = 2
        sampwidth = 2  # 16-bit
        n_frames = int(sample_rate * duration_sec)

        # Simple "chord" frequencies (A minor-ish vibe)
        freqs = [220.0, 261.63, 329.63]  # A3, C4, E4
        beat_hz = bpm / 60.0

        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)

            for i in range(n_frames):
                t = i / sample_rate

                # Basic amplitude envelope + beat pulsing
                pulse = 0.6 + 0.4 * (0.5 * (1.0 + math.sin(2 * math.pi * beat_hz * t)))
                env = min(1.0, t / 0.5) * min(1.0, (duration_sec - t) / 0.5)
                amp = 0.22 * pulse * env

                # Layered sines
                s = 0.0
                for f in freqs:
                    s += math.sin(2 * math.pi * f * t)

                # Add a tiny bit of "movement"
                s += 0.35 * math.sin(2 * math.pi * 110.0 * t)  # A2
                s /= 4.0

                # Stereo slight offset
                left = amp * s
                right = amp * (s * 0.98 + 0.02 * math.sin(2 * math.pi * 2.0 * t))

                # 16-bit PCM
                def clamp(x: float) -> int:
                    return max(-32768, min(32767, int(x * 32767)))

                frame = struct.pack("<hh", clamp(left), clamp(right))
                wf.writeframesraw(frame)

        return StubGenResult(
            duration_sec=duration_sec,
            bpm=bpm,
            title=title,
            mood=mood,
            genre=genre,
        )
