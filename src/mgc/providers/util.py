from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_wav_sine(
    path: Path,
    *,
    seconds: float = 1.0,
    freq_hz: float = 440.0,
    sample_rate: int = 22050,
) -> Tuple[float, int]:
    """
    Minimal deterministic 16-bit PCM mono WAV.
    Returns (duration_seconds, sample_rate_hz).
    """
    import struct

    n_samples = max(1, int(sample_rate * seconds))
    amp = 0.2
    samples = bytearray()

    for i in range(n_samples):
        t = i / sample_rate
        v = int(amp * 32767 * math.sin(2 * math.pi * freq_hz * t))
        samples += struct.pack("<h", v)

    data_size = len(samples)
    byte_rate = sample_rate * 1 * 16 // 8
    block_align = 1 * 16 // 8

    header = bytearray()
    header += b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)
    header += struct.pack("<H", 1)           # PCM
    header += struct.pack("<H", 1)           # mono
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", 16)          # 16-bit
    header += b"data"
    header += struct.pack("<I", data_size)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + bytes(samples))
    return (n_samples / sample_rate), sample_rate
