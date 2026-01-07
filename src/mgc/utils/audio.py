from __future__ import annotations
import subprocess
from pathlib import Path

def ensure_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
    except Exception as e:
        raise RuntimeError("ffmpeg not found. Install it (macOS: brew install ffmpeg).") from e

def make_preview_clip(full_path: Path, preview_path: Path, start_sec: float = 0.0, duration_sec: float = 20.0) -> None:
    """
    Creates an MP3 preview clip from the full track using ffmpeg.
    """
    ensure_ffmpeg()
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    # -y overwrite, -ss seek, -t duration, -vn no video, -ac 2 stereo
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(full_path),
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        "-b:a", "192k",
        str(preview_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
