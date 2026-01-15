from __future__ import annotations

from pathlib import Path

from .base import Provider, TrackArtifact
from .util import sha256_file, write_wav_sine


class StubProvider(Provider):
    name = "stub"

    def generate(
        self,
        *,
        out_dir: Path,
        track_id: str,
        context: str,
        seed: int,
        deterministic: bool,
        now_iso: str,
        schedule: str,
        period_key: str,
    ) -> TrackArtifact:
        tracks_dir = out_dir / "tracks"
        wav_path = tracks_dir / f"{track_id}.wav"

        # stable mapping but varies across context/period/seed
        base = (hash(f"{context}|{schedule}|{period_key}|{seed}") % 200) + 220  # 220..419
        freq_hz = float(base)

        dur, sr = write_wav_sine(wav_path, seconds=2.0, freq_hz=freq_hz, sample_rate=22050)
        h = sha256_file(wav_path)

        title = f"{context.title()} Track"
        return TrackArtifact(
            track_id=track_id,
            artifact_path=str(wav_path),
            sha256=h,
            provider=self.name,
            title=title,
            mood=context,
            genre="stub",
            duration_seconds=dur,
            sample_rate_hz=sr,
            meta={"freq_hz": freq_hz, "schedule": schedule, "period_key": period_key},
        )
