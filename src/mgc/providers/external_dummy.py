#!/usr/bin/env python3
"""
src/mgc/providers/external_dummy.py

Placeholder for real external generators (Suno, Diff-Singer, Riffusion).

For now:
- Deterministically copies a fixture WAV
- Keeps CI + determinism intact
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict


class ExternalDummyProvider:
    name = "external_dummy"

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
    ) -> Dict[str, Any]:
        out_dir = Path(out_dir)
        tracks_dir = out_dir / "tracks"
        tracks_dir.mkdir(parents=True, exist_ok=True)

        wav_path = tracks_dir / f"{track_id}.wav"

        # Deterministic fixture selection
        fixture_dir = Path(__file__).parent / "fixtures"
        fixture = fixture_dir / "external_dummy.wav"
        if not fixture.exists():
            raise RuntimeError("missing external_dummy.wav fixture")

        data = fixture.read_bytes()
        wav_path.write_bytes(data)

        sha = hashlib.sha256(data).hexdigest()

        return {
            "artifact_path": str(wav_path),
            "track_id": track_id,
            "provider": self.name,
            "sha256": sha,
            "meta": {
                "context": context,
                "schedule": schedule,
                "period_key": period_key,
                "external": True,
            },
            "genre": "stub",
            "mood": context,
            "title": f"{context.title()} Track (External)",
        }
