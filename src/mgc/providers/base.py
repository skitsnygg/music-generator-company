from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrackArtifact:
    """
    Contract returned by providers.

    - artifact_path must be a real file on disk (wav/mp3/etc).
    - sha256 must be computed over artifact_path bytes.
    - meta is provider-specific extras (JSON-serializable).
    """
    track_id: str
    artifact_path: str
    sha256: str
    provider: str
    title: str
    mood: str
    genre: str
    duration_seconds: Optional[float] = None
    sample_rate_hz: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


class Provider(abc.ABC):
    """
    Providers generate or fetch audio for a given context.

    Determinism:
    - If deterministic=True, provider MUST produce the same bytes for same inputs.
    """
    name: str

    @abc.abstractmethod
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
        raise NotImplementedError
