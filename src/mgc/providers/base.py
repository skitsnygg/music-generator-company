from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


class ProviderError(RuntimeError):
    """Raised when a provider cannot be constructed or fails generation."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class GenerateRequest:
    """Normalized request passed to providers."""

    # Required-ish fields used across the codebase
    track_id: str
    context: str
    seed: int

    # Optional metadata (present in some call paths)
    run_id: Optional[str] = None
    prompt: str = ""
    deterministic: bool = False
    ts: str = ""  # ISO timestamp string
    schedule: str = ""  # e.g. daily/weekly
    period_key: str = ""  # deterministic period label

    # Output location hint (some providers want to know intended path)
    out_rel: str = ""  # absolute or repo-relative path string
    out_dir: str = ""  # work/output dir (not repo storage)


@dataclass(frozen=True)
class GenerateResult:
    """Normalized provider output.

    Providers should return bytes. The pipeline will decide where to write them.
    """

    provider: str
    artifact_bytes: bytes

    # File naming hints
    ext: str = ""  # include leading dot, e.g. .wav
    mime: str = ""  # e.g. audio/wav

    # Provider-specific metadata (must be JSON-serializable)
    meta: Optional[Dict[str, Any]] = None

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.artifact_bytes)


# ---------------------------------------------------------------------------
# Legacy contract (kept for back-compat with older code paths)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackArtifact:
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
    """Preferred provider interface."""

    name: str

    @abc.abstractmethod
    def generate(self, req: GenerateRequest) -> GenerateResult:
        raise NotImplementedError
