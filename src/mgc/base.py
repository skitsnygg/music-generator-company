from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class GenerateRequest:
    track_id: str
    run_id: str
    context: str
    seed: str
    prompt: str
    deterministic: bool
    ts: str
    out_rel: str  # where the artifact will live (relative path like data/tracks/...)


@dataclass(frozen=True)
class GenerateResult:
    provider: str
    artifact_bytes: bytes
    mime: str
    ext: str
    meta: Dict[str, Any]


class MusicProvider(Protocol):
    name: str
    def generate(self, req: GenerateRequest) -> GenerateResult: ...
