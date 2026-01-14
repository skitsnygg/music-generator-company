from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import hashlib


# ---------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateRequest:
    track_id: str
    run_id: str
    context: str
    seed: str
    prompt: str
    deterministic: bool
    ts: str
    out_rel: str


@dataclass
class GenerateResult:
    artifact_bytes: bytes
    ext: str
    provider: str
    meta: Dict[str, object]


class BaseProvider:
    name: str = "base"

    def generate(self, req: GenerateRequest) -> GenerateResult:
        raise NotImplementedError


# ---------------------------------------------------------------------
# Stub provider (CI + determinism backbone)
# ---------------------------------------------------------------------

class StubProvider(BaseProvider):
    name = "stub"

    def generate(self, req: GenerateRequest) -> GenerateResult:
        # Deterministic pseudo-audio bytes
        key = f"{req.track_id}|{req.context}|{req.seed}|{req.prompt}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()

        # Fake WAV header + deterministic payload
        wav_header = b"RIFF" + b"\x00" * 36 + b"WAVEfmt "
        payload = digest * 1000  # small but non-empty

        return GenerateResult(
            artifact_bytes=wav_header + payload,
            ext=".wav",
            provider=self.name,
            meta={
                "context": req.context,
                "seed": req.seed,
                "prompt_hash": hashlib.sha256(req.prompt.encode()).hexdigest(),
                "deterministic": req.deterministic,
            },
        )


# ---------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------

_PROVIDERS = {
    "stub": StubProvider(),
}


def get_provider(name: str) -> BaseProvider:
    name = (name or "stub").lower()
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")
    return _PROVIDERS[name]
