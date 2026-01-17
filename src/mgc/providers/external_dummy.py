from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import GenerateRequest, GenerateResult, ProviderError, sha256_bytes


class ExternalDummyProvider:
    """Placeholder external provider.

    Uses a deterministic fixture WAV to keep the pipeline real in CI/dev.
    """

    name = "external_dummy"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        if req is None:
            req = GenerateRequest(
                track_id=str(kwargs.get("track_id") or "track"),
                context=str(kwargs.get("context") or "focus"),
                seed=int(kwargs.get("seed") or 1),
                deterministic=bool(kwargs.get("deterministic") or False),
                schedule=str(kwargs.get("schedule") or ""),
                period_key=str(kwargs.get("period_key") or ""),
                out_dir=str(kwargs.get("out_dir") or ""),
                out_rel=str(kwargs.get("out_rel") or ""),
                run_id=kwargs.get("run_id"),
                prompt=str(kwargs.get("prompt") or ""),
                ts=str(kwargs.get("ts") or kwargs.get("now_iso") or ""),
            )

        fixture_dir = Path(__file__).parent / "fixtures"
        fixture = fixture_dir / "external_dummy.wav"
        if not fixture.exists():
            raise ProviderError("missing fixtures/external_dummy.wav")

        b = fixture.read_bytes()

        meta: Dict[str, Any] = {
            "context": req.context,
            "schedule": req.schedule,
            "period_key": req.period_key,
            "external": True,
            "bytes": len(b),
            "sha256": sha256_bytes(b),
            "genre": "stub",
        }

        return GenerateResult(provider=self.name, artifact_bytes=b, ext=".wav", mime="audio/wav", meta=meta)
