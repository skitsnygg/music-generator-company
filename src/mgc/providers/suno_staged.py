from __future__ import annotations

from mgc.providers.base import GenerateRequest, GenerateResult


class SunoProvider:
    name = "suno"

    def generate(self, req: GenerateRequest) -> GenerateResult:
        raise RuntimeError("Suno provider is staged but not configured (no API key). Use stub or riffusion.")
