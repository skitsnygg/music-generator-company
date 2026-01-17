from __future__ import annotations

from typing import Any

from .base import GenerateRequest, GenerateResult, ProviderError


class SunoProvider:
    name = "suno"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        raise ProviderError(
            "Suno provider is staged but not configured. "
            "Wire an API client + key later. "
            "Use MGC_PROVIDER=stub or MGC_PROVIDER=filesystem (or riffusion if available)."
        )
