from __future__ import annotations

from typing import Any

from .base import GenerateRequest, GenerateResult, ProviderError


class DiffSingerProvider:
    name = "diffsinger"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        raise ProviderError(
            "DiffSinger provider is staged but not configured yet. "
            "Use MGC_PROVIDER=stub or MGC_PROVIDER=filesystem (or riffusion if you have a server running)."
        )
