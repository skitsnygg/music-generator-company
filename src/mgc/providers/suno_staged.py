from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .base import GenerateRequest, GenerateResult, ProviderError


@dataclass(frozen=True)
class SunoConfig:
    api_key: str
    api_base: str
    model: str
    output_format: str


def _load_config() -> SunoConfig:
    api_key = (os.environ.get("MGC_SUNO_API_KEY") or "").strip()
    api_base = (os.environ.get("MGC_SUNO_API_BASE") or "").strip()
    model = (os.environ.get("MGC_SUNO_MODEL") or "auto").strip()
    output_format = (os.environ.get("MGC_SUNO_OUTPUT_FORMAT") or "mp3").strip().lstrip(".")

    missing = []
    if not api_key:
        missing.append("MGC_SUNO_API_KEY")

    if missing:
        raise ProviderError(
            "Suno provider not configured. Missing: "
            + ", ".join(missing)
            + ". Set env vars and retry, or use MGC_PROVIDER=stub."
        )

    return SunoConfig(
        api_key=api_key,
        api_base=api_base,
        model=model,
        output_format=output_format,
    )


class SunoProvider:
    name = "suno"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        _ = _load_config()
        # Scaffold only: the real API integration will be added later.
        prompt = ""
        if req is not None:
            prompt = str(req.prompt or "")
        else:
            prompt = str(kwargs.get("prompt") or "")

        raise ProviderError(
            "Suno provider scaffolded but not implemented yet. "
            "Provide an API client + auth wiring when ready. "
            f"prompt_len={len(prompt)}."
        )
