from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import GenerateRequest, GenerateResult, ProviderError


@dataclass(frozen=True)
class DiffSingerConfig:
    endpoint: str
    model_dir: str
    api_key: str
    voice: str


def _load_config() -> DiffSingerConfig:
    endpoint = (os.environ.get("MGC_DIFFSINGER_ENDPOINT") or "").strip()
    model_dir = (os.environ.get("MGC_DIFFSINGER_MODEL_DIR") or "").strip()
    api_key = (os.environ.get("MGC_DIFFSINGER_API_KEY") or "").strip()
    voice = (os.environ.get("MGC_DIFFSINGER_VOICE") or "").strip()

    if not endpoint and not model_dir:
        raise ProviderError(
            "DiffSinger provider not configured. "
            "Set MGC_DIFFSINGER_ENDPOINT (remote) or MGC_DIFFSINGER_MODEL_DIR (local)."
        )

    if model_dir:
        p = Path(model_dir).expanduser()
        if not p.exists():
            raise ProviderError(f"DiffSinger model dir not found: {p}")

    return DiffSingerConfig(
        endpoint=endpoint,
        model_dir=model_dir,
        api_key=api_key,
        voice=voice,
    )


class DiffSingerProvider:
    name = "diffsinger"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        cfg = _load_config()
        prompt = ""
        if req is not None:
            prompt = str(req.prompt or "")
        else:
            prompt = str(kwargs.get("prompt") or "")

        raise ProviderError(
            "DiffSinger provider scaffolded but not implemented yet. "
            f"mode={'remote' if cfg.endpoint else 'local'} "
            f"voice={cfg.voice or '<unset>'} prompt_len={len(prompt)}. "
            "Wire the inference call when ready."
        )
