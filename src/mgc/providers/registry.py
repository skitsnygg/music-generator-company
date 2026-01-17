from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .base import Provider, ProviderError


def list_providers() -> List[str]:
    # Keep stable ordering for CLI/help output
    return [
        "stub",
        "filesystem",
        "external_dummy",
        "riffusion",
        "suno",
        "diffsinger",
    ]


def get_provider(name: str | None = None) -> Provider:
    n = (name or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()

    if n == "stub":
        from .stub import StubProvider

        return StubProvider()

    if n == "filesystem":
        from .filesystem import FilesystemProvider

        src = (os.environ.get("MGC_FS_PROVIDER_DIR") or "").strip()
        if not src:
            raise ProviderError("MGC_FS_PROVIDER_DIR is required for filesystem provider")
        return FilesystemProvider(root_dir=Path(src).expanduser().resolve())

    if n in ("external_dummy", "dummy"):
        from .external_dummy import ExternalDummyProvider

        return ExternalDummyProvider()

    if n == "riffusion":
        from .riffusion_adapter import RiffusionAdapter

        return RiffusionAdapter()

    if n == "suno":
        from .suno_staged import SunoProvider

        return SunoProvider()

    if n == "diffsinger":
        from .diffsinger_staged import DiffSingerProvider

        return DiffSingerProvider()

    raise ProviderError(f"Unknown provider: {n}. Available: {', '.join(list_providers())}")
