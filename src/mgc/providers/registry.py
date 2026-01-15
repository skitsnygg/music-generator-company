from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from .base import Provider, ProviderError
from .stub import StubProvider
from .filesystem import FilesystemProvider


def list_providers() -> List[str]:
    return ["stub", "filesystem"]


def get_provider(name: str | None) -> Provider:
    n = (name or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()

    if n == "stub":
        return StubProvider()

    if n == "filesystem":
        src = (os.environ.get("MGC_FS_PROVIDER_DIR") or "").strip()
        if not src:
            raise ProviderError("MGC_FS_PROVIDER_DIR is required for filesystem provider")
        return FilesystemProvider(Path(src).expanduser().resolve())

    raise ProviderError(f"Unknown provider: {n}. Available: {', '.join(list_providers())}")
