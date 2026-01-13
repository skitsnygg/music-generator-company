from __future__ import annotations

from typing import Dict

from .base import MusicProvider
from .stub_provider import StubProvider


def build_provider_registry() -> Dict[str, MusicProvider]:
    # Minimal registry: just stub for now.
    p = StubProvider()
    return {p.name: p}


def get_provider(name: str) -> MusicProvider:
    reg = build_provider_registry()
    n = (name or "").strip().lower() or "stub"
    if n not in reg:
        raise KeyError(f"unknown provider: {n} (known: {', '.join(sorted(reg.keys()))})")
    return reg[n]
