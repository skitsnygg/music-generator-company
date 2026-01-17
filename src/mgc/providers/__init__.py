"""src/mgc/providers/__init__.py

Public provider surface.

run_cli.py expects:
  from mgc.providers import ProviderError, get_provider

All registry logic lives in mgc.providers.registry.
"""

from __future__ import annotations

from .base import GenerateRequest, GenerateResult, Provider, ProviderError, TrackArtifact, sha256_bytes
from .registry import get_provider, list_providers

__all__ = [
    "ProviderError",
    "Provider",
    "GenerateRequest",
    "GenerateResult",
    "TrackArtifact",
    "sha256_bytes",
    "get_provider",
    "list_providers",
]
