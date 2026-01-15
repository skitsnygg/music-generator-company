"""
src/mgc/providers/__init__.py

Provider registry for MGC.

run_cli.py expects:
  from mgc.providers import ProviderError, get_provider

Providers are selected by name (usually via MGC_PROVIDER env var).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Raised when a provider cannot be constructed or fails generation."""
    pass


@runtime_checkable
class Provider(Protocol):
    name: str

    def generate(
        self,
        *,
        out_dir: str,
        track_id: str,
        context: str,
        seed: int,
        deterministic: bool,
        now_iso: str,
        schedule: str,
        period_key: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ...


@dataclass
class _ProviderEntry:
    name: str
    factory: Any  # callable returning a Provider


def _stub_factory() -> Provider:
    from mgc.providers.stub import StubProvider  # local import to avoid import cycles
    return StubProvider()


def _filesystem_factory() -> Provider:
    """
    Optional provider. Only works if your repo includes mgc.providers.filesystem.
    """
    try:
        from mgc.providers.filesystem import FilesystemProvider  # type: ignore
    except Exception as e:
        raise ProviderError(
            "filesystem provider not available (missing mgc.providers.filesystem or import failed)"
        ) from e
    return FilesystemProvider()  # type: ignore


_REGISTRY: Dict[str, _ProviderEntry] = {
    "stub": _ProviderEntry(name="stub", factory=_stub_factory),
    "filesystem": _ProviderEntry(name="filesystem", factory=_filesystem_factory),
}


def get_provider(name: str) -> Provider:
    key = (name or "").strip().lower()
    if not key:
        key = "stub"
    entry = _REGISTRY.get(key)
    if entry is None:
        raise ProviderError(f"Unknown provider: {key}")
    try:
        p = entry.factory()
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(f"Failed to construct provider {key}: {e}") from e
    return p


def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = [
    "ProviderError",
    "Provider",
    "get_provider",
    "list_providers",
]
