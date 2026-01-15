#!/usr/bin/env python3
"""
src/mgc/agents/music_agent.py

Music Agent: generates tracks via the provider registry.

Aligned with the project's current provider contract:
- mgc.providers.get_provider(name) -> provider
- provider.generate(...) -> dict with:
    artifact_path, sha256, track_id, provider, meta, genre, mood, title

Key rules:
- No direct imports of provider implementations here.
- Deterministic mode must not depend on wall clock randomness.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from mgc.providers import get_provider


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _fixed_now_iso(deterministic: bool) -> str:
    if deterministic:
        fixed = (os.environ.get("MGC_FIXED_TIME") or "2020-01-01T00:00:00Z").strip()
        return fixed.replace("Z", "+00:00")
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class TrackArtifact:
    track_id: str
    provider: str
    artifact_path: str
    sha256: str
    title: str
    mood: str
    genre: str
    meta: Dict[str, Any]
    preview_path: str = ""  # optional; can be set later by pipeline/web tools


@dataclass
class MusicAgentConfig:
    provider: str = "stub"
    strict_provider: bool = False


class MusicAgent:
    """
    Generates a TrackArtifact via provider registry.

    Expected workflow:
      agent = MusicAgent(provider="stub")
      track = agent.generate(
          track_id=...,
          context="focus",
          seed=1,
          deterministic=True,
          schedule="daily",
          period_key="2020-01-01",
          out_dir="artifacts/run"
      )
    """

    def __init__(self, provider: Optional[str] = None, *, strict_provider: Optional[bool] = None):
        p = (provider or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()
        strict = _env_bool("MGC_STRICT_PROVIDER", False) if strict_provider is None else bool(strict_provider)
        self.cfg = MusicAgentConfig(provider=p, strict_provider=strict)

    def _resolve_provider_name(self) -> str:
        return (self.cfg.provider or "stub").strip().lower()

    def _resolve_provider(self):
        name = self._resolve_provider_name()
        try:
            return get_provider(name)
        except Exception as e:
            if self.cfg.strict_provider:
                raise
            if name != "stub":
                self.cfg.provider = "stub"
                return get_provider("stub")
            raise e

    def generate(
        self,
        *,
        track_id: str,
        context: str,
        seed: int,
        deterministic: bool,
        schedule: str,
        period_key: str,
        out_dir: str | Path,
        now_iso: Optional[str] = None,
    ) -> TrackArtifact:
        provider = self._resolve_provider()
        ts = now_iso or _fixed_now_iso(deterministic)

        art = provider.generate(
            out_dir=out_dir,
            track_id=str(track_id),
            context=str(context),
            seed=int(seed),
            deterministic=bool(deterministic),
            now_iso=str(ts),
            schedule=str(schedule),
            period_key=str(period_key),
        )

        if not isinstance(art, dict):
            raise TypeError(f"Provider returned non-dict artifact: {type(art)}")

        artifact_path = str(art.get("artifact_path") or "")
        if not artifact_path:
            raise ValueError("Provider artifact missing artifact_path")

        sha = str(art.get("sha256") or "")
        if not sha:
            sha = _sha256_file(Path(artifact_path))

        meta = dict(art.get("meta") or {})
        meta.setdefault("context", context)
        meta.setdefault("seed", int(seed))
        meta.setdefault("schedule", schedule)
        meta.setdefault("period_key", period_key)
        meta.setdefault("deterministic", bool(deterministic))

        return TrackArtifact(
            track_id=str(art.get("track_id") or track_id),
            provider=str(art.get("provider") or getattr(provider, "name", "unknown")),
            artifact_path=artifact_path,
            sha256=sha,
            title=str(art.get("title") or f"{context.title()} Track"),
            mood=str(art.get("mood") or context),
            genre=str(art.get("genre") or "unknown"),
            meta=meta,
            preview_path=str(meta.get("preview_path") or ""),
        )
