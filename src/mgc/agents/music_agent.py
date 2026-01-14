#!/usr/bin/env python3
"""
src/mgc/agents/music_agent.py

Music Agent: generate tracks via the provider registry.

Key rule:
- This module must NOT directly import or instantiate provider implementations
  (riffusion/stub/suno/etc). It must ONLY use:
      from mgc.providers import GenerateRequest, get_provider

Why:
- Single source of truth for provider selection and configuration
- Avoid import-time NameError/Module drift
- CI friendliness (registry can expose only stub; others can be staged)
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from mgc.context import build_prompt
from mgc.providers import GenerateRequest, GenerateResult, MusicProvider, get_provider


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _stable_sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Public agent
# ---------------------------------------------------------------------------

@dataclass
class MusicAgentConfig:
    provider: str = "stub"
    strict_provider: bool = False  # if True, unknown provider is a hard error


class MusicAgent:
    """
    Generates a track via the provider registry.

    Expected workflow:
      agent = MusicAgent(provider="stub")
      res = agent.generate(
          track_id=...,
          run_id=...,
          context="focus",
          seed="1",
          deterministic=True,
          ts="2020-01-01T00:00:00+00:00",
          out_rel="data/tracks/<id>.wav",
      )
    """

    def __init__(self, provider: Optional[str] = None, *, strict_provider: Optional[bool] = None):
        p = (provider or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()
        strict = _env_bool("MGC_STRICT_PROVIDER", False) if strict_provider is None else bool(strict_provider)
        self.cfg = MusicAgentConfig(provider=p, strict_provider=strict)

    def _resolve_provider(self) -> MusicProvider:
        """
        Resolve provider from registry. If provider isn't present:
          - strict_provider=True => raise
          - else fallback to stub
        """
        try:
            return get_provider(self.cfg.provider)
        except Exception as e:
            if self.cfg.strict_provider:
                raise
            # fallback to stub (CI-safe)
            if self.cfg.provider != "stub":
                # avoid noisy prints in library code; carry info in exception-friendly way
                # callers can log this if desired
                self.cfg.provider = "stub"
            try:
                return get_provider("stub")
            except Exception:
                # If even stub isn't available, propagate the original error
                raise e

    def generate(
        self,
        *,
        track_id: str,
        run_id: str,
        context: str,
        seed: str,
        deterministic: bool,
        ts: str,
        out_rel: str,
        prompt: Optional[str] = None,
    ) -> GenerateResult:
        """
        Generate one track via provider registry.

        prompt:
          - if None, uses mgc.context.build_prompt(context)
        """
        provider = self._resolve_provider()
        prompt_text = prompt if isinstance(prompt, str) and prompt.strip() else build_prompt(context)

        req = GenerateRequest(
            track_id=track_id,
            run_id=run_id,
            context=context,
            seed=str(seed),
            prompt=prompt_text,
            deterministic=bool(deterministic),
            ts=ts,
            out_rel=out_rel,
        )
        res = provider.generate(req)

        # Ensure required fields are present/sane
        if not isinstance(res, GenerateResult):
            raise TypeError(f"Provider {getattr(provider, 'name', '?')} returned non-GenerateResult: {type(res)}")
        if not isinstance(res.artifact_bytes, (bytes, bytearray)):
            raise TypeError("GenerateResult.artifact_bytes must be bytes")
        if not isinstance(res.ext, str) or not res.ext.strip():
            raise ValueError("GenerateResult.ext must be a non-empty string")
        if not res.ext.startswith("."):
            res.ext = "." + res.ext

        # Add some helpful meta without changing provider output semantics
        res.meta = dict(res.meta or {})
        res.meta.setdefault("run_id", run_id)
        res.meta.setdefault("track_id", track_id)
        res.meta.setdefault("context", context)
        res.meta.setdefault("seed", str(seed))
        res.meta.setdefault("prompt_hash", _stable_sha256_hex(prompt_text))
        res.meta.setdefault("deterministic", bool(deterministic))

        return res

    # Backwards-compatible helper: if older code called agent.run(...) or similar,
    # this provides a stable minimal return shape without importing specific providers.
    def run(
        self,
        *,
        track_id: str,
        run_id: str,
        context: str,
        seed: str,
        deterministic: bool,
        ts: str,
        out_rel: str,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper returning a dict (useful for pipelines/tests).
        """
        res = self.generate(
            track_id=track_id,
            run_id=run_id,
            context=context,
            seed=seed,
            deterministic=deterministic,
            ts=ts,
            out_rel=out_rel,
            prompt=prompt,
        )
        return {
            "provider": res.provider,
            "ext": res.ext,
            "artifact_bytes_len": len(res.artifact_bytes),
            "meta": dict(res.meta or {}),
        }