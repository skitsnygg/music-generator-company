from __future__ import annotations

from pathlib import Path
from typing import Any

import mgc.providers.riffusion_adapter as riff


class _LegacyProvider:
    def __init__(self, *, url: str, steps: int, guidance: float, denoising: float, timeout_s: float | None = None):
        self.url = url
        self.steps = steps
        self.guidance = guidance
        self.denoising = denoising
        self.timeout_s = timeout_s

    def generate(self, *, prompt: str, seed: int, out_mp3_path: str, **_kwargs: Any) -> dict:
        # Minimal provider signature that expects out_mp3_path
        Path(out_mp3_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_mp3_path).write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00\x00")
        return {"provider": "legacy"}


class _ModernProvider:
    def __init__(self, *, server_url: str):
        self.server_url = server_url

    def generate(self, *, prompt: str, seed: int, out_mp3: str, **_kwargs: Any) -> dict:
        Path(out_mp3).parent.mkdir(parents=True, exist_ok=True)
        Path(out_mp3).write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00\x00")
        return {"provider": "modern"}


def _run_adapter_with_provider(monkeypatch, provider_cls):
    monkeypatch.setattr(riff, "RiffusionProvider", provider_cls)
    monkeypatch.setattr(riff, "_probe_duration_seconds", lambda _p: 1.0)
    monkeypatch.setattr(riff, "_which", lambda _cmd: None)
    monkeypatch.setenv("MGC_RIFFUSION_TARGET_SECONDS", "0")
    monkeypatch.setenv("MGC_RIFFUSION_MAX_SEGMENTS", "1")

    adapter = riff.RiffusionAdapter()
    result = adapter.generate(
        track_id="t1",
        context="focus",
        seed=1,
        deterministic=True,
        out_dir="/tmp",
        out_rel="/tmp/track",
        prompt="test",
        ts="2024-01-01T00:00:00Z",
        schedule="daily",
        period_key="2024-01-01",
    )
    assert isinstance(result, dict)
    assert result.get("artifact_bytes")


def test_adapter_with_legacy_provider(monkeypatch):
    _run_adapter_with_provider(monkeypatch, _LegacyProvider)


def test_adapter_with_modern_provider(monkeypatch):
    _run_adapter_with_provider(monkeypatch, _ModernProvider)
