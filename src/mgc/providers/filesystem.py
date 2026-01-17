from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import GenerateRequest, GenerateResult, ProviderError


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        raise ValueError("n must be > 0")
    h = hashlib.sha256(key.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big", signed=False)
    return v % n


_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def _list_audio_files(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
            files.append(p)
    return files


def _guess_mime(ext: str) -> str:
    e = ext.lower()
    if e == ".wav":
        return "audio/wav"
    if e == ".mp3":
        return "audio/mpeg"
    if e in (".m4a", ".aac"):
        return "audio/aac"
    if e == ".flac":
        return "audio/flac"
    if e == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"


@dataclass(frozen=True)
class FilesystemProvider:
    """Selects an existing audio file from a directory and returns its bytes.

    Env:
      MGC_FS_PROVIDER_DIR: directory containing audio files (recursively)

    Determinism:
      Uses stable hashing over (seed, context, period_key) to pick the file.
    """

    root_dir: Path
    name: str = "filesystem"

    @classmethod
    def from_env(cls) -> "FilesystemProvider":
        d = (os.environ.get("MGC_FS_PROVIDER_DIR") or "").strip()
        if not d:
            raise ProviderError("MGC_FS_PROVIDER_DIR is not set (filesystem provider requires it)")
        return cls(root_dir=Path(d).expanduser().resolve())

    def _pick_source(self, *, seed: int, context: str, period_key: str) -> Path:
        files = _list_audio_files(self.root_dir)
        if not files:
            raise ProviderError(f"No audio files found under: {self.root_dir}")

        # Prefer WAVs if present.
        wavs = [p for p in files if p.suffix.lower() == ".wav"]
        pool = wavs if wavs else files

        key = f"filesystem|seed={seed}|context={context}|period={period_key}"
        idx = _stable_index(key, len(pool))
        return pool[idx]

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        if req is None:
            req = GenerateRequest(
                track_id=str(kwargs.get("track_id") or "track"),
                context=str(kwargs.get("context") or "focus"),
                seed=int(kwargs.get("seed") or 1),
                deterministic=bool(kwargs.get("deterministic") or False),
                schedule=str(kwargs.get("schedule") or ""),
                period_key=str(kwargs.get("period_key") or ""),
                out_dir=str(kwargs.get("out_dir") or ""),
                out_rel=str(kwargs.get("out_rel") or ""),
                run_id=kwargs.get("run_id"),
                prompt=str(kwargs.get("prompt") or ""),
                ts=str(kwargs.get("ts") or kwargs.get("now_iso") or ""),
            )

        period_key = req.period_key or str(kwargs.get("period_key") or "")
        src = self._pick_source(seed=int(req.seed), context=str(req.context), period_key=str(period_key))
        b = src.read_bytes()
        ext = src.suffix if src.suffix.startswith(".") else f".{src.suffix}"
        mime = _guess_mime(ext)

        meta: Dict[str, Any] = {
            "context": str(req.context),
            "seed": int(req.seed),
            "period_key": str(period_key),
            "deterministic": bool(req.deterministic),
            "source_path": str(src),
            "bytes": len(b),
            "sha256": _sha256_bytes(b),
        }

        return GenerateResult(provider=self.name, artifact_bytes=b, ext=ext, mime=mime, meta=meta)


__all__ = ["FilesystemProvider"]
