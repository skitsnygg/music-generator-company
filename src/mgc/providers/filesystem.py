# src/mgc/providers/filesystem.py
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_index(key: str, n: int) -> int:
    """
    Deterministically map a string key to an index in [0, n).
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    h = hashlib.sha256(key.encode("utf-8")).digest()
    # take first 8 bytes as unsigned int
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


@dataclass(frozen=True)
class FilesystemProvider:
    """
    Filesystem-backed provider compatible with mgc.run_cli provider.generate(...) calling style.

    Env:
      MGC_FS_PROVIDER_DIR: directory containing audio files (recursively)

    Contract:
      - deterministically selects a source file from the pool using (seed, context, period_key)
      - copies it into out_dir/tracks/<track_id>.wav (or to an explicit dst_path if provided)
      - returns an "art" dict describing the artifact and hashes
    """

    root_dir: Path

    @classmethod
    def from_env(cls) -> "FilesystemProvider":
        d = (os.environ.get("MGC_FS_PROVIDER_DIR") or "").strip()
        if not d:
            raise ValueError("MGC_FS_PROVIDER_DIR is not set (filesystem provider requires it)")
        return cls(root_dir=Path(d).expanduser().resolve())

    def _pick_source(self, *, seed: int, context: str, period_key: str) -> Path:
        files = _list_audio_files(self.root_dir)
        if not files:
            raise FileNotFoundError(f"No audio files found under: {self.root_dir}")

        # Prefer WAVs if present (keeps downstream assumptions happier).
        wavs = [p for p in files if p.suffix.lower() == ".wav"]
        pool = wavs if wavs else files

        key = f"filesystem|seed={seed}|context={context}|period={period_key}"
        idx = _stable_index(key, len(pool))
        return pool[idx]

    def generate(
        self,
        *,
        out_dir: Path,
        seed: int,
        context: str,
        deterministic: bool = False,
        period_key: str,
        track_id: Any = None,
        dst_path: Optional[Path] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Provider interface expected by run_cli.

        Accepts extra kwargs (ignored) so run_cli can evolve without breaking providers.
        """
        out_dir = Path(out_dir)
        src = self._pick_source(seed=seed, context=context, period_key=period_key)

        # run_cli typically wants a .wav path; if it passes a dst_path, honor it.
        if dst_path is None:
            tid = str(track_id) if track_id is not None else "track"
            dst_path = out_dir / "tracks" / f"{tid}.wav"
        else:
            dst_path = Path(dst_path)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_path)

        sha = _sha256_file(dst_path)
        size = dst_path.stat().st_size

        # NOTE: source_path is absolute; if you later want strict portability,
        # switch this to a relative path or omit it.
        return {
            "provider": "filesystem",
            "deterministic": bool(deterministic),
            "source_path": str(src),
            "artifact_path": str(dst_path),
            "sha256": sha,
            "bytes": size,
        }


__all__ = ["FilesystemProvider"]
