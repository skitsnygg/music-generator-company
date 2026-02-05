#!/usr/bin/env python3
"""
src/mgc/agents/music_agent.py

Music Agent: generates tracks via the provider registry.

Provider contract:
- provider.generate(...) -> dict (preferred) or dataclass/GenerateResult-like object with keys/fields like:
    artifact_path, artifact_bytes, sha256, track_id, provider, meta, genre, mood, title, ext, mime
- optional preview:
    preview_path OR preview_bytes (+ preview_ext/preview_mime)

Rules:
- If provider returns bytes (artifact_bytes), materialize to disk under out_dir/<track_id>.<ext>.
- ext selection order:
    1) art["ext"]
    2) mime (art["mime"] or art["meta"]["mime"])
    3) default ".wav"
- If provider returns preview_bytes, materialize to out_dir/<track_id>.<preview_ext> (default .jpg)
  and set meta["preview_path"].

MP3 normalization:
- Some providers may write an MP3 file with leading junk bytes before the first real MP3 header.
- If artifact is .mp3, scan the first few KB for "ID3" or MPEG frame sync and trim leading bytes in-place.
"""

from __future__ import annotations

import hashlib
import os
import socket
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from mgc.providers import get_provider
from mgc.providers.base import ProviderError


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _fallback_to_stub_enabled() -> bool:
    return _env_bool("MGC_FALLBACK_TO_STUB", False) or _env_bool("MGC_DEMO_FALLBACK_TO_STUB", False)


def _riffusion_url_from_env() -> str:
    url = (
        os.environ.get("MGC_RIFFUSION_URL")
        or os.environ.get("RIFFUSION_URL")
        or "http://127.0.0.1:3013/run_inference/"
    )
    url = (url or "").strip()
    return url or "http://127.0.0.1:3013/run_inference/"


def _riffusion_reachable(url: str, timeout_s: float = 1.5) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def _preflight_riffusion() -> str:
    url = _riffusion_url_from_env()
    try:
        timeout_s = float(os.environ.get("MGC_RIFFUSION_PREFLIGHT_TIMEOUT") or "1.5")
    except Exception:
        timeout_s = 1.5
    if _riffusion_reachable(url, timeout_s=timeout_s):
        return ""
    if _fallback_to_stub_enabled():
        _eprint(f"[provider] riffusion unreachable at {url}; preflight fallback to stub")
        return "riffusion"
    raise ProviderError(f"riffusion not reachable at {url} (set MGC_FALLBACK_TO_STUB=1 to continue)")


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


def _infer_ext_from_mime(mime: str) -> str:
    m = (mime or "").strip().lower()
    if m in ("audio/mpeg", "audio/mp3"):
        return ".mp3"
    if m in ("audio/wav", "audio/x-wav", "audio/wave", "audio/x-pn-wav"):
        return ".wav"
    if m == "audio/flac":
        return ".flac"
    if m == "audio/aac":
        return ".aac"
    if m in ("audio/mp4", "audio/x-m4a", "audio/m4a"):
        return ".m4a"
    if m in ("audio/ogg", "application/ogg"):
        return ".ogg"
    if m in ("audio/aiff", "audio/x-aiff"):
        return ".aiff"
    return ""


def _pick_ext_for_materialize(art: Dict[str, Any]) -> str:
    # 1) explicit ext
    ext = str(art.get("ext") or "").strip().lower()
    if ext and not ext.startswith("."):
        ext = "." + ext
    if ext:
        return ext

    # 2) infer from mime / meta mime
    mime = str(art.get("mime") or "").strip().lower()
    meta_mime = str((art.get("meta") or {}).get("mime") or "").strip().lower()
    inferred = _infer_ext_from_mime(mime or meta_mime)
    if inferred:
        return inferred

    # 3) default
    return ".wav"


def _pick_preview_ext(art: Dict[str, Any]) -> str:
    ext = str(art.get("preview_ext") or "").strip().lower()
    if ext and not ext.startswith("."):
        ext = "." + ext
    if ext:
        return ext

    mime = str(art.get("preview_mime") or "").strip().lower()
    if mime in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if mime == "image/png":
        return ".png"
    return ".jpg"


def _normalize_mp3_in_place(path: Path, *, scan_bytes: int = 4096) -> None:
    """
    Trim any leading junk bytes before the first real MP3 header.

    Heuristics:
    - "ID3" tag header
    - MPEG frame sync bytes: 0xFF 0xFB / 0xF3 / 0xF2
    """
    if path.suffix.lower() != ".mp3":
        return
    if not path.exists() or not path.is_file():
        raise ValueError(f"mp3 normalize: missing file: {path}")

    data = path.read_bytes()
    if not data:
        raise ValueError(f"mp3 normalize: empty file: {path}")

    limit = min(len(data), int(scan_bytes))
    header_off: Optional[int] = None

    for i in range(limit):
        if data[i : i + 3] == b"ID3":
            header_off = i
            break
        if data[i : i + 2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            header_off = i
            break

    if header_off is None:
        raise ValueError(f"mp3 normalize: no MP3 header found in first {limit} bytes: {path}")

    if header_off > 0:
        path.write_bytes(data[header_off:])


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
    preview_path: str = ""


@dataclass
class MusicAgentConfig:
    provider: str = "riffusion"
    strict_provider: bool = False


class MusicAgent:
    def __init__(self, provider: Optional[str] = None, *, strict_provider: Optional[bool] = None):
        p = (provider or os.environ.get("MGC_PROVIDER") or "riffusion").strip().lower()
        strict = _env_bool("MGC_STRICT_PROVIDER", False) if strict_provider is None else bool(strict_provider)
        self.cfg = MusicAgentConfig(provider=p, strict_provider=strict)

    def _resolve_provider_name(self) -> str:
        return (self.cfg.provider or "riffusion").strip().lower()

    def _resolve_provider(self) -> tuple[Any, str]:
        name = self._resolve_provider_name()
        fallback_from = ""
        if name == "riffusion":
            fallback_from = _preflight_riffusion()
            if fallback_from:
                name = "stub"
                self.cfg.provider = "stub"
        try:
            return get_provider(name), fallback_from
        except Exception as e:
            if self.cfg.strict_provider:
                raise
            if name != "stub":
                if not fallback_from:
                    fallback_from = name
                self.cfg.provider = "stub"
                return get_provider("stub"), fallback_from
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
        provider, fallback_from = self._resolve_provider()
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

        # Normalize provider return -> dict
        if isinstance(art, dict):
            art_dict = art
        elif is_dataclass(art):
            art_dict = asdict(art)
        elif hasattr(art, "__dict__"):
            art_dict = dict(vars(art))
        else:
            raise TypeError(f"Provider returned unsupported artifact type: {type(art)}")

        art = art_dict

        out_dir_p = Path(out_dir).expanduser().resolve()
        out_dir_p.mkdir(parents=True, exist_ok=True)

        # Materialize audio if needed
        if (not art.get("artifact_path")) and art.get("artifact_bytes"):
            ext = _pick_ext_for_materialize(art)
            out_path = out_dir_p / f"{track_id}{ext}"
            b = art["artifact_bytes"]
            if isinstance(b, bytearray):
                b = bytes(b)
            out_path.write_bytes(b)
            art["artifact_path"] = str(out_path)

            # Preserve ext/mime for downstream consumers
            art.setdefault("ext", ext)
            if not art.get("mime"):
                if ext == ".mp3":
                    art["mime"] = "audio/mpeg"
                elif ext == ".wav":
                    art["mime"] = "audio/wav"

        # Materialize preview if provided as bytes (and preview_path not already set)
        if (not art.get("preview_path")) and art.get("preview_bytes"):
            prev_ext = _pick_preview_ext(art)
            prev_path = out_dir_p / f"{track_id}{prev_ext}"
            pb = art["preview_bytes"]
            if isinstance(pb, bytearray):
                pb = bytes(pb)
            prev_path.write_bytes(pb)
            art["preview_path"] = str(prev_path)

        if not art.get("artifact_path"):
            raise TypeError(
                f"Provider artifact missing required 'artifact_path' key: keys={sorted(list(art.keys()))}"
            )

        artifact_path = str(art.get("artifact_path") or "")
        if not artifact_path:
            raise ValueError("Provider artifact missing artifact_path")

        # Normalize MP3s in-place (trim any leading junk before ID3/frame sync)
        ap = Path(artifact_path)
        if ap.suffix.lower() == ".mp3":
            # default ON; allow opt-out if needed
            if _env_bool("MGC_NORMALIZE_MP3", True):
                _normalize_mp3_in_place(ap)

        sha = str(art.get("sha256") or "")
        if not sha:
            sha = _sha256_file(Path(artifact_path))

        meta = dict(art.get("meta") or {})
        meta.setdefault("context", context)
        meta.setdefault("seed", int(seed))
        meta.setdefault("schedule", schedule)
        meta.setdefault("period_key", period_key)
        meta.setdefault("deterministic", bool(deterministic))
        if fallback_from:
            meta.setdefault("provider_fallback_from", fallback_from)

        # Carry ext/mime and preview_path into meta for downstream tools
        if art.get("ext") and not meta.get("ext"):
            meta["ext"] = str(art.get("ext"))
        if art.get("mime") and not meta.get("mime"):
            meta["mime"] = str(art.get("mime"))
        if art.get("preview_path") and not meta.get("preview_path"):
            meta["preview_path"] = str(art.get("preview_path"))

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
