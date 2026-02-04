from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mgc.context import build_prompt, get_context_spec

from .base import GenerateRequest, GenerateResult, ProviderError, sha256_bytes


@dataclass(frozen=True)
class DiffSingerConfig:
    endpoint: str
    model_dir: str
    api_key: str
    voice: str
    cmd: str
    cmd_args: str
    cmd_mode: str
    timeout_s: int
    output_format: str
    duration_sec: int


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return (v if v is not None else default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, "")
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _normalize_format(fmt: str) -> str:
    f = (fmt or "").strip().lower().lstrip(".")
    return f or "wav"


def _load_config() -> DiffSingerConfig:
    endpoint = _env_str("MGC_DIFFSINGER_ENDPOINT")
    model_dir = _env_str("MGC_DIFFSINGER_MODEL_DIR")
    api_key = _env_str("MGC_DIFFSINGER_API_KEY")
    voice = _env_str("MGC_DIFFSINGER_VOICE")
    cmd = _env_str("MGC_DIFFSINGER_CMD")
    cmd_args = _env_str("MGC_DIFFSINGER_CMD_ARGS")
    cmd_mode = (_env_str("MGC_DIFFSINGER_CMD_MODE", "args") or "args").lower()
    timeout_s = _env_int("MGC_DIFFSINGER_TIMEOUT", 120)
    output_format = _normalize_format(_env_str("MGC_DIFFSINGER_OUTPUT_FORMAT", "wav"))
    duration_sec = _env_int("MGC_DIFFSINGER_DURATION_SEC", 30)

    if not endpoint and not cmd and not model_dir:
        raise ProviderError(
            "DiffSinger provider not configured. "
            "Set MGC_DIFFSINGER_ENDPOINT (remote) or MGC_DIFFSINGER_CMD (local wrapper)."
        )

    if model_dir:
        p = Path(model_dir).expanduser()
        if not p.exists():
            raise ProviderError(f"DiffSinger model dir not found: {p}")

    if model_dir and not cmd and not endpoint:
        raise ProviderError(
            "DiffSinger model dir set but no runner configured. "
            "Set MGC_DIFFSINGER_CMD to a local wrapper or set MGC_DIFFSINGER_ENDPOINT."
        )

    if cmd_mode not in ("args", "stdin_json"):
        raise ProviderError("MGC_DIFFSINGER_CMD_MODE must be 'args' or 'stdin_json'.")

    return DiffSingerConfig(
        endpoint=endpoint,
        model_dir=model_dir,
        api_key=api_key,
        voice=voice,
        cmd=cmd,
        cmd_args=cmd_args,
        cmd_mode=cmd_mode,
        timeout_s=int(timeout_s),
        output_format=output_format,
        duration_sec=int(duration_sec),
    )


def _guess_ext_mime(audio_bytes: bytes, ext_hint: str = "", mime_hint: str = "") -> Tuple[str, str]:
    ext = (ext_hint or "").strip().lower()
    if ext and not ext.startswith("."):
        ext = "." + ext

    mime = (mime_hint or "").strip().lower()

    if not ext:
        if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
            ext = ".wav"
        elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            ext = ".mp3"
        else:
            ext = ".wav"

    if not mime:
        if ext == ".mp3":
            mime = "audio/mpeg"
        else:
            mime = "audio/wav"

    return ext, mime


def _decode_audio_base64(raw: Any) -> Optional[bytes]:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, str):
        try:
            return base64.b64decode(raw, validate=False)
        except Exception:
            return None
    return None


def _read_audio_path(p: str) -> Optional[bytes]:
    try:
        path = Path(p).expanduser()
        if path.is_file():
            return path.read_bytes()
    except Exception:
        return None
    return None


def _fetch_url_bytes(url: str, *, timeout_s: int, api_key: str) -> Tuple[bytes, str]:
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        content_type = str(r.headers.get("Content-Type") or "")
        return r.read(), content_type


def _parse_json_audio(obj: Dict[str, Any], *, timeout_s: int, api_key: str) -> Tuple[bytes, str, str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    for k in ("bpm", "duration_sec", "duration_s", "voice", "model", "sample_rate"):
        if k in obj:
            meta[k] = obj[k]

    audio_bytes = None
    for key in ("audio_base64", "audio_b64", "audio", "audio_bytes"):
        if key in obj:
            audio_bytes = _decode_audio_base64(obj.get(key))
            if audio_bytes:
                break

    if audio_bytes is None:
        audio_path = obj.get("audio_path") or obj.get("path")
        if isinstance(audio_path, str) and audio_path:
            audio_bytes = _read_audio_path(audio_path)

    if audio_bytes is None:
        audio_url = obj.get("audio_url") or obj.get("url")
        if isinstance(audio_url, str) and audio_url:
            audio_bytes, content_type = _fetch_url_bytes(audio_url, timeout_s=timeout_s, api_key=api_key)
            ext, mime = _guess_ext_mime(audio_bytes, ext_hint="", mime_hint=content_type)
            return audio_bytes, ext, mime, meta

    if audio_bytes is None:
        raise ProviderError("DiffSinger response missing audio bytes (expected audio_base64 or audio_url).")

    ext = str(obj.get("ext") or obj.get("extension") or "").strip()
    mime = str(obj.get("mime") or obj.get("content_type") or "").strip()
    ext, mime = _guess_ext_mime(audio_bytes, ext_hint=ext, mime_hint=mime)
    return audio_bytes, ext, mime, meta


def _http_generate(cfg: DiffSingerConfig, payload: Dict[str, Any]) -> Tuple[bytes, str, str, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(cfg.endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if cfg.api_key:
        req.add_header("Authorization", f"Bearer {cfg.api_key}")
        req.add_header("X-API-Key", cfg.api_key)

    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as r:
            content_type = str(r.headers.get("Content-Type") or "")
            raw = r.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        raise ProviderError(f"DiffSinger HTTP {e.code}: {body[:200]}") from e
    except Exception as e:
        raise ProviderError(f"DiffSinger request failed: {e}") from e

    if content_type.startswith("application/json"):
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise ProviderError(f"DiffSinger returned invalid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise ProviderError("DiffSinger JSON response must be an object.")
        return _parse_json_audio(obj, timeout_s=cfg.timeout_s, api_key=cfg.api_key)

    audio_bytes = raw
    ext, mime = _guess_ext_mime(audio_bytes, mime_hint=content_type)
    return audio_bytes, ext, mime, {}


def _cmd_generate(cfg: DiffSingerConfig, payload: Dict[str, Any]) -> Tuple[bytes, str, str, Dict[str, Any]]:
    if not cfg.cmd:
        raise ProviderError("DiffSinger command mode requested but MGC_DIFFSINGER_CMD is unset.")

    with tempfile.TemporaryDirectory(prefix="mgc_diffsinger_") as td:
        out_path = Path(td) / f"{payload.get('track_id')}.{cfg.output_format}"

        base_cmd = shlex.split(cfg.cmd)

        if cfg.cmd_args:
            args = [arg.format(**payload, output=str(out_path)) for arg in shlex.split(cfg.cmd_args)]
        else:
            args = [
                "--output",
                str(out_path),
                "--prompt",
                str(payload.get("prompt") or ""),
                "--voice",
                str(payload.get("voice") or ""),
                "--seed",
                str(payload.get("seed") or 0),
                "--context",
                str(payload.get("context") or ""),
                "--bpm",
                str(payload.get("bpm") or ""),
                "--duration-seconds",
                str(payload.get("duration_sec") or ""),
            ]

        cmd = base_cmd + args

        if cfg.cmd_mode == "stdin_json":
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=cfg.timeout_s,
                check=False,
            )
        else:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=cfg.timeout_s,
                check=False,
            )

        if proc.returncode != 0:
            raise ProviderError(
                "DiffSinger command failed: "
                f"code={proc.returncode} stderr={proc.stderr.decode('utf-8', errors='ignore')[:200]}"
            )

        if out_path.exists() and out_path.stat().st_size > 0:
            audio_bytes = out_path.read_bytes()
            ext, mime = _guess_ext_mime(audio_bytes, ext_hint=out_path.suffix, mime_hint="")
            return audio_bytes, ext, mime, {}

        stdout = proc.stdout.decode("utf-8", errors="ignore").strip()
        if stdout:
            try:
                obj = json.loads(stdout)
                if isinstance(obj, dict):
                    return _parse_json_audio(obj, timeout_s=cfg.timeout_s, api_key=cfg.api_key)
            except Exception:
                if Path(stdout).expanduser().is_file():
                    audio_bytes = Path(stdout).expanduser().read_bytes()
                    ext, mime = _guess_ext_mime(audio_bytes, ext_hint=Path(stdout).suffix, mime_hint="")
                    return audio_bytes, ext, mime, {}

        raise ProviderError("DiffSinger command produced no audio output.")


class DiffSingerProvider:
    name = "diffsinger"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> GenerateResult:
        cfg = _load_config()

        if req is None:
            req = GenerateRequest(
                track_id=str(kwargs.get("track_id") or "track"),
                context=str(kwargs.get("context") or "focus"),
                seed=int(kwargs.get("seed") or 1),
                deterministic=bool(kwargs.get("deterministic") or False),
                schedule=str(kwargs.get("schedule") or ""),
                period_key=str(kwargs.get("period_key") or ""),
                prompt=str(kwargs.get("prompt") or ""),
                ts=str(kwargs.get("ts") or ""),
                out_dir=str(kwargs.get("out_dir") or ""),
                out_rel=str(kwargs.get("out_rel") or ""),
                run_id=kwargs.get("run_id"),
            )

        spec = get_context_spec(req.context)
        prompt = str(req.prompt or "").strip()
        if not prompt:
            prompt = build_prompt(req.context)

        payload: Dict[str, Any] = {
            "track_id": str(req.track_id),
            "context": str(req.context),
            "seed": int(req.seed),
            "deterministic": bool(req.deterministic),
            "schedule": str(req.schedule),
            "period_key": str(req.period_key),
            "prompt": prompt,
            "voice": cfg.voice,
            "bpm": int(spec.bpm_max),
            "bpm_min": int(spec.bpm_min),
            "bpm_max": int(spec.bpm_max),
            "duration_sec": int(cfg.duration_sec),
            "output_format": cfg.output_format,
            "model_dir": cfg.model_dir,
        }

        if cfg.endpoint:
            audio_bytes, ext, mime, extra = _http_generate(cfg, payload)
        else:
            audio_bytes, ext, mime, extra = _cmd_generate(cfg, payload)

        meta: Dict[str, Any] = {
            "provider": self.name,
            "sha256": sha256_bytes(audio_bytes),
            "bytes": len(audio_bytes),
            "prompt": prompt,
            "voice": cfg.voice,
            "context": req.context,
            "seed": int(req.seed),
            "deterministic": bool(req.deterministic),
            "schedule": str(req.schedule),
            "period_key": str(req.period_key),
            "output_format": cfg.output_format,
            "bpm": int(spec.bpm_max),
        }
        if cfg.endpoint:
            meta["endpoint"] = cfg.endpoint
        if cfg.cmd:
            meta["cmd"] = cfg.cmd
        if cfg.model_dir:
            meta["model_dir"] = cfg.model_dir
        if extra:
            meta.update(extra)
        if "duration_sec" not in meta and "duration_s" not in meta:
            meta["duration_sec"] = int(cfg.duration_sec)

        return GenerateResult(
            provider=self.name,
            artifact_bytes=audio_bytes,
            ext=str(ext),
            mime=str(mime),
            meta=meta,
        )
