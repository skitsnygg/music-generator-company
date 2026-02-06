from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from mgc.context import get_context_spec
from mgc.providers.base import GenerateRequest, ProviderError, sha256_bytes
from mgc.providers.riffusion_provider import RiffusionProvider


def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    u = u.rstrip("/")
    if "/run_inference" not in u:
        return u + "/run_inference/"
    return u.rstrip("/") + "/"


def _stable_int_from_key(key: str, lo: int, hi: int) -> int:
    import hashlib

    if hi <= lo:
        return lo
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return lo + (h % (hi - lo + 1))


def _env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return (v if v is not None else default).strip()


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _has_kw(fn: Any, key: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return key in sig.parameters
    except Exception:
        return False


def _which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def _parse_mp3_quality(quality: str) -> tuple[str, Optional[int], str]:
    q = (quality or "").strip().lower()
    if not q or q == "server":
        return ("server", None, "server")
    if q in ("v0", "0", "vbr0"):
        return ("vbr", 0, "v0")
    if q in ("v1", "1", "vbr1"):
        return ("vbr", 1, "v1")
    if q in ("v2", "2", "vbr2"):
        return ("vbr", 2, "v2")
    if q in ("v3", "3", "vbr3"):
        return ("vbr", 3, "v3")
    if q in ("v4", "4", "vbr4"):
        return ("vbr", 4, "v4")
    if q in ("v5", "5", "vbr5"):
        return ("vbr", 5, "v5")
    if q in ("v6", "6", "vbr6"):
        return ("vbr", 6, "v6")
    if q in ("v7", "7", "vbr7"):
        return ("vbr", 7, "v7")
    if q in ("v8", "8", "vbr8"):
        return ("vbr", 8, "v8")
    if q in ("v9", "9", "vbr9"):
        return ("vbr", 9, "v9")
    if q in ("320", "320k", "cbr320"):
        return ("cbr", 320, "320")
    if q in ("256", "256k", "cbr256"):
        return ("cbr", 256, "256")
    if q in ("192", "192k", "cbr192"):
        return ("cbr", 192, "192")
    raise ProviderError(
        f"Unsupported MGC_MP3_QUALITY='{quality}'. Use 'v0', 'v2', 'v4', '320', or 'server'."
    )


def _encode_mp3_with_lame(wav_path: Path, mp3_path: Path, *, quality: str) -> Dict[str, Any]:
    """
    Deterministic MP3 encoding via LAME.

    quality:
      - "v0"/"v2"/"v4": VBR levels (0 = best)
      - "320": CBR 320kbps
    """
    lame = _which("lame")
    if not lame:
        raise ProviderError("MGC_MP3_QUALITY requested LAME encoding but 'lame' was not found on PATH.")

    mode, qval, qnorm = _parse_mp3_quality(quality)
    if mode == "server":
        raise ProviderError("MGC_MP3_QUALITY='server' does not require LAME encoding.")

    cmd = [
        lame,
        "--noreplaygain",
        "--nohist",
        "--silent",
    ]

    if mode == "vbr":
        qval_i = int(qval or 0)
        cmd += [f"-V{qval_i}"]
    else:
        cmd += ["--cbr", "-b", str(int(qval or 320))]

    cmd += [str(wav_path), str(mp3_path)]
    subprocess.run(cmd, check=True)

    # keep meta small + path-free (determinism / privacy)
    return {
        "mp3_encoder": "lame",
        "mp3_quality": qnorm,
        "mp3_flags": " ".join(cmd[1:7]),
    }


def _encode_mp3_with_ffmpeg(wav_path: Path, mp3_path: Path, *, quality: str) -> Dict[str, Any]:
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise ProviderError("MGC_MP3_QUALITY requested MP3 encoding but 'ffmpeg' was not found on PATH.")

    mode, qval, qnorm = _parse_mp3_quality(quality)
    if mode == "server":
        raise ProviderError("MGC_MP3_QUALITY='server' does not require ffmpeg encoding.")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav_path),
        "-vn",
        "-map_metadata",
        "-1",
        "-write_id3v2",
        "0",
        "-codec:a",
        "libmp3lame",
    ]

    if mode == "vbr":
        cmd += ["-q:a", str(int(qval or 0))]
    else:
        cmd += ["-b:a", f"{int(qval or 320)}k"]

    cmd += [str(mp3_path)]
    subprocess.run(cmd, check=True)

    return {
        "mp3_encoder": "ffmpeg",
        "mp3_quality": qnorm,
        "mp3_flags": " ".join(cmd[1:9]),
    }


def _encode_mp3(wav_path: Path, mp3_path: Path, *, quality: str) -> Dict[str, Any]:
    if _which("lame"):
        try:
            return _encode_mp3_with_lame(wav_path, mp3_path, quality=quality)
        except Exception:
            pass
    return _encode_mp3_with_ffmpeg(wav_path, mp3_path, quality=quality)


def _probe_duration_seconds(path: Path) -> float:
    try:
        if path.suffix.lower() == ".wav":
            import wave

            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or 1
                return float(frames) / float(rate)
    except Exception:
        pass

    ffprobe = _which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode("utf-8", errors="replace").strip() or 0.0)
    except Exception:
        return 0.0


def _stitch_segments_with_crossfade(
    segments: list[Path],
    out_mp3: Path,
    *,
    crossfade_s: float,
    target_s: float,
    mp3_quality: str,
) -> Dict[str, Any]:
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise ProviderError("Crossfades require ffmpeg, but it was not found on PATH.")

    if not segments:
        raise ProviderError("no riffusion segments to stitch")
    if len(segments) == 1:
        shutil.copy2(segments[0], out_mp3)
        return {"mp3_encoder": "copy", "mp3_quality": "server"}

    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]

    parts: list[str] = []
    prev = "[0:a]"
    for i in range(1, len(segments)):
        out = f"[a{i}]"
        parts.append(f"{prev}[{i}:a]acrossfade=d={crossfade_s}:c1=tri:c2=tri{out}")
        prev = out

    filter_complex = ";".join(parts)

    mode, qval, qnorm = _parse_mp3_quality(mp3_quality)
    if mode == "server":
        mode = "vbr"
        qval = 0
        qnorm = "v0"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        prev,
    ]
    if target_s > 0:
        cmd += ["-t", f"{float(target_s):.3f}"]
    cmd += [
        "-vn",
        "-map_metadata",
        "-1",
        "-write_id3v2",
        "0",
        "-codec:a",
        "libmp3lame",
    ]
    if mode == "vbr":
        cmd += ["-q:a", str(int(qval or 0))]
    else:
        cmd += ["-b:a", f"{int(qval or 320)}k"]
    cmd += [str(out_mp3)]

    subprocess.run(cmd, check=True)

    return {
        "mp3_encoder": "ffmpeg",
        "mp3_quality": qnorm,
        "segment_count": int(len(segments)),
        "crossfade_seconds": float(crossfade_s),
    }


class RiffusionAdapter:
    """
    Adapter returns a dict so downstream kwarg filtering doesn't drop ext/mime.

    We prefer: server WAV -> VBR encode -> MP3 bytes.
    Fallback: server MP3 bytes (quality="server") when re-encode isn't possible.
    """

    name = "riffusion"

    def generate(self, req: GenerateRequest | None = None, **kwargs: Any) -> Dict[str, Any]:
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

        spec = get_context_spec(req.context)

        seed_int: Optional[int] = None
        if req.deterministic:
            seed_int = _stable_int_from_key(f"{req.track_id}|{req.seed}|{req.context}", 1, 2_000_000_000)

        bpm = (
            _stable_int_from_key(f"{req.track_id}|bpm", spec.bpm_min, spec.bpm_max)
            if req.deterministic
            else spec.bpm_max
        )

        title = f"{spec.name.title()} Track {req.seed}"
        mood = spec.mood
        genre = spec.genre

        server_url = _normalize_url(
            _env_str("MGC_RIFFUSION_URL")
            or _env_str("RIFFUSION_URL")
            or "http://127.0.0.1:3013/run_inference/"
        )
        prov = RiffusionProvider(server_url=server_url)

        prompt = req.prompt or spec.prompt

        ctx = str(req.context or "").strip().lower()
        is_workout = ctx == "workout"
        is_focus = ctx == "focus"

        if is_workout:
            target_seconds_default = 180.0
            segment_seconds_default = 12.0
            crossfade_seconds_default = 1.0
            max_segments_default = 48.0
        elif is_focus:
            target_seconds_default = 180.0
            segment_seconds_default = 12.0
            crossfade_seconds_default = 1.5
            max_segments_default = 48.0
        else:
            target_seconds_default = 120.0
            segment_seconds_default = 8.0
            crossfade_seconds_default = 1.5
            max_segments_default = 32.0

        target_seconds = _env_float("MGC_RIFFUSION_TARGET_SECONDS", target_seconds_default)
        segment_seconds = _env_float("MGC_RIFFUSION_SEGMENT_SECONDS", segment_seconds_default)
        crossfade_seconds = _env_float("MGC_RIFFUSION_CROSSFADE_SECONDS", crossfade_seconds_default)
        max_segments = int(_env_float("MGC_RIFFUSION_MAX_SEGMENTS", max_segments_default))
        if max_segments <= 0:
            max_segments = 1

        longform_hint = _env_str("MGC_RIFFUSION_LONGFORM_PROMPT", "1").strip()
        if target_seconds > 0:
            hint_text = ""
            if longform_hint.lower() in ("0", "false", "no", "off"):
                hint_text = ""
            elif longform_hint.lower() in ("1", "true", "yes", "on"):
                tags = ", ".join(spec.tags)
                hint_text = (
                    f"Playlist context: {spec.name} ({tags}). "
                    f"Long-form, cohesive, evolving structure, ~{int(target_seconds)}s. No vocals."
                )
            else:
                hint_text = longform_hint
            if hint_text:
                prompt = f"{prompt}\n{hint_text}".strip()

        # optional overrides
        steps_env = _env_str("RIFFUSION_STEPS") or _env_str("MGC_RIFFUSION_STEPS")
        guidance_env = _env_str("RIFFUSION_GUIDANCE") or _env_str("MGC_RIFFUSION_GUIDANCE")
        denoise_env = _env_str("RIFFUSION_DENOISE") or _env_str("MGC_RIFFUSION_DENOISE")
        timeout_env = _env_str("RIFFUSION_TIMEOUT") or _env_str("MGC_RIFFUSION_TIMEOUT")

        if steps_env:
            num_steps = int(steps_env)
        elif is_workout:
            num_steps = 60
        elif is_focus:
            num_steps = 55
        else:
            num_steps = None

        if guidance_env:
            guidance = float(guidance_env)
        elif is_workout:
            guidance = 5.5
        elif is_focus:
            guidance = 5.0
        else:
            guidance = None

        if denoise_env:
            denoise = float(denoise_env)
        elif is_workout:
            denoise = 0.45
        elif is_focus:
            denoise = 0.4
        else:
            denoise = None
        timeout_s = int(timeout_env) if timeout_env else None

        # MP3 quality selection:
        #   server: trust server MP3
        #   v0: VBR V0 (default)
        #   320: CBR 320kbps
        mp3_quality = (_env_str("MGC_MP3_QUALITY", "v0") or "v0").lower()
        mode, _, mp3_quality_norm = _parse_mp3_quality(mp3_quality)
        want_reencode = mode != "server"

        mp3_bytes: bytes
        preview_bytes: Optional[bytes] = None
        enc_meta: Dict[str, Any] = {}
        duration_sec = 0.0
        segment_count = 1
        xfade_used = 0.0

        with tempfile.TemporaryDirectory(prefix="mgc_riffusion_") as td:
            td_path = Path(td)
            out_preview = td_path / f"{req.track_id}.jpg"
            out_mp3 = td_path / f"{req.track_id}.mp3"

            segments: list[Path] = []
            seg_durations: list[float] = []
            total_seconds = 0.0
            longform = target_seconds > 0

            for i in range(max_segments):
                seg_mp3 = td_path / f"{req.track_id}_seg{i}.mp3"
                seg_wav = td_path / f"{req.track_id}_seg{i}.wav"
                seg_preview = out_preview if i == 0 else None

                seg_seed = seed_int
                if seed_int is not None:
                    seg_seed = _stable_int_from_key(
                        f"{req.track_id}|seg={i}|seed={req.seed}|context={req.context}",
                        1,
                        2_000_000_000,
                    )

                call_kwargs = dict(
                    out_mp3=seg_mp3,
                    out_preview_jpg=seg_preview,
                    title=title,
                    mood=mood,
                    genre=genre,
                    bpm=int(bpm),
                    prompt=prompt,
                    seed=seg_seed,
                    num_inference_steps=num_steps,
                    guidance=guidance,
                    denoising=denoise,
                    timeout_s=timeout_s,
                )

                if _has_kw(prov.generate, "out_wav") and (want_reencode or longform):
                    call_kwargs["out_wav"] = seg_wav

                try:
                    prov.generate(**call_kwargs)
                except TypeError:
                    # Signature mismatch (older provider): retry with MP3-only
                    call_kwargs.pop("out_wav", None)
                    prov.generate(**call_kwargs)
                except Exception as e:
                    raise ProviderError(f"riffusion.generate failed: {e}") from e

                if i == 0 and seg_preview and seg_preview.exists() and seg_preview.stat().st_size > 0:
                    preview_bytes = seg_preview.read_bytes()

                seg_path: Optional[Path] = None
                if seg_wav.exists() and seg_wav.stat().st_size > 0:
                    seg_path = seg_wav
                elif seg_mp3.exists() and seg_mp3.stat().st_size > 0:
                    seg_path = seg_mp3

                if not seg_path:
                    raise ProviderError("riffusion did not produce an audio artifact (missing or empty).")

                segments.append(seg_path)
                seg_dur = _probe_duration_seconds(seg_path)
                if seg_dur <= 0:
                    seg_dur = float(segment_seconds)
                seg_durations.append(seg_dur)

                if i == 0:
                    total_seconds += seg_dur
                else:
                    total_seconds += max(0.0, seg_dur - crossfade_seconds)

                if not longform:
                    break
                if total_seconds >= target_seconds:
                    break

            if not segments:
                raise ProviderError("riffusion did not produce any segments.")

            segment_count = len(segments)
            if len(segments) > 1:
                min_seg = min(seg_durations) if seg_durations else float(segment_seconds)
                xfade = max(0.1, min(float(crossfade_seconds), max(0.1, min_seg * 0.5)))
                xfade_used = float(xfade)
                trim_target = float(target_seconds) if total_seconds >= target_seconds else 0.0
                enc_meta = _stitch_segments_with_crossfade(
                    segments,
                    out_mp3,
                    crossfade_s=xfade,
                    target_s=trim_target,
                    mp3_quality=mp3_quality,
                )
                if not out_mp3.exists() or out_mp3.stat().st_size <= 0:
                    raise ProviderError("riffusion crossfade output missing or empty.")
                mp3_bytes = out_mp3.read_bytes()
                duration_sec = _probe_duration_seconds(out_mp3) or (trim_target or total_seconds)
            else:
                seg = segments[0]
                if seg.suffix.lower() == ".wav":
                    try:
                        enc_meta = _encode_mp3(seg, out_mp3, quality=mp3_quality)
                    except Exception as e:
                        raise ProviderError(f"riffusion mp3 re-encode failed: {e}") from e
                    if not out_mp3.exists() or out_mp3.stat().st_size <= 0:
                        raise ProviderError("riffusion mp3 encode output missing or empty.")
                    mp3_bytes = out_mp3.read_bytes()
                    duration_sec = _probe_duration_seconds(out_mp3)
                else:
                    mp3_bytes = seg.read_bytes()
                    duration_sec = _probe_duration_seconds(seg)
                    if want_reencode:
                        enc_meta = {
                            "mp3_encoder": "server",
                            "mp3_quality": "server",
                            "mp3_note": "no_wav_available",
                        }

            if duration_sec <= 0 and seg_durations:
                duration_sec = float(seg_durations[0])
            if duration_sec <= 0 and target_seconds > 0:
                duration_sec = float(target_seconds)

        meta: Dict[str, Any] = {
            "provider": self.name,
            "riffusion_url": server_url,
            "bpm": int(bpm),
            "title": title,
            "mood": mood,
            "genre": genre,
            "prompt": prompt,
            "deterministic": bool(req.deterministic),
            "seed": int(req.seed),
            "context": str(req.context),
            "schedule": str(req.schedule),
            "period_key": str(req.period_key),
            "ext": ".mp3",
            "mime": "audio/mpeg",
            "sha256": sha256_bytes(mp3_bytes),
            "bytes": len(mp3_bytes),
            "mp3_quality": enc_meta.get("mp3_quality") or mp3_quality_norm,
            "duration_sec": float(duration_sec),
            "riffusion_target_seconds": float(target_seconds) if target_seconds > 0 else 0.0,
            "riffusion_segment_seconds": float(segment_seconds),
            "riffusion_crossfade_seconds": float(xfade_used),
            "riffusion_segment_count": int(segment_count),
        }
        if enc_meta:
            meta.update(enc_meta)

        out: Dict[str, Any] = {
            "provider": self.name,
            "track_id": str(req.track_id),
            "artifact_bytes": mp3_bytes,
            "ext": ".mp3",
            "mime": "audio/mpeg",
            "sha256": meta["sha256"],
            "title": meta["title"],
            "mood": meta["mood"],
            "genre": meta["genre"],
            "meta": meta,
        }

        if preview_bytes:
            out["preview_bytes"] = preview_bytes
            out["preview_ext"] = ".jpg"
            out["preview_mime"] = "image/jpeg"
            out["meta"] = dict(meta)
            out["meta"].update(
                {
                    "preview_sha256": sha256_bytes(preview_bytes),
                    "preview_bytes": len(preview_bytes),
                    "preview_ext": ".jpg",
                    "preview_mime": "image/jpeg",
                }
            )

        return out
