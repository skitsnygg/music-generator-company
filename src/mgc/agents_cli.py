#!/usr/bin/env python3
"""src/mgc/agents_cli.py

Agents CLI (v1): music generation + marketing planning.

Commands:
  mgc agents music generate --count 3 --context focus --schedule daily --deterministic --json
  mgc agents marketing plan --drop-dir <dir> --out-dir <dir> --teaser-seconds 15 --json

Design goals:
- No DB mutation (v1).
- Deterministic IDs in --deterministic mode.
- Append-only JSONL receipts.

Important:
- mgc.main hoists --seed as a global flag for CI compatibility. Do NOT define a
  subcommand-local --seed here; read args.seed instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
import wave
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mgc.providers import ProviderError, get_provider  # type: ignore


# -----------------------------
# Utilities
# -----------------------------


def _eprint(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_stable_json_dumps(obj) + "\n")


def _derive_period_key(schedule: str, now_iso: str) -> str:
    dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
    if schedule == "weekly":
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"
    return dt.strftime("%Y-%m-%d")


def _uuid5(seed_material: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed_material))


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "y", "on")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_provider_result(res: Any) -> Tuple[bytes, str, str, Dict[str, Any]]:
    """Accept both legacy dict results and GenerateResult-like objects.

    Returns: (artifact_bytes, ext, mime, meta_dict)
    """

    if isinstance(res, dict):
        artifact_bytes = res.get("artifact_bytes")
        ext = str(res.get("ext") or "wav").lstrip(".")
        mime = str(res.get("mime") or "")
        meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
        if not isinstance(artifact_bytes, (bytes, bytearray)):
            raise ProviderError("Provider dict result missing artifact_bytes (bytes)")
        return (bytes(artifact_bytes), ext, mime, meta)

    if is_dataclass(res):
        d = asdict(res)
        artifact_bytes = d.get("artifact_bytes") or d.get("audio_bytes") or d.get("bytes")
        ext = str(d.get("ext") or d.get("extension") or "wav").lstrip(".")
        mime = str(d.get("mime") or d.get("content_type") or "")
        meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        if not isinstance(artifact_bytes, (bytes, bytearray)):
            raise ProviderError("Provider dataclass result missing artifact_bytes (bytes)")
        return (bytes(artifact_bytes), ext, mime, meta)

    artifact_bytes = getattr(res, "artifact_bytes", None)
    if artifact_bytes is None:
        artifact_bytes = getattr(res, "audio_bytes", None)
    ext = getattr(res, "ext", None) or getattr(res, "extension", None) or "wav"
    mime = getattr(res, "mime", None) or getattr(res, "content_type", None) or ""
    meta = getattr(res, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(artifact_bytes, (bytes, bytearray)):
        raise ProviderError(f"Provider returned unsupported result type: {type(res)} (missing bytes)")
    return (bytes(artifact_bytes), str(ext).lstrip("."), str(mime), meta)


# -----------------------------
# Music Agent
# -----------------------------


def cmd_agents_music_generate(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))

    provider_name = str(getattr(args, "provider", None) or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()
    provider = get_provider(provider_name)

    context = str(getattr(args, "context", "focus"))
    schedule = str(getattr(args, "schedule", "daily"))
    deterministic = bool(getattr(args, "deterministic", False) or _truthy_env("MGC_DETERMINISTIC"))

    seed_val = getattr(args, "seed", None)
    seed = int(seed_val) if seed_val is not None else int(os.environ.get("MGC_SEED") or "1")

    count = int(getattr(args, "count", 1))
    prompt = getattr(args, "prompt", None)
    out_dir = Path(str(getattr(args, "out_dir", "artifacts/run/agents/music"))).expanduser().resolve()

    now_iso = str(getattr(args, "now", None) or _utc_now_iso())
    period_key = str(getattr(args, "period_key", None) or _derive_period_key(schedule, now_iso))

    receipts_path = out_dir / "receipts.jsonl"
    result_path = out_dir / "result.json"

    tracks_written: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for i in range(count):
        if deterministic:
            material = f"{provider_name}|{context}|{schedule}|{period_key}|seed={seed}|i={i}|prompt={prompt or ''}"
            track_id = _uuid5(material)
        else:
            track_id = str(uuid.uuid4())

        try:
            res = provider.generate(
                track_id=track_id,
                context=context,
                seed=seed,
                prompt=prompt,
                deterministic=deterministic,
                schedule=schedule,
                period_key=period_key,
                ts=now_iso,
                out_dir=str(out_dir),
                out_rel=None,
                run_id=None,
            )

            audio_bytes, ext, mime, meta = _normalize_provider_result(res)

            rel_track_path = f"tracks/{track_id}.{ext}"
            abs_track_path = out_dir / rel_track_path
            _write_bytes(abs_track_path, audio_bytes)

            receipt = {
                "ok": True,
                "ts": now_iso,
                "provider": provider_name,
                "context": context,
                "schedule": schedule,
                "period_key": period_key,
                "deterministic": deterministic,
                "seed": seed,
                "prompt": prompt,
                "track_id": track_id,
                "paths": {"out_dir": str(out_dir), "track": rel_track_path},
                "meta": meta,
            }
            _append_jsonl(receipts_path, receipt)

            tracks_written.append({"track_id": track_id, "path": rel_track_path, "ext": ext, "mime": mime, "meta": meta})

        except Exception as e:
            err = {
                "ok": False,
                "ts": now_iso,
                "provider": provider_name,
                "context": context,
                "schedule": schedule,
                "period_key": period_key,
                "deterministic": deterministic,
                "seed": seed,
                "prompt": prompt,
                "track_id": track_id,
                "error": str(e),
            }
            _append_jsonl(receipts_path, err)
            errors.append(err)
            if deterministic:
                break

    out_obj: Dict[str, Any] = {
        "ok": (len(errors) == 0),
        "cmd": "agents.music.generate",
        "provider": provider_name,
        "context": context,
        "schedule": schedule,
        "period_key": period_key,
        "deterministic": deterministic,
        "seed": seed,
        "count": count,
        "paths": {"out_dir": str(out_dir), "receipts": str(receipts_path), "result": str(result_path)},
        "tracks": tracks_written,
        "errors": errors,
    }

    _write_text(result_path, _stable_json_dumps(out_obj) + "\n")

    if json_mode:
        sys.stdout.write(_stable_json_dumps(out_obj) + "\n")
    else:
        _eprint(f"[agents.music.generate] ok={out_obj['ok']} tracks={len(tracks_written)} errors={len(errors)} out_dir={out_dir}")

    return 0 if out_obj["ok"] else 2


# -----------------------------
# Marketing Agent
# -----------------------------


def _load_playlist(drop_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    playlist_path = drop_dir / "drop_bundle" / "playlist.json"
    if not playlist_path.exists():
        playlist_path = drop_dir / "playlist.json"
    if not playlist_path.exists():
        raise ValueError(f"playlist.json not found under: {drop_dir}")
    obj = _read_json(playlist_path)
    if not isinstance(obj, dict) or "tracks" not in obj:
        raise ValueError(f"Invalid playlist format: {playlist_path}")
    return playlist_path, obj


def _pick_lead_track(playlist: Dict[str, Any]) -> Dict[str, Any]:
    tracks = playlist.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("playlist.tracks is empty")
    lead_id = playlist.get("lead_track_id")
    if lead_id:
        for t in tracks:
            if isinstance(t, dict) and t.get("track_id") == lead_id:
                return t
    # fallback: first track
    t0 = tracks[0]
    if not isinstance(t0, dict):
        raise ValueError("playlist.tracks[0] is not an object")
    return t0


def _resolve_audio_path(repo_root: Path, track_obj: Dict[str, Any]) -> Path:
    # Expected keys (based on your playlist builder history): artifact_path or path
    p = track_obj.get("artifact_path") or track_obj.get("path")
    if not isinstance(p, str) or not p:
        raise ValueError("track is missing artifact_path/path")
    path = Path(p)
    if path.is_absolute():
        return path
    # Most of your artifacts are repo-relative
    return (repo_root / path).resolve()


def _write_teaser_wav(src_wav: Path, dst_wav: Path, seconds: float) -> Dict[str, Any]:
    if seconds <= 0:
        raise ValueError("teaser seconds must be > 0")

    with wave.open(str(src_wav), "rb") as r:
        nchannels = r.getnchannels()
        sampwidth = r.getsampwidth()
        framerate = r.getframerate()
        nframes_total = r.getnframes()
        max_frames = int(min(nframes_total, int(seconds * framerate)))
        frames = r.readframes(max_frames)

    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst_wav), "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(frames)

    return {
        "src": str(src_wav),
        "dst": str(dst_wav),
        "seconds": float(seconds),
        "channels": nchannels,
        "sample_rate": framerate,
        "sampwidth": sampwidth,
        "frames": max_frames,
    }


def _write_cover_png(dst_png: Path, seed_material: str, size_px: int = 1024) -> Dict[str, Any]:
    # No text. Deterministic abstract cover.
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        raise RuntimeError(f"Pillow not available: {e}")

    # Deterministic RNG from uuid5
    u = uuid.uuid5(uuid.NAMESPACE_URL, seed_material)
    seed_int = u.int & ((1 << 32) - 1)

    def rnd() -> int:
        nonlocal seed_int
        seed_int = (1664525 * seed_int + 1013904223) & 0xFFFFFFFF
        return seed_int

    img = Image.new("RGB", (size_px, size_px), (rnd() % 256, rnd() % 256, rnd() % 256))
    d = ImageDraw.Draw(img)

    # Draw deterministic shapes
    for _ in range(48):
        x0 = rnd() % size_px
        y0 = rnd() % size_px
        x1 = rnd() % size_px
        y1 = rnd() % size_px
        col = (rnd() % 256, rnd() % 256, rnd() % 256)
        if rnd() % 2 == 0:
            d.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=col, width=3)
        else:
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=col, width=3)

    dst_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(dst_png), format="PNG", optimize=True)

    return {"dst": str(dst_png), "size_px": size_px}


def cmd_agents_marketing_plan(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))

    drop_dir = Path(str(getattr(args, "drop_dir", ""))).expanduser().resolve()
    out_dir = Path(str(getattr(args, "out_dir", "artifacts/run/agents/marketing"))).expanduser().resolve()
    teaser_seconds = float(getattr(args, "teaser_seconds", 15.0))
    repo_root = Path(str(getattr(args, "repo_root", os.getcwd()))).expanduser().resolve()

    now_iso = str(getattr(args, "now", None) or _utc_now_iso())
    seed_val = getattr(args, "seed", None)
    seed = int(seed_val) if seed_val is not None else int(os.environ.get("MGC_SEED") or "1")

    receipts_path = out_dir / "receipts.jsonl"
    plan_path = out_dir / "marketing_plan.json"

    try:
        playlist_path, playlist = _load_playlist(drop_dir)
        lead_track = _pick_lead_track(playlist)
        lead_track_id = str(lead_track.get("track_id") or "")
        audio_path = _resolve_audio_path(repo_root, lead_track)

        teaser_path = out_dir / "teaser.wav"
        cover_path = out_dir / "cover.png"

        teaser_info: Optional[Dict[str, Any]] = None
        if audio_path.suffix.lower() == ".wav" and audio_path.exists():
            teaser_info = _write_teaser_wav(audio_path, teaser_path, teaser_seconds)
        else:
            # If non-wav, we skip creating teaser for v1.
            teaser_info = {
                "skipped": True,
                "reason": "lead track is not a .wav or does not exist",
                "lead_audio": str(audio_path),
            }

        cover_info = _write_cover_png(cover_path, seed_material=f"{lead_track_id}|{now_iso}|seed={seed}")

        # Copy variants (no emojis)
        title = str(lead_track.get("title") or "New drop")
        context = str(playlist.get("context") or lead_track.get("mood") or "")
        schedule = str(playlist.get("schedule") or "")
        period_key = str(playlist.get("period_key") or "")

        posts = [
            f"New release: {title}. Context: {context or 'mix'}. Listen now.",
            f"Fresh drop out now ({schedule or 'release'} {period_key or now_iso[:10]}). {title}.", 
            f"New music for {context or 'your day'}: {title}. Press play.",
        ]

        post_paths: List[str] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, text in enumerate(posts, start=1):
            p = out_dir / f"post_{idx}.txt"
            _write_text(p, text.strip() + "\n")
            post_paths.append(str(p))

        plan = {
            "ok": True,
            "cmd": "agents.marketing.plan",
            "ts": now_iso,
            "seed": seed,
            "paths": {
                "out_dir": str(out_dir),
                "playlist": str(playlist_path),
                "lead_audio": str(audio_path),
                "teaser": str(teaser_path),
                "cover": str(cover_path),
                "plan": str(plan_path),
                "receipts": str(receipts_path),
                "posts": post_paths,
            },
            "lead_track_id": lead_track_id,
            "teaser": teaser_info,
            "cover": cover_info,
        }

        _write_text(plan_path, _stable_json_dumps(plan) + "\n")
        _append_jsonl(receipts_path, plan)

        if json_mode:
            sys.stdout.write(_stable_json_dumps(plan) + "\n")
        else:
            _eprint(f"[agents.marketing.plan] ok out_dir={out_dir}")
        return 0

    except Exception as e:
        err = {
            "ok": False,
            "cmd": "agents.marketing.plan",
            "ts": now_iso,
            "seed": seed,
            "drop_dir": str(drop_dir),
            "out_dir": str(out_dir),
            "error": str(e),
        }
        _append_jsonl(receipts_path, err)
        _write_text(plan_path, _stable_json_dumps(err) + "\n")
        if json_mode:
            sys.stdout.write(_stable_json_dumps(err) + "\n")
        else:
            _eprint(f"[agents.marketing.plan] ERROR: {e}")
        return 2


# -----------------------------
# Registrar
# -----------------------------


def register_agents_subcommand(subparsers: argparse._SubParsersAction) -> None:
    agents = subparsers.add_parser("agents", help="Agent tools (music/marketing/billing orchestration)")
    a = agents.add_subparsers(dest="agents_cmd", required=True)

    # music
    music = a.add_parser("music", help="Music agent")
    ms = music.add_subparsers(dest="music_cmd", required=True)

    gen = ms.add_parser("generate", help="Generate N tracks via selected provider and write receipts")
    gen.add_argument("--count", type=int, default=1, help="Number of tracks to generate")
    gen.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"))
    gen.add_argument("--schedule", default="daily", choices=["daily", "weekly"], help="Schedule label (affects period_key)")
    gen.add_argument("--period-key", default=None, help="Override derived period key")
    gen.add_argument("--prompt", default=None, help="Optional prompt for generators that support it")
    gen.add_argument("--out-dir", default="artifacts/run/agents/music", help="Output directory")
    gen.add_argument("--now", default=None, help="Override timestamp ISO (supports Z)")
    gen.add_argument("--deterministic", action="store_true", help="Deterministic IDs + fail-fast")
    gen.add_argument("--provider", default=None, help="Override provider name (default: MGC_PROVIDER or stub)")
    gen.set_defaults(func=cmd_agents_music_generate)

    # marketing
    marketing = a.add_parser("marketing", help="Marketing agent")
    mk = marketing.add_subparsers(dest="marketing_cmd", required=True)

    plan = mk.add_parser("plan", help="Generate teaser + cover + post copy from a drop bundle")
    plan.add_argument("--drop-dir", required=True, help="Drop output dir (contains drop_bundle/playlist.json) or a dir containing playlist.json")
    plan.add_argument("--out-dir", default="artifacts/run/agents/marketing", help="Output directory")
    plan.add_argument("--repo-root", default=os.getcwd(), help="Repo root to resolve relative track paths")
    plan.add_argument("--teaser-seconds", type=float, default=15.0, help="Teaser duration in seconds (wav only in v1)")
    plan.add_argument("--now", default=None, help="Override timestamp ISO (supports Z)")
    plan.set_defaults(func=cmd_agents_marketing_plan)
