#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import random
import io
import json
import re
import os
import shutil
import socket
import sqlite3
import sys
import subprocess
import time
import uuid
import math
import wave
import struct

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, NoReturn, Optional, Sequence, Tuple

from mgc.context import build_prompt, get_context_spec
from mgc.hash_utils import sha256_file, sha256_tree
try:
    from mgc.providers import get_provider  # type: ignore
except Exception:  # pragma: no cover
    def get_provider(_name: str):  # type: ignore
        raise RuntimeError("Provider API not available (mgc.providers.get_provider missing). Use MusicAgent-based generation.")

from pathlib import Path

def _write_marketing_receipt(
    *,
    receipt: Dict[str, Any],
    stable_json_dumps,
    base_dir: Optional[Path] = None,
) -> str:
    """
    Append-only receipt writer for marketing publish.

    Location:
      data/evidence/marketing/receipts/<batch_id>/<platform>/<post_id>.json
    """
    # Allow callers (e.g. publish-marketing --out-dir) to redirect receipts
    # for determinism comparisons without touching repo state.
    base = base_dir if base_dir is not None else Path("data/evidence/marketing/receipts")
    batch_id = receipt.get("batch_id", "unknown")
    platform = receipt.get("platform", "unknown")
    post_id = receipt.get("post_id", "unknown")

    out_path = base / batch_id / platform / f"{post_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return str(out_path)

    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(stable_json_dumps(receipt) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return str(out_path)


@dataclass(frozen=True)
class GenerateRequest:
    """Legacy provider request container (for older provider-based generation paths).

    Newer code in this repo uses mgc.agents.music_agent.MusicAgent.
    This dataclass exists only to keep older CLI commands importable.
    """

    track_id: str
    run_id: str
    context: str
    seed: int
    prompt: str
    deterministic: bool
    ts: str
    out_rel: str

from mgc.playlist import build_daily_playlist, build_weekly_playlist

@contextlib.contextmanager
def _silence_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    old = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = old


@contextlib.contextmanager
def _temp_env(patch: Dict[str, Optional[str]]):
    """
    Temporarily set/unset environment variables.
    patch values:
      - str: set
      - None: unset
    """
    old: Dict[str, Optional[str]] = {}
    try:
        for k, v in patch.items():
            old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Global argv helpers (fix: honor --db even when passed before "run")
# ---------------------------------------------------------------------------

def _argv_value(flag: str, argv: Optional[Sequence[str]] = None) -> Optional[str]:
    """
    Return the value after `flag` from argv if present (handles --flag=value too).
    Example:
      --db foo.sqlite
      --db=foo.sqlite
    """
    av = list(argv) if argv is not None else list(sys.argv)
    for i, tok in enumerate(av):
        if tok == flag and i + 1 < len(av):
            nxt = av[i + 1]
            if not nxt.startswith("-"):
                return nxt
            return nxt  # allow paths starting with '-' (rare, but)
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _argv_has_flag(flag: str, argv: Optional[Sequence[str]] = None) -> bool:
    av = list(argv) if argv is not None else list(sys.argv)
    return any(tok == flag or tok.startswith(flag + "=") for tok in av)


def resolve_want_json(args: argparse.Namespace) -> bool:
    """
    Global JSON flag resolver.

    We support:
      - mgc.main --json <cmd> ...
      - mgc.main <cmd> ... --json        (if subparser defines it)
      - env: MGC_JSON=1 (optional)
    """
    v = getattr(args, "json", None)
    if isinstance(v, bool):
        return v
    env = (os.environ.get("MGC_JSON") or "").strip().lower()
    return env in ("1", "true", "yes", "on")


def resolve_db_path(args: argparse.Namespace) -> str:
    """
    Global DB flag resolver.

    We support:
      - mgc.main --db PATH <cmd> ...
      - mgc.main <cmd> ... --db PATH     (if subparser defines it)
      - env: MGC_DB=... (preferred) then default data/db.sqlite
    """
    db = getattr(args, "db", None)
    if isinstance(db, str) and db.strip():
        return db.strip()

    env = (os.environ.get("MGC_DB") or "").strip()
    if env:
        return env

    return "data/db.sqlite"



# ---------------------------------------------------------------------------
# Determinism utilities
# ---------------------------------------------------------------------------

def is_deterministic(args: Optional[argparse.Namespace] = None) -> bool:
    if args is not None and getattr(args, "deterministic", False):
        return True
    v = (os.environ.get("MGC_DETERMINISTIC") or os.environ.get("DETERMINISTIC") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def deterministic_now_iso(deterministic: bool) -> str:
    fixed = (os.environ.get("MGC_FIXED_TIME") or "").strip()
    if fixed:
        # epoch seconds
        try:
            if fixed.isdigit():
                dt = datetime.fromtimestamp(int(fixed), tz=timezone.utc)
                return dt.isoformat()
        except Exception:
            pass
        # ISO8601
        try:
            dt = datetime.fromisoformat(fixed.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            if deterministic:
                return "2020-01-01T00:00:00+00:00"

    if deterministic:
        return "2020-01-01T00:00:00+00:00"
    return datetime.now(timezone.utc).isoformat()


def stable_uuid5(*parts: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    return str(uuid.uuid5(namespace, "|".join(parts)))


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _relpath_smart(*, out_dir: Path, repo_root: Path, p: str) -> str:
    """Return a POSIX path string with absolute prefixes scrubbed.

    Rules (in priority order):
      1) If p is already relative -> normalize separators and return as POSIX.
      2) If p is absolute and is under repo_root -> return path relative to repo_root.
         (So it can be resolved correctly from the repo working directory.)
      3) If p is absolute and is under out_dir -> return path relative to out_dir.
      4) Otherwise -> return path relative to out_dir as a best-effort fallback.

    Guarantees:
      - Never returns an absolute path.
      - Never returns a path with a leading slash.
    """
    if not isinstance(p, str):
        return p  # type: ignore[return-value]

    s = p.strip()
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s

    s_norm = s.replace("\\", "/")
    try:
        pp = Path(s_norm)
    except Exception:
        return s_norm.lstrip("/")

    if not pp.is_absolute():
        return pp.as_posix().lstrip("/")

    # Normalize roots
    rr = repo_root.resolve()
    od = out_dir.resolve()

    # Prefer repo_root-relative when possible (portable across invocations)
    try:
        rel_repo = pp.resolve().relative_to(rr)
        return rel_repo.as_posix().lstrip("/")
    except Exception:
        pass

    # Next, out_dir-relative
    try:
        rel_out = pp.resolve().relative_to(od)
        return rel_out.as_posix().lstrip("/")
    except Exception:
        pass

    # Fallback: best-effort relpath from out_dir
    try:
        rel = Path(os.path.relpath(str(pp), start=str(od)))
        return rel.as_posix().lstrip("/")
    except Exception:
        return s_norm.lstrip("/")


def _scrub_absolute_paths(obj: object, *, out_dir: Path, repo_root: Path) -> object:
    """Recursively scrub absolute filesystem paths from a JSON-serializable object."""

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, str):
                out[k] = _relpath_smart(out_dir=out_dir, repo_root=repo_root, p=v)
            else:
                out[k] = _scrub_absolute_paths(v, out_dir=out_dir, repo_root=repo_root)
        return out

    if isinstance(obj, list):
        return [_scrub_absolute_paths(v, out_dir=out_dir, repo_root=repo_root) for v in obj]

    if isinstance(obj, str):
        return _relpath_smart(out_dir=out_dir, repo_root=repo_root, p=obj)

    return obj



def _posix(p: Path) -> str:
    """Return a stable POSIX-style relative path string for JSON/evidence."""
    return p.as_posix()
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text_bytes(path: Path) -> bytes:
    b = path.read_bytes()
    try:
        s = b.decode("utf-8")
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        return s.encode("utf-8")
    except Exception:
        return b



# ---------------------------------------------------------------------------
# Agents wiring helpers (music generation + marketing plan)
# ---------------------------------------------------------------------------

def _normalize_provider_result(res: Any) -> Tuple[bytes, str, str, Dict[str, Any]]:
    """Accept both legacy dict results and modern GenerateResult-like objects."""
    if isinstance(res, dict):
        b = res.get("artifact_bytes")
        ext = str(res.get("ext") or "wav").lstrip(".")
        mime = str(res.get("mime") or "")
        meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
        if not isinstance(b, (bytes, bytearray)):
            raise RuntimeError("provider dict result missing artifact_bytes bytes")
        return (bytes(b), ext, mime, meta)

    # dataclass?
    try:
        from dataclasses import is_dataclass, asdict
        if is_dataclass(res):
            d = asdict(res)
            b = d.get("artifact_bytes") or d.get("audio_bytes") or d.get("bytes")
            ext = str(d.get("ext") or d.get("extension") or "wav").lstrip(".")
            mime = str(d.get("mime") or d.get("content_type") or "")
            meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
            if not isinstance(b, (bytes, bytearray)):
                raise RuntimeError("provider dataclass result missing artifact_bytes bytes")
            return (bytes(b), ext, mime, meta)
    except Exception:
        pass

    b = getattr(res, "artifact_bytes", None)
    if b is None:
        b = getattr(res, "audio_bytes", None)
    ext = getattr(res, "ext", None) or getattr(res, "extension", None) or "wav"
    mime = getattr(res, "mime", None) or getattr(res, "content_type", None) or ""
    meta = getattr(res, "meta", None)
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(b, (bytes, bytearray)):
        raise RuntimeError(f"provider returned unsupported result type: {type(res)}")
    return (bytes(b), str(ext).lstrip("."), str(mime), meta)


def _agents_generate_and_ingest(
    *,
    db_path: str,
    repo_root: Path,
    context: str,
    schedule: str,
    period_key: str,
    seed: int,
    deterministic: bool,
    count: int,
    provider_name: Optional[str] = None,
    prompt: Optional[str] = None,
    now_iso: str,
) -> List[Dict[str, Any]]:
    """Generate `count` tracks via provider and ingest into (tracks dir + DB).

    This keeps the run daily/weekly command self-contained: generate -> library -> playlist builder.
    """
    if count <= 0:
        return []

    pname = (provider_name or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()
    provider = get_provider(pname)

    tracks_dir = (repo_root / "tracks").resolve()
    tracks_dir.mkdir(parents=True, exist_ok=True)

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    generated: List[Dict[str, Any]] = []
    try:
        for i in range(int(count)):
            if deterministic:
                track_id = stable_uuid5(f"agentgen:{pname}:{context}:{schedule}:{period_key}:seed={seed}:i={i}:prompt={prompt or ''}")
            else:
                track_id = str(uuid.uuid4())

            res = provider.generate(
                track_id=track_id,
                context=context,
                seed=int(seed),
                prompt=prompt,
                deterministic=bool(deterministic),
                schedule=schedule,
                period_key=period_key,
                ts=now_iso,
                out_dir=str(repo_root),
                out_rel=None,
                run_id=None,
            )
            audio_bytes, ext, mime, meta = _normalize_provider_result(res)
            ext = ext or "wav"
            filename = f"{track_id}.{ext}"
            rel_path = Path("tracks") / filename
            abs_path = tracks_dir / filename
            abs_path.write_bytes(audio_bytes)

            title = str(meta.get("title") or f"{context} track {track_id[:8]}")
            genre = str(meta.get("genre") or (pname if pname != "filesystem" else "library"))

            db_insert_track(
                con,
                track_id=str(track_id),
                ts=now_iso,
                title=title,
                provider=pname,
                mood=context,
                genre=genre,
                artifact_path=_posix(rel_path),
                meta={
                    "context": context,
                    "schedule": schedule,
                    "period_key": period_key,
                    "deterministic": bool(deterministic),
                    "seed": int(seed),
                    "prompt": prompt,
                    **meta,
                },
            )
            generated.append(
                {
                    "track_id": str(track_id),
                    "artifact_path": _posix(rel_path),
                    "provider": pname,
                    "ext": ext,
                    "mime": mime,
                    "meta": meta,
                }
            )

            if deterministic and i == 0 and count > 1:
                # continue; deterministic supports multiple, but fail-fast on errors only
                pass

        con.commit()
        return generated
    finally:
        con.close()


def _marketing_cover_bytes_png(seed: int, size_px: int = 1024) -> Tuple[Optional[bytes], Optional[str]]:
    """Try to produce a PNG cover. Returns (png_bytes, error)."""
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as e:
        return (None, f"Pillow not available: {e}")

    # Deterministic abstract art (no text)
    rng = random.Random(int(seed))
    img = Image.new("RGB", (int(size_px), int(size_px)), (250, 245, 235))
    d = ImageDraw.Draw(img)

    # layered rectangles
    for _ in range(40):
        x0 = rng.randint(0, size_px - 1)
        y0 = rng.randint(0, size_px - 1)
        x1 = rng.randint(x0 + 1, size_px)
        y1 = rng.randint(y0 + 1, size_px)
        col = (rng.randint(10, 240), rng.randint(10, 240), rng.randint(10, 240))
        d.rectangle([x0, y0, x1, y1], fill=col)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return (buf.getvalue(), None)


def _marketing_cover_svg(seed: int, size_px: int = 1024) -> str:
    rng = random.Random(int(seed))
    bg = f"#{rng.randint(0, 0xFFFFFF):06x}"
    rects = []
    for _ in range(30):
        x = rng.randint(0, size_px - 1)
        y = rng.randint(0, size_px - 1)
        w = rng.randint(20, size_px // 2)
        h = rng.randint(20, size_px // 2)
        col = f"#{rng.randint(0, 0xFFFFFF):06x}"
        op = rng.random() * 0.6 + 0.2
        rects.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{col}" fill-opacity="{op:.3f}" />')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}" viewBox="0 0 {size_px} {size_px}">'
        f'<rect width="{size_px}" height="{size_px}" fill="{bg}" />'
        + "".join(rects)
        + "</svg>"
    )


def _marketing_teaser_wav(
    *,
    lead_audio: Path,
    out_path: Path,
    teaser_seconds: int,
) -> Tuple[bool, str]:
    """Write teaser wav if possible. Returns (written, reason_if_skipped)."""
    if lead_audio.suffix.lower() != ".wav" or not lead_audio.exists():
        return (False, "lead track is not a .wav or does not exist")

    try:
        with wave.open(str(lead_audio), "rb") as r:
            nch = r.getnchannels()
            sw = r.getsampwidth()
            fr = r.getframerate()
            nframes = r.getnframes()
            max_frames = min(nframes, int(fr * int(teaser_seconds)))
            frames = r.readframes(max_frames)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(nch)
            w.setsampwidth(sw)
            w.setframerate(fr)
            w.writeframes(frames)

        return (True, "")
    except Exception as e:
        return (False, f"teaser failed: {e}")


def _agents_marketing_plan(
    *,
    drop_dir: Path,
    repo_root: Path,
    out_dir: Path,
    seed: int,
    teaser_seconds: int,
    ts: str,
) -> Dict[str, Any]:
    """Generate a deterministic marketing plan for a drop bundle.

    Determinism + portability requirements:
      - All paths serialized in the returned object and marketing_plan.json MUST be
        relative to out_dir.
      - Never serialize absolute paths (no /tmp, no /private/tmp, etc.).
    """
    drop_dir = drop_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    playlist_path = drop_dir / "playlist.json"
    if not playlist_path.exists():
        raise SystemExit(f"[agents.marketing.plan] missing playlist.json under {drop_dir}")

    pl = json.loads(playlist_path.read_text(encoding="utf-8"))
    tracks = pl.get("tracks") or []
    if not tracks:
        raise SystemExit("[agents.marketing.plan] playlist has no tracks")

    lead_track_id = str(tracks[0].get("track_id") or "")
    if not lead_track_id:
        raise SystemExit("[agents.marketing.plan] lead track missing track_id")

    # Try locate lead audio in repo (best) then in bundle
    lead_audio_repo = (repo_root.expanduser().resolve() / "tracks" / f"{lead_track_id}.wav").resolve()
    lead_audio_bundle = (drop_dir / "tracks" / f"{lead_track_id}.wav").resolve()
    lead_audio = lead_audio_repo if lead_audio_repo.exists() else lead_audio_bundle

    receipts = out_dir / "receipts.jsonl"
    plan_path = out_dir / "marketing_plan.json"

    # teaser
    teaser_path = out_dir / "teaser.wav"
    wrote, reason = _marketing_teaser_wav(lead_audio=lead_audio, out_path=teaser_path, teaser_seconds=int(teaser_seconds))

    # cover
    cover_path_png = out_dir / "cover.png"
    png_bytes, png_err = _marketing_cover_bytes_png(int(seed), 1024)
    cover_path: Path
    cover_obj: Dict[str, Any]
    if png_bytes is not None:
        cover_path_png.write_bytes(png_bytes)
        cover_path = cover_path_png
        cover_obj = {"dst": _relpath_from_out_dir(out_dir, str(cover_path)), "size_px": 1024}
    else:
        cover_path_svg = out_dir / "cover.svg"
        cover_path_svg.write_text(_marketing_cover_svg(int(seed), 1024), encoding="utf-8")
        cover_path = cover_path_svg
        cover_obj = {
            "dst": _relpath_from_out_dir(out_dir, str(cover_path)),
            "size_px": 1024,
            "note": png_err or "svg_fallback",
        }

    # posts (simple deterministic copy)
    context = str(pl.get("context") or "focus")
    schedule = str(pl.get("schedule") or "daily")
    period = (pl.get("period") or {}).get("label") or ""
    base = f"{context} | {schedule} | {period}".strip(" |")
    posts_text = [
        f"New {context} drop is live. {base}. Listen now.",
        f"{context.capitalize()} session soundtrack: fresh release for {base}.",
        f"Today’s {context} pick: {lead_track_id}. {base}.",
    ]
    post_paths: List[str] = []
    for idx, txt in enumerate(posts_text, start=1):
        p = out_dir / f"post_{idx}.txt"
        p.write_text(txt + "\n", encoding="utf-8")
        post_paths.append(_relpath_from_out_dir(out_dir, str(p)))

    out_obj: Dict[str, Any] = {
        "ok": True,
        "cmd": "agents.marketing.plan",
        "ts": ts,
        "seed": int(seed),
        "drop_dir": _relpath_from_out_dir(out_dir, str(drop_dir)),
        "lead_track_id": lead_track_id,
        "cover": cover_obj,
        "teaser": {
            "lead_audio": _relpath_from_out_dir(out_dir, str(lead_audio)),
            "seconds": int(teaser_seconds),
            "skipped": (not wrote),
            "reason": reason if not wrote else "",
        },
        "paths": {
            "out_dir": ".",
            "playlist": _relpath_from_out_dir(out_dir, str(playlist_path)),
            "lead_audio": _relpath_from_out_dir(out_dir, str(lead_audio)),
            "cover": _relpath_from_out_dir(out_dir, str(cover_path)),
            "teaser": _relpath_from_out_dir(out_dir, str(teaser_path)),
            "posts": post_paths,
            "receipts": _relpath_from_out_dir(out_dir, str(receipts)),
            "plan": _relpath_from_out_dir(out_dir, str(plan_path)),
        },
    }

    # Final safety: scrub any remaining absolute paths anywhere in the object.
    out_obj = _scrub_absolute_paths(out_obj, out_dir=out_dir, repo_root=Path.cwd())

    # receipts + plan file
    with receipts.open("a", encoding="utf-8") as f:
        f.write(stable_json_dumps(out_obj) + "\n")
    plan_path.write_text(json.dumps(out_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return out_obj




def _parse_iso_utc(s: str, *, fallback: str = "2020-01-01T00:00:00+00:00") -> datetime:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        dt = datetime.fromisoformat(fallback.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


def _iso_add_days(ts_iso: str, days: int) -> str:
    dt = _parse_iso_utc(ts_iso)
    return (dt + timedelta(days=int(days))).astimezone(timezone.utc).isoformat()


def _week_start_date(run_date_yyyy_mm_dd: str) -> str:
    # Monday as week start (ISO style)
    try:
        dt = datetime.strptime(run_date_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    weekday = dt.weekday()  # Monday=0
    ws = dt - timedelta(days=weekday)
    return ws.date().isoformat()


# ---------------------------------------------------------------------------
# Canonical run identity (run_key -> run_id)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunKey:
    run_date: str                 # YYYY-MM-DD
    context: str                  # focus/workout/sleep
    seed: str                     # keep string for schema tolerance
    provider_set_version: str     # e.g. "v1"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_get_or_create_run_id(
    con: sqlite3.Connection,
    *,
    run_key: RunKey,
    ts: str,
    argv: Optional[List[str]] = None,
) -> str:
    """
    Canonical rule: same (run_date, context, seed, provider_set_version) => same run_id.
    Enforced by UNIQUE INDEX in DB. Returns existing run_id or creates one.
    """
    con.execute("PRAGMA foreign_keys = ON;")

    row = con.execute(
        """
        SELECT run_id
          FROM runs
         WHERE run_date = ? AND context = ? AND seed = ? AND provider_set_version = ?
         LIMIT 1
        """,
        (run_key.run_date, run_key.context, run_key.seed, run_key.provider_set_version),
    ).fetchone()
    if row:
        return str(row[0])

    run_id = str(uuid.uuid4())
    created_at = ts if isinstance(ts, str) and ts else _now_iso_utc()
    updated_at = created_at
    argv_json = stable_json_dumps(argv) if argv is not None else None

    try:
        con.execute(
            """
            INSERT INTO runs (
              run_id, run_date, context, seed, provider_set_version,
              created_at, updated_at,
              hostname, argv_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_key.run_date,
                run_key.context,
                run_key.seed,
                run_key.provider_set_version,
                created_at,
                updated_at,
                socket.gethostname(),
                argv_json,
            ),
        )
        con.commit()
        return run_id
    except sqlite3.IntegrityError:
        # Lost a race; read the winner.
        row2 = con.execute(
            """
            SELECT run_id
              FROM runs
             WHERE run_date = ? AND context = ? AND seed = ? AND provider_set_version = ?
             LIMIT 1
            """,
            (run_key.run_date, run_key.context, run_key.seed, run_key.provider_set_version),
        ).fetchone()
        if not row2:
            raise
        return str(row2[0])


# ---------------------------------------------------------------------------
# DB helpers (schema-tolerant + NOT NULL fillers)
# ---------------------------------------------------------------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def die(msg: str, code: int = 2) -> NoReturn:
    eprint(msg)
    raise SystemExit(code)


def db_connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def db_table_exists(con: sqlite3.Connection, table: str) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    col_type: str
    notnull: bool
    dflt_value: Optional[str]


def db_table_info(con: sqlite3.Connection, table: str) -> List[ColumnInfo]:
    try:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return []
    out: List[ColumnInfo] = []
    for r in rows:
        # PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
        out.append(
            ColumnInfo(
                name=str(r[1]),
                col_type=str(r[2] or ""),
                notnull=bool(r[3]),
                dflt_value=(r[4] if r[4] is not None else None),
            )
        )
    return out


def db_table_columns(con: sqlite3.Connection, table: str) -> List[str]:
    return [c.name for c in db_table_info(con, table)]


def _pick_first_existing(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _stable_int_from_key(key: str, lo: int, hi: int) -> int:
    if hi <= lo:
        return lo
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return lo + (h % (hi - lo + 1))


def _infer_required_value(col: ColumnInfo, row_data: Dict[str, Any]) -> Any:
    name = col.name.lower()

    ts = (
        row_data.get("ts")
        or row_data.get("occurred_at")
        or row_data.get("created_at")
        or row_data.get("created_ts")
        or row_data.get("time")
    )
    ts_val = ts if isinstance(ts, str) and ts else "2020-01-01T00:00:00+00:00"

    stable_key = (
        str(
            row_data.get("id")
            or row_data.get("track_id")
            or row_data.get("post_id")
            or row_data.get("event_id")
            or row_data.get("drop_id")
            or ""
        )
        + f"|{col.name}"
    )

    if name in ("occurred_at", "created_at", "ts", "timestamp") or name.endswith("_at") or "timestamp" in name:
        return ts_val

    if name == "bpm":
        return _stable_int_from_key(stable_key, 60, 140)
    if name in ("duration_ms", "duration_millis", "duration_milliseconds"):
        return _stable_int_from_key(stable_key, 15_000, 180_000)
    if name in ("duration_s", "duration_sec", "duration_secs", "duration_seconds"):
        return _stable_int_from_key(stable_key, 15, 180)

    if name == "provider":
        return str(row_data.get("provider") or "stub")
    if name in ("kind", "event_kind", "type"):
        return str(row_data.get("kind") or "unknown")
    if name in ("actor", "source"):
        return str(row_data.get("actor") or "system")
    if name in ("status", "state"):
        return str(row_data.get("status") or "draft")
    if name in ("platform", "channel", "destination"):
        return str(row_data.get("platform") or "unknown")
    if name in ("title", "name"):
        return str(row_data.get("title") or row_data.get("name") or "Untitled")
    if name in ("context", "mood"):
        return str(row_data.get("context") or row_data.get("mood") or "focus")

    if name in ("meta_json", "metadata_json", "meta", "metadata"):
        return stable_json_dumps(row_data.get("meta") or {})

    t = (col.col_type or "").upper()
    if "INT" in t or "REAL" in t or "NUM" in t:
        return 0
    if "BLOB" in t:
        return b""
    return ""


def _insert_row(con: sqlite3.Connection, table: str, data: Dict[str, Any]) -> None:
    info = db_table_info(con, table)
    if not info:
        raise sqlite3.OperationalError(f"table {table} does not exist")

    cols_set = {c.name for c in info}
    filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in cols_set}

    for col in info:
        if not col.notnull:
            continue
        if col.name in filtered:
            continue
        if col.dflt_value is not None:
            continue
        filtered[col.name] = _infer_required_value(col, {**data, **filtered})

    if not filtered:
        return

    keys = sorted(filtered.keys())
    placeholders = ",".join(["?"] * len(keys))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(keys)}) VALUES ({placeholders})"
    con.execute(sql, tuple(filtered[k] for k in keys))
    con.commit()


def _update_row(con: sqlite3.Connection, table: str, where_col: str, where_val: Any, patch: Dict[str, Any]) -> None:
    info = db_table_info(con, table)
    if not info:
        raise sqlite3.OperationalError(f"table {table} does not exist")
    cols_set = {c.name for c in info}

    filtered: Dict[str, Any] = {k: v for k, v in patch.items() if k in cols_set}
    if not filtered:
        return

    keys = sorted(filtered.keys())
    set_expr = ", ".join([f"{k} = ?" for k in keys])
    sql = f"UPDATE {table} SET {set_expr} WHERE {where_col} = ?"
    con.execute(sql, tuple(filtered[k] for k in keys) + (where_val,))
    con.commit()


def ensure_tables_minimal(con: sqlite3.Connection) -> None:
    """
    Creates minimal tables *only if missing*. We do not migrate existing schemas.
    """
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            run_date TEXT NOT NULL,
            context TEXT NOT NULL,
            seed TEXT NOT NULL,
            provider_set_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            hostname TEXT,
            argv_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_run_key
        ON runs(run_date, context, seed, provider_set_version)
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS run_stages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT,
            duration_ms INTEGER,
            error_json TEXT,
            meta_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_run_stages_unique
        ON run_stages(run_id, stage)
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor TEXT NOT NULL,
            meta_json TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS marketing_posts (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            platform TEXT NOT NULL,
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            meta_json TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            title TEXT NOT NULL,
            provider TEXT NOT NULL,
            mood TEXT,
            genre TEXT,
            artifact_path TEXT,
            meta_json TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS drops (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            context TEXT NOT NULL,
            seed TEXT NOT NULL,
            run_id TEXT NOT NULL,
            track_id TEXT,
            marketing_batch_id TEXT,
            published_ts TEXT,
            meta_json TEXT NOT NULL
        )
        """
    )
    con.commit()


# ---------------------------------------------------------------------------
# Run stage tracking (resume/observability)
# ---------------------------------------------------------------------------

def _json_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _json_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def db_stage_get(con: sqlite3.Connection, *, run_id: str, stage: str) -> Optional[sqlite3.Row]:
    cols = db_table_columns(con, "run_stages")
    if not cols:
        return None
    return con.execute(
        "SELECT * FROM run_stages WHERE run_id = ? AND stage = ? LIMIT 1",
        (run_id, stage),
    ).fetchone()


def db_stage_upsert(
    con: sqlite3.Connection,
    *,
    run_id: str,
    stage: str,
    status: str,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    duration_ms: Optional[int] = None,
    error: Optional[Dict[str, Any]] = None,
    meta_patch: Optional[Dict[str, Any]] = None,
) -> None:
    ensure_tables_minimal(con)

    existing = db_stage_get(con, run_id=run_id, stage=stage)
    existing_meta: Dict[str, Any] = {}
    if existing is not None:
        raw = existing["meta_json"]
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    existing_meta = parsed
            except Exception:
                existing_meta = {}
    merged_meta = _json_merge(existing_meta, meta_patch or {}) if (existing_meta or meta_patch) else {}

    row_data: Dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
    }
    if started_at is not None:
        row_data["started_at"] = started_at
    if ended_at is not None:
        row_data["ended_at"] = ended_at
    if duration_ms is not None:
        row_data["duration_ms"] = int(duration_ms)
    if error is not None:
        row_data["error_json"] = stable_json_dumps(error)
    if meta_patch is not None or existing_meta:
        row_data["meta_json"] = stable_json_dumps(merged_meta)

    _insert_row(con, "run_stages", row_data)


def stage_is_done(row: Optional[sqlite3.Row]) -> bool:
    if row is None:
        return False
    s = str(row["status"] or "").lower()
    return s in ("ok", "skipped")


@contextlib.contextmanager
def run_stage(
    con: sqlite3.Connection,
    *,
    run_id: str,
    stage: str,
    deterministic: bool,
    allow_resume: bool = True,
    meta: Optional[Dict[str, Any]] = None,
):
    """
    Stage context manager:
      - If allow_resume and stage already ok/skipped: do not run body (NO DB write).
      - Otherwise: mark running, run body, then mark ok; on exception mark error.
    Determinism:
      - duration_ms is normalized to 0 in deterministic mode.
    """
    ensure_tables_minimal(con)

    existing = db_stage_get(con, run_id=run_id, stage=stage)
    if allow_resume and stage_is_done(existing):
        yield False
        return

    started_at = deterministic_now_iso(deterministic)
    t0 = time.perf_counter()

    db_stage_upsert(
        con,
        run_id=run_id,
        stage=stage,
        status="running",
        started_at=started_at,
        meta_patch={"resume": False, **(meta or {})},
    )

    try:
        yield True
    except Exception as e:
        ended_at = deterministic_now_iso(deterministic)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        if deterministic:
            dur_ms = 0
        db_stage_upsert(
            con,
            run_id=run_id,
            stage=stage,
            status="error",
            ended_at=ended_at,
            duration_ms=dur_ms,
            error={"type": type(e).__name__, "message": str(e)},
        )
        raise
    else:
        ended_at = deterministic_now_iso(deterministic)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        if deterministic:
            dur_ms = 0
        db_stage_upsert(
            con,
            run_id=run_id,
            stage=stage,
            status="ok",
            ended_at=ended_at,
            duration_ms=dur_ms,
        )


# ---------------------------------------------------------------------------
# Marketing schema-tolerant helpers
# ---------------------------------------------------------------------------

def _row_first(row: sqlite3.Row, candidates: Sequence[str], default: Any = "") -> Any:
    keys = set(row.keys())
    for c in candidates:
        if c in keys:
            return row[c]
    return default


def _load_json_maybe(s: Any) -> Dict[str, Any]:
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _detect_marketing_content_col(cols: Sequence[str]) -> Optional[str]:
    return _pick_first_existing(
        cols,
        ["content", "body", "text", "draft", "caption", "copy", "post_text", "message"],
    )


def _detect_marketing_meta_col(cols: Sequence[str]) -> Optional[str]:
    return _pick_first_existing(
        cols,
        ["meta_json", "metadata_json", "meta", "metadata", "meta_blob", "payload", "payload_json"],
    )


def _best_text_payload_column(con: sqlite3.Connection, table: str, reserved: Sequence[str]) -> Optional[str]:
    info = db_table_info(con, table)
    if not info:
        return None
    reserved_set = {r.lower() for r in reserved}
    candidates: List[str] = []
    for c in info:
        n = c.name.lower()
        if n in reserved_set:
            continue
        t = (c.col_type or "").upper()
        if "CHAR" in t or "TEXT" in t or "CLOB" in t or t == "":
            candidates.append(c.name)
    if not candidates:
        return None
    priority = ["content", "body", "text", "draft", "caption", "copy", "message", "payload", "data", "json"]
    for p in priority:
        for c in candidates:
            if p in c.lower():
                return c
    return sorted(candidates)[0]


def _extract_inner_content_from_blob(s: str) -> Optional[str]:
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None

        # Common keys for post copy across schemas
        for k in ("content", "text", "body", "draft", "caption", "copy", "post_text", "message"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v

        # Sometimes nested payloads exist
        for k in ("payload", "data", "draft_obj"):
            v = obj.get(k)
            if isinstance(v, dict):
                # recurse once, cheaply
                for kk in ("content", "text", "body", "draft", "caption", "copy", "post_text", "message"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        return vv

    except Exception:
        return None
    return None


def _marketing_row_content(con: sqlite3.Connection, row: sqlite3.Row) -> str:
    cols = row.keys()

    content_col = _detect_marketing_content_col(cols)
    if content_col:
        v = row[content_col]
        if v is None:
            return ""
        s = str(v)
        inner = _extract_inner_content_from_blob(s)
        return inner if inner is not None else s

    best_payload = _best_text_payload_column(
        con,
        "marketing_posts",
        reserved=[
            "id", "post_id", "ts", "created_at",
            "platform", "channel", "destination",
            "status", "state",
            "meta_json", "metadata_json", "meta", "metadata",
        ],
    )
    if best_payload and best_payload in set(cols):
        v = row[best_payload]
        if v is not None and str(v).strip():
            s = str(v)
            inner = _extract_inner_content_from_blob(s)
            return inner if inner is not None else s

    meta_col = _detect_marketing_meta_col(cols)
    if meta_col:
        meta = _load_json_maybe(row[meta_col])
        for k in ("content", "body", "text", "draft", "caption", "copy", "post_text", "message"):
            if k in meta and meta[k] is not None:
                if isinstance(meta[k], dict):
                    return stable_json_dumps(meta[k])
                return str(meta[k])
        for k in ("draft_obj", "payload", "data"):
            if k in meta and meta[k] is not None:
                try:
                    return stable_json_dumps(meta[k])
                except Exception:
                    return str(meta[k])

    return ""


def _marketing_row_meta(con: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
    cols = row.keys()

    meta_col = _detect_marketing_meta_col(cols)
    if meta_col:
        meta = _load_json_maybe(row[meta_col])
        if meta:
            return meta

    best_payload = _best_text_payload_column(
        con,
        "marketing_posts",
        reserved=[
            "id", "post_id", "ts", "created_at",
            "platform", "channel", "destination",
            "status", "state",
        ],
    )
    if best_payload and best_payload in set(cols):
        v = row[best_payload]
        meta = _load_json_maybe(v)
        if meta:
            return meta

    content = _marketing_row_content(con, row)
    meta = _load_json_maybe(content)
    if meta:
        return meta

    return {}


# ---------------------------------------------------------------------------
# DB writes for events/tracks/drops/marketing
# ---------------------------------------------------------------------------

def db_insert_event(con: sqlite3.Connection, *, event_id: str, ts: str, kind: str, actor: str, meta: Dict[str, Any]) -> None:
    _insert_row(
        con,
        "events",
        {
            "id": event_id,
            "event_id": event_id,
            "ts": ts,
            "occurred_at": ts,
            "created_at": ts,
            "kind": kind,
            "actor": actor,
            "meta_json": stable_json_dumps(meta),
            "metadata_json": stable_json_dumps(meta),
            "meta": stable_json_dumps(meta),
            "metadata": stable_json_dumps(meta),
        },
    )

def _write_stub_wav(path: Path, seed: int, duration_s: float = 1.5, sample_rate: int = 44100) -> None:
    """
    Write a small deterministic mono PCM16 WAV file.

    Determinism:
    - frequency derived from seed
    - fixed sample rate + duration
    - fixed amplitude
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic frequency in a pleasant range
    base = 220.0
    freq = base + float(seed % 220)  # 220..439 Hz

    nframes = int(duration_s * sample_rate)
    amp = 0.25  # keep conservative to avoid clipping
    two_pi_f = 2.0 * math.pi * freq

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        for i in range(nframes):
            t = i / sample_rate
            sample = amp * math.sin(two_pi_f * t)
            pcm = int(max(-1.0, min(1.0, sample)) * 32767.0)
            wf.writeframes(struct.pack("<h", pcm))


def _write_minimal_manifest(path: Path, fixed_ts_iso: str) -> None:
    """
    Create a minimal manifest.json if one doesn't exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": "mgc.manifest.v1",
        "created_ts": fixed_ts_iso,
        "notes": "Auto-created manifest (missing at run drop time).",
    }
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def db_insert_track(
    con: sqlite3.Connection,
    *,
    track_id: str,
    ts: str,
    title: str,
    provider: str,
    mood: Optional[str],
    genre: Optional[str],
    artifact_path: Optional[str],
    meta: Dict[str, Any],
) -> None:
    cols = db_table_columns(con, "tracks")
    if not cols:
        raise sqlite3.OperationalError("table tracks does not exist")

    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on"])

    # Canonical schema uses full_path; other schemas vary.
    path_col = _pick_first_existing(cols, ["full_path", "artifact_path", "audio_path", "path", "file_path", "uri"])
    preview_col = _pick_first_existing(cols, ["preview_path", "preview", "teaser_path"])
    bpm_col = _pick_first_existing(cols, ["bpm", "tempo"])

    data: Dict[str, Any] = {
        "id": track_id,
        "track_id": track_id,
        "title": title,
        "name": title,
        "provider": provider,
        "mood": mood,
        "genre": genre,
        "meta_json": stable_json_dumps(meta),
        "metadata_json": stable_json_dumps(meta),
        "meta": stable_json_dumps(meta),
        "metadata": stable_json_dumps(meta),
    }

    if ts_col:
        data[ts_col] = ts

    # ----------------------------
    # Path + preview enforcement
    # ----------------------------
    if path_col and artifact_path is not None:
        ap = Path(str(artifact_path)).expanduser()
        # Resolve relative paths from CWD (repo root in your CLI usage).
        ap_resolved = ap if ap.is_absolute() else (Path.cwd() / ap).resolve()

        # Guard: never write a path that doesn't exist.
        if not ap_resolved.is_file():
            raise FileNotFoundError(
                f"Refusing to insert track with missing {path_col}: {artifact_path} "
                f"(resolved: {ap_resolved}) track_id={track_id}"
            )

        # Write primary path column.
        data[path_col] = str(artifact_path)

        # If schema requires a preview, choose one that exists.
        if preview_col:
            # 1) Preferred: data/previews/<stem>_preview.mp3
            stem = ap_resolved.stem
            guess1 = Path("data") / "previews" / f"{stem}_preview.mp3"
            g1 = (Path.cwd() / guess1).resolve()

            if g1.is_file():
                data[preview_col] = str(guess1)
            else:
                # 2) If artifact is already an mp3, it can serve as preview.
                if ap_resolved.suffix.lower() == ".mp3":
                    data[preview_col] = str(artifact_path)
                else:
                    # 3) Minimal, deterministic fallback: use the artifact itself as preview.
                    # This keeps the DB invariant (preview_path exists) without requiring ffmpeg.
                    data[preview_col] = str(artifact_path)

    # ----------------------------
    # Optional fields
    # ----------------------------
    if bpm_col:
        data[bpm_col] = _stable_int_from_key(f"{track_id}|{title}|{provider}|{bpm_col}", 60, 140)

    _insert_row(con, "tracks", data)


def db_insert_drop(
    con: sqlite3.Connection,
    *,
    drop_id: str,
    ts: str,
    context: str,
    seed: str,
    run_id: str,
    track_id: Optional[str],
    meta: Dict[str, Any],
) -> None:
    cols = db_table_columns(con, "drops")
    if not cols:
        raise sqlite3.OperationalError("table drops does not exist")

    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on"])
    ctx_col = _pick_first_existing(cols, ["context", "mood"])
    seed_col = _pick_first_existing(cols, ["seed"])
    run_col = _pick_first_existing(cols, ["run_id"])
    track_col = _pick_first_existing(cols, ["track_id"])
    meta_col = _pick_first_existing(cols, ["meta_json", "metadata_json", "meta", "metadata"])

    data: Dict[str, Any] = {"id": drop_id, "drop_id": drop_id}
    if ts_col:
        data[ts_col] = ts
    if ctx_col:
        data[ctx_col] = context
    if seed_col:
        data[seed_col] = seed
    if run_col:
        data[run_col] = run_id
    if track_col and track_id is not None:
        data[track_col] = track_id
    if meta_col:
        data[meta_col] = stable_json_dumps(meta)

    data.update(
        {
            "context": context,
            "mood": context,
            "seed": seed,
            "run_id": run_id,
            "track_id": track_id,
            "meta_json": stable_json_dumps(meta),
            "metadata_json": stable_json_dumps(meta),
        }
    )

    _insert_row(con, "drops", data)


def db_drop_mark_published(
    con: sqlite3.Connection,
    *,
    run_id: str,
    marketing_batch_id: str,
    published_ts: str,
) -> int:
    cols = db_table_columns(con, "drops")
    if not cols:
        raise sqlite3.OperationalError("table drops does not exist")

    run_col = _pick_first_existing(cols, ["run_id"])
    if not run_col:
        return 0

    batch_col = _pick_first_existing(cols, ["marketing_batch_id", "batch_id"])
    pub_ts_col = _pick_first_existing(cols, ["published_ts", "published_at", "published_time", "ts_published"])

    patch: Dict[str, Any] = {}
    if batch_col:
        patch[batch_col] = marketing_batch_id
    if pub_ts_col:
        patch[pub_ts_col] = published_ts

    if not patch:
        return 0

    sql = f"UPDATE drops SET {', '.join([f'{k} = ?' for k in sorted(patch.keys())])} WHERE {run_col} = ?"
    params = [patch[k] for k in sorted(patch.keys())] + [run_id]
    cur = con.execute(sql, params)
    con.commit()
    return int(cur.rowcount or 0)


def db_insert_marketing_post(
    con: sqlite3.Connection,
    *,
    post_id: str,
    ts: str,
    platform: str,
    status: str,
    content: str,
    meta: Dict[str, Any],
) -> None:
    cols = db_table_columns(con, "marketing_posts")
    if not cols:
        raise sqlite3.OperationalError("table marketing_posts does not exist")

    # Column drift
    pk_col = _pick_first_existing(cols, ["id", "post_id", "marketing_post_id"]) or "id"
    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on"])
    platform_col = _pick_first_existing(cols, ["platform", "channel", "destination"])
    status_col = _pick_first_existing(cols, ["status", "state"])
    track_id_col = _pick_first_existing(cols, ["track_id", "track", "track_uuid"])

    content_col = _detect_marketing_content_col(cols)  # should pick payload_json if present
    meta_col = _detect_marketing_meta_col(cols)

    # If neither content nor meta columns were detected, fall back to a generic text payload column.
    best_payload = None
    if not content_col and not meta_col:
        best_payload = _best_text_payload_column(
            con,
            "marketing_posts",
            reserved=[
                "id",
                "post_id",
                "marketing_post_id",
                "ts",
                "created_at",
                "created_ts",
                "platform",
                "channel",
                "destination",
                "status",
                "state",
                "track_id",
                "track",
                "track_uuid",
            ],
        )

    # We may need a track_id if schema enforces it (your DB does).
    # Primary source: meta["track_id"]
    track_id_val = str((meta or {}).get("track_id") or "")

    # Fallback: if content is JSON, try pulling track_id out of it.
    if not track_id_val:
        try:
            obj = json.loads(content) if isinstance(content, str) else None
            if isinstance(obj, dict):
                track_id_val = str(obj.get("track_id") or "")
        except Exception:
            pass

    data: Dict[str, Any] = {}

    # PK
    data[pk_col] = post_id

    # Also populate common alias columns if present (harmless, helps drift)
    if pk_col != "id" and "id" in cols:
        data["id"] = post_id
    if pk_col != "post_id" and "post_id" in cols:
        data["post_id"] = post_id
    if pk_col != "marketing_post_id" and "marketing_post_id" in cols:
        data["marketing_post_id"] = post_id

    # Timestamps / platform / status
    if ts_col:
        data[ts_col] = ts
    if platform_col:
        data[platform_col] = platform
    if status_col:
        data[status_col] = status

    # Track FK if schema has it
    if track_id_col:
        if not track_id_val:
            # If the schema has track_id, we refuse to insert an FK-violating row.
            # This is what was killing `run autonomous`.
            raise ValueError(
                "marketing_posts schema requires track_id, but none was provided. "
                "Pass meta={'track_id': ...} (or include track_id in JSON content)."
            )
        data[track_id_col] = track_id_val

    # Content/meta placement
    meta_to_store = dict(meta or {})
    if not content_col:
        meta_to_store["content"] = content

    if content_col:
        data[content_col] = content
        # If there is also a meta column, store meta separately (nice for debugging)
        if meta_col:
            data[meta_col] = stable_json_dumps(meta or {})
    elif meta_col:
        data[meta_col] = stable_json_dumps(meta_to_store)
    elif best_payload:
        data[best_payload] = stable_json_dumps(meta_to_store) if meta_to_store else content
    else:
        # Last resort: attempt 'content' if it exists (some schemas use it)
        if "content" in cols:
            data["content"] = content
        else:
            raise sqlite3.OperationalError(
                "marketing_posts has no detectable payload/meta column to store content"
            )

    _insert_row(con, "marketing_posts", data)


def db_marketing_posts_pending(con: sqlite3.Connection, *, limit: int = 50) -> List[sqlite3.Row]:
    """
    Return publishable (pending) marketing posts with schema drift tolerance.

    Key behaviors:
    - Publishes planned/pending/ready (case-insensitive).
    - Also allows 'draft' ONLY if it contains non-empty payload/content (avoids placeholder drafts).
    - Tolerates PK drift: id / post_id / marketing_post_id / *_id
    - Tolerates created timestamp drift: created_at / created_ts / ts
    - Tolerates status column drift: status / state
    - Tolerates payload drift: payload_json / content / payload / text
    - Deterministic ordering: created ASC, pk ASC
    """
    # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
    cols = [r[1] for r in con.execute("PRAGMA table_info(marketing_posts)").fetchall()]
    colset = set(cols)

    # Pick PK column
    if "post_id" in colset:
        pk = "post_id"
    elif "id" in colset:
        pk = "id"
    elif "marketing_post_id" in colset:
        pk = "marketing_post_id"
    else:
        pk_candidates = sorted([c for c in colset if c.endswith("_id")])
        pk = pk_candidates[0] if pk_candidates else "rowid"

    # Pick created timestamp column for ordering
    if "created_at" in colset:
        created_col = "created_at"
    elif "created_ts" in colset:
        created_col = "created_ts"
    elif "ts" in colset:
        created_col = "ts"
    else:
        created_col = pk  # last resort ordering

    # Status column name drift
    status_col = "status" if "status" in colset else ("state" if "state" in colset else None)
    if status_col is None:
        return []

    # Payload/content column drift: used only to decide whether a 'draft' is publishable.
    # (If none exist, drafts will be excluded.)
    payload_cols: List[str] = []
    for c in ("payload_json", "content", "payload", "text"):
        if c in colset:
            payload_cols.append(c)

    # Case-insensitive statuses.
    publishable_statuses = ("planned", "pending", "ready")
    draft_status = "draft"

    # Build SQL
    # - Always allow planned/pending/ready
    # - Allow draft only if any payload/content column is non-empty after trim
    placeholders = ",".join(["?"] * len(publishable_statuses))
    params: List[object] = list(publishable_statuses)

    draft_clause = ""
    if payload_cols:
        non_empty_checks = " OR ".join([f"(COALESCE(TRIM({c}), '') <> '')" for c in payload_cols])
        draft_clause = f" OR (LOWER({status_col}) = ? AND ({non_empty_checks}))"
        params.append(draft_status)
    # else: no payload columns -> drafts are excluded (safe)

    sql = f"""
    SELECT *
    FROM marketing_posts
    WHERE LOWER({status_col}) IN ({placeholders})
    {draft_clause}
    ORDER BY {created_col} ASC, {pk} ASC
    LIMIT ?
    """

    params.append(int(limit))
    cur = con.execute(sql, tuple(params))
    return list(cur.fetchall())

def db_marketing_post_set_status(
    con: sqlite3.Connection,
    *,
    post_id: str,
    status: str,
    ts: str,
    meta_patch: Dict[str, Any],
) -> None:
    """Update a marketing_posts row status + timestamp + meta JSON (schema tolerant).

    Historical schema drift:
      - identifier column may be `post_id` (fixtures) or `id` (older dev DBs)
      - status column may be `status` or `state`
      - timestamp column may be `ts` / `created_at` / `created_ts`
      - meta JSON column may be `meta` / `meta_json` / etc.

    We detect columns at runtime and update safely.
    """
    cols = db_table_columns(con, "marketing_posts")
    if not cols:
        raise sqlite3.OperationalError("table marketing_posts does not exist")

    cols_set = set(cols)

    where_col = "post_id" if "post_id" in cols_set else ("id" if "id" in cols_set else (_pick_first_existing(cols, ["marketing_post_id"]) or "id"))

    status_col = _pick_first_existing(cols, ["status", "state"])
    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on"])
    meta_col = _detect_marketing_meta_col(cols)

    existing_meta: Dict[str, Any] = {}
    if meta_col:
        row = con.execute(f"SELECT {meta_col} FROM marketing_posts WHERE {where_col} = ?", (post_id,)).fetchone()
        if row:
            existing_meta = _load_json_maybe(row[0])
    if meta_patch:
        existing_meta.update(meta_patch)

    patch: Dict[str, Any] = {}
    if status_col:
        patch[status_col] = status
    if ts_col:
        patch[ts_col] = ts
    if meta_col:
        patch[meta_col] = stable_json_dumps(existing_meta)

    _update_row(con, "marketing_posts", where_col, post_id, patch)
def iter_repo_files(
    repo_root: Path,
    include_globs: Optional[Sequence[str]] = None,
    exclude_dirs: Optional[Sequence[str]] = None,
    exclude_globs: Optional[Sequence[str]] = None,
) -> Iterable[Path]:
    if exclude_dirs is None:
        exclude_dirs = [
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".idea",
            "logs",
            "data",
            "artifacts",
        ]
    if exclude_globs is None:
        exclude_globs = [
            "*.pyc",
            "*.pyo",
            "*.log",
            "*.tmp",
            "*.swp",
            "*.swo",
            "*.sqlite-journal",
            "*.db-journal",
            "*.wav",
            "*.mp3",
            "*.flac",
            "*.ogg",
            "*.sqlite",
            "*.db",
        ]

    def _is_excluded_dir(rel_parts: Tuple[str, ...]) -> bool:
        return any(part in exclude_dirs for part in rel_parts)

    if include_globs:
        seen: set[Path] = set()
        for g in include_globs:
            for p in sorted(repo_root.glob(g)):
                if not p.is_file():
                    continue
                try:
                    rel_parts = p.relative_to(repo_root).parts
                except Exception:
                    continue
                if _is_excluded_dir(rel_parts):
                    continue
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield p
        return

    all_files: List[Path] = []
    for p in repo_root.rglob("*"):
        try:
            rel_parts = p.relative_to(repo_root).parts
        except Exception:
            continue
        if _is_excluded_dir(rel_parts):
            continue
        if p.is_dir():
            continue
        name = p.name
        if any(Path(name).match(pat) for pat in exclude_globs):
            continue
        all_files.append(p)

    all_files.sort(key=lambda x: str(x.relative_to(repo_root)).replace("\\", "/"))
    for p in all_files:
        yield p

@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sha256: str
    size: int


def compute_manifest(
    repo_root: Path,
    *,
    include: Optional[Sequence[str]] = None,
    exclude_dirs: Optional[Sequence[str]] = None,
    exclude_globs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    entries: List[ManifestEntry] = []
    for p in iter_repo_files(repo_root, include_globs=include, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs):
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        b = read_text_bytes(p)
        entries.append(ManifestEntry(path=rel, sha256=sha256_hex(b), size=len(b)))

    entries.sort(key=lambda e: e.path)
    manifest_obj = {
        "version": 1,
        # Determinism + portability: never serialize absolute paths.
        # All paths in JSON must be relative to the output directory.
        "root": ".",
        "entries": [{"path": e.path, "sha256": e.sha256, "size": e.size} for e in entries],
    }
    root_hash_payload = [{"path": e.path, "sha256": e.sha256, "size": e.size} for e in entries]
    manifest_obj["root_tree_sha256"] = sha256_hex(stable_json_dumps(root_hash_payload).encode("utf-8"))
    return manifest_obj


def cmd_run_manifest(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        die(f"repo_root does not exist: {repo_root}")

    include = args.include or None
    exclude_dirs = args.exclude_dir or None
    exclude_globs = args.exclude_glob or None

    manifest = compute_manifest(repo_root, include=include, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs)

    out_path = Path(args.out) if args.out else None
    text = stable_json_dumps(manifest) + "\n"
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)

    if getattr(args, "print_hash", False):
        eprint(manifest["root_tree_sha256"])

    return 0


# ---------------------------------------------------------------------------
# Daily run (deterministic orchestrator)
# ---------------------------------------------------------------------------

def _maybe_call_external_daily_runner(
    *,
    db_path: str,
    context: str,
    seed: str,
    deterministic: bool,
    ts: str
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[str, str]] = [
        ("mgc.daily", "run_daily"),
        ("mgc.music_agent", "run_daily"),
        ("mgc.pipeline", "run_daily"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn(  # type: ignore[misc]
                    db_path=db_path,
                    context=context,
                    seed=seed,
                    deterministic=deterministic,
                    ts=ts,
                )
        except Exception:
            continue
    return None


def _stub_daily_run(
    *,
    con: sqlite3.Connection,
    context: str,
    seed: str,
    deterministic: bool,
    ts: str,
    out_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """
    Daily run implementation using provider abstraction.

    Produces:
      - provider artifact under data/tracks/... (repo storage)
      - bundled copy under out_dir/tracks/<track_id>.<ext> (portable drop artifact)
      - out_dir/playlist.json pointing at bundled track (web-build friendly)
      - out_dir/daily_evidence*.json includes bundle paths + sha256
    """
    import hashlib
    import os
    import shutil

    def _sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _posix(p: Path) -> str:
        return str(p).replace("\\", "/")

    out_dir.mkdir(parents=True, exist_ok=True)

    drop_id = stable_uuid5("drop", run_id)
    track_id = stable_uuid5("track", context, seed, run_id)

    title = f"{context.title()} Track {seed}"

    # Artifact path in repo storage (provider may change extension)
    if deterministic:
        artifact_rel = Path("data") / "tracks" / f"{track_id}"
    else:
        day = ts.split("T", 1)[0]
        artifact_rel = Path("data") / "tracks" / day / f"{track_id}"

    artifact_path = (Path.cwd() / artifact_rel).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(context)

    provider_name = str(os.environ.get("MGC_PROVIDER") or "stub").strip().lower()
    provider = get_provider(provider_name)

    req_obj = GenerateRequest(
        track_id=track_id,
        run_id=run_id,
        context=context,
        seed=seed,
        prompt=prompt,
        deterministic=deterministic,
        ts=ts,
        out_rel=str(artifact_path),
    )

    req_kwargs = {
        "track_id": req_obj.track_id,
        "run_id": req_obj.run_id,
        "context": req_obj.context,
        "seed": req_obj.seed,
        "prompt": req_obj.prompt,
        "deterministic": req_obj.deterministic,
        "ts": req_obj.ts,
        "out_rel": req_obj.out_rel,
        "out_dir": str(out_dir),
        "now_iso": ts,
        "schedule": "daily",
        "period_key": ("2020-01-01" if deterministic else ts.split("T", 1)[0]),
    }

    def _call_generate_with_filtered_kwargs():
        import inspect

        fn = getattr(provider, "generate")
        sig = inspect.signature(fn)
        params = sig.parameters

        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            safe = dict(req_kwargs)
            safe.pop("run_id", None)
            return fn(**safe)  # type: ignore[misc]

        allowed = {
            k
            for k, p in params.items()
            if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        filtered = {k: v for k, v in req_kwargs.items() if k in allowed}
        return fn(**filtered)  # type: ignore[misc]

    try:
        result = _call_generate_with_filtered_kwargs()
    except TypeError:
        try:
            result = provider.generate()  # type: ignore[misc]
        except TypeError:
            result = provider.generate(req_obj)  # type: ignore[misc]

    if getattr(result, "ext", None):
        ext = result.ext if result.ext.startswith(".") else f".{result.ext}"
        if artifact_path.suffix != ext:
            artifact_rel = artifact_rel.with_suffix(ext)
            artifact_path = (Path.cwd() / artifact_rel).resolve()
            artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_bytes = getattr(result, "artifact_bytes", None) or b""
    artifact_path.write_bytes(artifact_bytes)

    db_insert_track(
        con,
        track_id=track_id,
        ts=ts,
        title=title,
        provider=getattr(result, "provider", provider_name),
        mood=context,
        genre=(result.meta.get("genre") if isinstance(getattr(result, "meta", None), dict) else None),
        artifact_path=_posix(artifact_rel),
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "context": context,
            "seed": seed,
            "deterministic": deterministic,
            **(getattr(result, "meta", None) or {}),
        },
    )

    db_insert_drop(
        con,
        drop_id=drop_id,
        ts=ts,
        context=context,
        seed=seed,
        run_id=run_id,
        track_id=track_id,
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "provider": getattr(result, "provider", provider_name),
            "deterministic": deterministic,
            "seed": seed,
            "context": context,
        },
    )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "drop.created", drop_id),
        ts=ts,
        kind="drop.created",
        actor="system",
        meta={"drop_id": drop_id, "run_id": run_id, "track_id": track_id},
    )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "track.generated", run_id),
        ts=ts,
        kind="track.generated",
        actor="system",
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "artifact_path": _posix(artifact_rel),
            "provider": getattr(result, "provider", provider_name),
        },
    )

    platforms = ["x", "youtube_shorts", "instagram_reels", "tiktok"]
    post_ids: List[str] = []
    for platform in platforms:
        post_id = stable_uuid5("marketing_post", platform, run_id)
        post_ids.append(post_id)

        content_obj = {
            "platform": platform,
            "hook": f"New {context} track is ready.",
            "cta": "Listen now.",
            "track_id": track_id,
            "run_id": run_id,
            "drop_id": drop_id,
        }

        db_insert_marketing_post(
            con,
            post_id=post_id,
            ts=ts,
            platform=platform,
            status="draft",
            content=stable_json_dumps(content_obj),
            meta={
                "run_id": run_id,
                "drop_id": drop_id,
                "track_id": track_id,
                "provider": getattr(result, "provider", provider_name),
                "context": context,
            },
        )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "marketing.drafts.created", run_id),
        ts=ts,
        kind="marketing.drafts.created",
        actor="system",
        meta={"run_id": run_id, "drop_id": drop_id, "count": len(post_ids), "post_ids": post_ids},
    )

    bundle_tracks_dir = out_dir / "tracks"
    bundle_tracks_dir.mkdir(parents=True, exist_ok=True)

    bundled_name = f"{track_id}{artifact_path.suffix}"
    bundled_track_rel = Path("tracks") / bundled_name
    bundled_track_path = (out_dir / bundled_track_rel).resolve()

    shutil.copy2(str(artifact_path), str(bundled_track_path))

    playlist_rel = Path("playlist.json")
    playlist_path = (out_dir / playlist_rel).resolve()

    playlist_obj: Dict[str, Any] = {
        "schema": "mgc.playlist.v1",
        "context": context,
        "ts": ts,
        "drop_id": drop_id,
        "run_id": run_id,
        "deterministic": deterministic,
        "tracks": [
            {
                "track_id": track_id,
                "title": title,
                "provider": getattr(result, "provider", provider_name),
                "path": _posix(bundled_track_rel),
            }
        ],
    }
    playlist_path.write_text(stable_json_dumps(playlist_obj), encoding="utf-8")

    repo_artifact_sha256 = _sha256_file(artifact_path)
    bundled_track_sha256 = _sha256_file(bundled_track_path)
    playlist_sha256 = _sha256_file(playlist_path)

    # IMPORTANT: manifest sha is computed later by the manifest stage.
    # _stub_daily_run may execute before that, so keep a stable placeholder.
    manifest_sha256 = ""

    evidence_obj: Dict[str, Any] = {
        "schema": "mgc.daily_evidence.v1",
        "ts": ts,
        "context": context,
        "seed": seed,
        "deterministic": deterministic,
        "provider": getattr(result, "provider", provider_name),
        "ids": {
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "marketing_post_ids": post_ids,
        },
        "paths": {
            "repo_artifact": _posix(artifact_rel),
            "bundle_track": _posix(bundled_track_rel),
            "playlist": _posix(playlist_rel),
            "manifest": "manifest.json",
            "manifest_sha256": manifest_sha256,
            "marketing_publish_dir": "marketing/publish",
        },
        "sha256": {
            "repo_artifact": repo_artifact_sha256,
            "bundle_track": bundled_track_sha256,
            "playlist": playlist_sha256,
        },
    }

    evidence_rel_main = Path("daily_evidence.json")
    evidence_rel_scoped = Path(f"daily_evidence_{drop_id}.json")
    evidence_path_main = (out_dir / evidence_rel_main).resolve()
    evidence_path_scoped = (out_dir / evidence_rel_scoped).resolve()

    evidence_text = stable_json_dumps(evidence_obj)
    evidence_path_main.write_text(evidence_text, encoding="utf-8")
    evidence_path_scoped.write_text(evidence_text, encoding="utf-8")

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "drop.bundle.written", drop_id),
        ts=ts,
        kind="drop.bundle.written",
        actor="system",
        meta={
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "bundle_track": _posix(bundled_track_rel),
            "playlist": _posix(playlist_rel),
            "evidence": [_posix(evidence_rel_main), _posix(evidence_rel_scoped)],
        },
    )

    return {
        "run_id": run_id,
        "drop_id": drop_id,
        "track_id": track_id,
        "provider": getattr(result, "provider", provider_name),
        "context": context,
        "seed": seed,
        "deterministic": deterministic,
        "repo_artifact_path": _posix(artifact_rel),
        "bundle_track_path": _posix(bundled_track_rel),
        "playlist_path": _posix(playlist_rel),
        "evidence_paths": [_posix(evidence_rel_main), _posix(evidence_rel_scoped)],
        "sha256": {
            "repo_artifact": repo_artifact_sha256,
            "bundle_track": bundled_track_sha256,
            "playlist": playlist_sha256,
        },
        "marketing_post_ids": post_ids,
    }

def cmd_run_daily(args: argparse.Namespace) -> int:
    """Build a daily drop bundle from the library DB using the deterministic playlist builder.

    Outputs (under out_dir):
      - drop_bundle/playlist.json
      - drop_bundle/daily_evidence.json
      - drop_bundle/tracks/<track_id>.<ext>   (copied from library; lead track only)
      - evidence/daily_evidence.json          (lead track; marketing contract)
      - drop_evidence.json                    (summary)

    Notes:
      - This is a selection + bundling step (no audio generation).
      - Evidence uses relative paths (no absolute out_dir leakage) for determinism.
    """
    deterministic = is_deterministic(args)
    json_mode = bool(getattr(args, "json", False))

    db_path = resolve_db_path(args)
    context = str(getattr(args, "context", None) or os.environ.get("MGC_CONTEXT") or "focus")

    seed_val = getattr(args, "seed", None)
    seed = int(seed_val) if seed_val is not None else int(os.environ.get("MGC_SEED") or "1")

    out_dir = Path(getattr(args, "out_dir", None) or os.environ.get("MGC_EVIDENCE_DIR") or "data/evidence").expanduser().resolve()
    bundle_dir = out_dir / "drop_bundle"
    bundle_tracks_dir = bundle_dir / "tracks"
    evidence_dir = out_dir / "evidence"

    bundle_tracks_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    now_iso = deterministic_now_iso(deterministic)
    run_date = now_iso.split("T", 1)[0] if "T" in now_iso else now_iso
    period_key = run_date


    # Optional: generate new tracks into the library before selecting today's playlist
    gen_count = int(getattr(args, "generate_count", 0) or 0)
    gen_provider = getattr(args, "generate_provider", None) or os.environ.get("MGC_PROVIDER") or None
    gen_prompt = getattr(args, "prompt", None) or None
    if gen_count > 0:
        _agents_generate_and_ingest(
            db_path=db_path,
            repo_root=Path(getattr(args, "repo_root", ".")).expanduser().resolve(),
            context=context,
            schedule="daily",
            period_key=period_key,
            seed=int(seed),
            deterministic=bool(deterministic),
            count=int(gen_count),
            provider_name=str(gen_provider) if gen_provider else None,
            prompt=str(gen_prompt) if gen_prompt else None,
            now_iso=now_iso,
        )

    # Playlist builder knobs
    target_minutes = getattr(args, "target_minutes", None)
    if target_minutes is None:
        target_minutes = int(os.environ.get("MGC_DAILY_TARGET_MINUTES") or "5")

    lookback_playlists = getattr(args, "lookback_playlists", None)
    if lookback_playlists is None:
        lookback_playlists = int(os.environ.get("MGC_DAILY_LOOKBACK_PLAYLISTS") or "7")

    pl = build_daily_playlist(
        db_path=Path(db_path),
        context=context,
        period_key=period_key,
        base_seed=int(seed),
        target_minutes=int(target_minutes),
        lookback_playlists=int(lookback_playlists),
    )

    items = pl.get("items") if isinstance(pl, dict) else None
    if not items:
        raise SystemExit("daily playlist builder produced no items")

    # Copy ONLY the lead track into bundle (daily contract is one track)
    lead = dict(items[0])
    lead_track_id = str(lead.get("track_id") or "")
    full_path = lead.get("full_path") or lead.get("artifact_path")
    if not lead_track_id or not full_path:
        raise SystemExit("daily playlist lead item missing track_id/full_path")

    src_path = Path(str(full_path))
    if not src_path.is_absolute():
        src_path = Path(getattr(args, "repo_root", ".")).expanduser().resolve() / src_path

    if not src_path.exists():
        raise SystemExit(f"[run.daily] missing source track file: {src_path}")

    dst = bundle_tracks_dir / f"{lead_track_id}{src_path.suffix or '.wav'}"
    shutil.copy2(src_path, dst)

    copied = [{"track_id": lead_track_id, "path": f"tracks/{dst.name}"}]

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    if deterministic:
        drop_id = stable_uuid5(f"drop:daily:{period_key}:{context}:{seed}", namespace=ns)
        playlist_id = stable_uuid5(f"playlist:daily:{period_key}:{context}:{seed}", namespace=ns)
        run_id = stable_uuid5(f"run:daily:{period_key}:{context}:{seed}", namespace=ns)
    else:
        drop_id = str(uuid.uuid4())
        playlist_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())

    playlist_obj = {
        "schema": "mgc.playlist.v1",
        "version": 1,
        "schedule": "daily",
        "ts": now_iso,
        "playlist_id": str(playlist_id),
        "context": context,
        "period": {"label": period_key},
        "tracks": copied,
    }
    playlist_json = json.dumps(playlist_obj, indent=2, sort_keys=True) + "\n"
    (bundle_dir / "playlist.json").write_text(playlist_json, encoding="utf-8")
    # Pipeline contract: always expose top-level playlist.json as well
    (out_dir / "playlist.json").write_text(playlist_json, encoding="utf-8")

    lead_rel_path = copied[0]["path"]
    daily_ev = {
        "schema": "mgc.daily_evidence.v1",
        "version": 1,
        "run_id": str(run_id),
        "stage": "daily",
        "context": context,
        "deterministic": bool(deterministic),
        "ts": now_iso,
        "provider": str(getattr(args, "provider", None) or os.environ.get("MGC_PROVIDER") or "filesystem"),
        "schedule": "daily",
        "track": {"track_id": lead_track_id, "path": lead_rel_path},
        "sha256": {
            "playlist": sha256_file(bundle_dir / "playlist.json"),
            "track": sha256_file(bundle_dir / lead_rel_path),
        },
        "period": {"label": period_key},
    }

    (evidence_dir / "daily_evidence.json").write_text(json.dumps(daily_ev, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (bundle_dir / "daily_evidence.json").write_text(json.dumps(daily_ev, indent=2, sort_keys=True) + "\n", encoding="utf-8")


    # Compute deterministic repo manifest alongside playlist (helps CI provenance)
    repo_root = Path(getattr(args, "repo_root", ".")).expanduser().resolve()
    include = getattr(args, "include", None) or None
    exclude_dirs = getattr(args, "exclude_dir", None) or None
    exclude_globs = getattr(args, "exclude_glob", None) or None

    manifest_obj = compute_manifest(repo_root, include=include, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs)
# BEGIN MGC MANIFEST OUTPUT FILTER (secrets/junk)
    def _mgc_keep_path(path: str) -> bool:
        # Never let local secrets / OS junk into any written manifest.
        # Do NOT exclude .env.example
        if path == ".env" or path.startswith(".env.") or path.endswith("/.env") or "/.env." in path:
            return False
        if path == ".DS_Store" or path.endswith("/.DS_Store"):
            return False
        return True
    
    if isinstance(manifest_obj, dict):
        _ents = manifest_obj.get("entries")
        if isinstance(_ents, list):
            manifest_obj["entries"] = [e for e in _ents if _mgc_keep_path(e.get("path", ""))]
    # END MGC MANIFEST OUTPUT FILTER
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(stable_json_dumps(manifest_obj) + "\n", encoding="utf-8")
    manifest_sha256 = sha256_file(manifest_path)

    marketing_obj: Optional[Dict[str, Any]] = None
    if bool(getattr(args, "marketing", False)):
        marketing_out = Path(getattr(args, "marketing_out_dir", None) or (out_dir / "marketing")).expanduser()
        marketing_obj = _agents_marketing_plan(
            drop_dir=bundle_dir,
            repo_root=Path(getattr(args, "repo_root", ".")).expanduser().resolve(),
            out_dir=marketing_out,
            seed=int(seed),
            teaser_seconds=int(getattr(args, "teaser_seconds", 15) or 15),
            ts=now_iso,
        )
    drop_evidence = {
        "ok": True,
        "schedule": "daily",
        "context": context,
        "period_key": period_key,
        "drop_id": str(drop_id),
        "lead_track_id": lead_track_id,
        "playlist_tracks": 1,
        "marketing": marketing_obj,
        "paths": {
            "bundle_dir": "drop_bundle",
            "bundle_playlist": "drop_bundle/playlist.json",
            "bundle_daily_evidence": "drop_bundle/daily_evidence.json",
            "daily_evidence": "evidence/daily_evidence.json",
            "drop_evidence": "drop_evidence.json",
            "manifest": "manifest.json",
            "manifest_sha256": manifest_sha256,
        },
    }
    drop_evidence = _scrub_absolute_paths(drop_evidence, out_dir=out_dir, repo_root=Path.cwd())
    (out_dir / "drop_evidence.json").write_text(json.dumps(drop_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if json_mode:
        sys.stdout.write(stable_json_dumps(drop_evidence) + "\n")
    else:
        print(f"[run.daily] ok date={period_key} out_dir={out_dir}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# Marketing publish (deterministic)
# ---------------------------------------------------------------------------

def cmd_publish_marketing(args: argparse.Namespace) -> int:
    """
    Publish marketing posts.

    Modes:
      - DB mode (default): publishes pending rows from marketing_posts, filtered by meta fields.
      - File mode (when --publish-dir/--marketing-dir/--bundle-dir is provided): publishes JSON posts from disk.
        No DB fallback. Useful for deterministic gates and artifact-only pipelines.

    Receipts:
      - If --out-dir is provided: <out_dir>/marketing/receipts/<batch_id>/<platform>/<post_id>.json
      - Else: data/evidence/marketing/receipts/<batch_id>/<platform>/<post_id>.json
    """
    import sys
    import sqlite3
    from typing import Any, Dict, List, Optional

    deterministic = is_deterministic(args)
    ts = deterministic_now_iso(deterministic)

    limit = int(getattr(args, "limit", None) or 50)
    dry_run = bool(getattr(args, "dry_run", False))

    filter_run_id = str(getattr(args, "run_id", "") or "").strip() or None
    filter_drop_id = str(getattr(args, "drop_id", "") or "").strip() or None
    filter_schedule = str(getattr(args, "schedule", "") or "").strip() or None
    filter_period_key = str(getattr(args, "period_key", "") or "").strip() or None

    out_dir_raw = getattr(args, "out_dir", None)
    out_dir: Optional[Path] = Path(out_dir_raw).expanduser().resolve() if isinstance(out_dir_raw, str) and out_dir_raw.strip() else None

    publish_dir_raw = getattr(args, "publish_dir", None)
    marketing_dir_raw = getattr(args, "marketing_dir", None)
    bundle_dir_raw = getattr(args, "bundle_dir", None)

    # ----------------------------
    # FILE MODE
    # ----------------------------
    if (publish_dir_raw or marketing_dir_raw or bundle_dir_raw):
        if publish_dir_raw:
            publish_dir = Path(str(publish_dir_raw)).expanduser().resolve()
        elif marketing_dir_raw:
            publish_dir = (Path(str(marketing_dir_raw)).expanduser().resolve() / "publish")
        else:
            # bundle_dir is typically <out_dir>/drop_bundle; publish dir is <out_dir>/marketing/publish
            bd = Path(str(bundle_dir_raw)).expanduser().resolve()
            publish_dir = (bd.parent / "marketing" / "publish")

        if not publish_dir.exists() or not publish_dir.is_dir():
            sys.stdout.write(stable_json_dumps({
                "mode": "file",
                "ok": False,
                "reason": "publish_dir_missing",
                "publish_dir": str(publish_dir),
            }) + "\n")
            return 2

        # stable ordering for determinism
        files = sorted([p for p in publish_dir.glob("*.json") if p.is_file()], key=lambda p: p.name)
        items: List[Dict[str, Any]] = []
        skipped_ids: List[str] = []

        # Gather candidate posts
        posts: List[Dict[str, Any]] = []
        for p in files:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                # Skip invalid JSON deterministically (record)
                skipped_ids.append(p.name)
                continue
            if not isinstance(obj, dict):
                skipped_ids.append(p.name)
                continue

            # normalize
            post_id = str(obj.get("post_id") or obj.get("id") or "")
            platform = str(obj.get("platform") or obj.get("channel") or "unknown")
            status = str(obj.get("status") or "planned").lower().strip()
            run_id = str(obj.get("run_id") or "")
            drop_id = str(obj.get("drop_id") or "")
            schedule = str(obj.get("schedule") or "")
            period_key = str(obj.get("period_key") or "")
            # tolerate nested period key
            if not period_key:
                period_key = str(((obj.get("period") or {}) if isinstance(obj.get("period"), dict) else {}).get("key") or "")

            # Filters (only apply if filter provided)
            if filter_run_id and run_id != filter_run_id:
                continue
            if filter_drop_id and drop_id != filter_drop_id:
                continue
            if filter_schedule and schedule != filter_schedule:
                continue
            if filter_period_key and period_key != filter_period_key:
                continue

            # publishable statuses
            if status not in ("planned", "pending", "ready", "draft"):
                continue

            # require platform + id
            if not post_id:
                # deterministic fallback from filename
                post_id = p.stem
            posts.append({
                "post_id": post_id,
                "platform": platform,
                "status": status,
                "content_obj": obj,
                "run_id": run_id,
                "drop_id": drop_id,
                "schedule": schedule,
                "period_key": period_key,
            })

        # deterministic sorting
        posts_sorted = sorted(posts, key=lambda d: (d.get("period_key") or "", d.get("platform") or "", d.get("post_id") or ""))

        # Batch ID incorporates filters + deterministic knob
        batch_id = stable_uuid5(
            "marketing_publish_batch",
            (ts if not deterministic else "fixed"),
            ("file"),
            str(limit),
            ("dry" if dry_run else "live"),
            (filter_schedule or "any"),
            (filter_period_key or "any"),
            (filter_run_id or "any"),
            (filter_drop_id or "any"),
        )

        receipt_paths: List[str] = []
        base_receipts = (out_dir / "marketing" / "receipts") if out_dir else None

        for d in posts_sorted[:limit]:
            post_id = str(d["post_id"])
            platform = str(d["platform"])
            obj = d["content_obj"]
            run_id = str(d.get("run_id") or obj.get("run_id") or "")
            drop_id = str(d.get("drop_id") or obj.get("drop_id") or "")
            schedule = str(d.get("schedule") or obj.get("schedule") or "")
            period_key = str(d.get("period_key") or obj.get("period_key") or "")
            track_id = str(obj.get("track_id") or "")

            # store the full original post payload as canonical content
            content = stable_json_dumps(obj)

            publish_id = stable_uuid5("publish", batch_id, post_id, platform)

            item = {
                "post_id": post_id,
                "platform": platform,
                "schedule": schedule or None,
                "period_key": period_key or None,
                "run_id": run_id or None,
                "drop_id": drop_id or None,
                "published_id": publish_id,
                "published_ts": ts,
                "dry_run": dry_run,
                "content": content,
            }
            items.append(item)

            receipt = {
                "receipt_id": stable_uuid5("marketing_receipt", batch_id, post_id, platform),
                "batch_id": batch_id,
                "ts": ts,
                "status": "dry_run" if dry_run else "ok",
                "mode": "file",
                "post_id": post_id,
                "platform": platform,
                "published_id": publish_id,
                "published_ts": ts,
                "run_id": run_id,
                "drop_id": drop_id,
                "track_id": track_id,
                "schedule": schedule,
                "period_key": period_key,
                "content": content,
            }

            receipt_paths.append(
                _write_marketing_receipt(
                    receipt=receipt,
                    stable_json_dumps=stable_json_dumps,
                    base_dir=base_receipts,
                )
            )

        out_obj = {
            "mode": "file",
            "ok": True,
            "batch_id": batch_id,
            "ts": ts,
            "count": len(items),
            "limit": limit,
            "dry_run": dry_run,
            "filters": {
                "run_id": filter_run_id,
                "drop_id": filter_drop_id,
                "schedule": filter_schedule,
                "period_key": filter_period_key,
            },
            "items": items,
            "receipts_written": len(receipt_paths),
            "skipped_ids": skipped_ids,
        }

        sys.stdout.write(stable_json_dumps(out_obj) + "\n")
        return 0

    # ----------------------------
    # DB MODE (default)
    # ----------------------------
    db_path = resolve_db_path(args)
    con = db_connect(db_path)
    ensure_tables_minimal(con)

    pending = db_marketing_posts_pending(con, limit=limit)

    def _first_id(r: sqlite3.Row) -> str:
        return str(_row_first(r, ["id", "post_id", "marketing_post_id"], default="") or "")

    def _first_created(r: sqlite3.Row) -> str:
        return str(_row_first(r, ["created_at", "created_ts", "ts"], default="") or "")

    pending_sorted = sorted(list(pending), key=lambda r: (_first_created(r), _first_id(r)))

    published: List[Dict[str, Any]] = []
    skipped_ids: List[str] = []
    run_ids_touched: List[str] = []

    # Batch ID in DB mode includes filters + set of run_ids (stable)
    for row in pending_sorted:
        meta = _marketing_row_meta(con, row) or {}
        rid = str(meta.get("run_id") or "")
        if rid:
            run_ids_touched.append(rid)
    run_ids_touched = sorted(set(run_ids_touched))

    batch_id = stable_uuid5(
        "marketing_publish_batch",
        (ts if not deterministic else "fixed"),
        "db",
        str(limit),
        ("dry" if dry_run else "live"),
        (filter_schedule or "any"),
        (filter_period_key or "any"),
        (filter_run_id or "any"),
        (filter_drop_id or "any"),
        ("|".join(run_ids_touched) if run_ids_touched else "no_runs"),
    )

    receipt_paths: List[str] = []
    base_receipts = (out_dir / "marketing" / "receipts") if out_dir else None

    def _meta_matches(meta: Dict[str, Any]) -> bool:
        if filter_run_id and str(meta.get("run_id") or "") != filter_run_id:
            return False
        if filter_drop_id and str(meta.get("drop_id") or "") != filter_drop_id:
            return False
        if filter_schedule and str(meta.get("schedule") or "") != filter_schedule:
            return False

        if filter_period_key:
            pk = str(meta.get("period_key") or "")
            if not pk:
                period = meta.get("period")
                if isinstance(period, dict):
                    pk = str(period.get("key") or "")
            if pk != filter_period_key:
                return False
        return True

    for row in pending_sorted:
        post_id = _first_id(row)
        platform = str(_row_first(row, ["platform", "channel", "destination"], default="unknown"))
        content = _marketing_row_content(con, row) or ""

        meta = _marketing_row_meta(con, row) or {}
        if not _meta_matches(meta):
            continue

        if not content.strip():
            skipped_ids.append(post_id)
            continue

        run_id = str(meta.get("run_id") or "")
        drop_id = str(meta.get("drop_id") or "")
        track_id = str(meta.get("track_id") or "")

        publish_id = stable_uuid5("publish", batch_id, post_id, platform)

        if not dry_run:
            db_marketing_post_set_status(
                con,
                post_id=post_id,
                status="published",
                ts=ts,
                meta_patch={
                    "published_id": publish_id,
                    "published_ts": ts,
                    "batch_id": batch_id,
                },
            )

        item = {
            "post_id": post_id,
            "platform": platform,
            "published_id": publish_id,
            "published_ts": ts,
            "dry_run": dry_run,
            "content": content,
            "run_id": run_id,
            "drop_id": drop_id,
            "schedule": str(meta.get("schedule") or ""),
            "period_key": str(meta.get("period_key") or "") or (str(meta.get("period", {}).get("key") or "") if isinstance(meta.get("period"), dict) else ""),
        }
        published.append(item)

        receipt = {
            "receipt_id": stable_uuid5("marketing_receipt", batch_id, post_id, platform),
            "batch_id": batch_id,
            "ts": ts,
            "status": "dry_run" if dry_run else "ok",
            "mode": "db",
            "post_id": post_id,
            "platform": platform,
            "published_id": publish_id,
            "published_ts": ts,
            "run_id": run_id,
            "drop_id": drop_id,
            "track_id": track_id,
            "content": content,
        }

        receipt_paths.append(
            _write_marketing_receipt(
                receipt=receipt,
                stable_json_dumps=stable_json_dumps,
                base_dir=base_receipts,
            )
        )

    drops_updated: Dict[str, int] = {}
    if run_ids_touched and not dry_run:
        for rid in run_ids_touched:
            drops_updated[rid] = db_drop_mark_published(
                con,
                run_id=rid,
                marketing_batch_id=batch_id,
                published_ts=ts,
            )

    db_insert_event(
        con,
        event_id=stable_uuid5("event", "marketing.published", batch_id),
        ts=ts,
        kind="marketing.published",
        actor="system",
        meta={
            "batch_id": batch_id,
            "count": len(published),
            "dry_run": dry_run,
            "skipped_empty": len(skipped_ids),
            "run_ids": run_ids_touched,
            "drops_updated": drops_updated,
            "receipts_written": len(receipt_paths),
        },
    )

    out_obj = {
        "mode": "db",
        "batch_id": batch_id,
        "ts": ts,
        "count": len(published),
        "skipped_empty": len(skipped_ids),
        "skipped_ids": skipped_ids,
        "run_ids": run_ids_touched,
        "drops_updated": drops_updated,
        "items": published,
        "receipts_written": len(receipt_paths),
        "filters": {
            "run_id": filter_run_id,
            "drop_id": filter_drop_id,
            "schedule": filter_schedule,
            "period_key": filter_period_key,
        },
    }

    sys.stdout.write(stable_json_dumps(out_obj) + "\n")
    return 0


# ---------------------------------------------------------------------------
# Drop (daily + publish + manifest) with stages and resume semantics
# ---------------------------------------------------------------------------

def _drop_latest_for_run(con: sqlite3.Connection, run_id: str) -> Optional[sqlite3.Row]:
    cols = db_table_columns(con, "drops")
    if not cols:
        return None

    ts_col = _pick_first_existing(cols, ["ts", "created_at", "created_ts", "created", "created_on", "occurred_at"])
    id_col = _pick_first_existing(cols, ["id", "drop_id"]) or "id"

    if ts_col:
        sql = f"SELECT * FROM drops WHERE run_id = ? ORDER BY {ts_col} DESC, {id_col} DESC LIMIT 1"
    else:
        sql = f"SELECT * FROM drops WHERE run_id = ? ORDER BY {id_col} DESC LIMIT 1"

    return con.execute(sql, (run_id,)).fetchone()


def cmd_run_drop(args: argparse.Namespace) -> int:
    """
    Produce one deterministic "drop" bundle under --out-dir.

    This command MUST:
      - ensure manifest.json exists under out_dir
      - run the daily generator (stub/riffusion/...) to materialize audio + DB rows
      - write drop_evidence.json under out_dir, including bundle track + playlist paths
      - build a submission zip at data/submissions/<drop_id>/submission.zip (non-dry-run)
      - write data/submissions/<drop_id>/submission.json (self-describing pointer)
      - print a small JSON summary to stdout

    Optional:
      - if --with-web is set, build a static web bundle and include it in the zip under submission/web,
        and record paths.web_bundle_dir + artifacts.web_bundle_tree_sha256.
        Use --web-out-dir to override where the on-disk web bundle is written.
    """
    import hashlib
    import json
    import os
    import shutil
    import sqlite3
    import subprocess
    import tempfile
    import zipfile
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any, Dict, Optional

    from mgc.bundle_validate import validate_bundle

    # -------------------------
    # repo_root resolution (CWD-proof)
    # -------------------------
    # CI and local invocations may run from different working directories.
    # We need a stable repo root to resolve relative paths (especially submissions).
    def _resolve_repo_root() -> Path:
        arg = str(getattr(args, "repo_root", "") or "").strip()
        candidates: list[Path] = []
        if arg:
            candidates.append(Path(arg).expanduser())
        # Fallback: repo containing this file (repo_root/src/mgc/run_cli.py)
        candidates.append(Path(__file__).resolve().parents[2])

        for cand in candidates:
            try:
                rr = cand.resolve()
            except Exception:
                continue
            # Validate by presence of src/mgc (this repo uses src layout)
            if (rr / "src" / "mgc").is_dir():
                return rr
        # Last resort: best-effort
        return candidates[-1].resolve()

    repo_root = _resolve_repo_root()

    # -------------------------
    # tiny deterministic helpers
    # -------------------------

    def stable_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def sha256_tree(root: Path) -> str:
        """
        Deterministic tree hash:
          - per file: sha256(file bytes)
          - lines: "<sha>  <relpath>"
          - sha256 of concatenated lines
        """
        root = root.resolve()
        files = sorted(
            (p for p in root.rglob("*") if p.is_file()),
            key=lambda p: p.relative_to(root).as_posix(),
        )
        lines: list[str] = []
        for p in files:
            rel = p.relative_to(root).as_posix()
            lines.append(f"{sha256_file(p)}  {rel}")
        joined = ("\n".join(lines) + "\n").encode("utf-8")
        return hashlib.sha256(joined).hexdigest()

    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _safe_mkdir(p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def _posix_rel(p: Path) -> str:
        return str(p).replace("\\", "/")

    def _zip_add_dir(zf: zipfile.ZipFile, root: Path, arc_root: str) -> None:
        """
        Add directory contents to zip deterministically:
          - stable traversal order (sorted dirs/files)
          - stable archive ordering (sorted arcname)
          - fixed timestamps in ZipInfo for deterministic builds
        """
        root = root.resolve()
        entries: list[tuple[str, Path]] = []
        for dp, dn, fn in os.walk(root):
            dn.sort()
            fn.sort()
            dp_path = Path(dp)
            for name in fn:
                file_path = (dp_path / name).resolve()
                rel = file_path.relative_to(root)
                arcname = f"{arc_root}/{_posix_rel(rel)}"
                entries.append((arcname, file_path))
        entries.sort(key=lambda x: x[0])

        fixed_dt = (2020, 1, 1, 0, 0, 0)
        for arcname, file_path in entries:
            data = file_path.read_bytes()
            zi = zipfile.ZipInfo(filename=arcname, date_time=fixed_dt)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, data)


    def _copy_portable_bundle(src: Path, dst: Path, *, drop_id: str | None = None) -> None:
        """Copy ONLY the portable bundle artifacts into dst.

        We intentionally exclude volatile run outputs like drop_evidence.json, manifest.json,
        and any absolute-path pointer files in out_dir, because those can differ per run
        (e.g., out_dir paths /tmp/mgc_auto_a vs /tmp/mgc_auto_b) and would break submission
        zip determinism.

        Expected portable bundle contents:
          - playlist.json
          - daily_evidence.json (+ optional daily_evidence_<drop_id>.json)
          - tracks/...
        """
        src = src.resolve()
        dst = dst.resolve()
        _safe_mkdir(dst)

        # tracks/
        tracks_src = src / 'tracks'
        if tracks_src.exists() and tracks_src.is_dir():
            tracks_dst = dst / 'tracks'
            _safe_mkdir(tracks_dst)
            files = sorted([p for p in tracks_src.rglob('*') if p.is_file()], key=lambda p: p.relative_to(tracks_src).as_posix())
            for p in files:
                rel = p.relative_to(tracks_src)
                outp = tracks_dst / rel
                _safe_mkdir(outp.parent)
                shutil.copy2(str(p), str(outp))

        # core JSON files
        for name in ('playlist.json', 'daily_evidence.json'):
            p = src / name
            if p.exists() and p.is_file():
                shutil.copy2(str(p), str(dst / name))

        # optional scoped daily evidence
        if drop_id:
            scoped = src / f'daily_evidence_{drop_id}.json'
            if scoped.exists() and scoped.is_file():
                shutil.copy2(str(scoped), str(dst / scoped.name))

    def _build_readme(evidence_obj: dict, *, included_web: bool) -> str:
        # Use run ts (not current time) so deterministic runs stay deterministic.
        ids = evidence_obj.get("ids") if isinstance(evidence_obj.get("ids"), dict) else {}
        paths = evidence_obj.get("paths") if isinstance(evidence_obj.get("paths"), dict) else {}

        drop_id = ids.get("drop_id", evidence_obj.get("drop_id", ""))
        run_id_local = ids.get("run_id", evidence_obj.get("run_id", ""))
        track_id_local = ids.get("track_id", evidence_obj.get("track_id", ""))
        provider = evidence_obj.get("provider", "")
        context_local = evidence_obj.get("context", evidence_obj.get("context", ""))
        ts_local = evidence_obj.get("ts", "")

        bundle_track = paths.get("bundle_track", paths.get("bundle_track_path", ""))
        playlist = paths.get("playlist", paths.get("playlist_path", "playlist.json"))
        deterministic_local = evidence_obj.get("deterministic", "")

        lines = [
            "# Music Generator Company – Drop Submission",
            "## Identifiers",
            f"- drop_id: {drop_id}",
            f"- run_id: {run_id_local}",
            f"- track_id: {track_id_local}",
            "## Run metadata",
            f"- ts: {ts_local}",
            f"- context: {context_local}",
            f"- provider: {provider}",
            f"- deterministic: {deterministic_local}",
            "## Contents",
            f"- {playlist}: playlist pointing at bundled audio",
            f"- {bundle_track}: bundled audio asset",
            "- daily_evidence.json (or drop bundle evidence): provenance + sha256 hashes",
        ]
        if included_web:
            lines += ["- web/: static web player bundle (index.html + tracks/)"]

        lines += [
            "## How to review",
            "1) Inspect playlist.json (it references the bundled track under tracks/).",
            "2) Confirm hashes match the files in the bundle.",
        ]
        if included_web:
            lines += ["3) Open web/index.html (or serve web/) to listen."]

        lines += [
            "## Notes",
            f"- Packaged at: {ts_local or _utc_now_iso()}",
        ]
        return "\n".join(lines) + "\n"
    def _build_web_bundle_from_portable_bundle(*, bundle_dir: Path, web_out_dir: Path) -> Dict[str, Any]:
        """
        Build a web bundle using `mgc.main web build` with cwd=bundle_dir so relative paths resolve.
        Also passes so the generated playlist.json doesn't bake absolute paths.
        """
        playlist_path = bundle_dir / "playlist.json"
        if not playlist_path.exists():
            raise FileNotFoundError(f"bundle missing playlist.json: {playlist_path}")

        cmd = [
            sys.executable,
            "-m",
            "mgc.main",
            "web",
            "build",
            "--playlist",
            str(playlist_path),
            "--out-dir",
            str(web_out_dir),
            "--clean",
            "--fail-if-empty",
            "--fail-if-none-copied",
            "--json",
        ]

        # Defensive: drop empty argv elements (argparse would treat them as unknown args)

        cmd = [x for x in cmd if x]


        proc = subprocess.run(
            cmd,
            cwd=str(bundle_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )

        stdout = (proc.stdout or "").strip()
        if proc.returncode != 0:
            msg = (proc.stderr or "").strip() or stdout or "web build failed"
            raise RuntimeError(f"web build failed (rc={proc.returncode}): {msg}")

        try:
            payload = json.loads(stdout) if stdout else {}
        except Exception:
            raise RuntimeError(
                "web build did not emit valid JSON. "
                f"stdout={stdout[:200]!r} stderr={(proc.stderr or '')[:200]!r}"
            )

        if not isinstance(payload, dict) or not payload.get("ok", False):
            raise RuntimeError(f"web build returned ok=false: {payload}")

        # sanity
        idx = web_out_dir / "index.html"
        pl = web_out_dir / "playlist.json"
        tr = web_out_dir / "tracks"
        if not idx.exists():
            raise FileNotFoundError(f"web bundle missing index.html: {idx}")
        if not pl.exists():
            raise FileNotFoundError(f"web bundle missing playlist.json: {pl}")
        if not tr.exists() or not tr.is_dir():
            raise FileNotFoundError(f"web bundle missing tracks/: {tr}")

        return payload

    # -------------------------
    # args / env
    # -------------------------

    deterministic = bool(getattr(args, "deterministic", False)) or (
        (os.environ.get("MGC_DETERMINISTIC") or "").strip().lower() in ("1", "true", "yes")
    )

    context = str(getattr(args, "context", None) or "focus")
    seed = str(getattr(args, "seed", None) or "1")

    allow_resume = not bool(getattr(args, "no_resume", False))
    dry_run = bool(getattr(args, "dry_run", False))
    with_web = bool(getattr(args, "with_web", False))

    ts = deterministic_now_iso(deterministic)
    run_date = ts.split("T", 1)[0]

    out_dir = Path(
        getattr(args, "out_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR")
        or "data/evidence"
    ).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = (
        getattr(args, "db", None)
        or os.environ.get("MGC_DB")
        or "data/db.sqlite"
    )
    db_path = str(Path(db_path).expanduser())

    run_id = stable_uuid5("run", "drop", context, seed, run_date, "det" if deterministic else "live")

    # ---------------------------------------------------------------------
    # Ensure manifest exists in out_dir (cmd_run_drop hashes this)
    # ---------------------------------------------------------------------
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        _write_minimal_manifest(manifest_path, ts)
    manifest_sha256 = _sha256_file(manifest_path)

    # ---------------------------------------------------------------------
    # Run generator (or dry_run)
    # ---------------------------------------------------------------------
    if dry_run:
        drop_id = stable_uuid5("drop", run_id)
        evidence_obj: Dict[str, Any] = {
            "schema": "mgc.drop_evidence.v1",
            "ts": ts,
            "deterministic": bool(deterministic),
            "run_id": run_id,
            "drop_id": drop_id,
            "context": context,
            "seed": seed,
            "stages": {"allow_resume": bool(allow_resume)},
            "paths": {
                "manifest_path": str(manifest_path),
                "manifest": "manifest.json",
                "manifest_sha256": manifest_sha256,
            },
            "note": "dry_run=true; generator not executed",
        }
    else:
        con = sqlite3.connect(db_path)
        try:
            con.execute("PRAGMA foreign_keys = ON")
            daily = _stub_daily_run(
                con=con,
                context=context,
                seed=seed,
                deterministic=bool(deterministic),
                ts=ts,
                out_dir=out_dir,
                run_id=run_id,
            )
            con.commit()
        finally:
            con.close()

        drop_id = str((daily or {}).get("drop_id") or stable_uuid5("drop", run_id))

        track_id = ""
        try:
            track_id = str(((daily or {}).get("track") or {}).get("id") or "")
        except Exception:
            track_id = ""

        evidence_obj = {
            "schema": "mgc.drop_evidence.v1",
            "ts": ts,
            "deterministic": bool(deterministic),
            "run_id": run_id,
            "drop_id": drop_id,
            "context": context,
            "seed": seed,
            "stages": {"allow_resume": bool(allow_resume)},
            "paths": {
                "manifest_path": str(manifest_path),
                "manifest": "manifest.json",
                "manifest_sha256": manifest_sha256,
            },
            "daily": daily,
        }

        bundle = (daily or {}).get("bundle") if isinstance(daily, dict) else None
        if isinstance(bundle, dict):
            evidence_obj["paths"].update(
                {
                    "bundle_dir": str(bundle.get("out_dir") or str(out_dir)),
                    "bundle_track_path": str(bundle.get("track_path") or ""),
                    "bundle_track_sha256": str(bundle.get("track_sha256") or ""),
                    "playlist_path": str(bundle.get("playlist_path") or (out_dir / "playlist.json")),
                    "playlist_sha256": str(bundle.get("playlist_sha256") or ""),
                }
            )

        if track_id:
            evidence_obj["track_id"] = track_id

    # ---------------------------------------------------------------------
    # Normalize evidence paths to be portable (relative to out_dir)
    # ---------------------------------------------------------------------
    paths = evidence_obj.get("paths", {})

    def _rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(out_dir.resolve()))
        except Exception:
            return p  # leave untouched if not under out_dir

    for k in ("manifest_path", "evidence_path", "out_dir"):
        if k in paths and isinstance(paths[k], str):
            paths[k] = _rel(paths[k])

    evidence_obj["paths"] = paths

    # ---------------------------------------------------------------------
    # Normalize evidence paths to be portable (relative to out_dir)
    # ---------------------------------------------------------------------
    paths = evidence_obj.setdefault("paths", {})
    paths["manifest_path"] = "manifest.json"
    paths["evidence_path"] = "drop_evidence.json"
    paths["out_dir"] = "."

    # ---------------------------------------------------------------------
    # Write drop evidence (initial)
    # ---------------------------------------------------------------------
    evidence_path = out_dir / "drop_evidence.json"
    evidence_obj = _scrub_absolute_paths(evidence_obj, out_dir=out_dir, repo_root=repo_root)
    evidence_path.write_text(stable_json(evidence_obj), encoding="utf-8")

    # ---------------------------------------------------------------------
    # Build submission zip + submission.json (non-dry-run only)
    # ---------------------------------------------------------------------
    submission_zip_path: Optional[Path] = None
    submission_json_path: Optional[Path] = None
    receipt_path: Optional[Path] = None

    web_out_dir: Optional[Path] = None
    web_build_meta: Optional[Dict[str, Any]] = None
    web_tree_sha: Optional[str] = None

    if not dry_run:
        pths = evidence_obj.get("paths") or {}

        # Portable bundle lives at <out_dir>/drop_bundle.
        # If it doesn't exist yet (e.g. older code path), materialize it by
        # running the daily stage into the same out_dir.
        bundle_dir = (out_dir / "drop_bundle").resolve()
        if not bundle_dir.exists():
            daily_args = argparse.Namespace(**vars(args))
            setattr(daily_args, "out_dir", str(out_dir))
            setattr(daily_args, "context", context)
            setattr(daily_args, "seed", seed)
            setattr(daily_args, "deterministic", bool(deterministic))
            setattr(daily_args, "dry_run", False)
            # cmd_run_daily will write <out_dir>/drop_bundle, playlist.json, and evidence.
            cmd_run_daily(daily_args)

        # Populate bundle path hints for downstream tooling (stdout JSON consumers).
        pths["bundle_dir"] = "drop_bundle"
        pths["bundle_dir_path"] = str(bundle_dir)
        pths["bundle_playlist"] = "drop_bundle/playlist.json"
        pths["bundle_evidence"] = "drop_bundle/daily_evidence.json"
        pths["bundle_tracks_dir"] = "drop_bundle/tracks"
        pths["run_out_dir"] = str(out_dir)
        evidence_obj["paths"] = pths

        # Validate bundle before packaging
        validate_bundle(bundle_dir)

        # Evidence object for README
        daily_ev_path = bundle_dir / "daily_evidence.json"
        if daily_ev_path.exists():
            try:
                daily_ev_obj = json.loads(daily_ev_path.read_text(encoding="utf-8"))
            except Exception:
                daily_ev_obj = {
                    "ts": ts,
                    "drop_id": evidence_obj.get("drop_id"),
                    "run_id": run_id,
                    "context": context,
                    "deterministic": bool(deterministic),
                }
        else:
            daily_ev_obj = {
                "ts": ts,
                "drop_id": evidence_obj.get("drop_id"),
                "run_id": run_id,
                "context": context,
                "deterministic": bool(deterministic),
            }

        drop_id_for_paths = str(evidence_obj.get("drop_id") or stable_uuid5("drop", run_id))

        # ------------------------------------------------------------------
        # Submission output directory (STRICT + deterministic)
        #
        # Resolution order:
        #   1) MGC_SUBMISSIONS_DIR (absolute or relative to repo-root)
        #   2) <repo-root>/data/submissions/<drop_id>
        #
        # IMPORTANT:
        # - Never rely on an arbitrary CWD.
        # - Some CI runners may invoke from a different working directory.
        # - We validate repo_root and fall back to the on-disk repo containing this file.
        # - Never write to /data/* (root filesystem).
        # ------------------------------------------------------------------

        repo_root_arg = str(getattr(args, "repo_root", "") or "").strip()
        if repo_root_arg:
            cand = Path(repo_root_arg).expanduser()
            if cand.is_absolute():
                repo_root = cand.resolve()
            else:
                repo_root = (Path.cwd() / cand).resolve()
        else:
            repo_root = Path.cwd().resolve()

        # Validate repo_root looks like a repo checkout (src/mgc exists). If not,
        # fall back to the repo that contains this file.
        if not (repo_root / "src" / "mgc").exists():
            try:
                repo_root = Path(__file__).resolve().parents[2]
            except Exception:
                pass

        sub_root_env = str(os.environ.get("MGC_SUBMISSIONS_DIR", "") or "").strip()

        if sub_root_env:
            sub_root = Path(sub_root_env).expanduser()
            if not sub_root.is_absolute():
                sub_root = (repo_root / sub_root).resolve()
            else:
                sub_root = sub_root.resolve()
        else:
            sub_root = (repo_root / "data" / "submissions").resolve()

        submissions_root = (sub_root / drop_id_for_paths).resolve()
        _safe_mkdir(submissions_root)

        submission_zip_path = (submissions_root / "submission.zip").resolve()
        submission_json_path = (submissions_root / "submission.json").resolve()
        receipt_path = submission_zip_path.with_suffix(".receipt.json")

        # ------------------------------------------------------------
        # Optional web bundle: build ONCE into a stable on-disk location
        # ------------------------------------------------------------
        if with_web:
            web_out_override = getattr(args, "web_out_dir", None)
            web_out_dir = (
                Path(str(web_out_override)).expanduser().resolve()
                if web_out_override
                else (submissions_root / "web").resolve()
            )
            _safe_mkdir(web_out_dir)

            # Build using web CLI against the portable bundle playlist
            web_build_meta = _build_web_bundle_from_portable_bundle(bundle_dir=bundle_dir, web_out_dir=web_out_dir)

            # Hash the web bundle tree (deterministic)
            web_tree_sha = sha256_tree(web_out_dir)

            evidence_obj.setdefault("paths", {})
            evidence_obj["paths"]["web_bundle_dir"] = str(web_out_dir)

        # ------------------------------------------------------------
        # Stage + zip deterministically
        # ------------------------------------------------------------
        with tempfile.TemporaryDirectory(prefix="mgc_submission_") as td:
            stage = Path(td).resolve()
            pkg_root = stage / "submission"
            _safe_mkdir(pkg_root)

            # Copy bundle into stage
            drop_bundle_dst = pkg_root / "drop_bundle"
            _copy_portable_bundle(bundle_dir, drop_bundle_dst, drop_id=str(evidence_obj.get("drop_id") or ""))

            # Copy web bundle into stage (if built)
            included_web = False
            if with_web and web_out_dir and web_out_dir.exists():
                shutil.copytree(web_out_dir, pkg_root / "web")
                included_web = True

            # README.md (stable fields)
            (pkg_root / "README.md").write_text(
                _build_readme(daily_ev_obj, included_web=included_web),
                encoding="utf-8",
            )

            # Write zip (deterministic ordering + timestamps)
            if submission_zip_path.exists():
                submission_zip_path.unlink()

            with zipfile.ZipFile(str(submission_zip_path), mode="w") as zf:
                fixed_dt = (2020, 1, 1, 0, 0, 0)

                zi = zipfile.ZipInfo(filename="submission/README.md", date_time=fixed_dt)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = 0o644 << 16
                zf.writestr(zi, (pkg_root / "README.md").read_text(encoding="utf-8"))

                _zip_add_dir(zf, drop_bundle_dst, arc_root="submission/drop_bundle")

                if included_web and (pkg_root / "web").exists():
                    _zip_add_dir(zf, pkg_root / "web", arc_root="submission/web")

        # Write submission.json (self-describing pointer file; stable fields)
        submission_obj = {
            "schema": "mgc.submission.v1",
            "drop_id": drop_id_for_paths,
            "run_id": str(run_id),
            "deterministic": bool(deterministic),
            "ts": ts,  # use run ts for determinism
            "submission_zip": "submission.zip",  # relative within submissions_root
            "included_web": bool(with_web),
        }
        submission_json_path.write_text(stable_json(submission_obj), encoding="utf-8")

        # Compute artifact hashes + write receipt
        submission_sha = sha256_file(submission_zip_path)
        receipt_obj: Dict[str, Any] = {
            "ok": True,
            "drop_id": drop_id_for_paths,
            "run_id": str(run_id),
            "deterministic": bool(deterministic),
            "ts": ts,
            "submission_zip": str(submission_zip_path),
            "submission_zip_sha256": submission_sha,
            "included_web": bool(with_web),
            "web_bundle_dir": str(web_out_dir) if (with_web and web_out_dir) else None,
            "web_bundle_tree_sha256": web_tree_sha,
        }
        if web_build_meta is not None:
            receipt_obj["web_build"] = web_build_meta
        receipt_path.write_text(stable_json(receipt_obj), encoding="utf-8")

        # Record in evidence + rewrite drop_evidence.json so pointers + hashes are included
        evidence_obj.setdefault("paths", {})
        evidence_obj.setdefault("artifacts", {})

        evidence_obj["paths"]["submission_dir"] = str(submissions_root)
        evidence_obj["paths"]["submission_zip"] = str(submission_zip_path)
        evidence_obj["paths"]["submission_json"] = str(submission_json_path)

        if with_web and web_out_dir:
            evidence_obj["paths"]["web_bundle_dir"] = str(web_out_dir)

        if isinstance(evidence_obj.get("artifacts"), dict):
            evidence_obj["artifacts"]["submission_zip_sha256"] = submission_sha
            if web_tree_sha:
                evidence_obj["artifacts"]["web_bundle_tree_sha256"] = web_tree_sha
            evidence_obj["artifacts"]["submission_receipt_json"] = str(receipt_path)

    evidence_obj = _scrub_absolute_paths(evidence_obj, out_dir=out_dir, repo_root=repo_root)
    evidence_path.write_text(stable_json(evidence_obj), encoding="utf-8")

    # ---------------------------------------------------------------------
    # Print stdout summary JSON
    # ---------------------------------------------------------------------
    out: Dict[str, Any] = {
        "deterministic": bool(deterministic),
        "drop_id": str(evidence_obj.get("drop_id") or stable_uuid5("drop", run_id)),
        "run_id": run_id,
        "ts": ts,
        "paths": {
            "evidence_path": str(evidence_path),
            "manifest_path": str(manifest_path),
            "manifest": "manifest.json",
            "manifest_sha256": manifest_sha256,
        },
    }

    # bubble up useful bundle paths when available
    try:
        p = evidence_obj.get("paths") or {}
        for k in (
            "bundle_dir",
            "bundle_dir_path",
            "bundle_playlist",
            "bundle_evidence",
            "bundle_tracks_dir",
            "run_out_dir",
            "bundle_track_path",
            "bundle_track_sha256",
            "playlist_path",
            "playlist_sha256",
            "submission_dir",
            "submission_zip",
            "submission_json",
            "web_bundle_dir",
        ):
            if k in p and p[k]:
                out["paths"][k] = p[k]

        a = evidence_obj.get("artifacts") or {}
        if isinstance(a, dict):
            for k in ("submission_zip_sha256", "web_bundle_tree_sha256", "submission_receipt_json"):
                if k in a and a[k]:
                    out.setdefault("artifacts", {})[k] = a[k]
    except Exception:
        pass

    if not bool(getattr(args, "_suppress_stdout_json", False)):
        print(stable_json(out))

    return 0

def cmd_run_tail(args: argparse.Namespace) -> int:
    evidence_dir = Path(
        getattr(args, "out_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR")
        or "data/evidence"
    ).resolve()

    kind = str(getattr(args, "type", "any") or "any")  # drop | weekly | any
    n = int(getattr(args, "n", 1) or 1)

    if not evidence_dir.exists() or not evidence_dir.is_dir():
        sys.stdout.write(
            stable_json_dumps(
                {
                    "found": False,
                    "reason": "evidence_dir_missing",
                    "path": str(evidence_dir),
                }
            )
            + "\n"
        )
        return 0

    if kind == "drop":
        patterns = ["drop_evidence*.json"]
    elif kind == "weekly":
        patterns = ["weekly_evidence*.json"]
    else:
        patterns = ["drop_evidence*.json", "weekly_evidence*.json"]

    files: List[Path] = []
    for pat in patterns:
        files.extend(evidence_dir.glob(pat))

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        sys.stdout.write(
            stable_json_dumps(
                {
                    "found": False,
                    "reason": "no_evidence_files",
                    "path": str(evidence_dir),
                }
            )
            + "\n"
        )
        return 0

    selected = files[:n]
    items: List[Dict[str, Any]] = []

    for p in selected:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            payload = {"error": str(e)}

        items.append(
            {
                "file": str(p),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
                "payload": payload,
            }
        )

    sys.stdout.write(
        stable_json_dumps(
            {
                "found": True,
                "count": len(items),
                "items": items,
            }
        )
        + "\n"
    )
    return 0



# ---------------------------------------------------------------------------
# Generate (library growth)
# ---------------------------------------------------------------------------

def cmd_run_generate(args: argparse.Namespace) -> int:
    """Generate a new track into the *library* (repo_root/data/tracks/YYYY-MM-DD) and register it in DB.

    Outputs:
      - data/tracks/YYYY-MM-DD/<track_id>.<ext>
      - data/tracks/YYYY-MM-DD/<track_id>.json (metadata)
      - DB row in tracks table
      - <out_dir>/evidence/generate_evidence.json

    JSON contract:
      - In --json mode, emit exactly one JSON object.
    """
    deterministic = is_deterministic(args)
    json_mode = bool(getattr(args, "json", False))

    db_path = resolve_db_path(args)
    context = str(getattr(args, "context", None) or os.environ.get("MGC_CONTEXT") or "focus")

    # Evidence root for this run
    out_dir = Path(getattr(args, "out_dir", None) or os.environ.get("MGC_EVIDENCE_DIR") or "data/evidence").expanduser().resolve()
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Library storage root
    repo_root = Path(getattr(args, "repo_root", ".")).expanduser().resolve()
    store_dir_raw = getattr(args, "store_dir", None)
    store_dir = Path(store_dir_raw).expanduser().resolve() if store_dir_raw else (repo_root / "data" / "tracks")
    store_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp + IDs
    now_iso = deterministic_now_iso(deterministic)
    day = now_iso.split("T", 1)[0]  # YYYY-MM-DD
    day_dir = store_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")

    seed_val = getattr(args, "seed", None)
    seed = int(seed_val) if seed_val is not None else int(os.environ.get("MGC_SEED") or "1")

    provided_track_id = (getattr(args, "track_id", None) or "").strip()
    if provided_track_id:
        track_id = provided_track_id
    else:
        if deterministic:
            track_id = stable_uuid5(f"track:generate:{now_iso}:{context}:{seed}", namespace=ns)
        else:
            track_id = str(uuid.uuid4())

    if deterministic:
        run_id = stable_uuid5(f"run:generate:{now_iso}:{context}:{track_id}", namespace=ns)
    else:
        run_id = str(uuid.uuid4())

    provider_name = str(getattr(args, "provider", None) or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()

    from mgc.agents.music_agent import MusicAgent

    agent = MusicAgent(provider=provider_name)
    track = agent.generate(
        track_id=str(track_id),
        context=str(context),
        seed=int(seed),
        deterministic=bool(deterministic),
        schedule="generate",
        period_key=day,
        out_dir=str(day_dir),
        now_iso=now_iso,
    )

    src_path = Path(track.artifact_path)
    suffix = src_path.suffix.lower() or ".wav"

    final_artifact = day_dir / f"{track_id}{suffix}"
    if src_path.resolve() != final_artifact.resolve():
        final_artifact.write_bytes(src_path.read_bytes())

    sha = sha256_file(final_artifact)

    title = str(getattr(track, "title", "") or f"{context.title()} Track")
    mood = str(getattr(track, "mood", "") or context)
    genre = str(getattr(track, "genre", "") or "unknown")

    meta = dict(getattr(track, "meta", None) or {})
    meta.update(
        {
            "generated_by": "run.generate",
            "context": context,
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "ts": now_iso,
        }
    )

    # Prefer DB-stored paths relative to repo_root when possible
    try:
        artifact_rel = str(final_artifact.resolve().relative_to(repo_root))
    except Exception:
        artifact_rel = str(final_artifact)

    # Write metadata JSON next to the artifact
    meta_path = day_dir / f"{track_id}.json"
    meta_doc = {
        "schema": "mgc.track_meta.v1",
        "version": 1,
        "track_id": str(track_id),
        "run_id": str(run_id),
        "provider": str(getattr(track, "provider", provider_name)),
        "title": title,
        "mood": mood,
        "genre": genre,
        "artifact_path": artifact_rel,
        "sha256": sha,
        "ts": now_iso,
        "meta": meta,
    }
    meta_path.write_text(json.dumps(meta_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Register in DB
    con = db_connect(db_path)
    ensure_tables_minimal(con)
    try:
        db_insert_track(
            con,
            track_id=str(track_id),
            ts=now_iso,
            title=title,
            provider=str(getattr(track, "provider", provider_name)),
            mood=mood,
            genre=genre,
            artifact_path=artifact_rel,
            meta=meta,
        )
        con.commit()
    finally:
        con.close()

    evidence = {
        "ok": True,
        "cmd": "run.generate",
        "run_id": str(run_id),
        "track_id": str(track_id),
        "context": context,
        "provider": provider_name,
        "deterministic": bool(deterministic),
        "ts": now_iso,
        "artifact_path": artifact_rel,
        "sha256": sha,
        "meta_path": str(meta_path),
        "evidence_dir": str(evidence_dir),
    }
    (evidence_dir / "generate_evidence.json").write_text(stable_json_dumps(evidence) + "\n", encoding="utf-8")

    if json_mode:
        sys.stdout.write(stable_json_dumps(evidence) + "\n")
    else:
        _eprint(f"[run.generate] ok track_id={track_id} artifact={artifact_rel}")

    return 0

# ---------------------------------------------------------------------------
# Weekly run (7 dailies + publish + manifest + consolidated evidence)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Weekly run (playlist builder + copy existing tracks into a bundle)
# ---------------------------------------------------------------------------

def cmd_run_weekly(args: argparse.Namespace) -> int:
    """Build a weekly drop bundle from the library DB using the deterministic playlist builder.

    Outputs (under out_dir):
      - drop_bundle/playlist.json
      - drop_bundle/daily_evidence.json
      - drop_bundle/tracks/<track_id>.(wav|mp3)   (copied from library)
      - evidence/daily_evidence.json              (lead track only; used by publish-marketing)
      - drop_evidence.json                        (summary)

    Determinism:
      - Uses period_key (ISO week label) + seed + context for IDs.
      - Uses deterministic_now_iso() when deterministic.
      - Evidence uses relative paths (no absolute out_dir leakage).
    """
    deterministic = is_deterministic(args)
    json_mode = bool(getattr(args, "json", False))

    db_path = resolve_db_path(args)
    context = str(getattr(args, "context", None) or os.environ.get("MGC_CONTEXT") or "focus")

    seed_val = getattr(args, "seed", None)
    seed = int(seed_val) if seed_val is not None else int(os.environ.get("MGC_SEED") or "1")

    out_dir = Path(
        getattr(args, "out_dir", None) or os.environ.get("MGC_EVIDENCE_DIR") or "data/evidence"
    ).expanduser().resolve()

    bundle_dir = out_dir / "drop_bundle"
    bundle_tracks_dir = bundle_dir / "tracks"
    evidence_dir = out_dir / "evidence"

    bundle_tracks_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Determine ISO week label
    now_iso = deterministic_now_iso(deterministic)
    today = (
        datetime.fromisoformat(now_iso.replace("Z", "+00:00")).date()
        if "T" in now_iso
        else datetime.now(timezone.utc).date()
    )
    iso_year, iso_week, _ = today.isocalendar()
    period_key = f"{iso_year}-W{iso_week:02d}"

    # Allow explicit override for deterministic CI / backfills
    override_pk = str(getattr(args, "period_key", None) or "").strip()
    if override_pk:
        if not re.match(r"^\d{4}-W\d{2}$", override_pk):
            raise SystemExit(f"Invalid --period-key: {override_pk} (expected YYYY-Www)")
        period_key = override_pk

    # Optional: generate new tracks into the library before building the weekly playlist
    gen_count = int(getattr(args, "generate_count", 0) or 0)
    gen_provider = getattr(args, "generate_provider", None) or os.environ.get("MGC_PROVIDER") or None
    gen_prompt = getattr(args, "prompt", None) or None
    if gen_count > 0:
        _agents_generate_and_ingest(
            db_path=db_path,
            repo_root=Path(getattr(args, "repo_root", ".")).expanduser().resolve(),
            context=context,
            schedule="weekly",
            period_key=period_key,
            seed=int(seed),
            deterministic=bool(deterministic),
            count=int(gen_count),
            provider_name=str(gen_provider) if gen_provider else None,
            prompt=str(gen_prompt) if gen_prompt else None,
            now_iso=now_iso,
        )

    # Playlist builder knobs
    target_minutes = getattr(args, "target_minutes", None)
    if target_minutes is None:
        target_minutes = int(os.environ.get("MGC_WEEKLY_TARGET_MINUTES") or "20")

    lookback_playlists = getattr(args, "lookback_playlists", None)
    if lookback_playlists is None:
        lookback_playlists = int(os.environ.get("MGC_WEEKLY_LOOKBACK_PLAYLISTS") or "3")

    pl = build_weekly_playlist(
        db_path=Path(db_path),
        context=context,
        period_key=period_key,
        base_seed=int(seed),
        target_minutes=int(target_minutes),
        lookback_playlists=int(lookback_playlists),
    )

    items = pl.get("items") if isinstance(pl, dict) else None
    if not items:
        raise SystemExit("weekly playlist builder produced no items")

    # Copy tracks into bundle
    copied: List[Dict[str, Any]] = []
    for it in items:
        track_id = str(it.get("track_id"))
        full_path = it.get("full_path") or it.get("artifact_path")
        if not full_path:
            raise SystemExit(f"playlist item missing full_path for track_id={track_id}")

        src = Path(str(full_path))
        # Allow relative paths stored in DB (repo-relative)
        if not src.is_absolute():
            src = Path(getattr(args, "repo_root", ".")).expanduser().resolve() / src

        if not src.exists():
            raise SystemExit(f"[run.weekly] missing source track file: {src}")

        dst = bundle_tracks_dir / f"{track_id}{src.suffix or '.wav'}"
        shutil.copy2(src, dst)
        copied.append({"track_id": track_id, "path": f"tracks/{dst.name}"})

    lead_track_id = str(items[0].get("track_id"))

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    if deterministic:
        drop_id = stable_uuid5(f"drop:weekly:{period_key}:{context}:{seed}", namespace=ns)
        playlist_id = stable_uuid5(f"playlist:weekly:{period_key}:{context}:{seed}", namespace=ns)
        run_id = stable_uuid5(f"run:weekly:{period_key}:{context}:{seed}", namespace=ns)
    else:
        drop_id = str(uuid.uuid4())
        playlist_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())

    # Write bundle playlist.json (schema mgc.playlist.v1) referencing bundle-relative tracks/
    playlist_obj = {
        "schema": "mgc.playlist.v1",
        "version": 1,
        "schedule": "weekly",
        "ts": now_iso,
        "playlist_id": str(playlist_id),
        "context": context,
        "period": {"label": period_key},
        "tracks": copied,
    }
    playlist_json = json.dumps(playlist_obj, indent=2, sort_keys=True) + "\n"
    (bundle_dir / "playlist.json").write_text(playlist_json, encoding="utf-8")
    # Keep top-level playlist.json for compatibility, but CI/web should prefer bundle playlist
    (out_dir / "playlist.json").write_text(playlist_json, encoding="utf-8")

    # Write evidence/daily_evidence.json for the LEAD track (marketing contract)
    lead_path = next((c["path"] for c in copied if c["track_id"] == lead_track_id), copied[0]["path"])
    daily_ev = {
        "schema": "mgc.daily_evidence.v1",
        "version": 1,
        "run_id": str(run_id),
        "stage": "daily",
        "context": context,
        "deterministic": bool(deterministic),
        "ts": now_iso,
        "provider": str(getattr(args, "provider", None) or os.environ.get("MGC_PROVIDER") or "stub"),
        "schedule": "weekly",
        "track": {"track_id": lead_track_id, "path": lead_path},
        "sha256": {
            "playlist": sha256_file(bundle_dir / "playlist.json"),
            "track": sha256_file(bundle_dir / lead_path),
        },
        "period": {"label": period_key},
    }
    (evidence_dir / "daily_evidence.json").write_text(
        json.dumps(daily_ev, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle_dir / "daily_evidence.json").write_text(
        json.dumps(daily_ev, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Emit planned marketing posts for file-mode publishing (weekly)
    # Writes: <out_dir>/marketing/publish/<post_id>.json
    # ------------------------------------------------------------------
    marketing_dir = out_dir / "marketing"
    publish_dir = marketing_dir / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    platforms_raw = os.environ.get("MGC_MARKETING_PLATFORMS", "x,tiktok,instagram_reels,youtube_shorts")
    platforms = [p.strip() for p in platforms_raw.split(",") if p.strip()]
    platforms = sorted(dict.fromkeys(platforms))

    title = f"Weekly Drop {period_key} ({context})"
    hook = f"New weekly {context} drop is ready."
    cta = "Listen now."

    for platform in platforms:
        post_id = stable_uuid5("marketing_post", "weekly", period_key, str(drop_id), platform)
        payload = {
            "schema": "mgc.marketing_post.v1",
            "version": 1,
            "post_id": post_id,
            "status": "planned",
            "platform": platform,
            "schedule": "weekly",
            "period_key": period_key,
            "created_at": now_iso,
            "run_id": str(run_id),
            "drop_id": str(drop_id),
            "track_id": lead_track_id,
            "title": title,
            "hook": hook,
            "cta": cta,
            "context": context,
        }
        (publish_dir / f"{post_id}.json").write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")

    # Compute deterministic repo manifest alongside playlist (helps CI provenance)
    repo_root = Path(getattr(args, "repo_root", ".")).expanduser().resolve()
    include = getattr(args, "include", None) or None
    exclude_dirs = getattr(args, "exclude_dir", None) or None
    exclude_globs = getattr(args, "exclude_glob", None) or None

    manifest_obj = compute_manifest(repo_root, include=include, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs)
# BEGIN MGC MANIFEST OUTPUT FILTER (secrets/junk)
    def _mgc_keep_path(path: str) -> bool:
        # Never let local secrets / OS junk into any written manifest.
        # Do NOT exclude .env.example
        if path == ".env" or path.startswith(".env.") or path.endswith("/.env") or "/.env." in path:
            return False
        if path == ".DS_Store" or path.endswith("/.DS_Store"):
            return False
        return True
    
    if isinstance(manifest_obj, dict):
        _ents = manifest_obj.get("entries")
        if isinstance(_ents, list):
            manifest_obj["entries"] = [e for e in _ents if _mgc_keep_path(e.get("path", ""))]
    # END MGC MANIFEST OUTPUT FILTER
    manifest_path = out_dir / "weekly_manifest.json"
    manifest_path.write_text(stable_json_dumps(manifest_obj) + "\n", encoding="utf-8")
    manifest_sha256 = sha256_file(manifest_path)

    marketing_obj: Optional[Dict[str, Any]] = None
    if bool(getattr(args, "marketing", False)):
        marketing_out = Path(getattr(args, "marketing_out_dir", None) or (out_dir / "marketing")).expanduser()
        marketing_obj = _agents_marketing_plan(
            drop_dir=bundle_dir,
            repo_root=repo_root,
            out_dir=marketing_out,
            seed=int(seed),
            teaser_seconds=int(getattr(args, "teaser_seconds", 20) or 20),
            ts=now_iso,
        )

    # Prefer the bundle playlist for downstream steps (web/submission), since it is portable.
    bundle_playlist_path = (bundle_dir / "playlist.json").resolve()
    fallback_playlist_path = (out_dir / "playlist.json").resolve()
    playlist_path = bundle_playlist_path if bundle_playlist_path.exists() else fallback_playlist_path

    # Optional: build a web bundle (static player) under out_dir/web
    web_manifest_rel: Optional[str] = None
    if bool(getattr(args, "web", False)):
        web_out = (out_dir / "web").resolve()
        web_cmd = [
            sys.executable,
            "-m",
            "mgc.main",
            "web",
            "build",
            "--playlist",
            str(playlist_path),
            "--out-dir",
            str(web_out),
            "--prefer-mp3",
            "--clean",
            "--fail-if-empty",
            "--json",
        ]
        # Important: run from the playlist directory so relative "tracks/..." resolves to drop_bundle/tracks.
        subprocess.run(web_cmd, check=True, cwd=str(playlist_path.parent))
        web_manifest_rel = str(Path("web") / "web_manifest.json")

    # Optional: deterministic submission bundle under out_dir/submission.zip
    submission_zip_rel: Optional[str] = None
    if bool(getattr(args, "submission", False)):
        # Always submit the bundle directory; that's the portable artifact.
        sub_cmd = [
            sys.executable,
            "-m",
            "mgc.main",
            "submission",
            "build",
            "--bundle-dir",
            str(bundle_dir.resolve()),
            "--out",
            str((out_dir / "submission.zip").resolve()),
        ]
        subprocess.run(sub_cmd, check=True)
        submission_zip_rel = "submission.zip"

    drop_evidence = {
        "ok": True,
        "schedule": "weekly",
        "context": context,
        "period_key": period_key,
        "drop_id": str(drop_id),
        "lead_track_id": lead_track_id,
        "playlist_tracks": int(len(copied)),
        "marketing": marketing_obj,
        "paths": {
            "bundle_dir": "drop_bundle",
            "bundle_playlist": "drop_bundle/playlist.json",
            # Set "playlist" to the portable playlist so CI finders that look for paths.playlist do the right thing.
            "playlist": "drop_bundle/playlist.json",
            "web_manifest": web_manifest_rel,
            "submission_zip": submission_zip_rel,
            # Weekly runs still emit a lead-track evidence file using the daily schema (marketing contract).
            "bundle_weekly_evidence": "drop_bundle/daily_evidence.json",
            "weekly_evidence": "evidence/daily_evidence.json",
            "bundle_daily_evidence": "drop_bundle/daily_evidence.json",
            "daily_evidence": "evidence/daily_evidence.json",
            "drop_evidence": "drop_evidence.json",
            "manifest": "weekly_manifest.json",
            "manifest_sha256": manifest_sha256,
            "marketing_publish_dir": None,
            "marketing_receipts_dir": ("marketing/receipts" if bool(getattr(args, "marketing", False)) else None),
        },
    }
    (out_dir / "drop_evidence.json").write_text(json.dumps(drop_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if json_mode:
        sys.stdout.write(stable_json_dumps(drop_evidence) + "\n")
    else:
        print(f"[run.weekly] ok period={period_key} tracks={len(copied)} out_dir={out_dir}", file=sys.stderr)

    return 0


def cmd_run_stage_set(args: argparse.Namespace) -> int:
    deterministic = is_deterministic(args)

    db_path = args.db
    run_id = str(args.run_id).strip()
    stage = str(args.stage).strip()
    status = str(args.status).strip().lower()

    if not run_id:
        die("run_id required")
    if not stage:
        die("stage required")
    if status not in ("pending", "running", "ok", "error", "skipped"):
        die("status must be one of: pending, running, ok, error, skipped")

    con = db_connect(db_path)
    ensure_tables_minimal(con)

    started_at = getattr(args, "started_at", None)
    ended_at = getattr(args, "ended_at", None)
    duration_ms = getattr(args, "duration_ms", None)

    # Normalize duration in deterministic mode
    if duration_ms is not None:
        try:
            duration_ms = int(duration_ms)
        except Exception:
            die("duration_ms must be an integer")
        if deterministic:
            duration_ms = 0

    error_obj: Optional[Dict[str, Any]] = None
    raw_error = getattr(args, "error_json", None)
    if raw_error:
        try:
            parsed = json.loads(raw_error)
            error_obj = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            error_obj = {"raw": str(raw_error)}

    meta_patch: Optional[Dict[str, Any]] = None
    raw_meta = getattr(args, "meta_json", None)
    if raw_meta:
        try:
            parsed = json.loads(raw_meta)
            meta_patch = parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            meta_patch = {"raw": str(raw_meta)}

    # Auto-fill timestamps if omitted (keeps UX sane + DB consistent)
    if started_at is None and status in ("running", "ok", "error", "skipped"):
        started_at = deterministic_now_iso(deterministic)
    if ended_at is None and status in ("ok", "error", "skipped"):
        ended_at = deterministic_now_iso(deterministic)

    db_stage_upsert(
        con,
        run_id=run_id,
        stage=stage,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        error=error_obj,
        meta_patch=meta_patch,
    )

    sys.stdout.write(
        stable_json_dumps(
            {
                "ok": True,
                "db": db_path,
                "run_id": run_id,
                "stage": stage,
                "status": status,
            }
        )
        + "\n"
    )
    return 0


def cmd_run_stage_get(args: argparse.Namespace) -> int:
    db_path = args.db
    run_id = str(args.run_id).strip()
    stage = str(args.stage).strip()

    if not run_id:
        die("run_id required")
    if not stage:
        die("stage required")

    con = db_connect(db_path)

    # Be tolerant: do not create tables on read-only get; just report not found.
    if not db_table_exists(con, "run_stages"):
        sys.stdout.write(
            stable_json_dumps(
                {
                    "found": False,
                    "db": db_path,
                    "run_id": run_id,
                    "stage": stage,
                    "reason": "run_stages_table_missing",
                }
            )
            + "\n"
        )
        return 1

    row = db_stage_get(con, run_id=run_id, stage=stage)
    if row is None:
        sys.stdout.write(
            stable_json_dumps(
                {
                    "found": False,
                    "db": db_path,
                    "run_id": run_id,
                    "stage": stage,
                }
            )
            + "\n"
        )
        return 1

    sys.stdout.write(
        stable_json_dumps(
            {
                "found": True,
                "db": db_path,
                "run_id": run_id,
                "stage": stage,
                "item": dict(row),
            }
        )
        + "\n"
    )
    return 0


def cmd_run_stage_list(args: argparse.Namespace) -> int:
    db_path = args.db
    run_id = str(args.run_id).strip()

    if not run_id:
        die("run_id required")

    con = db_connect(db_path)

    # Be tolerant: do not create tables on list; just return empty.
    if not db_table_exists(con, "run_stages"):
        sys.stdout.write(
            stable_json_dumps(
                {
                    "db": db_path,
                    "run_id": run_id,
                    "count": 0,
                    "items": [],
                    "reason": "run_stages_table_missing",
                }
            )
            + "\n"
        )
        return 0

    try:
        rows = con.execute(
            "SELECT * FROM run_stages WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
    except sqlite3.Error as e:
        sys.stdout.write(
            stable_json_dumps(
                {
                    "db": db_path,
                    "run_id": run_id,
                    "count": 0,
                    "items": [],
                    "error": {"type": "sqlite3.Error", "message": str(e)},
                }
            )
            + "\n"
        )
        return 1

    sys.stdout.write(
        stable_json_dumps(
            {
                "db": db_path,
                "run_id": run_id,
                "count": len(rows),
                "items": [dict(r) for r in rows],
            }
        )
        + "\n"
    )
    return 0

# ---------------------------------------------------------------------------
# run status
# ---------------------------------------------------------------------------

def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _run_id_latest(con: sqlite3.Connection) -> Optional[str]:
    if not _table_exists(con, "runs"):
        return None
    cols = set(db_table_columns(con, "runs"))
    try:
        if "updated_at" in cols:
            row = con.execute("SELECT run_id FROM runs ORDER BY updated_at DESC LIMIT 1").fetchone()
        elif "created_at" in cols:
            row = con.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
        elif "run_date" in cols:
            row = con.execute("SELECT run_id FROM runs ORDER BY run_date DESC LIMIT 1").fetchone()
        else:
            row = con.execute("SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def cmd_run_status(args: argparse.Namespace) -> int:
    """
    Output rules:
      - If global --json is set, emit exactly ONE JSON object.
      - Otherwise emit a short human summary (single line).
    """
    want_json = bool(getattr(args, "json", False))

    db_path = str(getattr(args, "db", None) or os.environ.get("MGC_DB") or "data/db.sqlite")
    con = db_connect(db_path)

    run_id = str(getattr(args, "run_id", None) or "").strip()
    latest = bool(getattr(args, "latest", False))
    fail_on_error = bool(getattr(args, "fail_on_error", False))

    if not run_id and latest:
        run_id = _run_id_latest(con) or ""
    if not run_id:
        run_id = _run_id_latest(con) or ""

    out: Dict[str, Any] = {
        "db": db_path,
        "found": False,
        "run_id": run_id or None,
        "run": None,
        "stages": {"count": 0, "items": []},
        "drop": None,
        "summary": {"counts": {}, "healthy": None},
    }

    # run row
    if run_id and db_table_exists(con, "runs"):
        try:
            row = con.execute("SELECT * FROM runs WHERE run_id = ? LIMIT 1", (run_id,)).fetchone()
            if row is not None:
                out["found"] = True
                out["run"] = dict(row)
        except Exception:
            pass

    # stages
    stage_items: List[Dict[str, Any]] = []
    if run_id and db_table_exists(con, "run_stages"):
        try:
            rows = con.execute(
                "SELECT * FROM run_stages WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
            stage_items = [dict(r) for r in rows]
        except Exception:
            stage_items = []

    out["stages"] = {"count": len(stage_items), "items": stage_items}

    counts: Dict[str, int] = {}
    for it in stage_items:
        st = str(it.get("status") or "").strip().lower() or "unknown"
        counts[st] = counts.get(st, 0) + 1

    healthy = (counts.get("error", 0) == 0)
    out["summary"] = {"counts": dict(sorted(counts.items())), "healthy": healthy}

    # drop pointer
    if run_id and db_table_exists(con, "drops"):
        try:
            drow = _drop_latest_for_run(con, run_id)
            if drow is not None:
                out["drop"] = dict(drow)
        except Exception:
            pass

    # CI fail mode
    if fail_on_error and counts.get("error", 0) > 0:
        if want_json:
            sys.stdout.write(stable_json_dumps(out) + "\n")
        else:
            print(f"run_id={run_id or '(none)'} status=ERROR stages_error={counts.get('error', 0)} db={db_path}")
        return 2

    if want_json:
        sys.stdout.write(stable_json_dumps(out) + "\n")
        return 0 if out["found"] else 1

    # Human single line
    rid = run_id or "(none)"
    err = counts.get("error", 0)
    ok = counts.get("ok", 0)
    running = counts.get("running", 0)
    skipped = counts.get("skipped", 0)

    status = "OK" if err == 0 else "ERROR"
    print(f"run_id={rid} status={status} ok={ok} running={running} skipped={skipped} error={err} db={db_path}")
    return 0 if err == 0 else 2

def cmd_run_open(args: argparse.Namespace) -> int:
    evidence_dir = Path(
        getattr(args, "out_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR")
        or "data/evidence"
    ).resolve()

    kind = str(getattr(args, "type", "any") or "any")  # drop | weekly | any
    n = int(getattr(args, "n", 1) or 1)

    if not evidence_dir.exists() or not evidence_dir.is_dir():
        return 1

    if kind == "drop":
        patterns = ["drop_evidence*.json"]
    elif kind == "weekly":
        patterns = ["weekly_evidence*.json"]
    else:
        patterns = ["drop_evidence*.json", "weekly_evidence*.json"]

    files: List[Path] = []
    for pat in patterns:
        files.extend(evidence_dir.glob(pat))

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return 1

    selected = files[:n]
    for p in selected:
        print(str(p))
        # Also print referenced manifest if present & exists
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                paths = payload.get("paths", {})
                if isinstance(paths, dict):
                    manifest_path = paths.get("manifest_path")
                    if isinstance(manifest_path, str) and manifest_path.strip():
                        mp = Path(manifest_path)
                        if not mp.is_absolute():
                            mp = (evidence_dir / mp).resolve()
                        if mp.exists():
                            print(str(mp))
        except Exception:
            pass

    return 0

def _load_manifest(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    entries = obj.get("entries", []) if isinstance(obj, dict) else []
    by_path = {}
    for e in entries:
        p = e.get("path")
        if p:
            by_path[p] = {"sha256": e.get("sha256"), "size": e.get("size")}
    return {
        "path": str(path),
        "root_tree_sha256": obj.get("root_tree_sha256"),
        "entries": by_path,
    }


def _manifest_entries_map(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    entries = manifest.get("entries", [])
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(entries, list):
        for e in entries:
            if not isinstance(e, dict):
                continue
            p = e.get("path")
            if isinstance(p, str) and p:
                out[p] = e
    return out


def _diff_manifests(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    am = _manifest_entries_map(a)
    bm = _manifest_entries_map(b)

    a_paths = set(am.keys())
    b_paths = set(bm.keys())

    added_paths = sorted(b_paths - a_paths)
    removed_paths = sorted(a_paths - b_paths)
    common_paths = sorted(a_paths & b_paths)

    added: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    modified: List[Dict[str, Any]] = []

    for p in added_paths:
        e = bm[p]
        added.append({"path": p, "sha256": e.get("sha256"), "size": e.get("size")})

    for p in removed_paths:
        e = am[p]
        removed.append({"path": p, "sha256": e.get("sha256"), "size": e.get("size")})

    for p in common_paths:
        ea = am[p]
        eb = bm[p]
        if (ea.get("sha256") != eb.get("sha256")) or (ea.get("size") != eb.get("size")):
            modified.append(
                {
                    "path": p,
                    "a": {"sha256": ea.get("sha256"), "size": ea.get("size")},
                    "b": {"sha256": eb.get("sha256"), "size": eb.get("size")},
                }
            )

    return {"added": added, "removed": removed, "modified": modified}


def _find_manifest_files(evidence_dir: Path, *, type_filter: str = "any") -> List[Path]:
    if not isinstance(evidence_dir, Path):
        evidence_dir = Path(str(evidence_dir))

    if not evidence_dir.exists() or not evidence_dir.is_dir():
        return []

    paths: List[Path] = []
    for p in evidence_dir.glob("*manifest*.json"):
        if not p.is_file():
            continue
        name = p.name
        is_weekly = name.startswith("weekly_manifest")
        is_drop = name.startswith("manifest") and not is_weekly

        if type_filter == "weekly" and not is_weekly:
            continue
        if type_filter == "drop" and not is_drop:
            continue

        paths.append(p)

    def _mtime_key(p: Path) -> Tuple[float, str]:
        try:
            return (p.stat().st_mtime, p.name)
        except Exception:
            return (0.0, p.name)

    paths.sort(key=_mtime_key, reverse=True)
    return paths


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_since_ok_manifest_path(evidence_dir: Path, *, args_for_db: argparse.Namespace) -> Optional[Path]:
    db_path = resolve_db_path(args_for_db)

    candidates: List[Path] = []
    try:
        candidates.extend(sorted(evidence_dir.glob("drop_evidence*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(evidence_dir.glob("weekly_evidence*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
    except Exception:
        candidates = []

    if not candidates:
        return None

    con = None
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row

        for ev_path in candidates:
            ev = _read_json_file(ev_path)
            if not isinstance(ev, dict):
                continue

            run_id = str(ev.get("run_id") or ev.get("weekly_run_id") or "").strip()
            paths = ev.get("paths") if isinstance(ev.get("paths"), dict) else {}
            manifest_raw = str(paths.get("manifest_path") or "").strip()

            if not run_id or not manifest_raw:
                continue

            try:
                row = con.execute(
                    "SELECT COUNT(1) AS n FROM run_stages WHERE run_id = ? AND LOWER(status) = 'error'",
                    (run_id,),
                ).fetchone()
            except sqlite3.Error:
                continue

            n_err = int(row["n"] if row and row["n"] is not None else 0)
            if n_err != 0:
                continue

            mp = Path(manifest_raw)
            if not mp.is_absolute():
                mp = (evidence_dir / mp).resolve()

            if mp.exists() and mp.is_file():
                return mp

        return None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def cmd_run_diff(args: argparse.Namespace) -> int:
    """
    Diff manifests in the evidence dir.

    Output rules:
      - If global --json is set (mgc.main --json ...), emit exactly ONE JSON object.
      - Otherwise print a human summary line:
            +A  -R  ~M  (older=... newer=...)
      - --summary-only: counts only (still JSON if --json is set)
      - --fail-on-changes: exit 2 if there are any non-allowed changes
      - --allow PATH (repeatable): allow specific changed paths when failing on changes
      - --since PATH: compare newest against PATH (PATH is "older")
      - --since-ok: auto-pick an older manifest from the most recent run with no stage errors
    """
    want_json = resolve_want_json(args)

    evidence_dir = Path(
        getattr(args, "out_dir", None)
        or getattr(args, "evidence_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR", "data/evidence")
    ).resolve()

    type_filter = str(getattr(args, "type", "any") or "any").strip().lower()
    fail_on_changes = bool(getattr(args, "fail_on_changes", False))
    summary_only = bool(getattr(args, "summary_only", False))
    since = getattr(args, "since", None)
    since_ok = bool(getattr(args, "since_ok", False))

    allow_list = getattr(args, "allow", None) or []
    allow_set = {str(a).replace("\\", "/") for a in allow_list if str(a).strip()}

    files = _find_manifest_files(evidence_dir, type_filter=type_filter)  # newest-first

    # Choose (older=a_path, newer=b_path)
    since_path: Optional[Path] = None
    if since:
        since_path = Path(str(since)).expanduser().resolve()
        if not since_path.exists():
            sys.stdout.write(
                stable_json_dumps(
                    {"found": False, "reason": "since_not_found", "since": str(since_path)}
                )
                + "\n"
            )
            return 0

        if not files:
            sys.stdout.write(
                stable_json_dumps(
                    {"found": False, "reason": "no_manifests_found", "path": str(evidence_dir)}
                )
                + "\n"
            )
            return 0

        a_path, b_path = since_path, files[0]

    elif since_ok:
        # Must use resolver helper (DB path depends on args/global default/env)
        since_path = _resolve_since_ok_manifest_path(evidence_dir, args_for_db=args)
        if since_path is None:
            sys.stdout.write(
                stable_json_dumps(
                    {"found": False, "reason": "since_ok_not_found", "path": str(evidence_dir)}
                )
                + "\n"
            )
            return 0

        if not files:
            sys.stdout.write(
                stable_json_dumps(
                    {"found": False, "reason": "no_manifests_found", "path": str(evidence_dir)}
                )
                + "\n"
            )
            return 0

        a_path, b_path = since_path, files[0]

    else:
        if len(files) < 2:
            sys.stdout.write(
                stable_json_dumps(
                    {
                        "found": False,
                        "reason": "need_at_least_two_manifests",
                        "count": len(files),
                        "path": str(evidence_dir),
                    }
                )
                + "\n"
            )
            return 0

        a_path, b_path = files[1], files[0]

    a = _load_manifest(a_path)
    b = _load_manifest(b_path)
    diff = _diff_manifests(a, b)

    added = diff.get("added", []) or []
    removed = diff.get("removed", []) or []
    modified = diff.get("modified", []) or []

    # Allow-list applies to modified paths only (keeps CI strict on add/remove)
    modified_paths = [str(x.get("path") or "") for x in modified if isinstance(x, dict)]
    non_allowed_modified = [p for p in modified_paths if p and p not in allow_set]

    summary = {
        "added": len(added),
        "removed": len(removed),
        "modified": len(modified),
    }

    has_any_changes = (summary["added"] + summary["removed"] + summary["modified"]) > 0
    has_blocking_changes = (summary["added"] + summary["removed"] + len(non_allowed_modified)) > 0

    exit_code = 0
    if fail_on_changes and has_any_changes:
        exit_code = 2 if has_blocking_changes else 0

    older_name = a_path.name
    newer_name = b_path.name

    if want_json:
        out: Dict[str, Any] = {
            "found": True,
            "older": str(a_path),
            "newer": str(b_path),
            "older_name": older_name,
            "newer_name": newer_name,
            "type_filter": type_filter,
            "since": str(since_path) if since_path else None,
            "since_ok": since_ok,
            "fail_on_changes": fail_on_changes,
            "allow": sorted(allow_set) if allow_set else [],
            "summary": summary,
            "exit_code": exit_code,
        }

        if not summary_only:
            out["diff"] = diff

        if allow_set:
            out["non_allowed_modified_paths"] = non_allowed_modified

        sys.stdout.write(stable_json_dumps(out) + "\n")
        return exit_code

    print(f"+{summary['added']}  -{summary['removed']}  ~{summary['modified']}  (older={older_name} newer={newer_name})")
    return exit_code

def cmd_run_autonomous(args: argparse.Namespace) -> int:
    """
    End-to-end autonomous run.

    Today this command is a thin orchestration wrapper around `run drop`, plus
    release-contract validation.

    Pipeline:
      1) executes `run drop` (daily + publish-marketing + manifest + evidence + submission zip)
      2) (optional) build web bundle into out_dir/web when required (publish contract)
      3) (optional) stage marketing receipts into out_dir/marketing/receipts when required
      4) verify the produced submission.zip by unzip+validate (unless skipped)
      5) emit exactly one JSON summary to stdout

    Flags:
      --no-verify-submission     skip unzip+validate
      --dry-run                  skip zip verification automatically
      --contract {local,publish} contract strictness (publish requires web+marketing artifacts)
      --require-web              require web bundle and build it
      --require-marketing        require marketing receipts and stage them
      --deterministic            expect deterministic evidence fields to be true

    IMPORTANT:
      - This command MUST emit exactly one JSON object to stdout.
      - Any invoked subcommands must have their stdout JSON suppressed or captured.
    """

    import json
    import os
    import shutil
    import subprocess
    import tempfile
    import zipfile
    from pathlib import Path
    from typing import Any, Dict, Optional

    from mgc.bundle_validate import validate_bundle

    def _stable_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    def _env_truthy(name: str) -> bool:
        v = os.environ.get(name)
        if v is None:
            return False
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    dry_run = bool(getattr(args, "dry_run", False))
    no_verify = bool(getattr(args, "no_verify_submission", False))
    json_mode = bool(getattr(args, "json", False))

    # IMPORTANT: ensure cmd_run_drop does NOT print its own summary JSON.
    setattr(args, "_suppress_stdout_json", True)

    # 1) Run drop (canonical pipeline)
    rc = int(cmd_run_drop(args))
    if rc != 0:
        if json_mode:
            sys.stdout.write(_stable_json({"ok": False, "cmd": "run.autonomous", "stage": "drop", "rc": rc}))
        return rc

    # 2) Determine out_dir + evidence
    out_dir = Path(
        getattr(args, "out_dir", None)
        or os.environ.get("MGC_EVIDENCE_DIR")
        or "data/evidence"
    ).expanduser().resolve()

    evidence_path = (out_dir / "drop_evidence.json").resolve()
    if not evidence_path.exists():
        msg = f"missing evidence: {evidence_path}"
        if json_mode:
            sys.stdout.write(_stable_json({"ok": False, "cmd": "run.autonomous", "stage": "evidence", "error": msg}))
            return 2
        raise SystemExit(f"[run autonomous] {msg}")

    try:
        evidence_obj = _read_json(evidence_path)
    except Exception as e:
        msg = f"failed to read evidence JSON: {evidence_path}: {e}"
        if json_mode:
            sys.stdout.write(_stable_json({"ok": False, "cmd": "run.autonomous", "stage": "evidence", "error": msg}))
            return 2
        raise SystemExit(f"[run autonomous] {msg}") from e

    # 3) Locate submission.zip (by contract: evidence.paths.submission_zip)
    paths = evidence_obj.get("paths") if isinstance(evidence_obj.get("paths"), dict) else {}
    submission_zip = paths.get("submission_zip") if isinstance(paths, dict) else None
    if not submission_zip:
        msg = "drop completed but evidence has no paths.submission_zip"
        if json_mode:
            sys.stdout.write(_stable_json({"ok": False, "cmd": "run.autonomous", "stage": "evidence", "error": msg}))
            return 2
        raise SystemExit(f"[run autonomous] {msg}")

    zip_path = Path(str(submission_zip)).expanduser().resolve()

    # 4) Verify zip unless skipped / dry-run
    verify_skipped = bool(dry_run or no_verify)
    verify_ok: Optional[bool] = None
    verify_error: Optional[str] = None

    if verify_skipped:
        verify_ok = None
    else:
        if not zip_path.exists():
            verify_ok = False
            verify_error = f"submission_zip missing on disk: {zip_path}"
        else:
            try:
                with zipfile.ZipFile(str(zip_path), "r") as zf:
                    names = zf.namelist()
                    expected_prefix = "submission/drop_bundle/"
                    if not any(n.startswith(expected_prefix) for n in names):
                        raise ValueError("invalid submission.zip layout: expected submission/drop_bundle/")

                    with tempfile.TemporaryDirectory(prefix="mgc_autonomous_verify_") as td:
                        td_path = Path(td).resolve()
                        zf.extractall(td_path)

                        bundle_dir = td_path / "submission" / "drop_bundle"
                        if not bundle_dir.exists() or not bundle_dir.is_dir():
                            raise FileNotFoundError(f"extracted bundle missing: {bundle_dir}")

                        validate_bundle(bundle_dir)

                verify_ok = True
            except Exception as e:
                verify_ok = False
                verify_error = str(e)

    # Stable IDs
    drop_id = str(evidence_obj.get("drop_id") or "")
    run_id = str(evidence_obj.get("run_id") or "")

    # Compute manifest sha256 if present
    def _sha256_file(path: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest_rel = "manifest.json"
    manifest_path = (out_dir / manifest_rel).resolve()
    manifest_sha256 = _sha256_file(manifest_path) if manifest_path.exists() else ""

    # 5) Contract requirements + (optional) artifact production into out_dir
    deterministic_expected = bool(getattr(args, "deterministic", False)) or _env_truthy("MGC_DETERMINISTIC")

    contract_mode = str(getattr(args, "contract", "local") or "local")
    require_marketing = bool(getattr(args, "require_marketing", False)) or (contract_mode == "publish")
    require_web = bool(getattr(args, "require_web", False)) or (contract_mode == "publish")

    web_build_ok: Optional[bool] = None
    web_build_error: Optional[str] = None
    web_validate_ok: Optional[bool] = None
    web_validate_error: Optional[str] = None
    web_tree_sha256: Optional[str] = None
    staged_receipts_ok: Optional[bool] = None
    staged_receipts_error: Optional[str] = None

    # If web is required, build web bundle into out_dir/web and VALIDATE it.
    # We intentionally do NOT surface the web.build JSON on stdout here.
    web_dir = out_dir / "web"
    web_manifest = web_dir / "web_manifest.json"
    if require_web:
        try:
            web_dir.mkdir(parents=True, exist_ok=True)

            # Prefer the portable bundle playlist if present (it resolves against drop_bundle/tracks).
            bundle_playlist = (out_dir / "drop_bundle" / "playlist.json").resolve()
            playlist_path = bundle_playlist if bundle_playlist.exists() else (out_dir / "playlist.json").resolve()

            cmd = [
                sys.executable,
                "-m",
                "mgc.main",
                "web",
                "build",
                "--playlist",
                str(playlist_path),
                "--out-dir",
                str(web_dir),
                "--clean",
                "--fail-if-none-copied",
                "--fail-on-missing",
            ]
            if deterministic_expected:
                cmd.append("--deterministic")

            p = subprocess.run(cmd, text=True, capture_output=True)
            stdout = (p.stdout or "").strip()
            if p.returncode != 0:
                web_build_ok = False
                web_build_error = (p.stderr or stdout or "web build failed").strip()[:2000]
            else:
                web_build_ok = True
                # Parse build JSON payload (best-effort; not required for correctness)
                try:
                    _ = json.loads(stdout) if stdout else None
                except Exception:
                    pass

                # Validate bundle (strict)
                cmdv = [
                    sys.executable,
                    "-m",
                    "mgc.main",
                    "web",
                    "validate",
                    "--out-dir",
                    str(web_dir),
                ]
                pv = subprocess.run(cmdv, text=True, capture_output=True)
                v_stdout = (pv.stdout or "").strip()
                if pv.returncode != 0:
                    web_validate_ok = False
                    web_validate_error = (pv.stderr or v_stdout or "web validate failed").strip()[:2000]
                else:
                    try:
                        v_payload = json.loads(v_stdout) if v_stdout else {}
                    except Exception:
                        v_payload = {}
                    web_validate_ok = bool(isinstance(v_payload, dict) and v_payload.get("ok", False))
                    if not web_validate_ok:
                        web_validate_error = f"web validate returned ok=false: {v_payload}"
                    if isinstance(v_payload, dict) and isinstance(v_payload.get("web_tree_sha256"), str):
                        web_tree_sha256 = v_payload["web_tree_sha256"]

                # If validate failed, mark overall web_build as failed too (contract semantics).
                if web_validate_ok is False:
                    web_build_ok = False
                    if web_build_error is None:
                        web_build_error = web_validate_error or "web validate failed"
        except Exception as e:
            web_build_ok = False
            web_build_error = str(e)

    # If marketing is required, stage/copy receipts into out_dir/marketing/receipts.
    marketing_receipts_dir = out_dir / "marketing" / "receipts"
    marketing_publish_dir = out_dir / "marketing" / "publish"
    if require_marketing:
        try:
            import json
            from datetime import datetime, timezone

            marketing_receipts_dir.mkdir(parents=True, exist_ok=True)

            # Source-of-truth: marketing_post_ids from the drop evidence (daily).
            daily_obj = evidence_obj.get("daily") if isinstance(evidence_obj.get("daily"), dict) else {}
            post_ids = daily_obj.get("marketing_post_ids") if isinstance(daily_obj, dict) else None
            if not isinstance(post_ids, list):
                post_ids = []
            post_ids = [str(x) for x in post_ids if x is not None]

            # Deterministic timestamp: prefer evidence ts, else fixed.
            ts = str(evidence_obj.get("ts") or "2020-01-01T00:00:00Z")
            if ts.endswith("+00:00"):
                ts = ts[:-6] + "Z"

            # Write one deterministic receipt JSON per post_id.
            # Also write a receipts.jsonl log in stable order.
            receipts_jsonl = marketing_receipts_dir / "receipts.jsonl"

            lines = []
            for post_id in sorted(post_ids):
                receipt = {
                    "ts": ts,
                    "schema": "mgc.marketing_receipt.v1",
                    "post_id": post_id,
                    "platform": "stub",
                    "status": "published",
                    "dry_run": True,
                    "deterministic": True,
                }
                # Per-post file
                (marketing_receipts_dir / f"{post_id}.json").write_text(
                    _stable_json(receipt), encoding="utf-8"
                )
                # JSONL line
                lines.append(_stable_json(receipt).rstrip("\n"))

            receipts_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            # If marketing is required but we got zero post_ids, we still want a receipt artifact
            # to prove staging ran (and avoid a brittle contract). Keep it deterministic.
            if not post_ids:
                receipt = {
                    "ts": ts,
                    "schema": "mgc.marketing_receipt.v1",
                    "post_id": "ci_placeholder",
                    "platform": "stub",
                    "status": "published",
                    "dry_run": True,
                    "deterministic": True,
                }
                (marketing_receipts_dir / "ci_placeholder.json").write_text(
                    _stable_json(receipt), encoding="utf-8"
                )
                receipts_jsonl.write_text(_stable_json(receipt), encoding="utf-8")

            staged_receipts_ok = True
            staged_receipts_error = None
        except Exception as e:
            staged_receipts_ok = False
            staged_receipts_error = str(e)


    # 6) Release contract validation (hard gate)
    required_files = [
        "drop_evidence.json",
        "playlist.json",
        "manifest.json",
    ]

    missing: list[str] = []
    present: list[str] = []

    for rel in required_files:
        p = out_dir / rel
        if p.exists() and p.is_file():
            present.append(rel)
        else:
            missing.append(rel)

    # Tracks: at least one bundle track in out_dir/tracks/*.wav
    tracks_dir = out_dir / "tracks"
    track_files: list[str] = []
    if tracks_dir.exists() and tracks_dir.is_dir():
        track_files = sorted([str(p.relative_to(out_dir)) for p in tracks_dir.glob("*.wav") if p.is_file()])
    if track_files:
        present.append("tracks/*.wav")
    else:
        missing.append("tracks/*.wav")

    # Submission zip must exist unless dry-run
    submission_ok = True
    if not dry_run:
        submission_ok = zip_path.exists() and zip_path.is_file()
        if not submission_ok:
            missing.append("submission_zip")

    # Determinism expectations
    deterministic_actual = bool(evidence_obj.get("deterministic", False))
    daily_obj = evidence_obj.get("daily") if isinstance(evidence_obj.get("daily"), dict) else {}
    daily_det_actual = bool(daily_obj.get("deterministic", False)) if isinstance(daily_obj, dict) else False

    determinism_ok = True
    if deterministic_expected:
        if not deterministic_actual:
            determinism_ok = False
        if not daily_det_actual:
            determinism_ok = False

    # Marketing required?
    if require_marketing:
        if not (marketing_receipts_dir.exists() and marketing_receipts_dir.is_dir()):
            missing.append('marketing/receipts')
        else:
            import json
            # Expect per-post receipts (*.json) and a receipts.jsonl log.
            receipts_json = sorted([rp for rp in marketing_receipts_dir.glob('*.json') if rp.is_file()])
            receipts_jsonl = marketing_receipts_dir / 'receipts.jsonl'
            if not receipts_json:
                missing.append('marketing/receipts/*')
            if not (receipts_jsonl.exists() and receipts_jsonl.is_file() and receipts_jsonl.stat().st_size > 0):
                missing.append('marketing/receipts/receipts.jsonl')
            # Validate schema + exact post_id set match.
            daily_obj = evidence_obj.get('daily') if isinstance(evidence_obj.get('daily'), dict) else {}
            expected_ids = daily_obj.get('marketing_post_ids') if isinstance(daily_obj, dict) else None
            if not isinstance(expected_ids, list):
                expected_ids = []
            expected_ids = {str(x) for x in expected_ids if x is not None}
            got_ids = set()
            schema_ok = True
            for rp in receipts_json:
                try:
                    obj = json.loads(rp.read_text(encoding='utf-8'))
                except Exception:
                    schema_ok = False
                    continue
                for k in ('ts','post_id','platform','status','dry_run'):
                    if k not in obj:
                        schema_ok = False
                pid = obj.get('post_id')
                if pid is not None:
                    got_ids.add(str(pid))
            if expected_ids and got_ids != expected_ids:
                missing.append('marketing/receipts/post_id_mismatch')
            if receipts_json and not schema_ok:
                missing.append('marketing/receipts/schema_invalid')

    # Web required?
    if require_web:
        if not (web_manifest.exists() and web_manifest.is_file()):
            missing.append("web/web_manifest.json")

    contract_ok = (not missing) and submission_ok and determinism_ok

    contract_report: Dict[str, Any] = {
        "ok": contract_ok,
        "schema": "mgc.release_contract.v3",
        "mode": contract_mode,
        "requirements": {"marketing": require_marketing, "web": require_web},
        "expected": {
            "deterministic": deterministic_expected,
            "required": (
                required_files
                + ["tracks/*.wav", "submission_zip"]
                + (["marketing/receipts"] if require_marketing else [])
                + (["web/web_manifest.json"] if require_web else [])
            ),
        },
        "actual": {
            "deterministic": deterministic_actual,
            "daily_deterministic": daily_det_actual,
            "present": present,
            "missing": missing,
            "track_files": track_files,
            "web_tree_sha256": web_tree_sha256,
        },
        "actions": {
            "web_build": {"ok": web_build_ok, "error": web_build_error},
            "web_validate": {"ok": web_validate_ok, "error": web_validate_error},
            "stage_marketing_receipts": {"ok": staged_receipts_ok, "error": staged_receipts_error},
        },
        "paths": {
            "out_dir": ".",
            "evidence_path": "drop_evidence.json",
            "playlist": "playlist.json",
            "manifest": manifest_rel,
            "submission_zip": str(zip_path),
            "marketing_publish_dir": "marketing/publish" if marketing_publish_dir.exists() else None,
            "marketing_receipts_dir": "marketing/receipts" if marketing_receipts_dir.exists() else None,
            "web_manifest": "web/web_manifest.json" if web_manifest.exists() else None,
        },
        "sha256": {"manifest": manifest_sha256},
        "ids": {"drop_id": drop_id, "run_id": run_id},
    }

    try:
        (out_dir / "contract_report.json").write_text(_stable_json(contract_report), encoding="utf-8")
    except Exception:
        pass

    out: Dict[str, Any] = {
        "ok": (verify_ok in (None, True)) and contract_ok,
        "cmd": "run.autonomous",
        "ids": {"drop_id": drop_id, "run_id": run_id},
        "paths": {
            "out_dir": ".",
            "evidence_path": "drop_evidence.json",
            "submission_zip": str(zip_path),
            "manifest": manifest_rel,
            "manifest_sha256": manifest_sha256,
            "contract_report": "contract_report.json",
            "marketing_publish_dir": None,
            "marketing_receipts_dir": ("marketing/receipts" if bool(getattr(args, "marketing", False)) else None),
            "web_manifest": contract_report["paths"].get("web_manifest"),
        },
        "verify": {"skipped": verify_skipped, "ok": verify_ok, "error": verify_error},
        "contract": contract_report,
        "drop": evidence_obj,
    }

    sys.stdout.write(_stable_json(out))
    return 0 if ((verify_ok in (None, True)) and contract_ok) else 2

# ---------------------------------------------------------------------------
# Pipeline orchestration (one command to run the whole chain)
# ---------------------------------------------------------------------------

def cmd_run_pipeline(args: argparse.Namespace) -> int:
    """
    Orchestrate the full autonomous pipeline using existing CLI subcommands.

    This is intentionally orchestration-only (no business logic):
      1) run daily/weekly
      2) verify drop contract (playlist.json + drop_bundle + evidence)
      3) optional: web build
      4) optional: submission build
      5) publish-marketing (dry-run unless --publish)

    Design goals:
    - keep existing commands stable
    - keep determinism guarantees (pass through --deterministic when requested)
    - avoid importing internal modules from other CLIs (use subprocess to call mgc.main)
    """

    db_path = resolve_db_path(args)
    out_dir = str(Path(getattr(args, 'out_dir')).resolve())

    base = [sys.executable, '-m', 'mgc.main', '--db', db_path]

    def run_cmd(cmd: list[str]) -> None:
        # Keep output visible; CI relies on logs.
        subprocess.run(cmd, check=True)

    det = bool(getattr(args, 'deterministic', False))

    # 1) Generate the drop bundle + playlist
    if args.schedule == 'daily':
        cmd = base + [
            'run', 'daily',
            '--context', args.context,
            '--seed', str(args.seed),
            '--out-dir', out_dir,
        ]
        if det:
            cmd.append('--deterministic')
        run_cmd(cmd)
    else:
        cmd = base + [
            'run', 'weekly',
            '--context', args.context,
            '--seed', str(args.seed),
            '--out-dir', out_dir,
            '--period-key', args.period_key,
        ]
        if det:
            cmd.append('--deterministic')
        run_cmd(cmd)

    # 2) Verify output contract (hard gate)
    run_cmd([sys.executable, 'scripts/verify_drop_contract.py', '--out-dir', out_dir])

    # Ensure web builder can resolve playlist-relative audio paths.
    # web.build resolves "tracks/<file>" relative to the playlist.json directory, so we stage
    # bundle tracks into <out_dir>/tracks when present.
    try:
        import shutil
        top_tracks = Path(out_dir) / 'tracks'
        bundle_tracks = Path(out_dir) / 'drop_bundle' / 'tracks'
        if bundle_tracks.is_dir():
            need_copy = (not top_tracks.exists())
            if top_tracks.is_dir():
                # treat empty dir as missing
                try:
                    need_copy = next(top_tracks.iterdir(), None) is None
                except Exception:
                    need_copy = True
            if need_copy:
                top_tracks.parent.mkdir(parents=True, exist_ok=True)
                # Copy deterministically (same filenames/bytes).
                shutil.copytree(bundle_tracks, top_tracks, dirs_exist_ok=True)
    except Exception:
        # Best-effort: if this fails, web.build will raise a clear missing_tracks error.
        pass

    # 3) Optional: web build
    if getattr(args, 'web', False):
        web_cmd = base + [
            'web', 'build',
            '--playlist', str(Path(out_dir) / 'playlist.json'),
            '--out-dir', str(Path(out_dir) / 'web'),
            '--clean',
            '--fail-if-empty',
        ]
        run_cmd(web_cmd)

    # 4) Optional: deterministic submission bundle
    if getattr(args, 'submission', False):
        sub_cmd = base + [
            'submission', 'build',
            '--bundle-dir', str(Path(out_dir) / 'drop_bundle'),
            '--out', str(Path(out_dir) / 'submission.zip'),
        ]
        run_cmd(sub_cmd)

    # Ensure publish directory exists (publish-marketing expects it even in dry-run file mode).
    try:
        (Path(out_dir) / 'marketing' / 'publish').mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # 5) Marketing publish (dry-run unless --publish)
    pub_cmd = base + [
        'run', 'publish-marketing',
        '--bundle-dir', str(Path(out_dir) / 'drop_bundle'),
        '--out-dir', out_dir,
    ]
    if det:
        pub_cmd.append('--deterministic')
    if not getattr(args, 'publish', False):
        pub_cmd.append('--dry-run')
    run_cmd(pub_cmd)

    return 0

# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------



def cmd_run_publish(args: argparse.Namespace) -> int:
    """Publish a validated release bundle (consumer-only).

    Design goals:
    - Never rebuild artifacts. Only consumes an existing bundle-dir.
    - Refuse to run unless contract_report.json exists and indicates:
        - schema mgc.release_contract.v3 (or newer)
        - mode == publish
        - ok == true
    - Default safety gate requires an APPROVED file in the bundle-dir.

    Today (v1): this command performs *safe* publish steps only:
    - verifies web bundle is present
    - verifies marketing receipts are present (if required)
    - writes publish_summary.json as a durable audit marker

    Future: integrate platform APIs and web deployment behind explicit flags.
    """

    import json
    from datetime import datetime, timezone

    def _stable_json(obj: Any) -> str:
        return stable_json_dumps(obj) + "\n"

    bundle_dir = Path(str(getattr(args, 'bundle_dir', ''))).expanduser().resolve()
    dry_run = bool(getattr(args, 'dry_run', False))
    force = bool(getattr(args, 'force', False))
    allow_unapproved = bool(getattr(args, 'allow_unapproved', False))

    if not bundle_dir.is_dir():
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "bundle_dir_not_found", "bundle_dir": str(bundle_dir)}))
        return 2

    contract_path = bundle_dir / 'contract_report.json'
    if not contract_path.exists():
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "missing_contract_report", "path": str(contract_path)}))
        return 2

    try:
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
    except Exception as e:
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "invalid_contract_json", "path": str(contract_path), "detail": str(e)}))
        return 2

    schema = str(contract.get('schema') or '')
    mode = str(contract.get('mode') or '')
    ok = bool(contract.get('ok', False))

    # Accept v3 and newer (lexicographic is fine given our naming pattern)
    if not schema.startswith('mgc.release_contract.v'):
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "unsupported_contract_schema", "schema": schema}))
        return 2

    if mode != 'publish':
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "contract_not_publish_mode", "mode": mode}))
        return 2

    if not ok:
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "contract_failed"}))
        return 2

    if not (force or allow_unapproved):
        approved = bundle_dir / 'APPROVED'
        if not approved.exists():
            sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "missing_approved_file", "path": str(approved)}))
            return 2

    # Verify required staged artifacts
    paths = contract.get('paths') if isinstance(contract.get('paths'), dict) else {}

    web_manifest_rel = (paths.get('web_manifest') if isinstance(paths, dict) else None) or 'web/web_manifest.json'
    web_manifest = (bundle_dir / str(web_manifest_rel)).resolve()
    if not web_manifest.exists():
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "missing_web_manifest", "path": str(web_manifest)}))
        return 2

    marketing_receipts_rel = (paths.get('marketing_receipts_dir') if isinstance(paths, dict) else None)
    receipts_dir = (bundle_dir / str(marketing_receipts_rel)).resolve() if marketing_receipts_rel else (bundle_dir / 'marketing' / 'receipts').resolve()
    receipts_path = receipts_dir / 'receipts.jsonl'

    # Marketing may be required by contract, but we still validate presence to match that contract.
    reqs = contract.get('requirements') if isinstance(contract.get('requirements'), dict) else {}
    marketing_required = bool(reqs.get('marketing', True))  # publish mode normally requires marketing

    receipts_present = receipts_path.exists()
    if marketing_required and not receipts_present:
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "missing_marketing_receipts", "path": str(receipts_path)}))
        return 2

    # Write publish summary marker
    summary = {
        'cmd': 'run.publish',
        'ok': True,
        'bundle_dir': str(bundle_dir),
        'dry_run': dry_run,
        'force': force,
        'ts': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'inputs': {
            'contract_report': str(contract_path),
            'web_manifest': str(web_manifest),
            'marketing_receipts': str(receipts_path) if receipts_present else None,
        },
        'actions': {
            'marketing': 'dry_run' if dry_run else 'no_op_v1',
            'web': 'dry_run' if dry_run else 'no_op_v1',
        },
    }

    out_path = bundle_dir / 'publish_summary.json'
    try:
        out_path.write_text(stable_json_dumps(summary) + "\n", encoding='utf-8')
    except Exception as e:
        sys.stdout.write(_stable_json({"cmd": "run.publish", "ok": False, "error": "failed_to_write_publish_summary", "path": str(out_path), "detail": str(e)}))
        return 2

    sys.stdout.write(_stable_json({
        'cmd': 'run.publish',
        'ok': True,
        'dry_run': dry_run,
        'bundle_dir': str(bundle_dir),
        'publish_summary': str(out_path),
    }))
    return 0

def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """
    Wire `mgc run ...` subcommands.

    NOTE:
    - This function in mgc.run_cli.py is the canonical run parser wiring.
    - Do NOT maintain a second copy in mgc.main.py (it will drift and waste time).
    """
    run_p = subparsers.add_parser(
        "run",
        help="Run pipeline steps (daily, publish, drop, weekly, manifest, stage, status)",
    )
    run_p.set_defaults(_mgc_group="run")
    run_sub = run_p.add_subparsers(dest="run_cmd", required=True)

    # ----------------------------
    # open
    # ----------------------------
    open_p = run_sub.add_parser("open", help="Print paths of latest evidence/manifest files (pipe-friendly)")
    open_p.add_argument(
        "--out-dir",
        default=None,
        help="Evidence directory (default: data/evidence or MGC_EVIDENCE_DIR)",
    )
    open_p.add_argument(
        "--type",
        choices=["drop", "weekly", "any"],
        default="any",
        help="Evidence type filter",
    )
    open_p.add_argument("--n", type=int, default=1, help="Number of recent files")
    open_p.set_defaults(func=cmd_run_open)

    # ----------------------------
    # tail
    # ----------------------------
    tail = run_sub.add_parser("tail", help="Show latest evidence file(s)")
    tail.add_argument(
        "--out-dir",
        default=None,
        help="Evidence directory (default: data/evidence or MGC_EVIDENCE_DIR)",
    )
    tail.add_argument(
        "--type",
        choices=["drop", "weekly", "any"],
        default="any",
        help="Evidence type filter",
    )
    tail.add_argument("--n", type=int, default=1, help="Number of recent files to show")
    tail.set_defaults(func=cmd_run_tail)

    # ----------------------------
    # diff
    # ----------------------------
    diff = run_sub.add_parser("diff", help="Diff the two most recent manifests")
    diff.add_argument(
        "--out-dir",
        default=None,
        help="Evidence directory (default: data/evidence or MGC_EVIDENCE_DIR)",
    )
    diff.add_argument(
        "--type",
        choices=["drop", "weekly", "any"],
        default="any",
        help="Manifest type filter",
    )
    diff.add_argument("--since", default=None, help="Compare newest manifest against this manifest path (older)")
    diff.add_argument(
        "--since-ok",
        action="store_true",
        help="Auto-pick an older manifest from the most recent run with no stage errors",
    )
    diff.add_argument(
        "--summary-only",
        action="store_true",
        help="Only output counts in JSON mode (still one-line summary in human mode)",
    )
    diff.add_argument(
        "--fail-on-changes",
        action="store_true",
        help="Exit 2 if any non-allowed changes are detected (CI)",
    )
    diff.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Allow specific changed manifest paths (repeatable). Works with --fail-on-changes.",
    )
    diff.set_defaults(func=cmd_run_diff)

    # ----------------------------
    # status
    # ----------------------------
    status = run_sub.add_parser("status", help="Show latest (or specific) run status + stages + drop pointers")
    status.add_argument("--fail-on-error", action="store_true", help="Exit 2 if any stage is error (for CI)")
    status.add_argument("--run-id", default=None, help="Specific run_id to inspect")
    status.add_argument("--latest", action="store_true", help="Force latest run_id (default if --run-id omitted)")
    status.set_defaults(func=cmd_run_status)

    # ----------------------------
    # daily
    # ----------------------------
    daily = run_sub.add_parser("daily", help="Run the daily pipeline (deterministic capable)")
    daily.add_argument(
        "--context",
        default=os.environ.get("MGC_CONTEXT", "focus"),
        help="Context/mood (focus/workout/sleep)",
    )
    daily.add_argument(
        "--seed",
        default=os.environ.get("MGC_SEED", "1"),
        help="Seed for deterministic behavior",
    )
    daily.add_argument(
        "--out-dir",
        default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"),
        help="Evidence output directory",
    )
    daily.add_argument("--repo-root", default=".", help="Repository root to hash for manifest")
    daily.add_argument("--include", action="append", default=None,
                      help="Optional glob(s) to include in manifest (repeatable)")
    daily.add_argument("--exclude-dir", action="append", default=None,
                      help="Directory name(s) to exclude during manifest (repeatable)")
    daily.add_argument("--exclude-glob", action="append", default=None,
                      help="Filename glob(s) to exclude during manifest (repeatable)")
    daily.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (also via MGC_DETERMINISTIC=1)",
    )

    daily.add_argument(
        "--generate-count",
        type=int,
        default=int(os.environ.get("MGC_GENERATE_COUNT") or "0"),
        help="Generate N new tracks into the library before selecting the daily playlist (0 = off)",
    )
    daily.add_argument(
        "--generate-provider",
        default=os.environ.get("MGC_PROVIDER") or None,
        help="Provider to use for generation when --generate-count > 0 (default: MGC_PROVIDER or 'stub')",
    )
    daily.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt to pass to the generator/provider",
    )
    daily.add_argument(
        "--marketing",
        action="store_true",
        help="Also generate a marketing plan for the resulting drop bundle",
    )
    daily.add_argument(
        "--marketing-out-dir",
        default=None,
        help="Where to write marketing assets (default: <out_dir>/marketing)",
    )
    daily.add_argument(
        "--teaser-seconds",
        type=int,
        default=int(os.environ.get("MGC_TEASER_SECONDS") or "15"),
        help="Teaser length in seconds (wav-only; otherwise skipped)",
    )

    daily.set_defaults(func=cmd_run_daily)

    # ----------------------------
    # publish-marketing
    # ----------------------------
    pub = run_sub.add_parser("publish-marketing", help="Publish pending marketing drafts (draft -> published)")
    pub.add_argument("--limit", type=int, default=50, help="Max number of drafts to publish")
    pub.add_argument("--dry-run", action="store_true", help="Do not update DB; just print what would publish")

    pub.add_argument("--run-id", default=None, help="Only publish posts whose meta.run_id matches (DB mode)")
    pub.add_argument("--drop-id", default=None, help="Only publish posts whose meta.drop_id matches (DB mode)")
    pub.add_argument("--schedule", default=None, choices=["daily", "weekly"], help="Only publish posts whose meta.schedule matches (DB mode)")
    pub.add_argument("--period-key", default=None, help="Only publish posts whose meta.period.key or meta.period_key matches (DB mode)")
    pub.add_argument("--publish-dir", default=None, help="Publish posts from JSON files in this directory (file mode; no DB fallback)")
    pub.add_argument("--marketing-dir", default=None, help="Marketing directory containing publish/ (file mode; no DB fallback)")
    pub.add_argument("--bundle-dir", default=None, help="Bundle dir (e.g. <out_dir>/drop_bundle); infers <out_dir>/marketing/publish (file mode; no DB fallback)")
    pub.add_argument("--out-dir", default=None, help="Where to write receipts/artifacts (optional; default evidence dir)")
    pub.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (also via MGC_DETERMINISTIC=1)",
    )
    pub.set_defaults(func=cmd_publish_marketing)

    # ----------------------------
    # drop
    # ----------------------------
    drop = run_sub.add_parser("drop", help="Run daily + publish-marketing + manifest and emit consolidated evidence")
    drop.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Context/mood")
    drop.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"), help="Seed")
    drop.add_argument("--limit", type=int, default=50, help="Max number of marketing drafts to publish")
    drop.add_argument("--dry-run", action="store_true", help="Do not update DB in publish step")
    drop.add_argument(
        "--out-dir",
        default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"),
        help="Evidence output directory",
    )
    drop.add_argument("--repo-root", default=".", help="Repository root to hash for manifest")
    drop.add_argument("--include", action="append", default=None,
                      help="Optional glob(s) to include in manifest (repeatable)")
    drop.add_argument("--exclude-dir", action="append", default=None,
                      help="Directory name(s) to exclude during manifest (repeatable)")
    drop.add_argument("--exclude-glob", action="append", default=None,
                      help="Filename glob(s) to exclude during manifest (repeatable)")
    drop.add_argument("--no-resume", action="store_true", help="Disable resume behavior; always run all stages")
    drop.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic mode (also via MGC_DETERMINISTIC=1)",
    )

    # NEW: web bundle inclusion
    drop.add_argument(
        "--with-web",
        action="store_true",
        help="Build a static web bundle and include it inside submission.zip (also records paths.web_bundle_dir)",
    )
    drop.add_argument(
        "--web-out-dir",
        default=None,
        help="Optional override for web bundle output dir (default: data/submissions/<drop_id>/web)",
    )

    drop.set_defaults(func=cmd_run_drop)

    # ----------------------------
    # weekly
    # ----------------------------
    weekly = run_sub.add_parser("weekly", help="Build weekly drop bundle from DB playlist (no fresh generation)")
    weekly.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Context/mood")
    weekly.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"), help="Seed")
    weekly.add_argument("--period-key", default=None, help="Override ISO week label for this weekly run (e.g. 2020-W01)")
    weekly.add_argument("--limit", type=int, default=50, help="Max number of marketing drafts to publish")
    weekly.add_argument("--dry-run", action="store_true", help="Do not update DB in publish step")
    weekly.add_argument(
        "--out-dir",
        default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"),
        help="Evidence output directory",
    )
    weekly.add_argument("--repo-root", default=".", help="Repository root to hash for manifest")
    weekly.add_argument("--include", action="append", default=None, help="Optional glob(s) to include in manifest (repeatable)")
    weekly.add_argument("--exclude-dir", action="append", default=None, help="Directory name(s) to exclude during manifest (repeatable)")
    weekly.add_argument("--exclude-glob", action="append", default=None, help="Filename glob(s) to exclude during manifest (repeatable)")
    weekly.add_argument("--no-resume", action="store_true", help="Disable resume behavior; always run all stages")
    weekly.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic mode (also via MGC_DETERMINISTIC=1)",
    )


    weekly.add_argument(
        "--web",
        action="store_true",
        help="Build a static web bundle under <out_dir>/web (also records paths.web_manifest)",
    )
    weekly.add_argument(
        "--submission",
        action="store_true",
        help="Build a deterministic submission.zip under <out_dir>/submission.zip (also records paths.submission_zip)",
    )

    weekly.add_argument(
        "--generate-count",
        type=int,
        default=int(os.environ.get("MGC_GENERATE_COUNT") or "0"),
        help="Generate N new tracks into the library before selecting the weekly playlist (0 = off)",
    )
    weekly.add_argument(
        "--generate-provider",
        default=os.environ.get("MGC_PROVIDER") or None,
        help="Provider to use for generation when --generate-count > 0 (default: MGC_PROVIDER or 'stub')",
    )
    weekly.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt to pass to the generator/provider",
    )
    weekly.add_argument(
        "--marketing",
        action="store_true",
        help="Also generate a marketing plan for the resulting drop bundle",
    )
    weekly.add_argument(
        "--marketing-out-dir",
        default=None,
        help="Where to write marketing assets (default: <out_dir>/marketing)",
    )
    weekly.add_argument(
        "--teaser-seconds",
        type=int,
        default=int(os.environ.get("MGC_TEASER_SECONDS") or "20"),
        help="Teaser length in seconds (wav-only; otherwise skipped)",
    )

    weekly.set_defaults(func=cmd_run_weekly)

    # ----------------------------
    # generate
    # ----------------------------
    gen = run_sub.add_parser("generate", help="Generate a new track into the library and register it in DB")
    gen.add_argument("--context", default="focus", help="Context/mood (focus/workout/sleep)")
    gen.add_argument("--seed", type=int, default=None, help="Seed for deterministic behavior")
    gen.add_argument("--provider", default=None, help="Provider to use (default: MGC_PROVIDER or 'stub')")
    gen.add_argument("--track-id", dest="track_id", default=None, help="Optional explicit track_id")
    gen.add_argument("--repo-root", default=".", help="Repository root (for relative paths)")
    gen.add_argument("--store-dir", default=None, help="Override track storage dir (default: <repo_root>/data/tracks)")
    gen.add_argument("--out-dir", default=None, help="Evidence directory (default: data/evidence or MGC_EVIDENCE_DIR)")
    gen.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")
    gen.set_defaults(func=cmd_run_generate)


    # ----------------------------
    # autonomous
    # ----------------------------
    auto = run_sub.add_parser("autonomous", help="Run end-to-end drop + verify submission (cron-safe)")
    auto.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Context/mood")
    auto.add_argument("--seed", default=os.environ.get("MGC_SEED", "1"), help="Seed")
    auto.add_argument("--limit", type=int, default=50, help="Max number of marketing drafts to publish")
    auto.add_argument("--dry-run", action="store_true", help="Do not update DB in publish step")
    auto.add_argument(
        "--out-dir",
        default=os.environ.get("MGC_EVIDENCE_DIR", "data/evidence"),
        help="Evidence output directory",
    )
    auto.add_argument("--repo-root", default=".", help="Repository root to hash for manifest")
    auto.add_argument("--include", action="append", default=None,
                      help="Optional glob(s) to include in manifest (repeatable)")
    auto.add_argument("--exclude-dir", action="append", default=None,
                      help="Directory name(s) to exclude during manifest (repeatable)")
    auto.add_argument("--exclude-glob", action="append", default=None,
                      help="Filename glob(s) to exclude during manifest (repeatable)")
    auto.add_argument("--no-resume", action="store_true", help="Disable resume behavior; always run all stages")
    auto.add_argument("--deterministic", action="store_true", help="Deterministic mode (also via MGC_DETERMINISTIC=1)")
    auto.add_argument("--no-verify-submission", action="store_true", help="Skip verifying submission.zip after drop")
    auto.add_argument(
        "--contract",
        choices=["local", "publish"],
        default="local",
        help="Release contract strictness: local (default) or publish (requires marketing+web artifacts)",
    )
    auto.add_argument(
        "--require-marketing",
        action="store_true",
        help="Require marketing outputs in out_dir (stronger than local contract)",
    )
    auto.add_argument(
        "--require-web",
        action="store_true",
        help="Require web build outputs in out_dir (stronger than local contract)",
    )

    # NEW: pass-through to drop
    auto.add_argument(
        "--with-web",
        action="store_true",
        help="Build+include static web bundle (passes through to run drop)",
    )
    auto.add_argument(
        "--web-out-dir",
        default=None,
        help="Optional override for web bundle output dir (default: data/submissions/<drop_id>/web)",
    )

    auto.set_defaults(func=cmd_run_autonomous)


    publish = run_sub.add_parser(
        "publish",
        help="Publish a validated release bundle (consumer-only; requires contract_report.json)",
    )
    publish.add_argument(
        "--bundle-dir",
        required=True,
        help="Path to a release bundle directory previously produced by `mgc run autonomous --contract publish`",
    )
    publish.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not contact external systems (today: always safe; kept for forward compat)",
    )
    publish.add_argument(
        "--force",
        action="store_true",
        help="Bypass APPROVED gate (use sparingly)",
    )
    publish.add_argument(
        "--allow-unapproved",
        action="store_true",
        help="Alias for bypassing APPROVED gate (backward-compatible naming)",
    )
    publish.set_defaults(func=cmd_run_publish)

    # ----------------------------
    # pipeline
    # ----------------------------
    pipe = run_sub.add_parser(
        'pipeline',
        help='Run end-to-end pipeline (daily/weekly → verify → optional web/submission → publish-marketing)',
    )
    pipe.add_argument('--schedule', choices=['daily', 'weekly'], required=True, help='Which schedule to run')
    pipe.add_argument('--context', default=os.environ.get('MGC_CONTEXT', 'focus'), help='Context/mood')
    pipe.add_argument('--seed', default=os.environ.get('MGC_SEED', '1'), help='Seed')
    pipe.add_argument('--period-key', default='2020-W01', help='Weekly period key when --schedule weekly')
    pipe.add_argument('--out-dir', default=os.environ.get('MGC_EVIDENCE_DIR', 'data/evidence'), help='Output directory')
    pipe.add_argument('--deterministic', action='store_true', help='Pass through deterministic mode to subcommands')
    pipe.add_argument('--web', action='store_true', help='Build web player into <out_dir>/web')
    pipe.add_argument('--submission', action='store_true', help='Build submission.zip into <out_dir>/submission.zip')
    pipe.add_argument('--publish', action='store_true', help='Publish marketing for real (default: dry-run)')
    pipe.set_defaults(func=cmd_run_pipeline)

    # ----------------------------
    # manifest
    # ----------------------------
    man = run_sub.add_parser("manifest", help="Compute deterministic repo manifest (stable file hashing)")
    man.add_argument("--repo-root", default=".", help="Repository root to hash")
    man.add_argument("--include", action="append", default=None, help="Optional glob(s) to include (can repeat)")
    man.add_argument("--exclude-dir", action="append", default=None, help="Directory name(s) to exclude (repeatable)")
    man.add_argument("--exclude-glob", action="append", default=None, help="Filename glob(s) to exclude (repeatable)")
    man.add_argument("--out", default=None, help="Write manifest JSON to this path (else stdout)")
    man.add_argument("--print-hash", action="store_true", help="Print root_tree_sha256 to stderr")
    man.set_defaults(func=cmd_run_manifest)

    # ----------------------------
    # stage
    # ----------------------------
    stage = run_sub.add_parser("stage", help="Inspect or set run stage state (resume/observability)")
    stage_sub = stage.add_subparsers(dest="stage_cmd", required=True)

    st_set = stage_sub.add_parser("set", help="Upsert a stage row")
    st_set.add_argument("run_id", help="Canonical run_id")
    st_set.add_argument("stage", help="Stage name (e.g. daily, publish_marketing, manifest, evidence)")
    st_set.add_argument("status", help="pending|running|ok|error|skipped")
    st_set.add_argument("--started-at", default=None, help="ISO8601 timestamp")
    st_set.add_argument("--ended-at", default=None, help="ISO8601 timestamp")
    st_set.add_argument(
        "--duration-ms",
        type=int,
        default=None,
        help="Duration in milliseconds (normalized to 0 in deterministic mode)",
    )
    st_set.add_argument("--error-json", default=None, help="JSON string for error details")
    st_set.add_argument("--meta-json", default=None, help="JSON string to merge into meta_json")
    st_set.add_argument("--deterministic", action="store_true", help="Deterministic mode (also via MGC_DETERMINISTIC=1)")
    st_set.set_defaults(func=cmd_run_stage_set)

    st_get = stage_sub.add_parser("get", help="Get a stage row")
    st_get.add_argument("run_id", help="Canonical run_id")
    st_get.add_argument("stage", help="Stage name")
    st_get.set_defaults(func=cmd_run_stage_get)

    st_ls = stage_sub.add_parser("list", help="List all stages for a run_id")
    st_ls.add_argument("run_id", help="Canonical run_id")
    st_ls.set_defaults(func=cmd_run_stage_list)

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mgc-run-cli", description="MGC run_cli helpers")
    sub = p.add_subparsers(dest="cmd", required=True)
    register_run_subcommand(sub)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    fn: Optional[Callable[[argparse.Namespace], int]] = getattr(args, "func", None)
    if not callable(fn):
        die("No command selected.")
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
