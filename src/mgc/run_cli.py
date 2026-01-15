#!/usr/bin/env python3
"""
src/mgc/run_cli.py

Run/pipeline CLI for Music Generator Company (MGC).

CI contracts:
- In --json mode, each command must emit EXACTLY ONE JSON object to stdout.
- Human logs must go to stderr.
- Composite commands (weekly/autonomous) must not leak sub-step logs/JSON.

Determinism:
- evidence/* must not contain absolute out_dir-dependent paths.
- Weekly determinism must hold even if --out-dir differs.

ci_root_determinism.sh contract:
- `python -m mgc.main --json run drop ...` output JSON must include paths.manifest_sha256.

Agents:
- Music generation goes through mgc.agents.music_agent.MusicAgent
- Marketing planning goes through mgc.agents.marketing_agent.MarketingAgent
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import uuid
from mgc.playlist import build_playlist
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Output discipline
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _emit_json(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, sort_keys=True) + "\n")
    sys.stdout.flush()


@contextlib.contextmanager
def _silence_stdio(enabled: bool) -> None:
    """
    Suppress stdout+stderr for nested sub-steps when parent command is in --json mode.
    """
    if not enabled:
        yield
        return
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


# -----------------------------------------------------------------------------
# small I/O helpers
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _write_jsonl_overwrite(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _as_rel_posix(path_like: str) -> str:
    s = (path_like or "").replace("\\", "/").strip()
    s = s.lstrip("./")
    if not s or s.startswith("/") or s.startswith("..") or "/../" in s or s.endswith("/.."):
        raise ValueError(f"invalid relative path: {path_like!r}")
    return s


# -----------------------------------------------------------------------------
# determinism helpers
# -----------------------------------------------------------------------------

def is_deterministic(args: Optional[argparse.Namespace] = None) -> bool:
    if args is not None and getattr(args, "deterministic", False):
        return True
    v = (os.environ.get("MGC_DETERMINISTIC") or os.environ.get("DETERMINISTIC") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def stable_uuid(namespace: uuid.UUID, name: str, deterministic: bool) -> uuid.UUID:
    return uuid.uuid5(namespace, name) if deterministic else uuid.uuid4()


def _deterministic_now_utc() -> datetime:
    fixed = (os.environ.get("MGC_FIXED_TIME") or "").strip()
    if fixed:
        s = fixed.replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    return datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _iso_week_period(d: date) -> Tuple[date, date, str]:
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    iso_year, iso_week, _ = d.isocalendar()
    label = f"{iso_year}-W{iso_week:02d}"
    return monday, sunday, label


# -----------------------------------------------------------------------------
# DB helpers (kept for compatibility)
# -----------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    cur = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,))
    return cur.fetchone() is not None


def _ensure_run_stages_table(con: sqlite3.Connection) -> None:
    if _table_exists(con, "run_stages"):
        return
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS run_stages (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          started_ts TEXT NOT NULL,
          finished_ts TEXT,
          ok INTEGER,
          meta_json TEXT
        )
        """.strip()
    )
    con.commit()


# -----------------------------------------------------------------------------
# core run context
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    db_path: Path
    out_dir: Path
    evidence_dir: Path
    deterministic: bool
    now: datetime
    seed: int
    provider: str


def _require_db(args: argparse.Namespace) -> Path:
    db = getattr(args, "db", None)
    if not db:
        raise SystemExit("Missing --db. Use: mgc --db <path> run ... or mgc run ... --db <path>")
    return Path(db).resolve()


def _build_run_context(args: argparse.Namespace) -> RunContext:
    repo_root = Path(getattr(args, "repo_root", ".")).resolve()
    db_path = _require_db(args)
    out_dir = Path(getattr(args, "out_dir", "artifacts/run")).resolve()

    evidence_dir_val = getattr(args, "evidence_dir", None)
    evidence_dir = Path(evidence_dir_val).resolve() if evidence_dir_val else (out_dir / "evidence")

    det = is_deterministic(args)
    now = _deterministic_now_utc() if det else datetime.now(timezone.utc)

    seed = int(getattr(args, "seed", 1) or 1)
    provider_name = (getattr(args, "provider", None) or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()

    _ensure_dir(out_dir)
    _ensure_dir(evidence_dir)

    return RunContext(
        repo_root=repo_root,
        db_path=db_path,
        out_dir=out_dir,
        evidence_dir=evidence_dir,
        deterministic=det,
        now=now,
        seed=seed,
        provider=provider_name,
    )


# -----------------------------------------------------------------------------
# Daily work (no-emit helper + emitting wrapper)
# -----------------------------------------------------------------------------

def _run_daily_no_emit(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Do daily generation + write evidence/daily_evidence.json, but DO NOT print anything.
    Returns {run_id, track_id, rel_track_path, artifact_path, sha256}.
    """
    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    run_id = stable_uuid(ns, f"run:daily:{ctx.now.isoformat()}:{context}", ctx.deterministic)
    track_id = stable_uuid(ns, f"track:daily:{ctx.now.date().isoformat()}:{context}", ctx.deterministic)

    from mgc.agents.music_agent import MusicAgent

    agent = MusicAgent(provider=ctx.provider)
    track = agent.generate(
        track_id=str(track_id),
        context=str(context),
        seed=int(ctx.seed),
        deterministic=bool(ctx.deterministic),
        schedule="daily",
        period_key=ctx.now.date().isoformat(),
        out_dir=str(ctx.out_dir),
        now_iso=ctx.now.isoformat(),
    )

    wav_path = Path(track.artifact_path)
    rel_track_path = _as_rel_posix(f"tracks/{wav_path.name}")

    ev = {
        "schema": "mgc.daily_evidence.v1",
        "version": 1,
        "run_id": str(run_id),
        "stage": "daily",
        "context": context,
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "provider": track.provider,
        "track": {"track_id": track.track_id, "path": rel_track_path},
        "sha256": {"track": str(track.sha256)},
        "meta": track.meta or {},
    }
    _json_dump(ev, ctx.evidence_dir / "daily_evidence.json")

    return {
        "run_id": str(run_id),
        "track_id": track.track_id,
        "rel_track_path": rel_track_path,
        "artifact_path": track.artifact_path,
        "sha256": track.sha256,
    }


def cmd_run_daily(args: argparse.Namespace) -> int:
    info = _run_daily_no_emit(args)
    if getattr(args, "json", False):
        _emit_json(
            {
                "ok": True,
                "stage": "daily",
                "run_id": info["run_id"],
                "track_id": info["track_id"],
                "paths": {"daily_evidence": "evidence/daily_evidence.json"},
            }
        )
    else:
        _log(f"[run.daily] ok run_id={info['run_id']}")
    return 0


# -----------------------------------------------------------------------------
# Manifest (for ci_root_determinism.sh)
# -----------------------------------------------------------------------------

def _write_manifest(ctx: RunContext, files: List[Tuple[str, Path]]) -> Tuple[Path, str]:
    entries: List[Dict[str, Any]] = []
    for rel, abs_path in files:
        rel2 = _as_rel_posix(rel)
        entries.append({"path": rel2, "sha256": _sha256_file(abs_path), "bytes": abs_path.stat().st_size})
    entries.sort(key=lambda x: x["path"])

    manifest = {
        "schema": "mgc.manifest.v1",
        "version": 1,
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "files": entries,
    }
    manifest_path = ctx.evidence_dir / "manifest.json"
    _json_dump(manifest, manifest_path)
    return manifest_path, _sha256_file(manifest_path)


# -----------------------------------------------------------------------------
# Drop bundle writer
# -----------------------------------------------------------------------------

def _write_drop_bundle(*, ctx: RunContext, context: str, schedule: str) -> Tuple[uuid.UUID, Dict[str, Path]]:
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")

    if schedule == "weekly":
        period_start, period_end, period_label = _iso_week_period(ctx.now.date())
        period_key = period_label
    else:
        period_start = ctx.now.date()
        period_end = ctx.now.date()
        period_label = ctx.now.date().isoformat()
        period_key = ctx.now.date().isoformat()

    drop_id = stable_uuid(ns, f"drop:{schedule}:{period_key}:{context}", ctx.deterministic)
    run_id = stable_uuid(ns, f"run:{schedule}:{ctx.now.isoformat()}:{context}", ctx.deterministic)
    bundle_track_id = stable_uuid(ns, f"track:{schedule}:{period_key}:{context}", ctx.deterministic)
    playlist_id = stable_uuid(ns, f"playlist:{schedule}:{period_key}:{context}", ctx.deterministic)

    from mgc.agents.music_agent import MusicAgent
    agent = MusicAgent(provider=ctx.provider)
    track = agent.generate(
        track_id=str(bundle_track_id),
        context=str(context),
        seed=int(ctx.seed),
        deterministic=bool(ctx.deterministic),
        schedule=str(schedule),
        period_key=str(period_key),
        out_dir=str(ctx.out_dir),
        now_iso=ctx.now.isoformat(),
    )

    src_path = Path(track.artifact_path)

    bundle_dir = ctx.out_dir / "drop_bundle"
    bundle_tracks = bundle_dir / "tracks"
    _ensure_dir(bundle_tracks)

    dst_path = bundle_tracks / src_path.name
    dst_path.write_bytes(src_path.read_bytes())

    rel_track_path = _as_rel_posix(f"tracks/{dst_path.name}")
    bundle_track_sha = _sha256_file(dst_path)

    playlist_path = bundle_dir / "playlist.json"
    playlist = {
        "schema": "mgc.playlist.v1",
        "version": 1,
        "playlist_id": str(playlist_id),
        "context": context,
        "schedule": schedule,
        "period": {"key": period_key, "label": period_label, "start_date": period_start.isoformat(), "end_date": period_end.isoformat()},
        "ts": ctx.now.isoformat(),
        "tracks": [{
            "track_id": track.track_id,
            "title": track.title,
            "path": rel_track_path,
            "artifact_path": rel_track_path,
            "provider": track.provider,
            "genre": track.genre,
            "mood": track.mood,
        }],
    }
    _json_dump(playlist, playlist_path)
    playlist_sha = _sha256_file(playlist_path)

    bundle_daily_ev_path = bundle_dir / "daily_evidence.json"
    bundle_daily = {
        "schema": "mgc.daily_evidence.v1",
        "version": 1,
        "stage": "daily",
        "context": context,
        "deterministic": ctx.deterministic,
        "run_id": str(run_id),
        "ts": ctx.now.isoformat(),
        "schedule": schedule,
        "period": {"key": period_key, "label": period_label, "start_date": period_start.isoformat(), "end_date": period_end.isoformat()},
        "track": {"track_id": track.track_id, "path": rel_track_path},
        "sha256": {"track": bundle_track_sha, "playlist": playlist_sha},
        "meta": track.meta or {},
    }
    _json_dump(bundle_daily, bundle_daily_ev_path)

    drop_ev = {
        "drop": {"id": str(drop_id), "ts": ctx.now.isoformat(), "context": context, "schedule": schedule,
                 "period": {"key": period_key, "label": period_label, "start_date": period_start.isoformat(), "end_date": period_end.isoformat()}},
        "deterministic": ctx.deterministic,
        "paths": {"bundle_dir": "drop_bundle", "bundle_playlist": "drop_bundle/playlist.json", "bundle_daily_evidence": "drop_bundle/daily_evidence.json"},
    }
    drop_ev_path = ctx.out_dir / "drop_evidence.json"
    _json_dump(drop_ev, drop_ev_path)

    produced = {
        "drop_evidence": drop_ev_path,
        "bundle_playlist": playlist_path,
        "bundle_daily_evidence": bundle_daily_ev_path,
        "bundle_track": dst_path,
    }
    return drop_id, produced


# -----------------------------------------------------------------------------
# Marketing wiring
# -----------------------------------------------------------------------------

def _load_daily_track_as_artifact(ctx: RunContext) -> Optional[Any]:
    """
    Load the latest daily evidence in the current out_dir and build a TrackArtifact.
    Returns TrackArtifact or None if daily evidence doesn't exist yet.
    """
    ev_path = ctx.evidence_dir / "daily_evidence.json"
    if not ev_path.exists():
        return None

    ev = _json_load(ev_path)
    track_info = (ev.get("track") or {})
    sha_info = (ev.get("sha256") or {})
    meta = dict(ev.get("meta") or {})
    provider = str(ev.get("provider") or meta.get("provider") or ctx.provider)

    rel = str(track_info.get("path") or "")
    if not rel:
        return None
    rel = _as_rel_posix(rel)
    abs_path = ctx.out_dir / rel

    from mgc.agents.music_agent import TrackArtifact

    track_id = str(track_info.get("track_id") or meta.get("track_id") or "")
    if not track_id:
        return None

    title = str(meta.get("title") or f"{(ev.get('context') or 'focus').title()} Track")
    mood = str(meta.get("mood") or ev.get("context") or "focus")
    genre = str(meta.get("genre") or "unknown")
    sha = str(sha_info.get("track") or (abs_path.exists() and _sha256_file(abs_path)) or "")

    return TrackArtifact(
        track_id=track_id,
        provider=provider,
        artifact_path=str(abs_path),
        sha256=sha,
        title=title,
        mood=mood,
        genre=genre,
        meta=meta,
        preview_path="",
    )


def _make_preview(ctx: RunContext, track: Any) -> str:
    """
    Create a deterministic preview asset.

    For now this is a WAV copy:
      marketing/previews/<track_id>.wav

    Returns relative preview path.
    """
    src = Path(track.artifact_path)
    previews_dir = ctx.out_dir / "marketing" / "previews"
    _ensure_dir(previews_dir)

    dst = previews_dir / f"{track.track_id}.wav"
    dst.write_bytes(src.read_bytes())

    return _as_rel_posix(f"marketing/previews/{dst.name}")


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


def cmd_publish_marketing(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    dry_run = bool(getattr(args, "dry_run", False))

    # Weekly/autonomous may pass a fixed timestamp to force determinism.
    fixed_now = (getattr(args, "fixed_now", None) or "").strip()
    now_iso = fixed_now or ctx.now.isoformat()

    # Build track artifact from daily evidence (produced by run daily or by weekly/autonomous).
    track = _load_daily_track_as_artifact(ctx)
    planned_posts: List[Dict[str, Any]] = []

    receipts_path = ctx.out_dir / "marketing" / "receipts.jsonl"
    receipts_sha_before = _sha256_file(receipts_path) if receipts_path.exists() else ""

    if track is not None:
        # Attach preview
        preview_rel = _make_preview(ctx, track)
        # TrackArtifact is frozen, but we can stash preview path in meta + pass separately to agent inputs.
        meta2 = dict(track.meta or {})
        meta2["preview_path"] = preview_rel

        from mgc.agents.music_agent import TrackArtifact
        track2 = TrackArtifact(
            track_id=track.track_id,
            provider=track.provider,
            artifact_path=track.artifact_path,
            sha256=track.sha256,
            title=track.title,
            mood=track.mood,
            genre=track.genre,
            meta=meta2,
            preview_path=preview_rel,
        )

        # Plan posts
        from mgc.agents.marketing_agent import MarketingAgent, StoragePaths

        storage = StoragePaths(root=str(ctx.out_dir))
        agent = MarketingAgent(storage=storage)

        # created_at must be stable for determinism (weekly passes fixed_now)
        planned_posts = agent.plan_posts(track2, deterministic=True, created_at=now_iso)

    # Turn planned posts into receipts
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    # Stable run_id when fixed_now is provided; otherwise stable per (now, track)
    track_id_for_id = ""
    if track is not None:
        track_id_for_id = str(getattr(track, "track_id", ""))
    run_id = stable_uuid(ns, f"run:publish_marketing:{now_iso}:{track_id_for_id}", True)

    receipts: List[Dict[str, Any]] = []
    for p in planned_posts:
        receipts.append(
            {
                "schema": "mgc.marketing_receipt.v1",
                "version": 1,
                "run_id": str(run_id),
                "deterministic": True,
                "ts": now_iso,
                "platform": p.get("platform", "unknown"),
                "status": "published" if not dry_run else "planned",
                "post_id": str(p.get("id") or ""),
                "track_id": str(p.get("track_id") or ""),
            }
        )

    # Stable ordering
    receipts.sort(key=lambda r: (r.get("platform") or "", r.get("post_id") or ""))

    if not dry_run:
        # Determinism gate: overwrite receipts each run.
        _write_jsonl_overwrite(receipts_path, receipts)

    receipts_sha_after = _sha256_file(receipts_path) if receipts_path.exists() else ""

    ev = {
        "run_id": str(run_id),
        "stage": "publish-marketing",
        "deterministic": True,
        "ts": now_iso,
        "dry_run": dry_run,
        "inputs": {"daily_evidence": "evidence/daily_evidence.json" if (ctx.evidence_dir / "daily_evidence.json").exists() else ""},
        "outputs": {
            "posts_dir": "marketing/posts",
            "preview_dir": "marketing/previews",
            "receipts_path": "marketing/receipts.jsonl",
        },
        "counts": {"planned_posts": len(planned_posts), "receipts": len(receipts)},
        "sha256": {
            "receipts_before": receipts_sha_before,
            "receipts_after": receipts_sha_after,
        },
    }
    _json_dump(ev, ctx.evidence_dir / "publish_marketing_evidence.json")

    if getattr(args, "json", False):
        _emit_json(
            {
                "ok": True,
                "stage": "publish-marketing",
                "run_id": str(run_id),
                "counts": {"planned_posts": len(planned_posts), "receipts": len(receipts)},
                "paths": {"publish_marketing_evidence": "evidence/publish_marketing_evidence.json"},
            }
        )
    else:
        _log(f"[run.publish-marketing] ok run_id={run_id} planned_posts={len(planned_posts)} receipts={len(receipts)}")
    return 0

def cmd_run_drop(args: argparse.Namespace) -> int:
    _run_daily_no_emit(args)

    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")
    drop_id, produced = _write_drop_bundle(ctx=ctx, context=context, schedule="daily")

    files: List[Tuple[str, Path]] = [
        ("drop_evidence.json", produced["drop_evidence"]),
        ("drop_bundle/playlist.json", produced["bundle_playlist"]),
        ("drop_bundle/daily_evidence.json", produced["bundle_daily_evidence"]),
        (f"drop_bundle/tracks/{produced['bundle_track'].name}", produced["bundle_track"]),
    ]

    daily_ev = ctx.evidence_dir / "daily_evidence.json"
    if daily_ev.exists():
        files.append(("evidence/daily_evidence.json", daily_ev))

    pm_ev = ctx.evidence_dir / "publish_marketing_evidence.json"
    if pm_ev.exists():
        files.append(("evidence/publish_marketing_evidence.json", pm_ev))

    _manifest_path, manifest_sha = _write_manifest(ctx, files)

    if getattr(args, "json", False):
        _emit_json(
            {
                "ok": True,
                "stage": "drop",
                "drop_id": str(drop_id),
                "paths": {
                    "manifest_path": "evidence/manifest.json",
                    "manifest_sha256": manifest_sha,
                    "drop_evidence": "drop_evidence.json",
                    "bundle_dir": "drop_bundle",
                },
            }
        )
    else:
        _log(f"[run.drop] ok drop_id={drop_id} evidence={produced['drop_evidence']}")
    return 0



def cmd_run_weekly(args: argparse.Namespace) -> int:
    """Weekly pipeline step (playlist-driven, deterministic).

    Produces:
      - drop_bundle/playlist.json (multi-track)
      - drop_bundle/daily_evidence.json
      - drop_bundle/tracks/<track_id>.wav (copied)
      - evidence/daily_evidence.json (lead track; for publish-marketing)
      - evidence/publish_marketing_evidence.json + marketing/* (lead track)
      - drop_evidence.json

    Key points:
      - No audio generation.
      - Stable IDs + stable timestamps.
      - publish-marketing is run deterministically with a frozen timestamp.
    """
    ctx = _build_run_context(args)
    json_mode = bool(getattr(args, "json", False))

    context = getattr(args, "context", "focus")

    # Weekly period
    period_start, period_end, period_label = _iso_week_period(ctx.now.date())
    period_key = period_label  # e.g. "2026-W03"

    # Stable timestamp for determinism: week start at 00:00:00Z
    stable_ts = f"{period_start.isoformat()}T00:00:00+00:00"

    # Optional knobs
    target_minutes = getattr(args, "target_minutes", None)
    if target_minutes is None:
        target_minutes = 20
    lookback_playlists = getattr(args, "lookback_playlists", None)
    if lookback_playlists is None:
        lookback_playlists = 0

    out_dir = Path(getattr(args, "out_dir", ctx.out_dir)).resolve()
    bundle_dir = out_dir / "drop_bundle"
    bundle_tracks_dir = bundle_dir / "tracks"
    out_tracks_dir = out_dir / "tracks"
    evidence_dir = out_dir / "evidence"

    _ensure_dir(out_dir)
    _ensure_dir(bundle_dir)
    _ensure_dir(bundle_tracks_dir)
    _ensure_dir(out_tracks_dir)
    _ensure_dir(evidence_dir)

    # Build deterministic playlist
    pl = build_playlist(
        db_path=ctx.db_path,
        context=context,
        period_key=period_key,
        base_seed=int(ctx.seed),
        target_minutes=int(target_minutes),
        lookback_playlists=int(lookback_playlists),
    )

    if not isinstance(pl, dict) or "tracks" not in pl:
        raise SystemExit("[run.weekly] build_playlist must return dict with 'tracks'")

    tracks = list(pl.get("tracks") or [])
    if not tracks:
        raise SystemExit(f"[run.weekly] build_playlist returned 0 tracks (context={context!r}, period_key={period_key!r})")

    lead = tracks[0]
    lead_track_id = getattr(lead, "id", None) or getattr(lead, "track_id", None)
    if not lead_track_id:
        raise SystemExit("[run.weekly] lead track missing id/track_id")
    lead_track_id = str(lead_track_id)

    # Stable IDs (force deterministic regardless of CLI flags)
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    playlist_id = stable_uuid(ns, f"playlist:weekly:{period_key}:{context}", True)
    run_id = stable_uuid(ns, f"run:weekly:{period_key}:{context}:{ctx.seed}", True)
    drop_id = stable_uuid(ns, f"drop:weekly:{period_key}:{context}", True)

    def _resolve_src(p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (Path(ctx.repo_root) / pp).resolve()

    playlist_tracks_json: List[Dict[str, Any]] = []
    lead_out_path: Optional[Path] = None
    lead_bundle_path: Optional[Path] = None

    for t in tracks:
        tid = getattr(t, "id", None) or getattr(t, "track_id", None)
        if not tid:
            raise SystemExit(f"[run.weekly] playlist track missing id: {t!r}")
        tid = str(tid)

        src_raw = getattr(t, "full_path", None) or getattr(t, "path", None) or getattr(t, "artifact_path", None)
        if not src_raw:
            raise SystemExit(f"[run.weekly] playlist track missing source path: {t!r}")

        src = _resolve_src(str(src_raw))
        if not src.exists():
            raise SystemExit(f"[run.weekly] missing source track file: {src}")

        suffix = src.suffix.lower() or ".wav"

        bundle_dst = bundle_tracks_dir / f"{tid}{suffix}"
        if not bundle_dst.exists():
            shutil.copy2(src, bundle_dst)

        if tid == lead_track_id:
            out_dst = out_tracks_dir / f"{tid}{suffix}"
            if not out_dst.exists():
                shutil.copy2(src, out_dst)
            lead_out_path = out_dst
            lead_bundle_path = bundle_dst

        playlist_tracks_json.append(
            {
                "track_id": tid,
                "title": getattr(t, "title", None),
                "mood": getattr(t, "mood", None),
                "genre": getattr(t, "genre", None),
                "bpm": getattr(t, "bpm", None),
                "duration_sec": getattr(t, "duration_sec", None),
                "path": f"tracks/{bundle_dst.name}",
            }
        )

    if lead_out_path is None or lead_bundle_path is None:
        raise SystemExit("[run.weekly] internal error: lead track was not copied")

    # Write playlist.json (bundle-local paths)
    playlist_doc: Dict[str, Any] = {
        "schema": "mgc.playlist.v1",
        "version": 1,
        "playlist_id": str(playlist_id),
        "context": context,
        "schedule": "weekly",
        "period": {
            "key": period_key,
            "label": period_label,
            "start_date": period_start.isoformat(),
            "end_date": period_end.isoformat(),
        },
        "ts": stable_ts,
        "deterministic": True,
        "lead_track_id": lead_track_id,
        "tracks": playlist_tracks_json,
    }

    # Carry through additional playlist metadata (but never override core keys)
    for k, v in (pl.items() if isinstance(pl, dict) else []):
        if k in ("tracks", "items"):
            continue
        if k not in playlist_doc:
            playlist_doc[k] = v

    bundle_playlist_path = bundle_dir / "playlist.json"
    _json_dump(playlist_doc, bundle_playlist_path)
    playlist_sha = _sha256_file(bundle_playlist_path)

    # Evidence for publish-marketing expects evidence/daily_evidence.json and track under out_dir/tracks/
    lead_rel_out = _as_rel_posix(f"tracks/{Path(lead_out_path).name}")
    lead_sha_out = _sha256_file(Path(lead_out_path))

    evidence_daily = {
        "schema": "mgc.daily_evidence.v1",
        "version": 1,
        "run_id": str(run_id),
        "stage": "daily",
        "context": context,
        "deterministic": True,
        "ts": stable_ts,
        "provider": str(getattr(ctx, "provider", "stub")),
        "schedule": "weekly",
        "period": {
            "key": period_key,
            "label": period_label,
            "start_date": period_start.isoformat(),
            "end_date": period_end.isoformat(),
        },
        "track": {"track_id": lead_track_id, "path": lead_rel_out},
        "sha256": {"track": str(lead_sha_out), "playlist": str(playlist_sha)},
        "meta": {
            "target_minutes": int(target_minutes),
            "lookback_playlists": int(lookback_playlists),
            "playlist_track_count": len(tracks),
            "lead_track_id": lead_track_id,
        },
    }
    _json_dump(evidence_daily, evidence_dir / "daily_evidence.json")

    # Bundle-local daily evidence
    lead_rel_bundle = _as_rel_posix(f"tracks/{Path(lead_bundle_path).name}")
    lead_sha_bundle = _sha256_file(Path(lead_bundle_path))

    bundle_daily = dict(evidence_daily)
    bundle_daily["track"] = {"track_id": lead_track_id, "path": lead_rel_bundle}
    bundle_daily["sha256"] = {"track": str(lead_sha_bundle), "playlist": str(playlist_sha)}
    _json_dump(bundle_daily, bundle_dir / "daily_evidence.json")

    # Write drop_evidence.json (top-level summary, stable)
    drop_ev = {
        "ok": True,
        "drop_id": str(drop_id),
        "context": context,
        "schedule": "weekly",
        "period_key": period_key,
        "lead_track_id": lead_track_id,
        "playlist_tracks": len(tracks),
        "paths": {
            "bundle_dir": "drop_bundle",
            "bundle_playlist": "drop_bundle/playlist.json",
            "bundle_daily_evidence": "drop_bundle/daily_evidence.json",
            "daily_evidence": "evidence/daily_evidence.json",
            "drop_evidence": "drop_evidence.json",
        },
    }
    _json_dump(drop_ev, out_dir / "drop_evidence.json")

    # Run publish-marketing deterministically for the lead track.
    # publish-marketing reads evidence/daily_evidence.json (out_dir/evidence/...).
    setattr(args, "out_dir", str(out_dir))
    setattr(args, "evidence_dir", str(evidence_dir))
    setattr(args, "fixed_now", stable_ts)
    setattr(args, "deterministic", True)

    with _silence_stdio(json_mode):
        cmd_publish_marketing(args)

    if getattr(args, "json", False):
        _emit_json(drop_ev)
    else:
        _log(f"[run.weekly] ok drop_id={drop_id} period_key={period_key} lead_track_id={lead_track_id} playlist_tracks={len(tracks)}")

    return 0

def cmd_run_autonomous(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))

    _run_daily_no_emit(args)

    # Silence nested sub-steps if parent is JSON.
    with _silence_stdio(json_mode):
        cmd_publish_marketing(args)
        cmd_run_drop(args)

    if getattr(args, "json", False):
        _emit_json({"ok": True, "stage": "autonomous", "paths": {"drop_evidence": "drop_evidence.json", "bundle_dir": "drop_bundle"}})
    else:
        ctx = _build_run_context(args)
        _log(f"[run.autonomous] ok out_dir={ctx.out_dir}")
    return 0


def cmd_run_stage(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    stage_name = args.stage_name

    con = _connect(ctx.db_path)
    try:
        _ensure_run_stages_table(con)
        ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
        stage_id = stable_uuid(ns, f"run_stage:{stage_name}:{ctx.now.isoformat()}", ctx.deterministic)

        started = ctx.now.isoformat()
        con.execute(
            "INSERT OR REPLACE INTO run_stages (id, name, started_ts, finished_ts, ok, meta_json) VALUES (?,?,?,?,?,?)",
            (str(stage_id), stage_name, started, None, None, json.dumps({"deterministic": ctx.deterministic})),
        )
        con.commit()

        finished = (_deterministic_now_utc() if ctx.deterministic else datetime.now(timezone.utc)).isoformat()
        con.execute(
            "UPDATE run_stages SET finished_ts=?, ok=?, meta_json=? WHERE id=?",
            (finished, 1, json.dumps({"note": "stub", "stage": stage_name}, sort_keys=True), str(stage_id)),
        )
        con.commit()

        if getattr(args, "json", False):
            _emit_json({"ok": True, "stage": "stage", "stage_id": str(stage_id), "name": stage_name})
        else:
            _log(f"[run.stage] ok stage={stage_name} id={stage_id}")
        return 0
    finally:
        con.close()


# -----------------------------------------------------------------------------
# argparse wiring
# -----------------------------------------------------------------------------

def cmd_run_dispatch(args: argparse.Namespace) -> int:
    cmd = getattr(args, "run_cmd", None)
    if cmd == "daily":
        return cmd_run_daily(args)
    if cmd == "weekly":
        return cmd_run_weekly(args)
    if cmd == "autonomous":
        return cmd_run_autonomous(args)
    if cmd == "publish-marketing":
        return cmd_publish_marketing(args)
    if cmd == "drop":
        return cmd_run_drop(args)
    if cmd == "stage":
        return cmd_run_stage(args)
    raise SystemExit(f"Unknown run subcommand: {cmd}")


def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p_run = subparsers.add_parser("run", help="Run pipeline steps")
    p_run.set_defaults(fn=cmd_run_dispatch, func=cmd_run_dispatch)
    run_sub = p_run.add_subparsers(dest="run_cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", help="DB path (optional if provided globally)", default=argparse.SUPPRESS)
        p.add_argument("--repo-root", default=argparse.SUPPRESS)
        p.add_argument("--out-dir", default="artifacts/run")
        p.add_argument("--evidence-dir", default=None)
        p.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"))
        p.add_argument("--provider", default=os.environ.get("MGC_PROVIDER", "stub"))
        p.add_argument("--seed", type=int, default=int(os.environ.get("MGC_SEED", "1")))
        p.add_argument("--deterministic", action="store_true")
        p.add_argument("--limit", type=int, default=int(os.environ.get("MGC_PUBLISH_LIMIT", "200")))
        p.add_argument("--dry-run", dest="dry_run", action="store_true")

    p = run_sub.add_parser("daily")
    add_common(p)
    p.set_defaults(fn=cmd_run_daily, func=cmd_run_daily)

    p = run_sub.add_parser("publish-marketing")
    add_common(p)
    p.set_defaults(fn=cmd_publish_marketing, func=cmd_publish_marketing)

    p = run_sub.add_parser("drop")
    add_common(p)
    p.set_defaults(fn=cmd_run_drop, func=cmd_run_drop)

    p = run_sub.add_parser("weekly")
    add_common(p)
    p.set_defaults(fn=cmd_run_weekly, func=cmd_run_weekly)

    p = run_sub.add_parser("autonomous")
    add_common(p)
    p.set_defaults(fn=cmd_run_autonomous, func=cmd_run_autonomous)

    p = run_sub.add_parser("stage")
    add_common(p)
    p.add_argument("stage_name")
    p.set_defaults(fn=cmd_run_stage, func=cmd_run_stage)
