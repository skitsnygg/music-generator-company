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
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mgc.playlist import build_playlist


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


def _ensure_tracks_table(con: sqlite3.Connection) -> None:
    """Ensure a minimal tracks table exists.

    This is intentionally conservative and backward compatible:
    - If a tracks table already exists, this is a no-op.
    - If it doesn't, we create a minimal schema used by the generator.
    """
    if _table_exists(con, "tracks"):
        return
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tracks (
          track_id TEXT PRIMARY KEY,
          artifact_path TEXT NOT NULL,
          ts TEXT,
          title TEXT,
          mood TEXT,
          genre TEXT,
          provider TEXT,
          meta_json TEXT
        )
        """.strip()
    )
    con.commit()


def _tracks_columns(con: sqlite3.Connection) -> set[str]:
    if not _table_exists(con, "tracks"):
        return set()
    rows = con.execute("PRAGMA table_info(tracks)").fetchall()
    out: set[str] = set()
    for r in rows:
        name = r[1] if len(r) > 1 else None
        if name:
            out.add(str(name))
    return out


def _insert_track_row(
    con: sqlite3.Connection,
    *,
    track_id: str,
    artifact_path: str,
    ts: str,
    title: str,
    mood: str,
    genre: str,
    provider: str,
    meta_json: str,
    bpm: Optional[int] = None,
    duration_sec: Optional[float] = None,
    preview_path: Optional[str] = None,
) -> None:
    """Insert a track row using only columns that exist in the current schema."""
    cols = _tracks_columns(con)
    if not cols:
        _ensure_tracks_table(con)
        cols = _tracks_columns(con)

    colmap = {
        "id": "id" if "id" in cols else None,
        "track_id": "track_id" if "track_id" in cols else ("id" if "id" in cols else None),
        "artifact_path": "artifact_path" if "artifact_path" in cols else ("full_path" if "full_path" in cols else None),
        "full_path": "full_path" if "full_path" in cols else ("artifact_path" if "artifact_path" in cols else None),
        "preview_path": "preview_path" if "preview_path" in cols else None,
        "duration_sec": "duration_sec" if "duration_sec" in cols else None,
        "bpm": "bpm" if "bpm" in cols else None,
        "created_at": "created_at" if "created_at" in cols else ("ts" if "ts" in cols else None),
        "ts": "ts" if "ts" in cols else ("created_at" if "created_at" in cols else None),
        "title": "title" if "title" in cols else None,
        "mood": "mood" if "mood" in cols else None,
        "genre": "genre" if "genre" in cols else None,
        "provider": "provider" if "provider" in cols else None,
        "meta_json": "meta_json" if "meta_json" in cols else ("meta" if "meta" in cols else None),
        "meta": "meta" if "meta" in cols else ("meta_json" if "meta_json" in cols else None),
    }

    fields: list[str] = []
    vals: list[Any] = []

    def add(key: str, value: Any) -> None:
        col = colmap.get(key)
        if not col:
            return
        fields.append(col)
        vals.append(value)

    add("track_id", str(track_id))
    add("id", str(track_id))
    add("artifact_path", str(artifact_path))
    add("full_path", str(artifact_path))
    add("preview_path", str(preview_path) if preview_path else None)
    add("duration_sec", float(duration_sec) if duration_sec is not None else None)
    add("bpm", int(bpm) if bpm is not None else None)
    add("created_at", str(ts))
    add("ts", str(ts))
    add("title", str(title))
    add("mood", str(mood))
    add("genre", str(genre))
    add("provider", str(provider))
    add("meta_json", str(meta_json))
    add("meta", str(meta_json))

    pruned_fields: list[str] = []
    pruned_vals: list[Any] = []
    for f, v in zip(fields, vals):
        if v is None:
            continue
        pruned_fields.append(f)
        pruned_vals.append(v)

    if not pruned_fields:
        raise RuntimeError("No compatible columns found to insert into tracks table")

    qmarks = ",".join(["?"] * len(pruned_fields))
    sql = f"INSERT OR REPLACE INTO tracks ({', '.join(pruned_fields)}) VALUES ({qmarks})"
    con.execute(sql, pruned_vals)
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


def cmd_run_generate(args: argparse.Namespace) -> int:
    """Generate a new track into the *library* (repo_root/data/tracks) and register it in DB.

    This is the Music Agent entrypoint for the autonomous system.

    Outputs:
      - data/tracks/<track_id>.wav
      - data/tracks/<track_id>.json  (metadata)
      - DB row in tracks table (best-effort across schemas)
      - evidence/generate_evidence.json (in out_dir/evidence)

    Determinism:
      - When --deterministic (or env MGC_DETERMINISTIC=1), IDs + timestamps are stable.
    """
    ctx = _build_run_context(args)
    json_mode = bool(getattr(args, "json", False))

    context = getattr(args, "context", "focus")
    store_dir_raw = getattr(args, "store_dir", None)
    store_dir = Path(store_dir_raw).resolve() if store_dir_raw else (ctx.repo_root / "data" / "tracks")
    _ensure_dir(store_dir)

    fixed_now = (getattr(args, "fixed_now", None) or "").strip()
    now = datetime.fromisoformat(fixed_now.replace("Z", "+00:00")).astimezone(timezone.utc) if fixed_now else ctx.now
    now_iso = now.isoformat()

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")

    provided_track_id = (getattr(args, "track_id", None) or "").strip()
    if provided_track_id:
        track_id = provided_track_id
    else:
        track_uuid = stable_uuid(ns, f"track:generate:{now_iso}:{context}:{ctx.seed}", ctx.deterministic)
        track_id = str(track_uuid)

    run_uuid = stable_uuid(ns, f"run:generate:{now_iso}:{context}:{track_id}", ctx.deterministic)
    run_id = str(run_uuid)

    from mgc.agents.music_agent import MusicAgent

    agent = MusicAgent(provider=ctx.provider)
    track = agent.generate(
        track_id=str(track_id),
        context=str(context),
        seed=int(ctx.seed),
        deterministic=bool(ctx.deterministic),
        schedule="generate",
        period_key=now.date().isoformat(),
        out_dir=str(store_dir),
        now_iso=now_iso,
    )

    wav_path = Path(track.artifact_path)

    suffix = wav_path.suffix.lower() or ".wav"
    final_wav = store_dir / f"{track_id}{suffix}"
    if wav_path.resolve() != final_wav.resolve():
        _ensure_dir(final_wav.parent)
        final_wav.write_bytes(wav_path.read_bytes())
        try:
            if wav_path.exists() and wav_path.is_file() and wav_path.parent != final_wav.parent:
                wav_path.unlink(missing_ok=True)  # py3.12+
        except TypeError:
            try:
                wav_path.unlink()
            except Exception:
                pass

    sha = _sha256_file(final_wav)

    title = str(getattr(track, "title", "") or f"{str(context).title()} Track")
    mood = str(getattr(track, "mood", "") or str(context))
    genre = str(getattr(track, "genre", "") or "unknown")

    meta = dict(getattr(track, "meta", None) or {})
    meta.update(
        {
            "deterministic": bool(ctx.deterministic),
            "generated_by": "run.generate",
            "context": str(context),
            "seed": int(ctx.seed),
            "ts": now_iso,
        }
    )

    meta_path = store_dir / f"{track_id}.json"
    meta_doc = {
        "schema": "mgc.track_meta.v1",
        "version": 1,
        "track_id": str(track_id),
        "provider": str(getattr(track, "provider", ctx.provider)),
        "title": title,
        "mood": mood,
        "genre": genre,
        "artifact_path": _as_rel_posix(str(final_wav.relative_to(ctx.repo_root)).replace("\\", "/"))
        if str(final_wav).startswith(str(ctx.repo_root))
        else str(final_wav),
        "sha256": sha,
        "ts": now_iso,
        "meta": meta,
    }
    _json_dump(meta_doc, meta_path)

    con = _connect(ctx.db_path)
    try:
        _ensure_tracks_table(con)
        _insert_track_row(
            con,
            track_id=str(track_id),
            artifact_path=_as_rel_posix(str(final_wav.relative_to(ctx.repo_root)).replace("\\", "/"))
            if str(final_wav).startswith(str(ctx.repo_root))
            else str(final_wav),
            ts=now_iso,
            title=title,
            mood=mood,
            genre=genre,
            provider=str(getattr(track, "provider", ctx.provider)),
            meta_json=json.dumps(meta, sort_keys=True),
            bpm=meta.get("bpm"),
            duration_sec=meta.get("duration_sec"),
            preview_path=meta.get("preview_path"),
        )
    finally:
        con.close()

    ev = {
        "schema": "mgc.generate_evidence.v1",
        "version": 1,
        "run_id": run_id,
        "stage": "generate",
        "context": context,
        "deterministic": bool(ctx.deterministic),
        "ts": now_iso,
        "provider": str(getattr(track, "provider", ctx.provider)),
        "track": {
            "track_id": str(track_id),
            "path": _as_rel_posix(str(final_wav.relative_to(ctx.repo_root)).replace("\\", "/"))
            if str(final_wav).startswith(str(ctx.repo_root))
            else str(final_wav),
            "sha256": sha,
            "meta_path": _as_rel_posix(str(meta_path.relative_to(ctx.repo_root)).replace("\\", "/"))
            if str(meta_path).startswith(str(ctx.repo_root))
            else str(meta_path),
        },
    }
    _json_dump(ev, ctx.evidence_dir / "generate_evidence.json")

    if json_mode:
        _emit_json(
            {
                "ok": True,
                "stage": "generate",
                "run_id": run_id,
                "track_id": str(track_id),
                "paths": {
                    "generate_evidence": "evidence/generate_evidence.json",
                    "track": str(final_wav),
                    "meta": str(meta_path),
                },
            }
        )
    else:
        _log(f"[run.generate] ok run_id={run_id} track_id={track_id} path={final_wav}")

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
# (rest of your existing file continues unchanged)
# ...
# -----------------------------------------------------------------------------
# argparse wiring
# -----------------------------------------------------------------------------

def cmd_run_dispatch(args: argparse.Namespace) -> int:
    cmd = getattr(args, "run_cmd", None)
    if cmd == "daily":
        return cmd_run_daily(args)
    if cmd == "generate":
        return cmd_run_generate(args)
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

    p = run_sub.add_parser("generate")
    add_common(p)
    p.add_argument("--store-dir", dest="store_dir", default=None, help="Library storage dir (default: repo_root/data/tracks)")
    p.add_argument("--track-id", dest="track_id", default=None, help="Optional explicit track_id")
    p.set_defaults(fn=cmd_run_generate, func=cmd_run_generate)

    # (rest unchanged)
