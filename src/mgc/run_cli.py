#!/usr/bin/env python3
"""
src/mgc/run_cli.py

Run/pipeline CLI for Music Generator Company (MGC).

CI requirements:
- `python -m mgc.main --db ... --repo-root ... --seed 1 --no-resume --json run autonomous ...` must succeed.
- Deterministic mode must produce byte-for-byte stable bundles + submission.zip across runs.

Bundle contract (submission validator):
- drop_bundle/playlist.json: schema=mgc.playlist.v1 and tracks[*].path present
- drop_bundle/daily_evidence.json: schema=mgc.daily_evidence.v1 with portable-only paths + sha256 keys
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mgc.providers import ProviderError, get_provider


# -----------------------------------------------------------------------------
# small I/O helpers
# -----------------------------------------------------------------------------

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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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
# Provider artifact normalization
# -----------------------------------------------------------------------------

def _art_get(art: Any, key: str, default: Any = None) -> Any:
    """Support both dict-returning providers and object/dataclass providers."""
    if art is None:
        return default
    if isinstance(art, dict):
        return art.get(key, default)
    return getattr(art, key, default)


def _normalize_art(
    art: Any,
    *,
    fallback_track_id: str,
    fallback_provider: str,
    fallback_context: str,
) -> Dict[str, Any]:
    """Return a stable dict with the fields run_cli expects."""
    out: Dict[str, Any] = {}

    out["provider"] = _art_get(art, "provider", fallback_provider)
    out["track_id"] = _art_get(art, "track_id", fallback_track_id)
    out["artifact_path"] = _art_get(art, "artifact_path")
    out["sha256"] = _art_get(art, "sha256")

    # Optional metadata (some providers don't supply these)
    out["meta"] = _art_get(art, "meta", {}) or {}
    out["genre"] = _art_get(art, "genre", "") or ""
    out["mood"] = _art_get(art, "mood", "") or ""
    out["title"] = _art_get(art, "title", "") or ""

    if not out["title"]:
        out["title"] = f"{fallback_context.title()} Track"

    if not out["genre"]:
        # Keep it non-empty for downstream JSON; harmless default.
        out["genre"] = out["provider"] or "unknown"

    if not out["mood"]:
        out["mood"] = fallback_context

    if not out["artifact_path"]:
        raise SystemExit("Provider returned no artifact_path")
    if not out["sha256"]:
        # If provider doesn't return sha256, compute it here.
        try:
            out["sha256"] = _sha256_file(Path(str(out["artifact_path"])))
        except Exception as e:
            raise SystemExit(f"Provider returned no sha256 and hashing failed: {e}") from e

    return out


# -----------------------------------------------------------------------------
# DB helpers
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
    )


def _write_drop_evidence_root(ctx: RunContext, payload: Dict[str, Any]) -> Path:
    p = ctx.out_dir / "drop_evidence.json"
    _json_dump(payload, p)
    return p


def _normalize_daily_evidence(obj: Dict[str, Any], *, out_dir: Path, evidence_dir: Path) -> Dict[str, Any]:
    out = dict(obj) if isinstance(obj, dict) else {"raw": obj}
    out.setdefault("schema", "mgc.daily_evidence.v1")
    out.setdefault("version", 1)

    paths = out.get("paths")
    if not isinstance(paths, dict):
        paths = {}
    paths.setdefault("out_dir", str(out_dir))
    paths.setdefault("evidence_dir", str(evidence_dir))
    out["paths"] = paths

    sha = out.get("sha256")
    if not isinstance(sha, dict):
        sha = {}
    out["sha256"] = sha

    return out


# -----------------------------------------------------------------------------
# commands
# -----------------------------------------------------------------------------

def cmd_run_daily(args: argparse.Namespace) -> int:
    """
    Daily stage:
    - calls provider to generate a track artifact under out_dir/tracks/
    - writes evidence/daily_evidence.json (this file may contain absolute paths; bundle version must not)
    """
    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")
    provider_name = (getattr(args, "provider", None) or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    run_id = stable_uuid(ns, f"run:daily:{ctx.now.isoformat()}:{context}", ctx.deterministic)

    # Daily track_id uses date (not ISO week) so day-to-day differs in non-fixed-time mode.
    # In deterministic CI (fixed epoch), still stable.
    track_id = stable_uuid(ns, f"track:daily:{ctx.now.date().isoformat()}:{context}", ctx.deterministic)

    # Provider generates artifact into out_dir/tracks/
    try:
        provider = get_provider(provider_name)
        art_raw = provider.generate(
            out_dir=ctx.out_dir,
            track_id=str(track_id),
            context=context,
            seed=ctx.seed,
            deterministic=ctx.deterministic,
            now_iso=ctx.now.isoformat(),
            schedule="daily",
            period_key=ctx.now.date().isoformat(),
        )
    except ProviderError as e:
        raise SystemExit(f"[run.daily] provider error: {e}") from e

    art = _normalize_art(
        art_raw,
        fallback_track_id=str(track_id),
        fallback_provider=provider_name,
        fallback_context=context,
    )

    wav_path = Path(str(art["artifact_path"]))
    track_sha = str(art["sha256"])

    ev_raw = {
        "run_id": str(run_id),
        "stage": "daily",
        "context": context,
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "provider": art["provider"],
        "track": {"track_id": art["track_id"], "path": str(wav_path)},
        "paths": {
            "out_dir": str(ctx.out_dir),
            "evidence_dir": str(ctx.evidence_dir),
            "tracks_dir": str((ctx.out_dir / "tracks").resolve()),
        },
        "sha256": {
            "track": track_sha,
        },
        "meta": art.get("meta") or {},
    }
    ev = _normalize_daily_evidence(ev_raw, out_dir=ctx.out_dir, evidence_dir=ctx.evidence_dir)

    ev_path = ctx.evidence_dir / "daily_evidence.json"
    _json_dump(ev, ev_path)

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "run_id": str(run_id), "track_id": art["track_id"], "evidence": str(ev_path)}))
    else:
        print(f"[run.daily] ok run_id={run_id}")
    return 0


def cmd_publish_marketing(args: argparse.Namespace) -> int:
    """
    Stub publisher:
    - writes publish_marketing_evidence.json
    - appends one receipt record to out_dir/marketing/receipts.jsonl
    """
    ctx = _build_run_context(args)
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    run_id = stable_uuid(ns, f"run:publish_marketing:{ctx.now.isoformat()}", ctx.deterministic)

    marketing_dir = ctx.out_dir / "marketing"
    receipts_path = marketing_dir / "receipts.jsonl"

    post_id = stable_uuid(ns, f"marketing_post:stub:{ctx.now.isoformat()}", ctx.deterministic)

    receipt = {
        "schema": "mgc.marketing_receipt.v1",
        "version": 1,
        "run_id": str(run_id),
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "platform": "stub",
        "status": "published",
        "post_id": str(post_id),
        "artifact": {"kind": "marketing_receipt", "path": "marketing/receipts.jsonl"},
    }

    _append_jsonl(receipts_path, receipt)
    receipts_sha = _sha256_file(receipts_path)

    ev = {
        "run_id": str(run_id),
        "stage": "publish-marketing",
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "note": "stub",
        "receipts": {
            "path": "marketing/receipts.jsonl",   # PORTABLE (no abs path)
            "sha256": receipts_sha,
            "appended": 1,
            "last_post_id": str(post_id),
        },
    }
    ev_path = ctx.evidence_dir / "publish_marketing_evidence.json"
    _json_dump(ev, ev_path)

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "run_id": str(run_id), "evidence": str(ev_path), "receipts_sha256": receipts_sha}))
    else:
        print(f"[run.publish-marketing] ok run_id={run_id}")
    return 0


def _write_drop_bundle(*, ctx: RunContext, context: str, schedule: str) -> Tuple[uuid.UUID, Path]:
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
    track_id = stable_uuid(ns, f"track:{schedule}:{period_key}:{context}", ctx.deterministic)
    playlist_id = stable_uuid(ns, f"playlist:{schedule}:{period_key}:{context}", ctx.deterministic)

    # Ensure a track exists for this schedule/period (provider handles deterministic bytes)
    provider_name = (getattr(args_global := ctx, "provider", None) or os.environ.get("MGC_PROVIDER") or "stub").strip().lower()
    # Provider generation is keyed by schedule+period so weekly and daily are distinct.
    try:
        provider = get_provider(provider_name)
        art_raw = provider.generate(
            out_dir=ctx.out_dir,
            track_id=str(track_id),
            context=context,
            seed=ctx.seed,
            deterministic=ctx.deterministic,
            now_iso=ctx.now.isoformat(),
            schedule=schedule,
            period_key=period_key,
        )
    except ProviderError as e:
        raise SystemExit(f"[run.{schedule}] provider error: {e}") from e

    art = _normalize_art(
        art_raw,
        fallback_track_id=str(track_id),
        fallback_provider=provider_name,
        fallback_context=context,
    )

    src_path = Path(str(art["artifact_path"]))

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
        "period": {
            "key": period_key,
            "label": period_label,
            "start_date": period_start.isoformat(),
            "end_date": period_end.isoformat(),
        },
        "ts": ctx.now.isoformat(),
        "tracks": [
            {
                "track_id": art["track_id"],
                "title": art["title"],
                "path": rel_track_path,          # REQUIRED by validator
                "artifact_path": rel_track_path, # back-compat
                "provider": art["provider"],
                "genre": art["genre"],
                "mood": art["mood"],
            }
        ],
    }
    _json_dump(playlist, playlist_path)
    playlist_sha = _sha256_file(playlist_path)

    # Bundle daily evidence (portable only)
    daily_ev_path = ctx.evidence_dir / "daily_evidence.json"
    bundle_daily_ev_path = bundle_dir / "daily_evidence.json"

    def _write_bundle_daily_evidence(base_obj: Dict[str, Any]) -> None:
        norm = _normalize_daily_evidence(base_obj, out_dir=ctx.out_dir, evidence_dir=ctx.evidence_dir)

        norm["schema"] = "mgc.daily_evidence.v1"
        norm["version"] = int(norm.get("version") or 1)
        norm["stage"] = "daily"
        norm["context"] = context
        norm["deterministic"] = ctx.deterministic
        norm["run_id"] = str(run_id)
        norm["ts"] = ctx.now.isoformat()
        norm["schedule"] = schedule
        norm["period"] = {
            "key": period_key,
            "label": period_label,
            "start_date": period_start.isoformat(),
            "end_date": period_end.isoformat(),
        }

        norm["track"] = {"track_id": art["track_id"], "path": rel_track_path}

        norm["paths"] = {
            "playlist": "playlist.json",
            "track": rel_track_path,
            "bundle_playlist": "playlist.json",
            "bundle_track": rel_track_path,
        }

        norm["sha256"] = {
            "track": bundle_track_sha,
            "playlist": playlist_sha,
            "bundle_track": bundle_track_sha,
            "bundle_playlist": playlist_sha,
        }

        _json_dump(norm, bundle_daily_ev_path)

    if daily_ev_path.exists():
        try:
            obj = _read_json(daily_ev_path)
            base = obj if isinstance(obj, dict) else {"raw": obj}
            _write_bundle_daily_evidence(base)
        except Exception:
            _write_bundle_daily_evidence({})
    else:
        _write_bundle_daily_evidence({})

    (bundle_dir / "README.txt").write_text(
        "MGC Drop Bundle (portable)\n"
        f"drop_id: {drop_id}\n"
        f"ts: {ctx.now.isoformat()}\n"
        f"context: {context}\n"
        f"schedule: {schedule}\n"
        f"period: {period_label} ({period_start.isoformat()}..{period_end.isoformat()})\n",
        encoding="utf-8",
    )

    root_ev = {
        "drop": {
            "id": str(drop_id),
            "ts": ctx.now.isoformat(),
            "context": context,
            "schedule": schedule,
            "period": {
                "key": period_key,
                "label": period_label,
                "start_date": period_start.isoformat(),
                "end_date": period_end.isoformat(),
            },
        },
        "deterministic": ctx.deterministic,
        "paths": {
            "out_dir": str(ctx.out_dir),
            "evidence_dir": str(ctx.evidence_dir),
            "bundle_dir": str(bundle_dir),
            "bundle_playlist": str(playlist_path),
            "bundle_daily_evidence": str(bundle_daily_ev_path),
        },
    }
    ev_path = _write_drop_evidence_root(ctx, root_ev)
    return drop_id, ev_path


def cmd_run_drop(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")
    drop_id, ev_path = _write_drop_bundle(ctx=ctx, context=context, schedule="daily")

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "drop_id": str(drop_id), "drop_evidence": str(ev_path), "bundle_dir": str(ctx.out_dir / "drop_bundle")}))
    else:
        print(f"[run.drop] ok drop_id={drop_id} evidence={ev_path}")
    return 0


def cmd_run_weekly(args: argparse.Namespace) -> int:
    rc = cmd_run_daily(args)
    if rc != 0:
        return rc
    rc = cmd_publish_marketing(args)
    if rc != 0:
        return rc

    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")
    drop_id, ev_path = _write_drop_bundle(ctx=ctx, context=context, schedule="weekly")

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "schedule": "weekly", "drop_id": str(drop_id), "drop_evidence": str(ev_path), "bundle_dir": str(ctx.out_dir / "drop_bundle")}))
    else:
        print(f"[run.weekly] ok drop_id={drop_id} evidence={ev_path}")
    return 0


def cmd_run_autonomous(args: argparse.Namespace) -> int:
    rc = cmd_run_daily(args)
    if rc != 0:
        return rc
    rc = cmd_publish_marketing(args)
    if rc != 0:
        return rc
    rc = cmd_run_drop(args)
    if rc != 0:
        return rc

    ctx = _build_run_context(args)
    drop_ev = ctx.out_dir / "drop_evidence.json"

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "out_dir": str(ctx.out_dir), "drop_evidence": str(drop_ev)}))
    else:
        print(f"[run.autonomous] ok out_dir={ctx.out_dir}")
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
            print(json.dumps({"ok": True, "stage_id": str(stage_id), "stage": stage_name}))
        else:
            print(f"[run.stage] ok stage={stage_name} id={stage_id}")
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

    print(f"Unknown run_cmd: {cmd}", file=sys.stderr)
    return 2


def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p_run = subparsers.add_parser(
        "run",
        help="Run pipeline steps (daily, weekly, autonomous, publish-marketing, drop, stage)",
    )
    p_run.set_defaults(fn=cmd_run_dispatch, func=cmd_run_dispatch)
    run_sub = p_run.add_subparsers(dest="run_cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", help="DB path (optional if provided globally)", default=argparse.SUPPRESS)
        p.add_argument("--repo-root", default=argparse.SUPPRESS, help="Repository root override (optional if provided globally)")
        p.add_argument("--out-dir", default="artifacts/run", help="Output directory for artifacts")
        p.add_argument("--evidence-dir", default=None, help="Evidence directory (default: <out-dir>/evidence)")
        p.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Generation context")
        p.add_argument("--provider", default=os.environ.get("MGC_PROVIDER", "stub"), help="Audio provider (stub|filesystem|...)")
        p.add_argument("--seed", type=int, default=int(os.environ.get("MGC_SEED", "1")), help="Seed")
        p.add_argument("--deterministic", action="store_true", help="Force deterministic mode (CI)")
        p.add_argument("--json", action="store_true", help="JSON output where supported")

    p = run_sub.add_parser("daily", help="Generate one track + daily evidence")
    add_common(p)
    p.set_defaults(fn=cmd_run_daily, func=cmd_run_daily)

    p = run_sub.add_parser("weekly", help="Weekly pipeline (daily -> publish-marketing -> weekly drop)")
    add_common(p)
    p.set_defaults(fn=cmd_run_weekly, func=cmd_run_weekly)

    p = run_sub.add_parser("autonomous", help="Daily pipeline (daily -> publish-marketing -> drop)")
    add_common(p)
    p.set_defaults(fn=cmd_run_autonomous, func=cmd_run_autonomous)

    p = run_sub.add_parser("publish-marketing", help="Publish pending marketing drafts (stub)")
    add_common(p)
    p.set_defaults(fn=cmd_publish_marketing, func=cmd_publish_marketing)

    p = run_sub.add_parser("drop", help="Create a daily drop bundle from latest artifacts")
    add_common(p)
    p.set_defaults(fn=cmd_run_drop, func=cmd_run_drop)

    p = run_sub.add_parser("stage", help="Run a named stage with run_stages tracking")
    add_common(p)
    p.add_argument("stage_name", help="Stage name")
    p.set_defaults(fn=cmd_run_stage, func=cmd_run_stage)
