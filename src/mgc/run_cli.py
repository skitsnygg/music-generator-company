#!/usr/bin/env python3
"""
src/mgc/run_cli.py

Run/pipeline CLI for Music Generator Company (MGC).

CI contracts:
- In --json mode, each command must emit EXACTLY ONE JSON object to stdout.
- Human logs must go to stderr.

Determinism:
- evidence/* must not contain absolute out_dir-dependent paths.
- Weekly determinism must hold even if --out-dir differs.

ci_root_determinism.sh contract:
- `python -m mgc.main --json run drop ...` output JSON must include paths.manifest_sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mgc.providers import ProviderError, get_provider


# -----------------------------------------------------------------------------
# Output discipline
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _emit_json(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, sort_keys=True) + "\n")
    sys.stdout.flush()


# -----------------------------------------------------------------------------
# small I/O helpers
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    out: Dict[str, Any] = {}

    out["provider"] = _art_get(art, "provider", fallback_provider)
    out["track_id"] = _art_get(art, "track_id", fallback_track_id)
    out["artifact_path"] = _art_get(art, "artifact_path")
    out["sha256"] = _art_get(art, "sha256")

    out["meta"] = _art_get(art, "meta", {}) or {}
    out["genre"] = _art_get(art, "genre", "") or ""
    out["mood"] = _art_get(art, "mood", "") or ""
    out["title"] = _art_get(art, "title", "") or ""

    if not out["title"]:
        out["title"] = f"{fallback_context.title()} Track"
    if not out["genre"]:
        out["genre"] = out["provider"] or "unknown"
    if not out["mood"]:
        out["mood"] = fallback_context

    if not out["artifact_path"]:
        raise SystemExit("Provider returned no artifact_path")
    if not out["sha256"]:
        out["sha256"] = _sha256_file(Path(str(out["artifact_path"])))

    return out


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
    Returns {run_id, track_id, rel_track_path}.
    """
    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    run_id = stable_uuid(ns, f"run:daily:{ctx.now.isoformat()}:{context}", ctx.deterministic)
    track_id = stable_uuid(ns, f"track:daily:{ctx.now.date().isoformat()}:{context}", ctx.deterministic)

    provider = get_provider(ctx.provider)
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

    art = _normalize_art(art_raw, fallback_track_id=str(track_id), fallback_provider=ctx.provider, fallback_context=context)

    wav_path = Path(str(art["artifact_path"]))
    rel_track_path = _as_rel_posix(f"tracks/{wav_path.name}")

    ev = {
        "schema": "mgc.daily_evidence.v1",
        "version": 1,
        "run_id": str(run_id),
        "stage": "daily",
        "context": context,
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "provider": art["provider"],
        "track": {"track_id": art["track_id"], "path": rel_track_path},
        "sha256": {"track": str(art["sha256"])},
        "meta": art.get("meta") or {},
    }
    _json_dump(ev, ctx.evidence_dir / "daily_evidence.json")
    return {"run_id": str(run_id), "track_id": art["track_id"], "rel_track_path": rel_track_path}


def cmd_run_daily(args: argparse.Namespace) -> int:
    info = _run_daily_no_emit(args)
    if getattr(args, "json", False):
        _emit_json({"ok": True, "stage": "daily", "run_id": info["run_id"], "track_id": info["track_id"], "paths": {"daily_evidence": "evidence/daily_evidence.json"}})
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
    track_id = stable_uuid(ns, f"track:{schedule}:{period_key}:{context}", ctx.deterministic)
    playlist_id = stable_uuid(ns, f"playlist:{schedule}:{period_key}:{context}", ctx.deterministic)

    provider = get_provider(ctx.provider)
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
    art = _normalize_art(art_raw, fallback_track_id=str(track_id), fallback_provider=ctx.provider, fallback_context=context)

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
        "period": {"key": period_key, "label": period_label, "start_date": period_start.isoformat(), "end_date": period_end.isoformat()},
        "ts": ctx.now.isoformat(),
        "tracks": [{
            "track_id": art["track_id"],
            "title": art["title"],
            "path": rel_track_path,
            "artifact_path": rel_track_path,
            "provider": art["provider"],
            "genre": art["genre"],
            "mood": art["mood"],
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
        "track": {"track_id": art["track_id"], "path": rel_track_path},
        "sha256": {"track": bundle_track_sha, "playlist": playlist_sha},
        "meta": art.get("meta") or {},
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
# Commands
# -----------------------------------------------------------------------------

def cmd_publish_marketing(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    dry_run = bool(getattr(args, "dry_run", False))

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
    }

    appended = 0
    if not dry_run:
        _append_jsonl(receipts_path, receipt)
        appended = 1

    receipts_sha = _sha256_file(receipts_path) if receipts_path.exists() else ""

    ev = {
        "run_id": str(run_id),
        "stage": "publish-marketing",
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "dry_run": dry_run,
        "receipts": {"path": "marketing/receipts.jsonl", "sha256": receipts_sha, "appended": appended},
    }
    _json_dump(ev, ctx.evidence_dir / "publish_marketing_evidence.json")

    if getattr(args, "json", False):
        _emit_json({"ok": True, "stage": "publish-marketing", "run_id": str(run_id), "paths": {"publish_marketing_evidence": "evidence/publish_marketing_evidence.json"}})
    else:
        _log(f"[run.publish-marketing] ok run_id={run_id}")
    return 0


def cmd_run_drop(args: argparse.Namespace) -> int:
    # DO daily work without emitting anything (so run drop outputs ONE JSON object).
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
    # daily + publish without extra JSON emissions
    _run_daily_no_emit(args)
    # publish-marketing should not emit JSON as part of weekly JSON output; keep it quiet:
    # Temporarily force args.json false for publish if we're in json mode.
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        setattr(args, "json", False)
    cmd_publish_marketing(args)
    if json_mode:
        setattr(args, "json", True)

    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")
    drop_id, produced = _write_drop_bundle(ctx=ctx, context=context, schedule="weekly")

    if getattr(args, "json", False):
        _emit_json({"ok": True, "stage": "weekly", "drop_id": str(drop_id), "paths": {"drop_evidence": "drop_evidence.json", "bundle_dir": "drop_bundle"}})
    else:
        _log(f"[run.weekly] ok drop_id={drop_id} evidence={produced['drop_evidence']}")
    return 0


def cmd_run_autonomous(args: argparse.Namespace) -> int:
    # In JSON mode autonomous should output ONE JSON object; keep sub-steps quiet.
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        setattr(args, "json", False)

    _run_daily_no_emit(args)
    cmd_publish_marketing(args)
    cmd_run_drop(args)

    if json_mode:
        setattr(args, "json", True)
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
        # compatibility flags used by CI scripts
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
