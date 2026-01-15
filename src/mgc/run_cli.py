#!/usr/bin/env python3
"""
src/mgc/run_cli.py

Run/pipeline CLI for Music Generator Company (MGC).

Key behavior:
- Global flags (from mgc.main) like --db and --repo-root should work when placed
  BEFORE subcommands, e.g.:
      python -m mgc.main --db fixtures/ci_db.sqlite run status

- Also allow passing --db AFTER the run subcommand (override), e.g.:
      python -m mgc.main run status --db fixtures/ci_db.sqlite

To make both work reliably with argparse, we register run-subcommand --db with
default=argparse.SUPPRESS so it does not clobber the global value when omitted.
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
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Small I/O helpers
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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------

def is_deterministic(args: Optional[argparse.Namespace] = None) -> bool:
    if args is not None and getattr(args, "deterministic", False):
        return True
    v = (os.environ.get("MGC_DETERMINISTIC") or os.environ.get("DETERMINISTIC") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def fixed_utc_now(args: Optional[argparse.Namespace] = None) -> datetime:
    fixed = (os.environ.get("MGC_FIXED_TIME") or "").strip()
    if fixed:
        s = fixed.replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def stable_uuid(namespace: uuid.UUID, name: str, deterministic: bool) -> uuid.UUID:
    return uuid.uuid5(namespace, name) if deterministic else uuid.uuid4()


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    cur = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    )
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
# Manifest + diff (deterministic repo hashing)
# -----------------------------------------------------------------------------

DEFAULT_MANIFEST_ALLOW_EXT: Tuple[str, ...] = (
    ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini",
    ".sh", ".bash", ".zsh", ".sql", ".html", ".css", ".js",
)


def _iter_files_for_manifest(repo_root: Path, include_hidden: bool = False) -> Iterable[Path]:
    skip_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "artifacts",
        "data/tracks",
        "data/playlists",
        "node_modules",
        "dist",
        "build",
    }

    def should_skip_dir(p: Path) -> bool:
        try:
            rel = p.relative_to(repo_root).as_posix()
        except Exception:
            return True
        for d in skip_dirs:
            if rel == d or rel.startswith(d + "/"):
                return True
        return False

    for root, dirs, files in os.walk(repo_root):
        root_p = Path(root)

        dirs[:] = sorted(dirs)
        files = sorted(files)

        if should_skip_dir(root_p):
            dirs[:] = []
            continue

        for fn in files:
            if not include_hidden and fn.startswith("."):
                continue
            p = root_p / fn
            if p.suffix.lower() in DEFAULT_MANIFEST_ALLOW_EXT:
                yield p


def compute_repo_manifest(repo_root: Path) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for p in _iter_files_for_manifest(repo_root):
        rel = p.relative_to(repo_root).as_posix()
        items.append({"path": rel, "sha256": _sha256_file(p), "bytes": p.stat().st_size})

    h = hashlib.sha256()
    for it in items:
        h.update(it["path"].encode("utf-8"))
        h.update(b"\0")
        h.update(it["sha256"].encode("utf-8"))
        h.update(b"\n")

    return {"version": 1, "root_sha256": h.hexdigest(), "file_count": len(items), "items": items}


def diff_manifests(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    old_map = {it["path"]: it["sha256"] for it in old.get("items", [])}
    new_map = {it["path"]: it["sha256"] for it in new.get("items", [])}

    added = sorted([p for p in new_map if p not in old_map])
    removed = sorted([p for p in old_map if p not in new_map])
    changed = sorted([p for p in new_map if p in old_map and new_map[p] != old_map[p]])

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "summary": {"added": len(added), "removed": len(removed), "changed": len(changed)},
    }


# -----------------------------------------------------------------------------
# Submission bundle helper (deterministic zip)
# -----------------------------------------------------------------------------

def _zip_write_deterministic(zip_path: Path, files: Sequence[Tuple[Path, str]]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files_sorted = sorted(files, key=lambda t: t[1])

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files_sorted:
            data = src.read_bytes()
            zi = zipfile.ZipInfo(arcname)
            zi.date_time = (1980, 1, 1, 0, 0, 0)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, data)


# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    db_path: Path
    out_dir: Path
    evidence_dir: Path
    deterministic: bool
    now: datetime


def _require_db(args: argparse.Namespace) -> Path:
    db = getattr(args, "db", None)
    if not db:
        raise SystemExit("Missing --db. Use global: mgc --db <path> run ... or local: mgc run ... --db <path>")
    return Path(db).resolve()


def _build_run_context(args: argparse.Namespace) -> RunContext:
    repo_root = Path(getattr(args, "repo_root", ".")).resolve()
    db_path = _require_db(args)
    out_dir = Path(getattr(args, "out_dir", "artifacts/run")).resolve()
    evidence_dir_val = getattr(args, "evidence_dir", None)
    evidence_dir = Path(evidence_dir_val).resolve() if evidence_dir_val else (out_dir / "evidence")

    det = is_deterministic(args)
    now = fixed_utc_now(args) if det else datetime.now(timezone.utc)

    _ensure_dir(out_dir)
    _ensure_dir(evidence_dir)

    return RunContext(repo_root=repo_root, db_path=db_path, out_dir=out_dir, evidence_dir=evidence_dir, deterministic=det, now=now)


def _write_drop_evidence_root(ctx: RunContext, payload: Dict[str, Any]) -> Path:
    p = ctx.out_dir / "drop_evidence.json"
    _json_dump(payload, p)
    return p


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def cmd_run_daily(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    run_id = stable_uuid(ns, f"run:daily:{ctx.now.isoformat()}:{getattr(args, 'context', 'focus')}", ctx.deterministic)

    evidence = {
        "run_id": str(run_id),
        "stage": "daily",
        "context": getattr(args, "context", "focus"),
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "paths": {"out_dir": str(ctx.out_dir), "evidence_dir": str(ctx.evidence_dir)},
        "note": "daily stage stub executed",
    }
    _json_dump(evidence, ctx.evidence_dir / "daily_evidence.json")

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "run_id": str(run_id), "evidence": str(ctx.evidence_dir / "daily_evidence.json")}))
    else:
        print(f"[run.daily] ok run_id={run_id}")
    return 0


def cmd_publish_marketing(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    run_id = stable_uuid(ns, f"run:publish_marketing:{ctx.now.isoformat()}", ctx.deterministic)

    evidence = {
        "run_id": str(run_id),
        "stage": "publish-marketing",
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "note": "publish-marketing stage stub executed",
    }
    _json_dump(evidence, ctx.evidence_dir / "publish_marketing_evidence.json")

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "run_id": str(run_id), "evidence": str(ctx.evidence_dir / "publish_marketing_evidence.json")}))
    else:
        print(f"[run.publish-marketing] ok run_id={run_id}")
    return 0


def cmd_run_drop(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    drop_id = stable_uuid(ns, f"drop:{ctx.now.date().isoformat()}:{getattr(args, 'context', 'focus')}", ctx.deterministic)

    bundle_dir = ctx.out_dir / "drop_bundle"
    _ensure_dir(bundle_dir)

    readme = bundle_dir / "README.txt"
    readme.write_text(
        "MGC Drop Bundle\n"
        f"drop_id: {drop_id}\n"
        f"ts: {ctx.now.isoformat()}\n"
        f"context: {getattr(args, 'context', 'focus')}\n",
        encoding="utf-8",
    )

    drop_evidence = {
        "drop": {"id": str(drop_id), "ts": ctx.now.isoformat(), "context": getattr(args, "context", "focus")},
        "deterministic": ctx.deterministic,
        "paths": {"out_dir": str(ctx.out_dir), "bundle_dir": str(bundle_dir), "bundle_readme": str(readme)},
    }
    ev_path = _write_drop_evidence_root(ctx, drop_evidence)

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "drop_id": str(drop_id), "drop_evidence": str(ev_path), "bundle_dir": str(bundle_dir)}))
    else:
        print(f"[run.drop] ok drop_id={drop_id} evidence={ev_path}")
    return 0


def cmd_run_autonomous(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)

    rc = cmd_run_daily(args)
    if rc != 0:
        return rc

    rc = cmd_publish_marketing(args)
    if rc != 0:
        return rc

    rc = cmd_run_drop(args)
    if rc != 0:
        return rc

    drop_ev = ctx.out_dir / "drop_evidence.json"
    if not drop_ev.exists():
        _write_drop_evidence_root(ctx, {"ok": False, "reason": "drop stage did not emit drop_evidence.json"})

    if getattr(args, "build_submission", False):
        submission_path = ctx.out_dir / "submission.zip"

        files: List[Tuple[Path, str]] = []
        if drop_ev.exists():
            files.append((drop_ev, "drop_evidence.json"))

        if ctx.evidence_dir.exists():
            for p in sorted(ctx.evidence_dir.rglob("*")):
                if p.is_file():
                    files.append((p, f"evidence/{p.relative_to(ctx.evidence_dir).as_posix()}"))

        bundle_dir = ctx.out_dir / "drop_bundle"
        if bundle_dir.exists():
            for p in sorted(bundle_dir.rglob("*")):
                if p.is_file():
                    files.append((p, f"drop_bundle/{p.relative_to(bundle_dir).as_posix()}"))

        _zip_write_deterministic(submission_path, files)

        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "out_dir": str(ctx.out_dir), "submission": str(submission_path), "drop_evidence": str(drop_ev)}))
        else:
            print(f"[run.autonomous] ok submission={submission_path}")
        return 0

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

        ok = True
        meta = {"note": "stage executed (stub)", "stage": stage_name}
        finished = fixed_utc_now(args).isoformat() if ctx.deterministic else datetime.now(timezone.utc).isoformat()

        con.execute(
            "UPDATE run_stages SET finished_ts=?, ok=?, meta_json=? WHERE id=?",
            (finished, 1 if ok else 0, json.dumps(meta, sort_keys=True), str(stage_id)),
        )
        con.commit()

        if getattr(args, "json", False):
            print(json.dumps({"ok": ok, "stage_id": str(stage_id), "stage": stage_name}))
        else:
            print(f"[run.stage] ok stage={stage_name} id={stage_id}")
        return 0
    finally:
        con.close()


def cmd_run_manifest(args: argparse.Namespace) -> int:
    repo_root = Path(getattr(args, "repo_root", ".")).resolve()
    out_path = Path(args.out).resolve()
    manifest = compute_repo_manifest(repo_root)
    _json_dump(manifest, out_path)

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "out": str(out_path), "root_sha256": manifest["root_sha256"], "file_count": manifest["file_count"]}))
    else:
        print(f"[run.manifest] ok out={out_path} root_sha256={manifest['root_sha256']} files={manifest['file_count']}")
    return 0


def cmd_run_diff(args: argparse.Namespace) -> int:
    old_p = Path(args.old).resolve()
    new_p = Path(args.new).resolve()
    old = _read_json(old_p)
    new = _read_json(new_p)
    d = diff_manifests(old, new)

    summary = d["summary"]
    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "summary": summary, "added": d["added"], "removed": d["removed"], "changed": d["changed"]}))
    else:
        print(f"+{summary['added']}  -{summary['removed']}  ~{summary['changed']}  (older={old_p.name} newer={new_p.name})")

    if getattr(args, "fail_on_changes", False) and (summary["added"] or summary["removed"] or summary["changed"]):
        return 2
    return 0


def cmd_run_status(args: argparse.Namespace) -> int:
    db_path = _require_db(args)

    out: Dict[str, Any] = {"db": str(db_path), "ok": True}
    if not db_path.exists():
        out["ok"] = False
        out["reason"] = "db_missing"
        if getattr(args, "json", False):
            print(json.dumps(out))
        else:
            print(f"[run.status] FAIL db missing: {db_path}")
        return 2

    con = _connect(db_path)
    try:
        tables = ["tracks", "playlists", "marketing_posts", "drops", "run_stages"]
        out["tables"] = {t: _table_exists(con, t) for t in tables}
    finally:
        con.close()

    if getattr(args, "json", False):
        print(json.dumps(out))
    else:
        tbls = ", ".join([f"{k}={'yes' if v else 'no'}" for k, v in out.get("tables", {}).items()])
        print(f"[run.status] ok db={db_path} {tbls}")
    return 0


# -----------------------------------------------------------------------------
# Argparse wiring
# -----------------------------------------------------------------------------

def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p_run = subparsers.add_parser(
        "run",
        help="Run pipeline steps (daily, autonomous, publish-marketing, drop, stage, manifest, diff, status)",
    )
    run_sub = p_run.add_subparsers(dest="run_cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        # IMPORTANT:
        # - We allow --db here, but with default=SUPPRESS so it doesn't overwrite global args.db.
        p.add_argument("--db", help="DB path (optional if provided globally)", default=argparse.SUPPRESS)
        p.add_argument("--repo-root", default=argparse.SUPPRESS, help="Repository root override (optional if provided globally)")
        p.add_argument("--out-dir", default="artifacts/run", help="Output directory for artifacts")
        p.add_argument("--evidence-dir", default=None, help="Evidence directory (default: <out-dir>/evidence)")
        p.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Generation context")
        p.add_argument("--deterministic", action="store_true", help="Force deterministic mode (CI)")
        p.add_argument("--json", action="store_true", help="JSON output where supported")

    # daily
    p = run_sub.add_parser("daily", help="Run the daily pipeline (deterministic capable)")
    add_common(p)
    p.set_defaults(fn=cmd_run_daily)

    # autonomous
    p = run_sub.add_parser("autonomous", help="Run autonomous pipeline stages (daily -> publish-marketing -> drop)")
    add_common(p)
    p.add_argument("--build-submission", action="store_true", help="Also build submission.zip under --out-dir")
    p.set_defaults(fn=cmd_run_autonomous)

    # publish-marketing
    p = run_sub.add_parser("publish-marketing", help="Publish pending marketing drafts (draft -> published)")
    add_common(p)
    p.set_defaults(fn=cmd_publish_marketing)

    # drop
    p = run_sub.add_parser("drop", help="Create a minimal drop bundle from the latest artifacts")
    add_common(p)
    p.set_defaults(fn=cmd_run_drop)

    # stage
    p = run_sub.add_parser("stage", help="Run a named stage with run_stages tracking")
    add_common(p)
    p.add_argument("stage_name", help="Stage name")
    p.set_defaults(fn=cmd_run_stage)

    # manifest (no db required)
    p = run_sub.add_parser("manifest", help="Compute deterministic repo manifest (stable file hashing)")
    p.add_argument("--repo-root", default=".", help="Repository root")
    p.add_argument("--out", default="data/evidence/manifest.json", help="Output manifest path")
    p.add_argument("--json", action="store_true", help="JSON output (command result)")
    p.set_defaults(fn=cmd_run_manifest)

    # diff (no db required)
    p = run_sub.add_parser("diff", help="Compare manifest files (CI gate helper)")
    p.add_argument("--old", default="data/evidence/manifest.json", help="Older manifest path")
    p.add_argument("--new", default="data/evidence/weekly_manifest.json", help="Newer manifest path")
    p.add_argument("--fail-on-changes", action="store_true", help="Exit non-zero if any changes exist")
    p.add_argument("--json", action="store_true", help="JSON diff output")
    p.set_defaults(fn=cmd_run_diff)

    # status (db required, but may come from global)
    p = run_sub.add_parser("status", help="Show run/pipeline status snapshot")
    p.add_argument("--db", help="DB path (optional if provided globally)", default=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(fn=cmd_run_status)
