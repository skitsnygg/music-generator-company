#!/usr/bin/env python3
"""
src/mgc/run_cli.py

Run/pipeline CLI for Music Generator Company (MGC).

Hard requirements for CI:
- `python -m mgc.main --db ... --repo-root ... --seed 1 --no-resume --json run autonomous ...`
  must succeed (CI passes global flags before the subcommand).
- Top-level dispatch must always find a handler; if mgc.main expects args.fn, we provide it.
- `run autonomous` must produce:
    out_dir/drop_evidence.json                (CI contract)
    out_dir/drop_bundle/playlist.json         (portable bundle)
    out_dir/drop_bundle/tracks/*.wav          (portable bundle)
    out_dir/drop_bundle/daily_evidence.json   (portable bundle)

Bundle contract for submission.build (validator expectations):
- drop_bundle/playlist.json must include:
    {"schema": "mgc.playlist.v1", ...}
- each playlist track entry must include:
    tracks[i].path   (REQUIRED by validator)
- drop_bundle/daily_evidence.json must include:
    {"schema": "mgc.daily_evidence.v1", "paths": {...}, "sha256": {...}, ...}

Determinism contract:
- In deterministic mode (CI), bundles + submission.zip must be byte-for-byte stable across runs,
  even when out_dir differs (e.g. /tmp/run1 vs /tmp/run2).
- Therefore the BUNDLED JSON must not contain absolute paths like out_dir/evidence_dir/tracks_dir.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _as_rel_posix(path_like: str) -> str:
    """
    Coerce a path string to a RELATIVE, forward-slash path (portable bundle contract).
    Reject absolute paths and traversal.
    """
    s = (path_like or "").replace("\\", "/").strip()
    s = s.lstrip("./")
    if not s or s.startswith("/") or s.startswith("..") or "/../" in s or s.endswith("/.."):
        raise ValueError(f"invalid relative path: {path_like!r}")
    return s


# -----------------------------------------------------------------------------
# Determinism helpers
# -----------------------------------------------------------------------------

def is_deterministic(args: Optional[argparse.Namespace] = None) -> bool:
    if args is not None and getattr(args, "deterministic", False):
        return True
    v = (os.environ.get("MGC_DETERMINISTIC") or os.environ.get("DETERMINISTIC") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def stable_uuid(namespace: uuid.UUID, name: str, deterministic: bool) -> uuid.UUID:
    return uuid.uuid5(namespace, name) if deterministic else uuid.uuid4()


def _deterministic_now_utc() -> datetime:
    """
    Deterministic clock for CI.

    - If MGC_FIXED_TIME is set, honor it.
    - Otherwise return a stable epoch (2020-01-01T00:00:00Z) so deterministic runs
      are reproducible without extra env configuration.
    """
    fixed = (os.environ.get("MGC_FIXED_TIME") or "").strip()
    if fixed:
        s = fixed.replace("Z", "+00:00")
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    return datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


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
# Minimal deterministic WAV generation (16-bit PCM mono)
# -----------------------------------------------------------------------------

def _write_wav_sine(path: Path, *, seconds: float = 1.0, freq_hz: float = 440.0, sample_rate: int = 22050) -> None:
    import struct

    n_samples = max(1, int(sample_rate * seconds))
    amp = 0.2
    samples = bytearray()

    for i in range(n_samples):
        t = i / sample_rate
        v = int(amp * 32767 * math.sin(2 * math.pi * freq_hz * t))
        samples += struct.pack("<h", v)

    data_size = len(samples)
    byte_rate = sample_rate * 1 * 16 // 8
    block_align = 1 * 16 // 8

    header = bytearray()
    header += b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)
    header += struct.pack("<H", 1)           # PCM
    header += struct.pack("<H", 1)           # mono
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", 16)          # 16-bit
    header += b"data"
    header += struct.pack("<I", data_size)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + bytes(samples))


# -----------------------------------------------------------------------------
# Core pipeline context
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

    _ensure_dir(out_dir)
    _ensure_dir(evidence_dir)

    return RunContext(
        repo_root=repo_root,
        db_path=db_path,
        out_dir=out_dir,
        evidence_dir=evidence_dir,
        deterministic=det,
        now=now,
    )


def _write_drop_evidence_root(ctx: RunContext, payload: Dict[str, Any]) -> Path:
    p = ctx.out_dir / "drop_evidence.json"
    _json_dump(payload, p)
    return p


def _normalize_daily_evidence(obj: Dict[str, Any], *, out_dir: Path, evidence_dir: Path) -> Dict[str, Any]:
    """
    Ensure daily_evidence.json matches validator expectations.
    The validator wants:
      - daily_evidence.json.paths to be an object
      - daily_evidence.json.sha256 to be an object

    NOTE: For the BUNDLED daily_evidence.json we overwrite paths/sha256 with portable-only keys.
    """
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
# Commands (leaf handlers)
# -----------------------------------------------------------------------------

def cmd_run_daily(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    run_id = stable_uuid(ns, f"run:daily:{ctx.now.isoformat()}:{context}", ctx.deterministic)

    tracks_dir = ctx.out_dir / "tracks"
    _ensure_dir(tracks_dir)

    track_id = stable_uuid(ns, f"track:{ctx.now.date().isoformat()}:{context}", ctx.deterministic)
    wav_path = tracks_dir / f"{track_id}.wav"
    _write_wav_sine(wav_path)

    track_sha = _sha256_file(wav_path)

    ev_raw = {
        "run_id": str(run_id),
        "stage": "daily",
        "context": context,
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "track": {"track_id": str(track_id), "path": str(wav_path)},
        "paths": {
            "out_dir": str(ctx.out_dir),
            "evidence_dir": str(ctx.evidence_dir),
            "tracks_dir": str(tracks_dir),
        },
        "sha256": {
            "track": track_sha,
        },
    }
    ev = _normalize_daily_evidence(ev_raw, out_dir=ctx.out_dir, evidence_dir=ctx.evidence_dir)

    ev_path = ctx.evidence_dir / "daily_evidence.json"
    _json_dump(ev, ev_path)

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "run_id": str(run_id), "track_id": str(track_id), "evidence": str(ev_path)}))
    else:
        print(f"[run.daily] ok run_id={run_id}")
    return 0


def cmd_publish_marketing(args: argparse.Namespace) -> int:
    ctx = _build_run_context(args)
    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")
    run_id = stable_uuid(ns, f"run:publish_marketing:{ctx.now.isoformat()}", ctx.deterministic)

    ev = {
        "run_id": str(run_id),
        "stage": "publish-marketing",
        "deterministic": ctx.deterministic,
        "ts": ctx.now.isoformat(),
        "note": "stub",
    }
    ev_path = ctx.evidence_dir / "publish_marketing_evidence.json"
    _json_dump(ev, ev_path)

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "run_id": str(run_id), "evidence": str(ev_path)}))
    else:
        print(f"[run.publish-marketing] ok run_id={run_id}")
    return 0


def cmd_run_drop(args: argparse.Namespace) -> int:
    """
    Create a portable bundle that `mgc submission build --bundle-dir <dir>` can package.
    """
    ctx = _build_run_context(args)
    context = getattr(args, "context", "focus")

    ns = uuid.UUID("00000000-0000-0000-0000-000000000000")

    # Stable ids (deterministic mode => uuid5; non-deterministic => uuid4)
    drop_id = stable_uuid(ns, f"drop:{ctx.now.date().isoformat()}:{context}", ctx.deterministic)
    run_id = stable_uuid(ns, f"run:daily:{ctx.now.isoformat()}:{context}", ctx.deterministic)
    track_id = stable_uuid(ns, f"track:{ctx.now.date().isoformat()}:{context}", ctx.deterministic)
    playlist_id = stable_uuid(ns, f"playlist:{ctx.now.date().isoformat()}:{context}", ctx.deterministic)

    # Ensure a deterministic track exists
    wav_src = ctx.out_dir / "tracks" / f"{track_id}.wav"
    if not wav_src.exists():
        _ensure_dir(wav_src.parent)
        _write_wav_sine(wav_src)

    # Bundle layout
    bundle_dir = ctx.out_dir / "drop_bundle"
    bundle_tracks = bundle_dir / "tracks"
    _ensure_dir(bundle_tracks)

    wav_dst = bundle_tracks / wav_src.name
    wav_dst.write_bytes(wav_src.read_bytes())

    rel_track_path = _as_rel_posix(f"tracks/{wav_dst.name}")
    bundle_track_sha = _sha256_file(wav_dst)

    # Write playlist.json (portable). Then force stable ts by rewriting (belt+suspenders).
    playlist_path = bundle_dir / "playlist.json"
    playlist = {
        "schema": "mgc.playlist.v1",
        "version": 1,
        "playlist_id": str(playlist_id),
        "context": context,
        "ts": ctx.now.isoformat(),
        "tracks": [
            {
                "track_id": str(track_id),
                "title": f"{context.title()} Track",
                "path": rel_track_path,           # REQUIRED by validator
                "artifact_path": rel_track_path,  # back-compat
                "provider": "stub",
                "genre": "stub",
                "mood": context,
            }
        ],
    }
    _json_dump(playlist, playlist_path)

    # Force stable ts (in case other code paths ever modify ctx.now formatting)
    try:
        pl = _read_json(playlist_path)
        if isinstance(pl, dict):
            pl["ts"] = ctx.now.isoformat()
            _json_dump(pl, playlist_path)
    except Exception:
        # If anything goes wrong reading/writing, keep the already-written file.
        pass

    playlist_sha = _sha256_file(playlist_path)

    # Bundle daily evidence (portable-only paths, validator keys present)
    daily_ev_path = ctx.evidence_dir / "daily_evidence.json"
    bundle_daily_ev_path = bundle_dir / "daily_evidence.json"

    def _write_bundle_daily_evidence(base_obj: Dict[str, Any]) -> None:
        norm = _normalize_daily_evidence(base_obj, out_dir=ctx.out_dir, evidence_dir=ctx.evidence_dir)

        # Force stable fields in the bundled artifact
        norm["schema"] = "mgc.daily_evidence.v1"
        norm["version"] = int(norm.get("version") or 1)
        norm["stage"] = "daily"
        norm["context"] = context
        norm["deterministic"] = ctx.deterministic
        norm["run_id"] = str(run_id)
        norm["ts"] = ctx.now.isoformat()

        # Track object must be portable
        norm["track"] = {
            "track_id": str(track_id),
            "path": rel_track_path,
        }

        # Paths MUST be portable-only (no absolute dirs)
        norm["paths"] = {
            # Validator-required keys:
            "playlist": "playlist.json",
            "track": rel_track_path,
            # Back-compat keys:
            "bundle_playlist": "playlist.json",
            "bundle_track": rel_track_path,
        }

        # Sha256 MUST include validator-required keys
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
        f"context: {context}\n",
        encoding="utf-8",
    )

    root_ev = {
        "drop": {"id": str(drop_id), "ts": ctx.now.isoformat(), "context": context},
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

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "drop_id": str(drop_id), "drop_evidence": str(ev_path), "bundle_dir": str(bundle_dir)}))
    else:
        print(f"[run.drop] ok drop_id={drop_id} evidence={ev_path}")
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

        # In deterministic mode, keep finished_ts stable too.
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
    d = diff_manifests(_read_json(old_p), _read_json(new_p))
    s = d["summary"]

    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "summary": s, "added": d["added"], "removed": d["removed"], "changed": d["changed"]}))
    else:
        print(f"+{s['added']}  -{s['removed']}  ~{s['changed']}  (older={old_p.name} newer={new_p.name})")

    if getattr(args, "fail_on_changes", False) and (s["added"] or s["removed"] or s["changed"]):
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
# Run dispatcher (covers both mgc.main dispatch styles)
# -----------------------------------------------------------------------------

def cmd_run_dispatch(args: argparse.Namespace) -> int:
    cmd = getattr(args, "run_cmd", None)
    if cmd == "daily":
        return cmd_run_daily(args)
    if cmd == "autonomous":
        return cmd_run_autonomous(args)
    if cmd == "publish-marketing":
        return cmd_publish_marketing(args)
    if cmd == "drop":
        return cmd_run_drop(args)
    if cmd == "stage":
        return cmd_run_stage(args)
    if cmd == "manifest":
        return cmd_run_manifest(args)
    if cmd == "diff":
        return cmd_run_diff(args)
    if cmd == "status":
        return cmd_run_status(args)

    print(f"Unknown run_cmd: {cmd}", file=sys.stderr)
    return 2


# -----------------------------------------------------------------------------
# Argparse wiring
# -----------------------------------------------------------------------------

def register_run_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p_run = subparsers.add_parser(
        "run",
        help="Run pipeline steps (daily, autonomous, publish-marketing, drop, stage, manifest, diff, status)",
    )

    # Provide a handler at the run level too (some mgc.main variants dispatch here).
    p_run.set_defaults(fn=cmd_run_dispatch, func=cmd_run_dispatch)

    run_sub = p_run.add_subparsers(dest="run_cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", help="DB path (optional if provided globally)", default=argparse.SUPPRESS)
        p.add_argument("--repo-root", default=argparse.SUPPRESS, help="Repository root override (optional if provided globally)")
        p.add_argument("--out-dir", default="artifacts/run", help="Output directory for artifacts")
        p.add_argument("--evidence-dir", default=None, help="Evidence directory (default: <out-dir>/evidence)")
        p.add_argument("--context", default=os.environ.get("MGC_CONTEXT", "focus"), help="Generation context")
        p.add_argument("--deterministic", action="store_true", help="Force deterministic mode (CI)")
        p.add_argument("--json", action="store_true", help="JSON output where supported")

    p = run_sub.add_parser("daily", help="Run the daily pipeline (deterministic capable)")
    add_common(p)
    p.set_defaults(fn=cmd_run_daily, func=cmd_run_daily)

    p = run_sub.add_parser("autonomous", help="Run autonomous pipeline stages (daily -> publish-marketing -> drop)")
    add_common(p)
    p.set_defaults(fn=cmd_run_autonomous, func=cmd_run_autonomous)

    p = run_sub.add_parser("publish-marketing", help="Publish pending marketing drafts (draft -> published)")
    add_common(p)
    p.set_defaults(fn=cmd_publish_marketing, func=cmd_publish_marketing)

    p = run_sub.add_parser("drop", help="Create a minimal drop bundle from the latest artifacts")
    add_common(p)
    p.set_defaults(fn=cmd_run_drop, func=cmd_run_drop)

    p = run_sub.add_parser("stage", help="Run a named stage with run_stages tracking")
    add_common(p)
    p.add_argument("stage_name", help="Stage name")
    p.set_defaults(fn=cmd_run_stage, func=cmd_run_stage)

    p = run_sub.add_parser("manifest", help="Compute deterministic repo manifest (stable file hashing)")
    p.add_argument("--repo-root", default=".", help="Repository root")
    p.add_argument("--out", default="data/evidence/manifest.json", help="Output manifest path")
    p.add_argument("--json", action="store_true", help="JSON output (command result)")
    p.set_defaults(fn=cmd_run_manifest, func=cmd_run_manifest)

    p = run_sub.add_parser("diff", help="Compare manifest files (CI gate helper)")
    p.add_argument("--old", default="data/evidence/manifest.json", help="Older manifest path")
    p.add_argument("--new", default="data/evidence/weekly_manifest.json", help="Newer manifest path")
    p.add_argument("--fail-on-changes", action="store_true", help="Exit non-zero if any changes exist")
    p.add_argument("--json", action="store_true", help="JSON diff output")
    p.set_defaults(fn=cmd_run_diff, func=cmd_run_diff)

    p = run_sub.add_parser("status", help="Show run/pipeline status snapshot")
    p.add_argument("--db", help="DB path (optional if provided globally)", default=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(fn=cmd_run_status, func=cmd_run_status)
