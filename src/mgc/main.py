#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from dotenv import load_dotenv

try:
    from mgc.logging_setup import setup_logging  # type: ignore
except Exception:
    setup_logging = None  # type: ignore

from mgc.analytics_cli import register_analytics_subcommand
from mgc.events import EventContext, EventWriter, new_run_id
from mgc.rebuild_cli import register_rebuild_subcommand
from mgc.db_helpers import (
    NotFoundError,
    PlaylistRow,
    ResolvedRef,
    TrackRow,
    compute_diff,
    db_build_playlist_json,
    db_duration_for_tracks,
    db_get_playlist,
    db_get_track,
    db_insert_playlist_run,
    db_list_events,
    db_list_playlist_runs,
    db_list_playlists,
    db_list_playlists_by_slug,
    db_list_tracks,
    db_playlist_track_ids,
    db_tracks_stats,
    ensure_playlist_runs_table,
    resolve_ref,
    resolve_json_path,
    sqlite_connect,
)


# ----------------------------
# basic utils
# ----------------------------

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def die(msg: str, code: int = 2) -> NoReturn:
    eprint(msg)
    raise SystemExit(code)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        die(f"JSON file not found: {path}")
    except json.JSONDecodeError as ex:
        die(f"Invalid JSON in {path}: {ex}")


def write_json_file(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_requests() -> Any:
    try:
        import requests  # type: ignore
        return requests
    except Exception:
        die("Missing dependency: requests. Install with: pip install requests")


def cmd_name(args: argparse.Namespace) -> str:
    parts: List[str] = []
    if getattr(args, "cmd", None):
        parts.append(str(args.cmd))
    if getattr(args, "db_cmd", None):
        parts.append(str(args.db_cmd))
    if getattr(args, "playlists_cmd", None):
        parts.append(str(args.playlists_cmd))
    if getattr(args, "tracks_cmd", None):
        parts.append(str(args.tracks_cmd))
    if getattr(args, "analytics_cmd", None):
        parts.append(str(args.analytics_cmd))
    if getattr(args, "events_cmd", None):
        parts.append(str(args.events_cmd))
    if getattr(args, "rebuild_cmd", None):
        parts.append(str(args.rebuild_cmd))
    if getattr(args, "manifest_cmd", None):
        parts.append(str(args.manifest_cmd))
    return " ".join(parts) if parts else "unknown"


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            if k == "func":
                continue
            out[str(k)] = _jsonable(v)
        return out
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted([_jsonable(v) for v in x])
    if callable(x):
        return getattr(x, "__name__", "<callable>")
    return str(x)


def args_payload(args: argparse.Namespace) -> Dict[str, Any]:
    d = vars(args).copy()
    d.pop("func", None)
    return _jsonable(d)


# ----------------------------
# export layout
# ----------------------------

def default_export_dir() -> Path:
    return Path(os.environ.get("MGC_PLAYLISTS_DIR", "data/playlists"))


def safe_slug(s: str) -> str:
    s = s.strip()
    out = "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_")).strip("-_")
    return out


def export_filename(playlist_id: str, slug: Optional[str] = None) -> str:
    if slug:
        return f"{safe_slug(slug)}_{playlist_id}.json"
    return f"{playlist_id}.json"


# ----------------------------
# validation
# ----------------------------

def validate_playlist_json(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "playlist JSON is not an object"
    tracks = obj.get("tracks")
    if not isinstance(tracks, list):
        return False, "missing tracks[] list"
    if len(tracks) == 0:
        return False, "tracks[] is empty"
    return True, "ok"


# ----------------------------
# push targets
# ----------------------------

@dataclass
class PushResult:
    target: str
    ok: bool
    detail: str


def push_local(src_path: Path, dst_dir: Path, overwrite: bool, dry_run: bool) -> PushResult:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src_path.name

    if dst_path.exists() and not overwrite:
        return PushResult("local", True, f"Skipped (exists): {dst_path}")

    if dry_run:
        return PushResult("local", True, f"Dry-run copy to: {dst_path}")

    dst_path.write_bytes(src_path.read_bytes())
    return PushResult("local", True, f"Copied to: {dst_path}")


def push_webhook(src_path: Path, url: str, timeout_s: int, dry_run: bool) -> PushResult:
    requests = ensure_requests()
    payload = read_json_file(src_path)
    headers = {"Content-Type": "application/json", "User-Agent": "mgc-cli/1.0"}

    if dry_run:
        return PushResult("webhook", True, f"Dry-run POST to: {url}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        if 200 <= int(resp.status_code) < 300:
            return PushResult("webhook", True, f"POST {resp.status_code}: {url}")
        return PushResult("webhook", False, f"POST {resp.status_code}: {resp.text[:400]}")
    except Exception as ex:
        return PushResult("webhook", False, f"POST failed: {ex}")


# ----------------------------
# playlist export helpers
# ----------------------------

def export_one_playlist(conn, pl: PlaylistRow, hookup_export_dir: Path, build: bool) -> Path:
    hookup_export_dir.mkdir(parents=True, exist_ok=True)
    out_path = hookup_export_dir / export_filename(pl.id, pl.slug)

    if build:
        obj = db_build_playlist_json(conn, pl, built_at=now_iso())
        obj.setdefault("exported_at", now_iso())
        write_json_file(out_path, obj)
        return out_path

    src = resolve_json_path(pl.json_path)
    if src.exists():
        obj = read_json_file(src)
        if isinstance(obj, dict):
            obj.setdefault("id", pl.id)
            obj.setdefault("slug", pl.slug)
            obj.setdefault("exported_at", now_iso())
            write_json_file(out_path, obj)
        else:
            out_path.write_bytes(src.read_bytes())
        return out_path

    obj = db_build_playlist_json(conn, pl, built_at=now_iso())
    obj.setdefault("exported_at", now_iso())
    write_json_file(out_path, obj)
    return out_path


def maybe_record_run(
    conn,
    ew: Optional[EventWriter],
    pl: PlaylistRow,
    export_path: Path,
    record: bool,
    notes: Optional[str],
) -> Optional[str]:
    if not record:
        return None

    ensure_playlist_runs_table(conn)
    track_ids = db_playlist_track_ids(conn, pl.id)

    rid = db_insert_playlist_run(
        conn=conn,
        created_at=now_iso(),
        playlist_id=pl.id,
        seed=pl.seed,
        track_ids=track_ids,
        export_path=export_path,
        notes=notes,
    )

    if ew:
        ew.emit(
            "playlist.run_recorded",
            "playlist",
            pl.id,
            {
                "playlist_id": pl.id,
                "slug": pl.slug,
                "run_id": rid,
                "seed": pl.seed,
                "track_count": len(track_ids),
                "export_path": str(export_path),
                "notes": notes,
            },
            occurred_at=now_iso(),
        )
    return rid


# ----------------------------
# commands
# ----------------------------

def events_list_cmd(args: argparse.Namespace) -> int:
    conn = sqlite_connect(args.db)
    try:
        rows = db_list_events(
            conn,
            limit=args.limit,
            run_id=args.run_id,
            event_type=args.event_type,
        )
    finally:
        conn.close()

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    if not rows:
        print("(no events)")
        return 0

    for r in rows:
        ts = r["occurred_at"]
        et = r["event_type"]
        ent = f"{r['entity_type']}:{r['entity_id']}" if r["entity_id"] else r["entity_type"]
        print(f"{ts}  {et:<28}  {ent}  run={r['run_id']}")
    return 0


def manifest_diff_cmd(args: argparse.Namespace) -> int:
    from mgc.manifest import diff_manifest
    return int(diff_manifest(args.committed, args.generated))

def db_schema_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        with conn:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r["name"] for r in cur.fetchall()]
            for t in tables:
                print(t)
                info = conn.execute(f"PRAGMA table_info({t})").fetchall()
                for r in info:
                    name = r["name"]
                    ctype = r["type"] or ""
                    pk = " pk" if r["pk"] else ""
                    nn = " notnull" if r["notnull"] else ""
                    print(f"  - {name} {ctype}{pk}{nn}")

        ew.emit("system.command_completed", "system", None, {"command": cmd_name(args), "ok": True}, occurred_at=now_iso())
        return 0

    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def playlists_list_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        with conn:
            pls = db_list_playlists(conn, limit=args.limit)

        if args.json:
            print(json.dumps([pl.__dict__ for pl in pls], indent=2, ensure_ascii=False))
        else:
            if not pls:
                print("(no playlists)")
            else:
                for i, pl in enumerate(pls, 1):
                    print(f"{i:>2}. {pl.id}  {pl.slug}  {pl.created_at}  tracks={pl.track_count}  json_path={pl.json_path}")

        ew.emit(
            "system.command_completed",
            "system",
            None,
            {"command": cmd_name(args), "ok": True, "result_count": len(pls)},
            occurred_at=now_iso(),
        )
        return 0

    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def playlists_reveal_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        with conn:
            pl = db_get_playlist(conn, args.id)

            if args.build:
                obj = db_build_playlist_json(conn, pl, built_at=now_iso())
                print(json.dumps(obj, indent=2, ensure_ascii=False))
                ew.emit(
                    "playlist.built_from_db",
                    "playlist",
                    pl.id,
                    {"playlist_id": pl.id, "slug": pl.slug, "track_count": len(obj.get("tracks", []))},
                    occurred_at=now_iso(),
                )
                ew.emit("system.command_completed", "system", None, {"command": cmd_name(args), "ok": True}, occurred_at=now_iso())
                return 0

            jp = resolve_json_path(pl.json_path)
            if jp.exists():
                obj = read_json_file(jp)
                print(json.dumps(obj, indent=2, ensure_ascii=False))
                ew.emit(
                    "playlist.revealed",
                    "playlist",
                    pl.id,
                    {"playlist_id": pl.id, "slug": pl.slug, "source": "json_path", "path": str(jp)},
                    occurred_at=now_iso(),
                )
                ew.emit("system.command_completed", "system", None, {"command": cmd_name(args), "ok": True}, occurred_at=now_iso())
                return 0

            obj = db_build_playlist_json(conn, pl, built_at=now_iso())
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            ew.emit(
                "playlist.revealed",
                "playlist",
                pl.id,
                {"playlist_id": pl.id, "slug": pl.slug, "source": "db_build"},
                occurred_at=now_iso(),
            )
            ew.emit("system.command_completed", "system", None, {"command": cmd_name(args), "ok": True}, occurred_at=now_iso())
            return 0

    except NotFoundError as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        die(str(e), 2)
    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def playlists_export_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    export_dir = Path(args.out_dir) if args.out_dir else default_export_dir()

    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        out_paths: List[str] = []
        run_ids: List[str] = []

        with conn:
            if args.id:
                pl = db_get_playlist(conn, args.id)
                out_path = export_one_playlist(conn, pl, export_dir, build=bool(args.build))
                out_paths.append(str(out_path))

                ew.emit(
                    "playlist.exported",
                    "playlist",
                    pl.id,
                    {"playlist_id": pl.id, "slug": pl.slug, "export_path": str(out_path), "build": bool(args.build), "out_dir": str(export_dir)},
                    occurred_at=now_iso(),
                )

                rid = maybe_record_run(conn, ew, pl, out_path, record=not args.no_record, notes=args.notes)
                if rid:
                    run_ids.append(rid)

            else:
                pls = db_list_playlists(conn, limit=args.limit)
                if not pls:
                    die("No playlists found to export.")

                for pl in pls:
                    out_path = export_one_playlist(conn, pl, export_dir, build=bool(args.build))
                    out_paths.append(str(out_path))

                    ew.emit(
                        "playlist.exported",
                        "playlist",
                        pl.id,
                        {"playlist_id": pl.id, "slug": pl.slug, "export_path": str(out_path), "build": bool(args.build), "out_dir": str(export_dir)},
                        occurred_at=now_iso(),
                    )

                    rid = maybe_record_run(conn, ew, pl, out_path, record=not args.no_record, notes=args.notes)
                    if rid:
                        run_ids.append(rid)

        if args.json:
            payload: Dict[str, Any] = {"exported": out_paths}
            if args.id:
                payload["run_id"] = (run_ids[0] if run_ids else None)
            else:
                payload["run_ids"] = run_ids
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            for pth in out_paths:
                print(pth)
            if run_ids:
                print("Recorded runs:")
                for rid in run_ids:
                    print(f"  {rid}")

        ew.emit(
            "system.command_completed",
            "system",
            None,
            {"command": cmd_name(args), "ok": True, "exported_count": len(out_paths), "recorded_runs": len(run_ids)},
            occurred_at=now_iso(),
        )
        return 0

    except NotFoundError as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        die(str(e), 2)
    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def playlists_history_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        with conn:
            ensure_playlist_runs_table(conn)

            runs = db_list_playlist_runs(conn, args.playlist_id, limit=args.limit)
            if runs:
                if args.json:
                    print(
                        json.dumps(
                            [
                                {
                                    "kind": "run",
                                    "id": r.id,
                                    "created_at": r.created_at,
                                    "playlist_id": r.playlist_id,
                                    "seed": r.seed,
                                    "track_count": len(r.track_ids),
                                    "export_path": r.export_path,
                                    "notes": r.notes,
                                }
                                for r in runs
                            ],
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                else:
                    for i, r in enumerate(runs, 1):
                        ep = r.export_path or ""
                        notes = (r.notes or "").strip()
                        notes_s = (notes[:60] + "…") if len(notes) > 60 else notes
                        print(f"{i:>2}. RUN {r.id}  {r.created_at}  seed={r.seed}  tracks={len(r.track_ids)}  {ep}  {notes_s}")

                ew.emit(
                    "system.command_completed",
                    "system",
                    None,
                    {"command": cmd_name(args), "ok": True, "mode": "playlist_runs", "count": len(runs)},
                    occurred_at=now_iso(),
                )
                return 0

            pl = db_get_playlist(conn, args.playlist_id)
            peers = db_list_playlists_by_slug(conn, pl.slug, limit=args.limit)

        if args.json:
            with conn:
                print(
                    json.dumps(
                        [
                            {
                                "kind": "playlist",
                                "id": p.id,
                                "created_at": p.created_at,
                                "playlist_id": p.id,
                                "seed": p.seed,
                                "track_count": len(db_playlist_track_ids(conn, p.id)),
                                "slug": p.slug,
                            }
                            for p in peers
                        ],
                        indent=2,
                        ensure_ascii=False,
                    )
                )
        else:
            print("(no recorded runs in playlist_runs; showing implicit history from playlists table by slug)")
            with conn:
                for i, p in enumerate(peers, 1):
                    tc = len(db_playlist_track_ids(conn, p.id))
                    print(f"{i:>2}. PL  {p.id}  {p.created_at}  seed={p.seed}  tracks={tc}  slug={p.slug}")

        ew.emit(
            "system.command_completed",
            "system",
            None,
            {"command": cmd_name(args), "ok": True, "mode": "slug_history", "count": len(peers)},
            occurred_at=now_iso(),
        )
        return 0

    except NotFoundError as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        die(str(e), 2)
    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def playlists_diff_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        with conn:
            ensure_playlist_runs_table(conn)

            a: ResolvedRef = resolve_ref(conn, args.a)
            b: ResolvedRef = resolve_ref(conn, args.b)

            diff = compute_diff(a.track_ids, b.track_ids)

            a_dur = db_duration_for_tracks(conn, a.track_ids)
            b_dur = db_duration_for_tracks(conn, b.track_ids)

        if args.json:
            print(
                json.dumps(
                    {
                        "a": a.__dict__,
                        "b": b.__dict__,
                        "duration_sec": {"a": a_dur, "b": b_dur, "delta": (b_dur - a_dur)},
                        "diff": diff,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(f"A: {a.kind.upper()} {a.id}  created={a.created_at}  seed={a.seed}  tracks={len(a.track_ids)}")
            print(f"B: {b.kind.upper()} {b.id}  created={b.created_at}  seed={b.seed}  tracks={len(b.track_ids)}")
            print(f"Duration: A={a_dur:.1f}s  B={b_dur:.1f}s  Δ={(b_dur - a_dur):+.1f}s")
            print(f"Added:   {len(diff['added'])}")
            print(f"Removed: {len(diff['removed'])}")
            print(f"Moved:   {len(diff['moved'])}")

            if args.verbose:
                if diff["added"]:
                    print("\nADDED:")
                    for tid in diff["added"]:
                        print(f"  + {tid}")
                if diff["removed"]:
                    print("\nREMOVED:")
                    for tid in diff["removed"]:
                        print(f"  - {tid}")
                if diff["moved"]:
                    print("\nMOVED:")
                    for m in diff["moved"][:200]:
                        print(f"  ~ {m['track_id']}  {m['from']} -> {m['to']}")
                    if len(diff["moved"]) > 200:
                        print(f"  ... ({len(diff['moved']) - 200} more)")

        ew.emit("system.command_completed", "system", None, {"command": cmd_name(args), "ok": True}, occurred_at=now_iso())
        return 0

    except NotFoundError as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        die(str(e), 2)
    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def tracks_list_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        with conn:
            rows = db_list_tracks(
                conn,
                limit=args.limit,
                mood=args.mood,
                genre=args.genre,
                status=args.status,
                bpm_min=args.bpm_min,
                bpm_max=args.bpm_max,
                q=args.q,
            )

        if args.json:
            print(json.dumps([r.__dict__ for r in rows], indent=2, ensure_ascii=False))
        else:
            if not rows:
                print("(no tracks)")
            else:
                for i, t in enumerate(rows, 1):
                    print(f"{i:>2}. {t.id}  {t.title}  mood={t.mood} genre={t.genre} bpm={t.bpm} dur={t.duration_sec:.1f}s status={t.status}")

        ew.emit(
            "system.command_completed",
            "system",
            None,
            {"command": cmd_name(args), "ok": True, "result_count": len(rows)},
            occurred_at=now_iso(),
        )
        return 0

    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def tracks_show_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        with conn:
            t: TrackRow = db_get_track(conn, args.id)

        print(json.dumps(t.__dict__, indent=2, ensure_ascii=False))
        ew.emit("system.command_completed", "system", None, {"command": cmd_name(args), "ok": True}, occurred_at=now_iso())
        return 0

    except NotFoundError as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        die(str(e), 2)
    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def tracks_stats_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))
    ew.emit(
        "system.command_started",
        "system",
        None,
        {"command": cmd_name(args), "args": args_payload(args)},
        occurred_at=now_iso(),
    )

    try:
        with conn:
            stats = db_tracks_stats(conn)

        print(json.dumps(stats, indent=2, ensure_ascii=False))
        ew.emit("system.command_completed", "system", None, {"command": cmd_name(args), "ok": True}, occurred_at=now_iso())
        return 0

    except Exception as e:
        ew.emit(
            "system.command_failed",
            "system",
            None,
            {"command": cmd_name(args), "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=now_iso(),
        )
        raise
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mgc", description="mgc CLI")
    p.add_argument("--env", default=".env", help="Path to .env (default: .env)")
    p.add_argument("--log-level", default=os.environ.get("MGC_LOG_LEVEL", "INFO"))

    sub = p.add_subparsers(dest="cmd", required=True)

    # manifest
    mg = sub.add_parser("manifest", help="Manifest utilities")
    mgs = mg.add_subparsers(dest="manifest_cmd", required=True)

    md = mgs.add_parser("diff", help="Diff committed vs generated manifest")
    md.add_argument("--committed", default="data/manifest.json", help="Path to committed manifest")
    md.add_argument("--generated", default="data/manifest.generated.json", help="Path to generated manifest")
    md.set_defaults(func=manifest_diff_cmd)

    # db
    dbg = sub.add_parser("db", help="DB utilities")
    dbs = dbg.add_subparsers(dest="db_cmd", required=True)

    db_schema = dbs.add_parser("schema", help="Print tables + columns")
    db_schema.add_argument("--db", default="data/db.sqlite")
    db_schema.set_defaults(func=db_schema_cmd)

    # playlists
    pg = sub.add_parser("playlists", help="Playlist operations")
    pgs = pg.add_subparsers(dest="playlists_cmd", required=True)

    pl_list = pgs.add_parser("list", help="List playlists")
    pl_list.add_argument("--db", default="data/db.sqlite")
    pl_list.add_argument("--limit", type=int, default=20)
    pl_list.add_argument("--json", action="store_true")
    pl_list.set_defaults(func=playlists_list_cmd)

    pl_rev = pgs.add_parser("reveal", help="Reveal playlist JSON (json_path) or build from DB")
    pl_rev.add_argument("id")
    pl_rev.add_argument("--db", default="data/db.sqlite")
    pl_rev.add_argument("--build", action="store_true")
    pl_rev.set_defaults(func=playlists_reveal_cmd)

    pl_exp = pgs.add_parser("export", help="Export playlists to JSON files")
    pl_exp.add_argument("--db", default="data/db.sqlite")
    pl_exp.add_argument("--id", default=None)
    pl_exp.add_argument("--limit", type=int, default=20)
    pl_exp.add_argument("--out-dir", default=None)
    pl_exp.add_argument("--json", action="store_true")
    pl_exp.add_argument("--build", action="store_true")
    pl_exp.add_argument("--no-record", action="store_true")
    pl_exp.add_argument("--notes", default=None)
    pl_exp.set_defaults(func=playlists_export_cmd)

    pl_hist = pgs.add_parser("history", help="List recorded runs for a playlist (fallback: playlist rows by slug)")
    pl_hist.add_argument("playlist_id")
    pl_hist.add_argument("--db", default="data/db.sqlite")
    pl_hist.add_argument("--limit", type=int, default=20)
    pl_hist.add_argument("--json", action="store_true")
    pl_hist.set_defaults(func=playlists_history_cmd)

    pl_diff = pgs.add_parser("diff", help="Diff two refs: run IDs or playlist IDs")
    pl_diff.add_argument("a")
    pl_diff.add_argument("b")
    pl_diff.add_argument("--db", default="data/db.sqlite")
    pl_diff.add_argument("--json", action="store_true")
    pl_diff.add_argument("--verbose", action="store_true")
    pl_diff.set_defaults(func=playlists_diff_cmd)

    # tracks
    tg = sub.add_parser("tracks", help="Track library")
    tgs = tg.add_subparsers(dest="tracks_cmd", required=True)

    tl = tgs.add_parser("list", help="List tracks")
    tl.add_argument("--db", default="data/db.sqlite")
    tl.add_argument("--limit", type=int, default=20)
    tl.add_argument("--mood", default=None)
    tl.add_argument("--genre", default=None)
    tl.add_argument("--status", default=None)
    tl.add_argument("--bpm-min", type=int, default=None)
    tl.add_argument("--bpm-max", type=int, default=None)
    tl.add_argument("--q", default=None)
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(func=tracks_list_cmd)

    ts = tgs.add_parser("show", help="Show track details")
    ts.add_argument("id")
    ts.add_argument("--db", default="data/db.sqlite")
    ts.set_defaults(func=tracks_show_cmd)

    tt = tgs.add_parser("stats", help="Track library stats")
    tt.add_argument("--db", default="data/db.sqlite")
    tt.set_defaults(func=tracks_stats_cmd)

    # analytics
    register_analytics_subcommand(sub)

    # events
    eg = sub.add_parser("events", help="Event log inspection")
    egs = eg.add_subparsers(dest="events_cmd", required=True)

    el = egs.add_parser("list", help="List events")
    el.add_argument("--db", default="data/db.sqlite")
    el.add_argument("--limit", type=int, default=20)
    el.add_argument("--run-id", default=None)
    el.add_argument("--type", dest="event_type", default=None)
    el.add_argument("--json", action="store_true")
    el.set_defaults(func=events_list_cmd)

    # rebuild
    register_rebuild_subcommand(sub)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    load_dotenv(args.env)

    if setup_logging:
        try:
            setup_logging(level=args.log_level)
        except Exception:
            pass

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
