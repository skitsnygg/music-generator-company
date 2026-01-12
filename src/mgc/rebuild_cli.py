# src/mgc/rebuild_cli.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mgc.db_helpers import (
    PlaylistRow,
    db_build_playlist_json,
    db_list_playlists,
    sqlite_connect,
)
from mgc.events import EventContext, EventWriter, canonical_json, new_run_id

# Deterministic stamp for rebuild outputs/manifests unless overridden.
FIXED_STAMP = "1970-01-01T00:00:00Z"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_slug(s: str) -> str:
    s = (s or "").strip()
    return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_")).strip("-_")


def _payload_bytes(obj: Dict[str, Any]) -> bytes:
    # Hash/bytes must match exactly what we write: canonical JSON + newline.
    return (canonical_json(obj) + "\n").encode("utf-8")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_payload_bytes(obj))


def _stable_time_fields(obj: Dict[str, Any], *, stamp: str) -> Dict[str, Any]:
    out = dict(obj)
    if "built_from_db_at" in out:
        out["built_from_db_at"] = stamp
    if "exported_at" in out:
        out["exported_at"] = stamp
    return out


def _repo_root() -> Path:
    # In this repo we run CLI from repo root in CI/local; keep it simple + deterministic.
    return Path.cwd().resolve()


def _relpath_from_repo(p: Path) -> str:
    rr = _repo_root()
    pr = p.resolve()
    return str(pr.relative_to(rr))


def _path_within_repo(p: Path) -> bool:
    rr = _repo_root()
    pr = p.resolve()
    return pr == rr or rr in pr.parents


def _normalize_manifest_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _validate_manifest_item_path(path_str: str, *, expected_out_dir: Path) -> None:
    """
    Strict path validation:
      - must be relative
      - must not escape repo root
      - must be under expected_out_dir (relative)
    """
    p = Path(path_str)
    if p.is_absolute():
        raise ValueError("manifest_path_absolute")

    abs_p = _normalize_manifest_path(path_str)
    if not _path_within_repo(abs_p):
        raise ValueError("manifest_path_escapes_repo")

    rr = _repo_root()
    expected_out_dir_abs = expected_out_dir.resolve()
    # expected_out_dir is passed as repo-relative in our CLI; ensure we compare absolute.
    if not expected_out_dir_abs.is_absolute():
        expected_out_dir_abs = rr / expected_out_dir_abs

    if expected_out_dir_abs not in abs_p.resolve().parents and abs_p.resolve() != expected_out_dir_abs:
        raise ValueError("manifest_path_outside_out_dir")


# ----------------------------
# CLEAN (safe delete)
# ----------------------------

def _is_within_repo(path: Path, *, repo_root: Path) -> bool:
    try:
        rr = repo_root.resolve()
        p = path.resolve()
        return p == rr or rr in p.parents
    except Exception:
        return False


def _rm_tree_or_file(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def rebuild_clean_cmd(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    scope = str(args.clean_scope)
    dry = bool(args.dry_run)
    yes = bool(args.yes)

    targets: List[Path] = []

    def add_scope(out_dir: str, manifest_name: str) -> None:
        d = Path(out_dir)
        targets.append(d)
        targets.append(d / manifest_name)

    if scope == "playlists":
        add_scope(args.playlists_out_dir, "_manifest.playlists.json")
    elif scope == "tracks":
        add_scope(args.tracks_out_dir, "_manifest.tracks.json")
    elif scope == "all":
        add_scope(args.playlists_out_dir, "_manifest.playlists.json")
        add_scope(args.tracks_out_dir, "_manifest.tracks.json")
    else:
        raise SystemExit(2)

    seen: set[str] = set()
    uniq: List[Path] = []
    for t in targets:
        ts = str(t)
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append(t)

    print(f"[rebuild.clean] scope: {scope}")
    for t in uniq:
        print(f"[rebuild.clean] target: {t}")

    for t in uniq:
        if not _is_within_repo(t, repo_root=repo_root):
            print(f"[rebuild.clean] ERROR: refusing to delete outside repo: {t}", file=os.sys.stderr)
            raise SystemExit(2)

    if dry:
        print("[rebuild.clean] dry_run: true")
        return 0

    if not yes:
        print("[rebuild.clean] Refusing to delete without --yes (or use --dry-run).")
        raise SystemExit(2)

    removed = 0
    for t in uniq:
        if t.exists():
            _rm_tree_or_file(t)
            removed += 1

    print(f"[rebuild.clean] removed: {removed}")
    return 0


# ----------------------------
# LS (manifest status)
# ----------------------------

def rebuild_ls_cmd(args: argparse.Namespace) -> int:
    scope = str(getattr(args, "ls_scope", "all") or "all")
    as_json = bool(args.json)

    def one(scope_name: str, out_dir: str, manifest_name: str) -> Dict[str, Any]:
        d = Path(out_dir)
        m = d / manifest_name
        info: Dict[str, Any] = {
            "scope": scope_name,
            "out_dir": str(d),
            "manifest": str(m),
            "manifest_exists": m.exists(),
        }
        if not m.exists():
            return info

        try:
            man = _load_manifest(m)
            info["manifest_scope"] = man.get("scope")
            info["count"] = int(man.get("count") or 0)
            info["fingerprint_sha256"] = str(man.get("fingerprint_sha256") or "")
            info["fingerprint_recomputed_sha256"] = _manifest_fingerprint(man)
        except Exception as e:
            info["error"] = str(e)
        return info

    results: List[Dict[str, Any]] = []
    if scope in ("playlists", "all"):
        results.append(one("playlists", args.playlists_out_dir, "_manifest.playlists.json"))
    if scope in ("tracks", "all"):
        results.append(one("tracks", args.tracks_out_dir, "_manifest.tracks.json"))

    if as_json:
        print(json.dumps({"ok": True, "results": results}, indent=2, ensure_ascii=False))
        return 0

    for r in results:
        print(f"scope: {r['scope']}")
        print(f"  out_dir: {r['out_dir']}")
        print(f"  manifest: {r['manifest']}")
        print(f"  manifest_exists: {r['manifest_exists']}")
        if r.get("error"):
            print(f"  error: {r['error']}")
        elif r["manifest_exists"]:
            print(f"  manifest_scope: {r.get('manifest_scope')}")
            print(f"  count: {r.get('count')}")
            print(f"  fingerprint_sha256: {r.get('fingerprint_sha256')}")
            print(f"  fingerprint_recomputed_sha256: {r.get('fingerprint_recomputed_sha256')}")
        print()

    return 0


# ----------------------------
# PLAYLISTS
# ----------------------------

def _playlist_out_path(export_dir: Path, pl: PlaylistRow) -> Path:
    safe = _safe_slug(pl.slug)
    name = f"{safe}_{pl.id}.json" if safe else f"{pl.id}.json"
    return export_dir / name


def _sum_duration_sec(obj: Dict[str, Any]) -> int:
    total = 0.0
    for t in (obj.get("tracks") or []):
        try:
            total += float(t.get("duration_sec") or 0.0)
        except Exception:
            pass
    return int(round(total))


def _quality_gate_playlist(pl: PlaylistRow, obj: Dict[str, Any]) -> None:
    tracks = obj.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("tracks_not_list")

    json_tc = len(tracks)
    db_tc = int(pl.track_count)
    if json_tc != db_tc:
        raise ValueError(f"track_count_mismatch db={db_tc} json={json_tc}")

    json_dur = _sum_duration_sec(obj)
    db_dur = int(pl.total_duration_sec)
    if json_dur != db_dur:
        raise ValueError(f"duration_mismatch db={db_dur} json={json_dur}")


def _build_playlist_obj(conn, pl: PlaylistRow, *, stamp: str) -> Dict[str, Any]:
    obj = db_build_playlist_json(conn, pl, built_at=stamp)
    obj["exported_at"] = stamp
    return _stable_time_fields(obj, stamp=stamp)


@dataclass(frozen=True)
class ManifestItem:
    scope_id: str
    slug: str
    path: str
    sha256: str
    bytes: int
    item_count: int  # track_count for playlists; 1 for tracks


def _manifest_path_str(out_path: Path, *, relative_paths: bool) -> str:
    if not relative_paths:
        return str(out_path)
    return _relpath_from_repo(out_path)


def _build_manifest_playlists(
    conn,
    *,
    export_dir: Path,
    limit: int,
    stamp: str,
    relative_paths: bool,
) -> Tuple[Dict[str, Any], List[Tuple[Path, Dict[str, Any]]]]:
    playlists = db_list_playlists(conn, limit=limit)
    playlists = sorted(playlists, key=lambda p: (p.slug, p.id))

    items: List[ManifestItem] = []
    to_write: List[Tuple[Path, Dict[str, Any]]] = []

    for pl in playlists:
        obj = _build_playlist_obj(conn, pl, stamp=stamp)
        _quality_gate_playlist(pl, obj)

        out_path = _playlist_out_path(export_dir, pl)

        data = _payload_bytes(obj)
        sha = _sha256_bytes(data)

        items.append(
            ManifestItem(
                scope_id=str(pl.id),
                slug=str(pl.slug or ""),
                path=_manifest_path_str(out_path, relative_paths=relative_paths),
                sha256=sha,
                bytes=len(data),
                item_count=len(obj.get("tracks") or []),
            )
        )
        to_write.append((out_path, obj))

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "scope": "playlists",
        "stamp": stamp,
        "count": len(items),
        "relative_paths": bool(relative_paths),
        "items": [
            {
                "playlist_id": i.scope_id,
                "slug": i.slug,
                "path": i.path,
                "sha256": i.sha256,
                "bytes": i.bytes,
                "track_count": i.item_count,
            }
            for i in items
        ],
    }
    manifest["fingerprint_sha256"] = _sha256_text(canonical_json(manifest))
    return manifest, to_write


# ----------------------------
# TRACKS (schema-agnostic)
# ----------------------------

def _table_columns(conn, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols: List[str] = []
    for r in rows:
        try:
            cols.append(str(r[1]))
        except Exception:
            pass
    return cols


def _row_to_jsonable(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (bytes, bytearray, memoryview)):
            out[k] = bytes(v).hex()
        else:
            out[k] = str(v)
    return out


def _list_tracks(conn, *, limit: int) -> List[Dict[str, Any]]:
    cols = _table_columns(conn, "tracks")
    if not cols:
        raise ValueError("tracks_table_has_no_columns")

    sql = f"SELECT {', '.join(cols)} FROM tracks LIMIT ?"
    cur = conn.execute(sql, (int(limit),))
    names = [d[0] for d in cur.description]  # type: ignore[union-attr]
    rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = {names[i]: r[i] for i in range(len(names))}
        out.append(_row_to_jsonable(d))
    return out


def _track_identity(row: Dict[str, Any]) -> Tuple[str, str]:
    for k in ("id", "track_id", "uuid"):
        if row.get(k) is not None:
            return str(row.get(k)), ""
    raw = canonical_json(row)
    return _sha256_text(raw)[:32], ""


def _track_out_path(export_dir: Path, track_id: str, slugish: str) -> Path:
    safe = _safe_slug(slugish)
    name = f"{safe}_{track_id}.json" if safe else f"{track_id}.json"
    return export_dir / name


def _quality_gate_track(row: Dict[str, Any]) -> None:
    track_id, _ = _track_identity(row)
    if not track_id:
        raise ValueError("track_missing_id")

    title = str(row.get("title") or row.get("name") or "").strip()
    if not title:
        raise ValueError("track_missing_title")

    if "duration_sec" not in row:
        raise ValueError("track_missing_duration_field")
    if row["duration_sec"] is None:
        raise ValueError("track_missing_duration")

    try:
        dur = float(row["duration_sec"])
    except Exception:
        raise ValueError("track_duration_not_numeric")
    if dur < 0:
        raise ValueError("track_duration_negative")


def _build_track_obj(row: Dict[str, Any], *, stamp: str) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "schema_version": 1,
        "scope": "track",
        "built_from_db_at": stamp,
        "exported_at": stamp,
        "track": dict(row),
    }
    return _stable_time_fields(obj, stamp=stamp)


def _build_manifest_tracks(
    conn,
    *,
    export_dir: Path,
    limit: int,
    stamp: str,
    relative_paths: bool,
) -> Tuple[Dict[str, Any], List[Tuple[Path, Dict[str, Any]]]]:
    rows = _list_tracks(conn, limit=limit)

    def sort_key(r: Dict[str, Any]) -> Tuple[str, str]:
        tid, _ = _track_identity(r)
        title = str(r.get("title") or r.get("name") or "")
        return (title, tid)

    rows = sorted(rows, key=sort_key)

    items: List[ManifestItem] = []
    to_write: List[Tuple[Path, Dict[str, Any]]] = []

    for r in rows:
        _quality_gate_track(r)

        track_id, slugish = _track_identity(r)
        out_path = _track_out_path(export_dir, track_id, slugish)
        obj = _build_track_obj(r, stamp=stamp)

        data = _payload_bytes(obj)
        sha = _sha256_bytes(data)

        items.append(
            ManifestItem(
                scope_id=track_id,
                slug=slugish,
                path=_manifest_path_str(out_path, relative_paths=relative_paths),
                sha256=sha,
                bytes=len(data),
                item_count=1,
            )
        )
        to_write.append((out_path, obj))

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "scope": "tracks",
        "stamp": stamp,
        "count": len(items),
        "relative_paths": bool(relative_paths),
        "items": [{"track_id": i.scope_id, "path": i.path, "sha256": i.sha256, "bytes": i.bytes} for i in items],
    }
    manifest["fingerprint_sha256"] = _sha256_text(canonical_json(manifest))
    return manifest, to_write


# ----------------------------
# Manifest helpers (shared)
# ----------------------------

def _load_manifest(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("manifest_not_object")
    return obj


def _manifest_fingerprint(manifest: Dict[str, Any]) -> str:
    m2 = dict(manifest)
    m2.pop("fingerprint_sha256", None)
    return _sha256_text(canonical_json(m2))


def _require_manifest_scope(m: Dict[str, Any], scope: str) -> None:
    if m.get("scope") != scope:
        raise ValueError(f"manifest_wrong_scope scope={m.get('scope')!r}")
    if not isinstance(m.get("items"), list):
        raise ValueError("manifest_items_not_list")


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _short(s: str, n: int = 12) -> str:
    s = s or ""
    return s[:n]


def _items_map(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    items = list(manifest.get("items") or [])
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = str(it.get("playlist_id") or it.get("track_id") or "")
        if not pid:
            continue
        out[pid] = it
    return out


def _diff_manifests(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a_map = _items_map(a)
    b_map = _items_map(b)

    a_ids = set(a_map.keys())
    b_ids = set(b_map.keys())

    added = sorted(list(b_ids - a_ids))
    removed = sorted(list(a_ids - b_ids))
    common = sorted(list(a_ids & b_ids))

    changed: List[Dict[str, Any]] = []
    same: List[str] = []

    for pid in common:
        ai = a_map[pid]
        bi = b_map[pid]

        a_sha = str(ai.get("sha256") or "")
        b_sha = str(bi.get("sha256") or "")
        a_bytes = _as_int(ai.get("bytes"))
        b_bytes = _as_int(bi.get("bytes"))
        a_path = str(ai.get("path") or "")
        b_path = str(bi.get("path") or "")

        if (a_sha == b_sha) and (a_bytes == b_bytes) and (a_path == b_path):
            same.append(pid)
            continue

        changed.append({"id": pid, "sha256_a": a_sha, "sha256_b": b_sha, "bytes_a": a_bytes, "bytes_b": b_bytes, "path_a": a_path, "path_b": b_path})

    return {
        "fingerprint": {"a": str(a.get("fingerprint_sha256") or ""), "b": str(b.get("fingerprint_sha256") or ""), "a_recomputed": _manifest_fingerprint(a), "b_recomputed": _manifest_fingerprint(b)},
        "counts": {"a": len(a_map), "b": len(b_map), "added": len(added), "removed": len(removed), "changed": len(changed), "same": len(same)},
        "added": added,
        "removed": removed,
        "changed": changed,
        "same": same,
    }


# ----------------------------
# Commands: playlists
# ----------------------------

def rebuild_playlists_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    export_dir = Path(args.out_dir)
    stamp = args.stamp or FIXED_STAMP
    limit = int(args.limit)
    relative_paths = bool(args.relative_paths)

    try:
        with conn:
            ew.emit(
                "rebuild.started",
                "system",
                None,
                {"scope": "playlists", "db": args.db, "out_dir": str(export_dir), "limit": limit, "stamp": stamp, "determinism_check": bool(args.determinism_check), "write": bool(args.write), "relative_paths": relative_paths},
                occurred_at=now_iso(),
            )

            try:
                manifest1, to_write1 = _build_manifest_playlists(conn, export_dir=export_dir, limit=limit, stamp=stamp, relative_paths=relative_paths)
            except ValueError as e:
                ew.emit("rebuild.quality_gate_failed", "system", None, {"scope": "playlists", "gate": "db_vs_export", "reason": str(e)}, occurred_at=now_iso())
                raise SystemExit(2)

            if args.determinism_check:
                try:
                    manifest2, _ = _build_manifest_playlists(conn, export_dir=export_dir, limit=limit, stamp=stamp, relative_paths=relative_paths)
                except ValueError as e:
                    ew.emit("rebuild.quality_gate_failed", "system", None, {"scope": "playlists", "gate": "db_vs_export", "reason": str(e)}, occurred_at=now_iso())
                    raise SystemExit(2)

                if manifest1["fingerprint_sha256"] != manifest2["fingerprint_sha256"]:
                    ew.emit("rebuild.quality_gate_failed", "system", None, {"scope": "playlists", "gate": "determinism", "fingerprint_1": manifest1["fingerprint_sha256"], "fingerprint_2": manifest2["fingerprint_sha256"]}, occurred_at=now_iso())
                    raise SystemExit(2)

            if args.write:
                export_dir.mkdir(parents=True, exist_ok=True)
                for path, obj in to_write1:
                    _write_json(path, obj)
                _write_json(export_dir / "_manifest.playlists.json", manifest1)

            ew.emit("rebuild.completed", "system", None, {"scope": "playlists", "ok": True, "fingerprint_sha256": manifest1["fingerprint_sha256"], "count": manifest1["count"], "wrote": bool(args.write)}, occurred_at=now_iso())

        if args.json:
            print(json.dumps(manifest1, indent=2, ensure_ascii=False))
        else:
            print(f"fingerprint_sha256: {manifest1['fingerprint_sha256']}")
            print(f"count: {manifest1['count']}")
            if args.write:
                print(f"wrote: {export_dir}/_manifest.playlists.json")

        return 0

    except SystemExit:
        raise
    finally:
        conn.close()


def rebuild_tracks_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    export_dir = Path(args.out_dir)
    stamp = args.stamp or FIXED_STAMP
    limit = int(args.limit)
    relative_paths = bool(args.relative_paths)

    try:
        with conn:
            ew.emit(
                "rebuild.started",
                "system",
                None,
                {"scope": "tracks", "db": args.db, "out_dir": str(export_dir), "limit": limit, "stamp": stamp, "determinism_check": bool(args.determinism_check), "write": bool(args.write), "relative_paths": relative_paths},
                occurred_at=now_iso(),
            )

            try:
                manifest1, to_write1 = _build_manifest_tracks(conn, export_dir=export_dir, limit=limit, stamp=stamp, relative_paths=relative_paths)
            except (ValueError, sqlite3.Error) as e:
                ew.emit("rebuild.quality_gate_failed", "system", None, {"scope": "tracks", "gate": "row_sanity", "reason": str(e)}, occurred_at=now_iso())
                raise SystemExit(2)

            if args.determinism_check:
                try:
                    manifest2, _ = _build_manifest_tracks(conn, export_dir=export_dir, limit=limit, stamp=stamp, relative_paths=relative_paths)
                except (ValueError, sqlite3.Error) as e:
                    ew.emit("rebuild.quality_gate_failed", "system", None, {"scope": "tracks", "gate": "row_sanity", "reason": str(e)}, occurred_at=now_iso())
                    raise SystemExit(2)

                if manifest1["fingerprint_sha256"] != manifest2["fingerprint_sha256"]:
                    ew.emit("rebuild.quality_gate_failed", "system", None, {"scope": "tracks", "gate": "determinism", "fingerprint_1": manifest1["fingerprint_sha256"], "fingerprint_2": manifest2["fingerprint_sha256"]}, occurred_at=now_iso())
                    raise SystemExit(2)

            if args.write:
                export_dir.mkdir(parents=True, exist_ok=True)
                for path, obj in to_write1:
                    _write_json(path, obj)
                _write_json(export_dir / "_manifest.tracks.json", manifest1)

            ew.emit("rebuild.completed", "system", None, {"scope": "tracks", "ok": True, "fingerprint_sha256": manifest1["fingerprint_sha256"], "count": manifest1["count"], "wrote": bool(args.write)}, occurred_at=now_iso())

        if args.json:
            print(json.dumps(manifest1, indent=2, ensure_ascii=False))
        else:
            print(f"fingerprint_sha256: {manifest1['fingerprint_sha256']}")
            print(f"count: {manifest1['count']}")
            if args.write:
                print(f"wrote: {export_dir}/_manifest.tracks.json")

        return 0

    except SystemExit:
        raise
    finally:
        conn.close()


def rebuild_verify_playlists_cmd(args: argparse.Namespace) -> int:
    return _rebuild_verify_generic(args, scope="playlists", default_manifest_name="_manifest.playlists.json")


def rebuild_verify_tracks_cmd(args: argparse.Namespace) -> int:
    return _rebuild_verify_generic(args, scope="tracks", default_manifest_name="_manifest.tracks.json")


def rebuild_diff_playlists_cmd(args: argparse.Namespace) -> int:
    return _rebuild_diff_generic(args, scope="playlists")


def rebuild_diff_tracks_cmd(args: argparse.Namespace) -> int:
    return _rebuild_diff_generic(args, scope="tracks")


def _rebuild_verify_generic(args: argparse.Namespace, *, scope: str, default_manifest_name: str) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest) if getattr(args, "manifest", None) else (out_dir / default_manifest_name)
    strict_paths = bool(getattr(args, "strict_paths", False))

    try:
        with conn:
            ew.emit("rebuild.verify_started", "system", None, {"scope": scope, "db": args.db, "out_dir": str(out_dir), "manifest": str(manifest_path), "strict_paths": strict_paths}, occurred_at=now_iso())

            if not manifest_path.exists():
                ew.emit("rebuild.verify_failed", "system", None, {"scope": scope, "reason": "manifest_missing", "manifest": str(manifest_path)}, occurred_at=now_iso())
                raise SystemExit(2)

            try:
                manifest = _load_manifest(manifest_path)
                _require_manifest_scope(manifest, scope)
            except Exception as e:
                ew.emit("rebuild.verify_failed", "system", None, {"scope": scope, "reason": "manifest_invalid", "error": str(e)}, occurred_at=now_iso())
                raise SystemExit(2)

            expected_fp = str(manifest.get("fingerprint_sha256") or "")
            recomputed_fp = _manifest_fingerprint(manifest)
            if expected_fp and expected_fp != recomputed_fp:
                ew.emit("rebuild.verify_failed", "system", None, {"scope": scope, "reason": "fingerprint_mismatch", "expected": expected_fp, "recomputed": recomputed_fp}, occurred_at=now_iso())
                raise SystemExit(2)

            items = list(manifest.get("items") or [])
            failures: List[Dict[str, Any]] = []
            checked = 0

            for it in items:
                if not isinstance(it, dict):
                    failures.append({"reason": "item_not_object"})
                    continue

                rel_or_abs = str(it.get("path") or "")
                expected_sha = str(it.get("sha256") or "")
                expected_bytes = it.get("bytes")

                if strict_paths:
                    try:
                        _validate_manifest_item_path(rel_or_abs, expected_out_dir=out_dir)
                    except Exception as e:
                        failures.append({"path": rel_or_abs, "reason": "path_invalid", "error": str(e)})
                        continue

                p = _normalize_manifest_path(rel_or_abs)
                if not p.exists():
                    failures.append({"path": rel_or_abs, "reason": "file_missing"})
                    continue

                data = p.read_bytes()
                got_bytes = len(data)
                got_sha = _sha256_bytes(data)

                ok_bytes = (expected_bytes is None) or (int(expected_bytes) == got_bytes)
                ok_sha = (not expected_sha) or (expected_sha == got_sha)

                checked += 1
                if not ok_bytes or not ok_sha:
                    failures.append({"path": rel_or_abs, "reason": "file_mismatch", "expected_bytes": int(expected_bytes) if expected_bytes is not None else None, "got_bytes": got_bytes, "expected_sha256": expected_sha, "got_sha256": got_sha})

            if failures:
                sample = failures[: min(20, len(failures))]
                ew.emit("rebuild.verify_failed", "system", None, {"scope": scope, "reason": "verification_failed", "checked": checked, "failures": len(failures), "sample": sample}, occurred_at=now_iso())
                raise SystemExit(2)

            ew.emit("rebuild.verify_completed", "system", None, {"scope": scope, "ok": True, "checked": checked, "fingerprint_sha256": expected_fp or recomputed_fp}, occurred_at=now_iso())

        if args.json:
            print(json.dumps({"ok": True, "checked": checked, "fingerprint_sha256": expected_fp or recomputed_fp, "manifest": str(manifest_path)}, indent=2, ensure_ascii=False))
        else:
            print("ok: true")
            print(f"checked: {checked}")
            print(f"fingerprint_sha256: {expected_fp or recomputed_fp}")
            print(f"manifest: {manifest_path}")

        return 0

    except SystemExit:
        raise
    finally:
        conn.close()


def _rebuild_diff_generic(args: argparse.Namespace, *, scope: str) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    a_path = Path(args.a)
    b_path = Path(args.b)

    if not a_path.is_absolute():
        a_path = Path.cwd() / a_path
    if not b_path.is_absolute():
        b_path = Path.cwd() / b_path

    show = int(args.show)

    try:
        with conn:
            ew.emit("rebuild.diff_started", "system", None, {"scope": scope, "a": str(a_path), "b": str(b_path)}, occurred_at=now_iso())

            if not a_path.exists() or not b_path.exists():
                ew.emit("rebuild.diff_failed", "system", None, {"scope": scope, "reason": "manifest_missing", "a_exists": a_path.exists(), "b_exists": b_path.exists(), "a": str(a_path), "b": str(b_path)}, occurred_at=now_iso())
                raise SystemExit(2)

            try:
                a = _load_manifest(a_path)
                b = _load_manifest(b_path)
                _require_manifest_scope(a, scope)
                _require_manifest_scope(b, scope)
            except Exception as e:
                ew.emit("rebuild.diff_failed", "system", None, {"scope": scope, "reason": "manifest_invalid", "error": str(e)}, occurred_at=now_iso())
                raise SystemExit(2)

            diff = _diff_manifests(a, b)

            ew.emit("rebuild.diff_completed", "system", None, {"scope": scope, "a": str(a_path), "b": str(b_path), "counts": diff["counts"], "fingerprint_a": diff["fingerprint"]["a"], "fingerprint_b": diff["fingerprint"]["b"]}, occurred_at=now_iso())

        if args.json:
            print(json.dumps(diff, indent=2, ensure_ascii=False))
            return 0

        fp = diff["fingerprint"]
        c = diff["counts"]

        print(f"A: {a_path}")
        print(f"B: {b_path}")
        print(f"fingerprint A: {fp['a']} (recomputed {fp['a_recomputed']})")
        print(f"fingerprint B: {fp['b']} (recomputed {fp['b_recomputed']})")
        print()
        print(f"counts: a={c['a']} b={c['b']}  added={c['added']} removed={c['removed']} changed={c['changed']} same={c['same']}")

        if diff["added"]:
            print("\nADDED (in B):")
            for pid in diff["added"][:show]:
                print(f"  + {pid}")
            if len(diff["added"]) > show:
                print(f"  ... ({len(diff['added']) - show} more)")

        if diff["removed"]:
            print("\nREMOVED (from A):")
            for pid in diff["removed"][:show]:
                print(f"  - {pid}")
            if len(diff["removed"]) > show:
                print(f"  ... ({len(diff['removed']) - show} more)")

        if diff["changed"]:
            print("\nCHANGED:")
            for ch in diff["changed"][:show]:
                pid = ch["id"]
                print(f"  ~ {pid}  sha {_short(ch['sha256_a'])} -> {_short(ch['sha256_b'])}  bytes {ch['bytes_a']} -> {ch['bytes_b']}")
            if len(diff["changed"]) > show:
                print(f"  ... ({len(diff['changed']) - show} more)")

        return 0

    except SystemExit:
        raise
    finally:
        conn.close()


# ----------------------------
# CLI wiring
# ----------------------------

def register_rebuild_subcommand(subparsers) -> None:
    rg = subparsers.add_parser("rebuild", help="Deterministic rebuilds + quality gates")
    rgs = rg.add_subparsers(dest="rebuild_cmd", required=True)

    # rebuild playlists
    rp = rgs.add_parser("playlists", help="Rebuild playlist exports deterministically (from DB)")
    rp.add_argument("--db", default="data/db.sqlite")
    rp.add_argument("--out-dir", default="data/playlists")
    rp.add_argument("--limit", type=int, default=100000)
    rp.add_argument("--json", action="store_true")
    rp.add_argument("--stamp", default=None, help=f"Override deterministic stamp (default: {FIXED_STAMP})")
    rp.add_argument("--determinism-check", action="store_true")
    rp.add_argument("--write", action="store_true")
    rp.add_argument("--relative-paths", action="store_true", help="Write manifest paths relative to repo root")
    rp.add_argument("--build-from-db", action="store_true", help=argparse.SUPPRESS)
    rp.set_defaults(func=rebuild_playlists_cmd)

    # rebuild tracks
    rt = rgs.add_parser("tracks", help="Rebuild track exports deterministically (from DB)")
    rt.add_argument("--db", default="data/db.sqlite")
    rt.add_argument("--out-dir", default="data/tracks")
    rt.add_argument("--limit", type=int, default=100000)
    rt.add_argument("--json", action="store_true")
    rt.add_argument("--stamp", default=None, help=f"Override deterministic stamp (default: {FIXED_STAMP})")
    rt.add_argument("--determinism-check", action="store_true")
    rt.add_argument("--write", action="store_true")
    rt.add_argument("--relative-paths", action="store_true", help="Write manifest paths relative to repo root")
    rt.set_defaults(func=rebuild_tracks_cmd)

    # rebuild verify
    rv = rgs.add_parser("verify", help="Verify rebuild artifacts against manifest")
    rvs = rv.add_subparsers(dest="rebuild_verify_cmd", required=True)

    rvp = rvs.add_parser("playlists", help="Verify playlist JSON files against _manifest.playlists.json")
    rvp.add_argument("--db", default="data/db.sqlite")
    rvp.add_argument("--out-dir", default="data/playlists")
    rvp.add_argument("--manifest", default=None, help="Path to manifest (default: <out-dir>/_manifest.playlists.json)")
    rvp.add_argument("--json", action="store_true")
    rvp.add_argument("--strict-paths", action="store_true", help="Fail if any manifest path is absolute or escapes repo/out-dir")
    rvp.set_defaults(func=rebuild_verify_playlists_cmd)

    rvt = rvs.add_parser("tracks", help="Verify track JSON files against _manifest.tracks.json")
    rvt.add_argument("--db", default="data/db.sqlite")
    rvt.add_argument("--out-dir", default="data/tracks")
    rvt.add_argument("--manifest", default=None, help="Path to manifest (default: <out-dir>/_manifest.tracks.json)")
    rvt.add_argument("--json", action="store_true")
    rvt.add_argument("--strict-paths", action="store_true", help="Fail if any manifest path is absolute or escapes repo/out-dir")
    rvt.set_defaults(func=rebuild_verify_tracks_cmd)

    # rebuild diff
    rd = rgs.add_parser("diff", help="Diff two manifests")
    rds = rd.add_subparsers(dest="rebuild_diff_cmd", required=True)

    rdp = rds.add_parser("playlists", help="Diff two playlist manifests")
    rdp.add_argument("a", help="Path to manifest A (old)")
    rdp.add_argument("b", help="Path to manifest B (new)")
    rdp.add_argument("--db", default="data/db.sqlite")
    rdp.add_argument("--json", action="store_true")
    rdp.add_argument("--show", type=int, default=50, help="Max items to print per section (default: 50)")
    rdp.set_defaults(func=rebuild_diff_playlists_cmd)

    rdt = rds.add_parser("tracks", help="Diff two track manifests")
    rdt.add_argument("a", help="Path to manifest A (old)")
    rdt.add_argument("b", help="Path to manifest B (new)")
    rdt.add_argument("--db", default="data/db.sqlite")
    rdt.add_argument("--json", action="store_true")
    rdt.add_argument("--show", type=int, default=50, help="Max items to print per section (default: 50)")
    rdt.set_defaults(func=rebuild_diff_tracks_cmd)

    # clean
    rc = rgs.add_parser("clean", help="Delete rebuild output artifacts (safe)")
    rcs = rc.add_subparsers(dest="clean_scope", required=True)

    rc.add_argument("--dry-run", action="store_true", help="Print what would be deleted")
    rc.add_argument("--yes", action="store_true", help="Actually delete files")
    rc.add_argument("--playlists-out-dir", default="data/playlists", help="Playlists output dir (default: data/playlists)")
    rc.add_argument("--tracks-out-dir", default="data/tracks", help="Tracks output dir (default: data/tracks)")

    rcp = rcs.add_parser("playlists", help="Delete playlists rebuild artifacts")
    rcp.set_defaults(func=rebuild_clean_cmd)

    rct = rcs.add_parser("tracks", help="Delete tracks rebuild artifacts")
    rct.set_defaults(func=rebuild_clean_cmd)

    rca = rcs.add_parser("all", help="Delete playlists + tracks rebuild artifacts")
    rca.set_defaults(func=rebuild_clean_cmd)

    # ls
    rls = rgs.add_parser("ls", help="Show rebuild output status + fingerprints")
    rls.add_argument("--json", action="store_true")
    rls.add_argument("--playlists-out-dir", default="data/playlists")
    rls.add_argument("--tracks-out-dir", default="data/tracks")
    rlss = rls.add_subparsers(dest="ls_scope", required=False)

    rlsp = rlss.add_parser("playlists", help="Show playlists manifest info")
    rlsp.set_defaults(func=rebuild_ls_cmd)

    rlst = rlss.add_parser("tracks", help="Show tracks manifest info")
    rlst.set_defaults(func=rebuild_ls_cmd)

    rlsa = rlss.add_parser("all", help="Show both manifests info")
    rlsa.set_defaults(func=rebuild_ls_cmd)

    rls.set_defaults(func=rebuild_ls_cmd, ls_scope="all")
