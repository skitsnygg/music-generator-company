# src/mgc/rebuild_cli.py
from __future__ import annotations

import argparse
import hashlib
import json
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


def _playlist_out_path(export_dir: Path, pl: PlaylistRow) -> Path:
    safe = _safe_slug(pl.slug)
    name = f"{safe}_{pl.id}.json" if safe else f"{pl.id}.json"
    return export_dir / name


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def _stable_playlist_obj(obj: Dict[str, Any], *, stamp: str) -> Dict[str, Any]:
    # Normalize time-ish fields so rebuilds are deterministic.
    out = dict(obj)
    if "built_from_db_at" in out:
        out["built_from_db_at"] = stamp
    if "exported_at" in out:
        out["exported_at"] = stamp
    return out


def _sum_duration_sec(obj: Dict[str, Any]) -> int:
    total = 0.0
    for t in (obj.get("tracks") or []):
        try:
            total += float(t.get("duration_sec") or 0.0)
        except Exception:
            pass
    return int(round(total))


def _quality_gate_playlist(pl: PlaylistRow, obj: Dict[str, Any]) -> None:
    """
    Gate #1: DB vs rebuilt JSON consistency.
      - tracks must be a list
      - len(tracks) must match playlists.track_count
      - sum(duration_sec) must match playlists.total_duration_sec (rounded)
    Raises ValueError on failure (no events here).
    """
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
    # Always build from DB for rebuilds.
    obj = db_build_playlist_json(conn, pl, built_at=stamp)
    obj["exported_at"] = stamp
    return _stable_playlist_obj(obj, stamp=stamp)


@dataclass(frozen=True)
class ManifestItem:
    playlist_id: str
    slug: str
    path: str
    sha256: str
    bytes: int
    track_count: int


def _build_manifest(
    conn,
    *,
    export_dir: Path,
    limit: int,
    stamp: str,
) -> Tuple[Dict[str, Any], List[Tuple[Path, Dict[str, Any]]]]:
    playlists = db_list_playlists(conn, limit=limit)
    playlists = sorted(playlists, key=lambda p: (p.slug, p.id))

    items: List[ManifestItem] = []
    to_write: List[Tuple[Path, Dict[str, Any]]] = []

    for pl in playlists:
        obj = _build_playlist_obj(conn, pl, stamp=stamp)

        # 🔒 Gate #1: DB vs export consistency (raise only)
        _quality_gate_playlist(pl, obj)

        payload = canonical_json(obj)
        sha = _sha256_text(payload)
        out_path = _playlist_out_path(export_dir, pl)

        items.append(
            ManifestItem(
                playlist_id=pl.id,
                slug=pl.slug,
                path=str(out_path),
                sha256=sha,
                bytes=len(payload.encode("utf-8")) + 1,  # + newline when written
                track_count=len(obj.get("tracks") or []),
            )
        )
        to_write.append((out_path, obj))

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "scope": "playlists",
        "stamp": stamp,
        "count": len(items),
        "items": [i.__dict__ for i in items],
    }
    # fingerprint is over the manifest WITHOUT the fingerprint field
    manifest["fingerprint_sha256"] = _sha256_text(canonical_json(manifest))
    return manifest, to_write


def _load_manifest(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("manifest_not_object")
    return obj


def _manifest_fingerprint(manifest: Dict[str, Any]) -> str:
    # Recompute fingerprint over manifest content excluding fingerprint_sha256
    m2 = dict(manifest)
    m2.pop("fingerprint_sha256", None)
    return _sha256_text(canonical_json(m2))


def _require_manifest_playlists(m: Dict[str, Any]) -> None:
    if m.get("scope") != "playlists":
        raise ValueError(f"manifest_wrong_scope scope={m.get('scope')!r}")
    if not isinstance(m.get("items"), list):
        raise ValueError("manifest_items_not_list")


def _items_by_playlist_id(items: List[Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = str(it.get("playlist_id") or "")
        if not pid:
            continue
        out[pid] = it
    return out


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _short(s: str, n: int = 12) -> str:
    s = s or ""
    return s[:n]


def _diff_manifests(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a_items = list(a.get("items") or [])
    b_items = list(b.get("items") or [])

    a_map = _items_by_playlist_id(a_items)
    b_map = _items_by_playlist_id(b_items)

    a_ids = set(a_map.keys())
    b_ids = set(b_map.keys())

    added = sorted(list(b_ids - a_ids))
    removed = sorted(list(a_ids - b_ids))
    common = sorted(list(a_ids & b_ids))

    changed: List[Dict[str, Any]] = []
    same: List[str] = []

    def key_fields(it: Dict[str, Any]) -> Tuple[str, str, Optional[int], Optional[int], Optional[int]]:
        return (
            str(it.get("sha256") or ""),
            str(it.get("path") or ""),
            _as_int(it.get("bytes")),
            _as_int(it.get("track_count")),
            _as_int(it.get("playlist_id")),  # never used, just keeps tuple stable-ish
        )

    for pid in common:
        ai = a_map[pid]
        bi = b_map[pid]

        a_sha = str(ai.get("sha256") or "")
        b_sha = str(bi.get("sha256") or "")
        a_bytes = _as_int(ai.get("bytes"))
        b_bytes = _as_int(bi.get("bytes"))
        a_path = str(ai.get("path") or "")
        b_path = str(bi.get("path") or "")
        a_tc = _as_int(ai.get("track_count"))
        b_tc = _as_int(bi.get("track_count"))
        a_slug = str(ai.get("slug") or "")
        b_slug = str(bi.get("slug") or "")

        if (a_sha == b_sha) and (a_bytes == b_bytes) and (a_path == b_path) and (a_tc == b_tc) and (a_slug == b_slug):
            same.append(pid)
            continue

        changed.append(
            {
                "playlist_id": pid,
                "slug_a": a_slug,
                "slug_b": b_slug,
                "sha256_a": a_sha,
                "sha256_b": b_sha,
                "bytes_a": a_bytes,
                "bytes_b": b_bytes,
                "track_count_a": a_tc,
                "track_count_b": b_tc,
                "path_a": a_path,
                "path_b": b_path,
            }
        )

    return {
        "fingerprint": {
            "a": str(a.get("fingerprint_sha256") or ""),
            "b": str(b.get("fingerprint_sha256") or ""),
            "a_recomputed": _manifest_fingerprint(a),
            "b_recomputed": _manifest_fingerprint(b),
        },
        "counts": {
            "a": len(a_map),
            "b": len(b_map),
            "added": len(added),
            "removed": len(removed),
            "changed": len(changed),
            "same": len(same),
        },
        "added": added,
        "removed": removed,
        "changed": changed,
        "same": same,
    }


# ----------------------------
# Commands
# ----------------------------

def rebuild_playlists_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    export_dir = Path(args.out_dir)
    stamp = args.stamp or FIXED_STAMP
    limit = int(args.limit)

    try:
        with conn:
            ew.emit(
                "rebuild.started",
                "system",
                None,
                {
                    "scope": "playlists",
                    "db": args.db,
                    "out_dir": str(export_dir),
                    "limit": limit,
                    "stamp": stamp,
                    "determinism_check": bool(args.determinism_check),
                    "write": bool(args.write),
                },
                occurred_at=now_iso(),
            )

            try:
                manifest1, to_write1 = _build_manifest(
                    conn,
                    export_dir=export_dir,
                    limit=limit,
                    stamp=stamp,
                )
            except ValueError as e:
                ew.emit(
                    "rebuild.quality_gate_failed",
                    "system",
                    None,
                    {"gate": "db_vs_export", "reason": str(e)},
                    occurred_at=now_iso(),
                )
                raise SystemExit(2)

            if args.determinism_check:
                try:
                    manifest2, _ = _build_manifest(
                        conn,
                        export_dir=export_dir,
                        limit=limit,
                        stamp=stamp,
                    )
                except ValueError as e:
                    ew.emit(
                        "rebuild.quality_gate_failed",
                        "system",
                        None,
                        {"gate": "db_vs_export", "reason": str(e)},
                        occurred_at=now_iso(),
                    )
                    raise SystemExit(2)

                if manifest1["fingerprint_sha256"] != manifest2["fingerprint_sha256"]:
                    ew.emit(
                        "rebuild.quality_gate_failed",
                        "system",
                        None,
                        {
                            "gate": "determinism",
                            "fingerprint_1": manifest1["fingerprint_sha256"],
                            "fingerprint_2": manifest2["fingerprint_sha256"],
                        },
                        occurred_at=now_iso(),
                    )
                    raise SystemExit(2)

            if args.write:
                export_dir.mkdir(parents=True, exist_ok=True)
                for path, obj in to_write1:
                    _write_json(path, obj)
                _write_json(export_dir / "_manifest.playlists.json", manifest1)

            ew.emit(
                "rebuild.completed",
                "system",
                None,
                {
                    "scope": "playlists",
                    "ok": True,
                    "fingerprint_sha256": manifest1["fingerprint_sha256"],
                    "count": manifest1["count"],
                    "wrote": bool(args.write),
                },
                occurred_at=now_iso(),
            )

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


def rebuild_verify_playlists_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest) if args.manifest else (out_dir / "_manifest.playlists.json")

    try:
        with conn:
            ew.emit(
                "rebuild.verify_started",
                "system",
                None,
                {
                    "scope": "playlists",
                    "db": args.db,
                    "out_dir": str(out_dir),
                    "manifest": str(manifest_path),
                },
                occurred_at=now_iso(),
            )

            if not manifest_path.exists():
                ew.emit(
                    "rebuild.verify_failed",
                    "system",
                    None,
                    {"reason": "manifest_missing", "manifest": str(manifest_path)},
                    occurred_at=now_iso(),
                )
                raise SystemExit(2)

            try:
                manifest = _load_manifest(manifest_path)
                _require_manifest_playlists(manifest)
            except Exception as e:
                ew.emit(
                    "rebuild.verify_failed",
                    "system",
                    None,
                    {"reason": "manifest_invalid", "error": str(e)},
                    occurred_at=now_iso(),
                )
                raise SystemExit(2)

            expected_fp = str(manifest.get("fingerprint_sha256") or "")
            recomputed_fp = _manifest_fingerprint(manifest)
            if expected_fp and expected_fp != recomputed_fp:
                ew.emit(
                    "rebuild.verify_failed",
                    "system",
                    None,
                    {
                        "reason": "fingerprint_mismatch",
                        "expected": expected_fp,
                        "recomputed": recomputed_fp,
                    },
                    occurred_at=now_iso(),
                )
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
                playlist_id = str(it.get("playlist_id") or "")
                slug = str(it.get("slug") or "")

                p = Path(rel_or_abs)
                if not p.is_absolute():
                    p = Path.cwd() / p

                if not p.exists():
                    failures.append({"playlist_id": playlist_id, "slug": slug, "path": rel_or_abs, "reason": "file_missing"})
                    continue

                data = p.read_bytes()
                got_bytes = len(data)
                got_sha = _sha256_bytes(data)

                ok_bytes = (expected_bytes is None) or (int(expected_bytes) == got_bytes)
                ok_sha = (not expected_sha) or (expected_sha == got_sha)

                checked += 1
                if not ok_bytes or not ok_sha:
                    failures.append(
                        {
                            "playlist_id": playlist_id,
                            "slug": slug,
                            "path": rel_or_abs,
                            "reason": "file_mismatch",
                            "expected_bytes": int(expected_bytes) if expected_bytes is not None else None,
                            "got_bytes": got_bytes,
                            "expected_sha256": expected_sha,
                            "got_sha256": got_sha,
                        }
                    )

            if failures:
                sample = failures[: min(20, len(failures))]
                ew.emit(
                    "rebuild.verify_failed",
                    "system",
                    None,
                    {"reason": "verification_failed", "checked": checked, "failures": len(failures), "sample": sample},
                    occurred_at=now_iso(),
                )
                raise SystemExit(2)

            ew.emit(
                "rebuild.verify_completed",
                "system",
                None,
                {"scope": "playlists", "ok": True, "checked": checked, "fingerprint_sha256": expected_fp or recomputed_fp},
                occurred_at=now_iso(),
            )

        if args.json:
            print(
                json.dumps(
                    {"ok": True, "checked": checked, "fingerprint_sha256": expected_fp or recomputed_fp, "manifest": str(manifest_path)},
                    indent=2,
                    ensure_ascii=False,
                )
            )
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


def rebuild_diff_playlists_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    a_path = Path(args.a)
    b_path = Path(args.b)

    # Allow relative paths; interpret relative to repo cwd.
    if not a_path.is_absolute():
        a_path = Path.cwd() / a_path
    if not b_path.is_absolute():
        b_path = Path.cwd() / b_path

    show = int(args.show)

    try:
        with conn:
            ew.emit(
                "rebuild.diff_started",
                "system",
                None,
                {"scope": "playlists", "a": str(a_path), "b": str(b_path)},
                occurred_at=now_iso(),
            )

            if not a_path.exists() or not b_path.exists():
                ew.emit(
                    "rebuild.diff_failed",
                    "system",
                    None,
                    {
                        "reason": "manifest_missing",
                        "a_exists": a_path.exists(),
                        "b_exists": b_path.exists(),
                        "a": str(a_path),
                        "b": str(b_path),
                    },
                    occurred_at=now_iso(),
                )
                raise SystemExit(2)

            try:
                a = _load_manifest(a_path)
                b = _load_manifest(b_path)
                _require_manifest_playlists(a)
                _require_manifest_playlists(b)
            except Exception as e:
                ew.emit(
                    "rebuild.diff_failed",
                    "system",
                    None,
                    {"reason": "manifest_invalid", "error": str(e)},
                    occurred_at=now_iso(),
                )
                raise SystemExit(2)

            diff = _diff_manifests(a, b)

            ew.emit(
                "rebuild.diff_completed",
                "system",
                None,
                {
                    "scope": "playlists",
                    "a": str(a_path),
                    "b": str(b_path),
                    "counts": diff["counts"],
                    "fingerprint_a": diff["fingerprint"]["a"],
                    "fingerprint_b": diff["fingerprint"]["b"],
                },
                occurred_at=now_iso(),
            )

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
                pid = ch["playlist_id"]
                print(f"  ~ {pid}  sha {_short(ch['sha256_a'])} -> {_short(ch['sha256_b'])}  bytes {ch['bytes_a']} -> {ch['bytes_b']}  tc {ch['track_count_a']} -> {ch['track_count_b']}")
            if len(diff["changed"]) > show:
                print(f"  ... ({len(diff['changed']) - show} more)")

        return 0

    except SystemExit:
        raise
    finally:
        conn.close()


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
    # Compatibility flag (ignored). Keeps old muscle-memory from erroring.
    rp.add_argument("--build-from-db", action="store_true", help=argparse.SUPPRESS)
    rp.set_defaults(func=rebuild_playlists_cmd)

    # rebuild verify playlists
    rv = rgs.add_parser("verify", help="Verify rebuild artifacts against manifest")
    rvs = rv.add_subparsers(dest="rebuild_verify_cmd", required=True)

    rvp = rvs.add_parser("playlists", help="Verify playlist JSON files against _manifest.playlists.json")
    rvp.add_argument("--db", default="data/db.sqlite")
    rvp.add_argument("--out-dir", default="data/playlists")
    rvp.add_argument("--manifest", default=None, help="Path to manifest (default: <out-dir>/_manifest.playlists.json)")
    rvp.add_argument("--json", action="store_true")
    rvp.set_defaults(func=rebuild_verify_playlists_cmd)

    # rebuild diff playlists
    rd = rgs.add_parser("diff", help="Diff two manifests")
    rds = rd.add_subparsers(dest="rebuild_diff_cmd", required=True)

    rdp = rds.add_parser("playlists", help="Diff two playlist manifests")
    rdp.add_argument("a", help="Path to manifest A (old)")
    rdp.add_argument("b", help="Path to manifest B (new)")
    rdp.add_argument("--db", default="data/db.sqlite")
    rdp.add_argument("--json", action="store_true")
    rdp.add_argument("--show", type=int, default=50, help="Max items to print per section (default: 50)")
    rdp.set_defaults(func=rebuild_diff_playlists_cmd)
