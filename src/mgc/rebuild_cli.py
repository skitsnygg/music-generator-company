from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mgc.db_helpers import (
    PlaylistRow,
    db_build_playlist_json,
    db_get_playlist,
    db_list_playlists,
    resolve_json_path,
    sqlite_connect,
)
from mgc.events import EventContext, EventWriter, canonical_json, new_run_id


FIXED_STAMP = "1970-01-01T00:00:00Z"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _stable_playlist_obj(obj: Dict[str, Any], *, stamp: str) -> Dict[str, Any]:
    """
    Normalize volatile fields so repeated rebuilds are byte-for-byte deterministic.
    """
    out = dict(obj)

    # Known volatile keys (keep them, but set to fixed stamp)
    if "built_from_db_at" in out:
        out["built_from_db_at"] = stamp
    if "exported_at" in out:
        out["exported_at"] = stamp

    # Optional: if your builder adds any other time-ish fields in the future,
    # normalize them here.
    return out


def _build_playlist_export_obj(conn, pl: PlaylistRow, *, build_from_db: bool, stamp: str) -> Dict[str, Any]:
    if build_from_db:
        obj = db_build_playlist_json(conn, pl, built_at=stamp)
        obj["exported_at"] = stamp
        return _stable_playlist_obj(obj, stamp=stamp)

    # If you want deterministic rebuilds, "build_from_db" should be the default.
    # This branch is kept for completeness, but it will not be deterministic if the json_path file changes.
    src = resolve_json_path(pl.json_path)
    if src.exists():
        raw = json.loads(src.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw.setdefault("id", pl.id)
            raw.setdefault("slug", pl.slug)
            raw["exported_at"] = stamp
            return _stable_playlist_obj(raw, stamp=stamp)

    obj = db_build_playlist_json(conn, pl, built_at=stamp)
    obj["exported_at"] = stamp
    return _stable_playlist_obj(obj, stamp=stamp)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # canonical_json gives stable ordering/separators.
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def _playlist_out_path(export_dir: Path, pl: PlaylistRow) -> Path:
    # match your existing export convention: <slug>_<playlist_id>.json
    # (keep it exactly the same so your old tooling continues to work)
    safe = "".join(ch for ch in pl.slug.strip() if ch.isalnum() or ch in ("-", "_")).strip("-_")
    name = f"{safe}_{pl.id}.json" if safe else f"{pl.id}.json"
    return export_dir / name


@dataclass(frozen=True)
class ManifestItem:
    playlist_id: str
    slug: str
    path: str
    sha256: str
    bytes: int
    track_count: int


def _build_manifest(conn, *, export_dir: Path, limit: int, build_from_db: bool, stamp: str) -> Tuple[Dict[str, Any], List[Tuple[Path, Dict[str, Any]]]]:
    pls = db_list_playlists(conn, limit=limit)
    # Deterministic ordering
    pls_sorted = sorted(pls, key=lambda p: (p.slug, p.id))

    to_write: List[Tuple[Path, Dict[str, Any]]] = []
    items: List[ManifestItem] = []

    for pl in pls_sorted:
        obj = _build_playlist_export_obj(conn, pl, build_from_db=build_from_db, stamp=stamp)
        out_path = _playlist_out_path(export_dir, pl)
        payload = canonical_json(obj)
        sha = _sha256_text(payload)
        items.append(
            ManifestItem(
                playlist_id=pl.id,
                slug=pl.slug,
                path=str(out_path),
                sha256=sha,
                bytes=len(payload.encode("utf-8")) + 1,  # + newline
                track_count=len(obj.get("tracks") or []),
            )
        )
        to_write.append((out_path, obj))

    manifest_obj: Dict[str, Any] = {
        "schema_version": 1,
        "scope": "playlists",
        "stamp": stamp,
        "count": len(items),
        "items": [i.__dict__ for i in items],
    }
    # Fingerprint is deterministic from the manifest contents (without fingerprint itself)
    fp = _sha256_text(canonical_json(manifest_obj))
    manifest_obj["fingerprint_sha256"] = fp

    return manifest_obj, to_write


def rebuild_playlists_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    export_dir = Path(args.out_dir)
    stamp = args.stamp if args.stamp else FIXED_STAMP
    build_from_db = bool(args.build_from_db)

    ew.emit(
        "rebuild.started",
        "system",
        None,
        {
            "scope": "playlists",
            "db": args.db,
            "out_dir": str(export_dir),
            "limit": int(args.limit),
            "build_from_db": build_from_db,
            "stamp": stamp,
            "determinism_check": bool(args.determinism_check),
            "write": bool(args.write),
        },
        occurred_at=args.occurred_at,
    )

    try:
        with conn:
            manifest1, to_write1 = _build_manifest(
                conn,
                export_dir=export_dir,
                limit=int(args.limit),
                build_from_db=build_from_db,
                stamp=stamp,
            )

            # Quality gate: determinism self-check (compute twice in-process and compare fingerprints)
            if args.determinism_check:
                manifest2, _ = _build_manifest(
                    conn,
                    export_dir=export_dir,
                    limit=int(args.limit),
                    build_from_db=build_from_db,
                    stamp=stamp,
                )
                if manifest1["fingerprint_sha256"] != manifest2["fingerprint_sha256"]:
                    ew.emit(
                        "rebuild.quality_gate_failed",
                        "system",
                        None,
                        {
                            "scope": "playlists",
                            "gate": "determinism",
                            "fingerprint_1": manifest1["fingerprint_sha256"],
                            "fingerprint_2": manifest2["fingerprint_sha256"],
                        },
                        occurred_at=args.occurred_at,
                    )
                    raise SystemExit(2)

            # Write outputs + manifest (optional)
            if args.write:
                export_dir.mkdir(parents=True, exist_ok=True)
                for path, obj in to_write1:
                    _write_json(path, obj)

                manifest_path = export_dir / "_manifest.playlists.json"
                _write_json(manifest_path, manifest1)

        if args.json:
            print(json.dumps(manifest1, indent=2, ensure_ascii=False))
        else:
            print(f"fingerprint_sha256: {manifest1['fingerprint_sha256']}")
            print(f"count: {manifest1['count']}")
            if args.write:
                print(f"wrote: {export_dir}/_manifest.playlists.json")

        ew.emit(
            "rebuild.completed",
            "system",
            None,
            {
                "scope": "playlists",
                "ok": True,
                "fingerprint_sha256": manifest1["fingerprint_sha256"],
                "count": manifest1["count"],
                "out_dir": str(export_dir),
                "write": bool(args.write),
            },
            occurred_at=args.occurred_at,
        )
        return 0

    except SystemExit:
        # already emitted quality gate if determinism failed
        raise
    except Exception as e:
        ew.emit(
            "rebuild.failed",
            "system",
            None,
            {"scope": "playlists", "ok": False, "error": str(e), "error_type": type(e).__name__},
            occurred_at=args.occurred_at,
        )
        raise
    finally:
        conn.close()


def register_rebuild_subcommand(subparsers) -> None:
    rg = subparsers.add_parser("rebuild", help="Deterministic rebuilds + quality gates")
    rgs = rg.add_subparsers(dest="rebuild_cmd", required=True)

    rp = rgs.add_parser("playlists", help="Rebuild playlist exports deterministically")
    rp.add_argument("--db", default="data/db.sqlite")
    rp.add_argument("--out-dir", default="data/playlists")
    rp.add_argument("--limit", type=int, default=100000)
    rp.add_argument("--json", action="store_true")

    # Determinism knobs
    rp.add_argument("--stamp", default=None, help=f"Timestamp used inside exported JSON (default: {FIXED_STAMP})")
    rp.add_argument("--build-from-db", action="store_true", help="Build exports from DB (recommended for determinism)")

    # Gates / mode
    rp.add_argument("--determinism-check", action="store_true", help="Compute fingerprint twice and compare")
    rp.add_argument("--write", action="store_true", help="Write JSON files + manifest (otherwise dry-run)")

    # Events occurred_at (kept separate from stamp; stamp controls JSON determinism)
    rp.add_argument("--occurred-at", default="", help="Event occurred_at value (optional)")

    rp.set_defaults(func=rebuild_playlists_cmd)
