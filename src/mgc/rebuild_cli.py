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
    Raises ValueError with a structured-ish string on failure.
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

        # 🔒 Gate #1: DB vs export consistency (no events here; raise only)
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
    manifest["fingerprint_sha256"] = _sha256_text(canonical_json(manifest))
    return manifest, to_write


def rebuild_playlists_cmd(args: argparse.Namespace) -> int:
    run_id = new_run_id()
    conn = sqlite_connect(args.db)
    ew = EventWriter(conn, EventContext(run_id=run_id, source="cli"))

    export_dir = Path(args.out_dir)
    stamp = args.stamp or FIXED_STAMP
    limit = int(args.limit)

    try:
        # IMPORTANT: emit events inside the transaction so they commit with the rebuild.
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
                # Quality gate failures land here (from _quality_gate_playlist).
                ew.emit(
                    "rebuild.quality_gate_failed",
                    "system",
                    None,
                    {
                        "gate": "db_vs_export",
                        "reason": str(e),
                    },
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
                        {
                            "gate": "db_vs_export",
                            "reason": str(e),
                        },
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

        # Print after commit so output matches persisted state.
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
    except Exception as e:
        # If we get here, the `with conn:` may have rolled back; still try to log failure.
        try:
            with conn:
                ew.emit(
                    "rebuild.failed",
                    "system",
                    None,
                    {"error": str(e), "error_type": type(e).__name__},
                    occurred_at=now_iso(),
                )
        except Exception:
            pass
        raise
    finally:
        conn.close()


def register_rebuild_subcommand(subparsers) -> None:
    rg = subparsers.add_parser("rebuild", help="Deterministic rebuilds + quality gates")
    rgs = rg.add_subparsers(dest="rebuild_cmd", required=True)

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
