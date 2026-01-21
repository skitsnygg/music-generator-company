#!/usr/bin/env python3
"""
src/mgc/web_cli.py

Static web player build.

FIX:
- Resolve track paths relative to the playlist directory when absolute / repo paths
  are missing.
- This allows autonomous publish + bundle-based releases to work deterministically.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def _load_playlist(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_track_path(track_path: str, playlist_dir: Path) -> Path:
    p = Path(track_path)

    # 1) Exact path
    if p.exists():
        return p

    # 2) Relative to playlist dir
    rel = (playlist_dir / p).resolve()
    if rel.exists():
        return rel

    # 3) Common bundle layout: playlist.json + tracks/<file>
    candidate = playlist_dir / "tracks" / p.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(track_path)


def cmd_web_build(args: argparse.Namespace) -> int:
    playlist_path = Path(args.playlist).resolve()
    out_dir = Path(args.out_dir).resolve()

    playlist = _load_playlist(playlist_path)
    tracks: List[Dict] = playlist.get("tracks") or []

    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_out = out_dir / "tracks"
    tracks_out.mkdir(exist_ok=True)

    copied = 0
    missing = 0

    for t in tracks:
        src_raw = t.get("artifact_path") or t.get("full_path")
        if not src_raw:
            missing += 1
            continue

        try:
            src = _resolve_track_path(src_raw, playlist_path.parent)
        except FileNotFoundError:
            missing += 1
            continue

        dst = tracks_out / src.name
        shutil.copyfile(src, dst)
        copied += 1

    if args.fail_if_empty and copied == 0:
        raise SystemExit(
            json.dumps(
                {
                    "ok": False,
                    "copied_count": copied,
                    "missing_count": missing,
                    "reason": "missing_tracks",
                    "out_dir": str(out_dir),
                }
            )
        )

    manifest = {
        "ok": True,
        "track_count": copied,
        "copied_count": copied,
        "missing_count": missing,
        "playlist": str(playlist_path),
        "out_dir": str(out_dir),
    }

    with (out_dir / "web_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(json.dumps(manifest))
    return 0


def register_web_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("web", help="Static web player tools")
    sp = p.add_subparsers(dest="web_cmd", required=True)

    b = sp.add_parser("build", help="Build static web player")
    b.add_argument("--playlist", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--clean", action="store_true")
    b.add_argument("--prefer-mp3", action="store_true")
    b.add_argument("--fail-if-empty", action="store_true")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_web_build)
