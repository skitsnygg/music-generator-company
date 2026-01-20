#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def _nonempty_file(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _pick_evidence(bundle_dir: Path) -> Optional[Path]:
    """
    Evidence naming varies by mode/history. Accept the common ones:
      - drop_evidence.json (preferred)
      - daily_evidence.json
      - weekly_evidence.json
      - anything matching *evidence*.json (fallback, deterministic)
    """
    preferred = [
        bundle_dir / "drop_evidence.json",
        bundle_dir / "daily_evidence.json",
        bundle_dir / "weekly_evidence.json",
    ]
    for p in preferred:
        if _nonempty_file(p):
            return p.resolve()

    # fallback: any *evidence*.json in the bundle dir (deterministic pick)
    candidates = sorted(bundle_dir.glob("*evidence*.json"))
    for p in candidates:
        if _nonempty_file(p):
            return p.resolve()

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify MGC drop output contract.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists():
        return _fail(f"out-dir does not exist: {out_dir}")

    playlist_top = out_dir / "playlist.json"
    if not _nonempty_file(playlist_top):
        return _fail(f"missing or empty: {playlist_top}")

    bundle_dir = out_dir / "drop_bundle"
    if not bundle_dir.is_dir():
        return _fail(f"missing drop_bundle dir: {bundle_dir}")

    playlist_bundle = bundle_dir / "playlist.json"
    if not _nonempty_file(playlist_bundle):
        return _fail(f"missing or empty: {playlist_bundle}")

    evidence_path = _pick_evidence(bundle_dir)
    if not evidence_path:
        # helpful debug listing
        existing = sorted([p.name for p in bundle_dir.glob("*.json")])
        return _fail(
            "missing evidence json in drop_bundle. Expected one of "
            "drop_evidence.json/daily_evidence.json/weekly_evidence.json or *evidence*.json. "
            f"Found json files: {existing}"
        )

    # Basic shape checks
    try:
        ev = _load_json(evidence_path)
    except Exception as e:
        return _fail(f"{evidence_path.name} is not valid JSON: {e}")

    paths = ev.get("paths") if isinstance(ev.get("paths"), dict) else {}
    # Soft check: if evidence declares a playlist path, it must exist.
    pl_path = paths.get("playlist_path") or paths.get("playlist")
    if isinstance(pl_path, str) and pl_path.strip():
        p = Path(pl_path.strip())
        if not p.is_absolute():
            p = (bundle_dir / p).resolve()
        if not p.exists():
            return _fail(f"evidence paths.playlist_path points to missing file: {p}")

    # Ensure playlist references are sane (at least one track)
    try:
        pl = _load_json(playlist_top)
    except Exception as e:
        return _fail(f"top-level playlist.json is not valid JSON: {e}")

    tracks = pl.get("tracks")
    if not isinstance(tracks, list) or len(tracks) < 1:
        return _fail("playlist.json must contain a non-empty 'tracks' array")

    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": str(out_dir),
                "playlist_top": str(playlist_top),
                "playlist_bundle": str(playlist_bundle),
                "evidence": str(evidence_path),
                "evidence_name": evidence_path.name,
                "track_count": len(tracks),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
