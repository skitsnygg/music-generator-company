#!/usr/bin/env python3
"""
scripts/find_artifacts.py

Small helper for CI and local debugging.

- find_evidence: locate drop_evidence.json under likely roots
- find_playlist: extract playlist path from evidence OR find a nearby playlist*.json

This replaces brittle YAML logic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def _dig(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def find_evidence(
    artifacts_dir: Path, evidence_dir: Path, runner_temp: Path, max_depth: int = 6
) -> Optional[Path]:
    candidates = [
        artifacts_dir / "auto" / "drop_evidence.json",
        evidence_dir / "drop_evidence.json",
        evidence_dir / "auto" / "drop_evidence.json",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p.resolve()

    # bounded search
    if runner_temp.exists():
        found = list(runner_temp.glob("**/drop_evidence.json"))
        # manual bounded filter
        best = []
        for p in found:
            try:
                rel = p.relative_to(runner_temp)
                if len(rel.parts) <= max_depth and p.is_file() and p.stat().st_size > 0:
                    best.append(p)
            except Exception:
                continue
        best.sort()
        if best:
            return best[0].resolve()

    return None


def find_playlist_from_evidence(evidence_path: Path) -> Optional[Path]:
    obj = json.loads(evidence_path.read_text(encoding="utf-8"))

    candidates = [
        _dig(obj, "paths", "playlist_path"),
        _dig(obj, "paths", "playlist"),
        _dig(obj, "paths", "playlist_json"),
        _dig(obj, "paths", "playlist_file"),
        _dig(obj, "drop", "paths", "playlist_path"),
        _dig(obj, "drop", "paths", "playlist"),
        obj.get("playlist_path") if isinstance(obj, dict) else None,
        obj.get("playlist") if isinstance(obj, dict) else None,
    ]

    raw = None
    for c in candidates:
        if isinstance(c, str) and c.strip():
            raw = c.strip()
            break

    if not raw:
        return None

    p = Path(raw)
    if not p.is_absolute():
        p = (evidence_path.parent / p).resolve()
    if p.exists() and p.stat().st_size > 0:
        return p
    return None


def find_playlist_near(evidence_path: Path) -> Optional[Path]:
    root = evidence_path.parent
    near = [
        root / "playlist.json",
        root / "drop_bundle" / "playlist.json",
        root.parent / "playlist.json",
        root.parent / "drop_bundle" / "playlist.json",
    ]
    for p in near:
        if p.exists() and p.stat().st_size > 0:
            return p.resolve()

    # bounded rglob
    for sr in (root, root.parent):
        if not sr.exists():
            continue
        for p in sr.rglob("playlist*.json"):
            try:
                if p.is_file() and p.stat().st_size > 0:
                    return p.resolve()
            except OSError:
                continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["find_evidence", "find_playlist"])
    ap.add_argument("--artifacts-dir", default=os.environ.get("MGC_ARTIFACTS_DIR", ""))
    ap.add_argument("--evidence-dir", default=os.environ.get("MGC_EVIDENCE_DIR", ""))
    ap.add_argument("--runner-temp", default=os.environ.get("RUNNER_TEMP", "/tmp"))
    ap.add_argument("--evidence-path", default="")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else Path(".").resolve()
    evidence_dir = Path(args.evidence_dir).resolve() if args.evidence_dir else Path(".").resolve()
    runner_temp = Path(args.runner_temp).resolve()

    if args.mode == "find_evidence":
        p = find_evidence(artifacts_dir, evidence_dir, runner_temp)
        if not p:
            print("ERROR: drop_evidence.json not found", file=sys.stderr)
            return 2
        print(str(p))
        return 0

    # find_playlist
    if args.evidence_path:
        evidence_path = Path(args.evidence_path).resolve()
    else:
        p = find_evidence(artifacts_dir, evidence_dir, runner_temp)
        if not p:
            print("ERROR: drop_evidence.json not found", file=sys.stderr)
            return 2
        evidence_path = p

    if not evidence_path.exists():
        print(f"ERROR: evidence missing: {evidence_path}", file=sys.stderr)
        return 2

    pl = find_playlist_from_evidence(evidence_path) or find_playlist_near(evidence_path)
    if not pl:
        print("ERROR: playlist not found from evidence or nearby files", file=sys.stderr)
        print(f"evidence_path={evidence_path}", file=sys.stderr)
        return 2

    print(str(pl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
