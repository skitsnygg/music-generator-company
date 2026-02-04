#!/usr/bin/env python3
"""
Local DiffSinger wrapper for MGC.

This script selects a pre-generated audio file from a directory pool and
copies it to the requested output path. It is intended for wiring
DiffSinger outputs into MGC without running inference on each call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aif", ".aiff"}


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


def _read_stdin_json() -> Optional[Dict[str, Any]]:
    if sys.stdin is None or sys.stdin.isatty():
        return None
    raw = sys.stdin.read()
    if not raw.strip():
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _iter_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def _die(msg: str) -> int:
    sys.stderr.write(msg.rstrip() + "\n")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick a DiffSinger wav from a pool and write to --output.")
    parser.add_argument("--output", default="", help="Destination path for generated audio")
    parser.add_argument("--prompt", default="", help="Prompt text (used for deterministic selection)")
    parser.add_argument("--voice", default="", help="Voice name (used for deterministic selection)")
    parser.add_argument("--seed", default="", help="Seed (used for deterministic selection)")
    parser.add_argument("--context", default="", help="Context (used for deterministic selection)")
    parser.add_argument("--bpm", default="", help="BPM (ignored)")
    parser.add_argument("--duration-seconds", default="", help="Duration (ignored)")
    parser.add_argument("--source-dir", default="", help="Directory containing pre-generated audio files")
    parser.add_argument("--stdin-json", action="store_true", help="Read payload JSON from stdin")
    args, _unknown = parser.parse_known_args()

    payload = _read_stdin_json() if args.stdin_json or (sys.stdin and not sys.stdin.isatty()) else None
    payload = payload or {}

    output_path = (
        args.output
        or str(payload.get("output") or "")
        or str(payload.get("out_path") or "")
        or str(payload.get("output_path") or "")
    )
    if not output_path:
        return _die("diffsinger_local: --output is required")

    source_dir = (
        args.source_dir
        or os.environ.get("MGC_DIFFSINGER_SAMPLE_DIR")
        or os.environ.get("DIFFSINGER_SAMPLE_DIR")
        or str(payload.get("sample_dir") or "")
    )
    if not source_dir:
        return _die("diffsinger_local: set MGC_DIFFSINGER_SAMPLE_DIR or pass --source-dir")

    root = Path(source_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return _die(f"diffsinger_local: sample dir not found: {root}")

    files: List[Path] = sorted(_iter_audio_files(root))
    if not files:
        return _die(f"diffsinger_local: no audio files found under {root}")

    prompt = args.prompt or str(payload.get("prompt") or "")
    voice = args.voice or str(payload.get("voice") or "")
    seed = str(args.seed or payload.get("seed") or "")
    context = args.context or str(payload.get("context") or "")

    key = f"{seed}|{prompt}|{context}|{voice}"
    src = files[_stable_index(key, len(files))]

    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
