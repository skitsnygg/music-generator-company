#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def find_mp3_start(b: bytes) -> int | None:
    # Prefer ID3 tag
    i = b.find(b"ID3")
    if i != -1:
        return i

    # Otherwise look for MPEG frame sync: 0xFFEx (covers common cases)
    # We scan for 0xFF followed by 0xE0..0xFF, and also accept 0xF? patterns.
    for j in range(len(b) - 1):
        if b[j] == 0xFF:
            nxt = b[j + 1]
            # Frame sync is 11 set bits: 0xFF and top 3 bits of next are 1.
            # So nxt should be 0xE0..0xFF.
            if (nxt & 0xE0) == 0xE0:
                return j
    return None


def clean_file(p: Path, *, write: bool) -> tuple[bool, str]:
    b = p.read_bytes()
    start = find_mp3_start(b)
    if start is None:
        return False, "no ID3 or frame sync found"

    if start == 0:
        return True, "already clean"

    cleaned = b[start:]
    if write:
        p.write_bytes(cleaned)
    return True, f"stripped {start} leading bytes"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="mp3 file(s) or directory(ies)")
    ap.add_argument("--write", action="store_true", help="actually rewrite files (default: dry-run)")
    args = ap.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.mp3")))
        else:
            files.append(p)

    ok = 0
    bad = 0
    for f in files:
        if not f.is_file():
            continue
        cleaned, msg = clean_file(f, write=args.write)
        if cleaned:
            ok += 1
            print(f"[ok] {f} :: {msg}")
        else:
            bad += 1
            print(f"[skip] {f} :: {msg}")

    print(f"done ok={ok} skip={bad} write={bool(args.write)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
