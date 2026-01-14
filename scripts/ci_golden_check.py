#!/usr/bin/env python3
"""
scripts/ci_golden_check.py

Compare a computed tree hash against fixtures/golden_hashes.json.

Usage:
  python scripts/ci_golden_check.py \
    --golden fixtures/golden_hashes.json \
    --key ci.rebuild.verify \
    --root artifacts/ci/data

Exit codes:
  0 = match
  2 = missing key
  3 = mismatch
  4 = root missing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from mgc.hash_tree import hash_tree


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _get_key(d: Dict[str, Any], dotted: str) -> Optional[Any]:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--golden", required=True, help="Path to golden_hashes.json")
    p.add_argument("--key", required=True, help="Dotted key path inside JSON")
    p.add_argument("--root", required=True, help="Root directory to hash")
    p.add_argument("--ignore", action="append", default=[], help="Additional ignore globs")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    golden_path = Path(args.golden)
    root = Path(args.root)

    if not root.exists():
        print(f"[ci_golden_check] FAIL: root does not exist: {root}")
        return 4

    golden = _load_json(golden_path)
    expected = _get_key(golden, args.key)

    if expected is None:
        print(f"[ci_golden_check] FAIL: missing key '{args.key}' in {golden_path}")
        return 2

    if not isinstance(expected, str) or len(expected) < 16:
        print(f"[ci_golden_check] FAIL: key '{args.key}' is not a plausible hash string")
        return 2

    res = hash_tree(root, ignore_globs=args.ignore or None)

    if res.sha256 != expected:
        print("[ci_golden_check] FAIL: golden hash mismatch")
        print(f"  key:      {args.key}")
        print(f"  root:     {res.root}")
        print(f"  expected: {expected}")
        print(f"  actual:   {res.sha256}")
        print(f"  files:    {res.file_count}")
        print(f"  bytes:    {res.total_bytes}")
        print("")
        print("If this change is intended, bless a new golden hash via:")
        print("  python scripts/ci_golden_bless.py --golden fixtures/golden_hashes.json "
              f"--key {args.key} --root {args.root}")
        return 3

    print("[ci_golden_check] OK")
    print(f"  key:   {args.key}")
    print(f"  hash:  {res.sha256}")
    print(f"  files: {res.file_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
