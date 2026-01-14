#!/usr/bin/env python3
"""
scripts/ci_golden_bless.py

Compute a tree hash and write it into fixtures/golden_hashes.json at --key.

Usage:
  python scripts/ci_golden_bless.py \
    --golden fixtures/golden_hashes.json \
    --key ci.rebuild.verify \
    --root artifacts/ci/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from mgc.hash_tree import hash_tree


def _load_or_init(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _set_key(d: Dict[str, Any], dotted: str, value: Any) -> None:
    cur: Dict[str, Any] = d
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


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

    res = hash_tree(root, ignore_globs=args.ignore or None)
    doc = _load_or_init(golden_path)

    _set_key(doc, args.key, res.sha256)

    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[ci_golden_bless] WROTE")
    print(f"  golden: {golden_path}")
    print(f"  key:    {args.key}")
    print(f"  hash:   {res.sha256}")
    print(f"  files:  {res.file_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
