#!/usr/bin/env python3
"""
src/mgc/hash_tree.py

Deterministic "tree hash" for a directory. This is the backbone for golden-hash CI.

Design:
- Hash is over (normalized relative path + NUL + file bytes) in sorted path order.
- Ignores:
  - dotfiles / dotdirs by default (configurable)
  - common junk (pyc, __pycache__, .DS_Store, etc.)
- Stable across platforms (path normalization to '/')
- Does NOT use mtimes, permissions, zip metadata, etc.

CLI:
  python -m mgc.hash_tree --root <dir> --json
  python -m mgc.hash_tree --root <dir> --print
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_IGNORE_GLOBS: List[str] = [
    # Python junk
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    # OS/editor junk
    "**/.DS_Store",
    "**/Thumbs.db",
    "**/.idea/**",
    "**/.vscode/**",
    # Git junk
    "**/.git/**",
    # Logs/temp
    "**/*.log",
    "**/*.tmp",
]


@dataclass(frozen=True)
class TreeHashResult:
    root: str
    file_count: int
    total_bytes: int
    sha256: str


def _norm_relpath(root: Path, p: Path) -> str:
    rel = p.relative_to(root).as_posix()
    # Ensure no leading "./"
    if rel.startswith("./"):
        rel = rel[2:]
    return rel


def _matches_any_glob(rel_posix: str, globs: Sequence[str]) -> bool:
    # fnmatch works with POSIX-ish patterns if we provide posix paths
    for g in globs:
        if fnmatch.fnmatch(rel_posix, g):
            return True
    return False


def iter_files(
    root: Path,
    ignore_globs: Sequence[str],
    ignore_dotfiles: bool = True,
) -> Iterable[Tuple[str, Path]]:
    """
    Yield (rel_posix, abs_path) for files under root, sorted by rel_posix.
    """
    root = root.resolve()
    if not root.exists():
        return
    files: List[Tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)

        # Optionally drop dotdirs from traversal
        if ignore_dotfiles:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for name in filenames:
            if ignore_dotfiles and name.startswith("."):
                continue
            ap = dp / name
            if not ap.is_file():
                continue
            rel = _norm_relpath(root, ap)

            # Apply ignore globs
            if _matches_any_glob(rel, ignore_globs):
                continue

            files.append((rel, ap))

    files.sort(key=lambda t: t[0])
    for rel, ap in files:
        yield rel, ap


def hash_tree(
    root: Path,
    ignore_globs: Optional[Sequence[str]] = None,
    ignore_dotfiles: bool = True,
    chunk_size: int = 1024 * 1024,
) -> TreeHashResult:
    """
    Compute deterministic SHA256 over a directory tree.

    Hash input stream is:
      for each file in sorted relpath order:
         sha.update(relpath_utf8)
         sha.update(b"\\0")
         sha.update(file_bytes)
         sha.update(b"\\0")
    """
    root = root.resolve()
    globs = list(ignore_globs or DEFAULT_IGNORE_GLOBS)

    sha = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    for rel, ap in iter_files(root, globs, ignore_dotfiles=ignore_dotfiles):
        file_count += 1
        rel_b = rel.encode("utf-8")
        sha.update(rel_b)
        sha.update(b"\0")

        with ap.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                sha.update(chunk)

        sha.update(b"\0")

    return TreeHashResult(
        root=str(root),
        file_count=file_count,
        total_bytes=total_bytes,
        sha256=sha.hexdigest(),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mgc.hash_tree", add_help=True)
    p.add_argument("--root", required=True, help="Root directory to hash")
    p.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Additional ignore glob(s). Can be provided multiple times.",
    )
    p.add_argument(
        "--no-ignore-dotfiles",
        action="store_true",
        help="Include dotfiles and dotdirs (default ignores them).",
    )
    p.add_argument("--json", action="store_true", help="Print JSON result")
    p.add_argument("--print", action="store_true", help="Print sha256 only")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)

    ignore_globs = list(DEFAULT_IGNORE_GLOBS)
    ignore_globs.extend(args.ignore or [])

    res = hash_tree(
        root=root,
        ignore_globs=ignore_globs,
        ignore_dotfiles=(not args.no_ignore_dotfiles),
    )

    if args.print:
        print(res.sha256)
        return 0

    if args.json or True:
        print(
            json.dumps(
                {
                    "root": res.root,
                    "file_count": res.file_count,
                    "total_bytes": res.total_bytes,
                    "sha256": res.sha256,
                    "ignore_dotfiles": (not args.no_ignore_dotfiles),
                    "ignore_globs": ignore_globs,
                },
                sort_keys=True,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
