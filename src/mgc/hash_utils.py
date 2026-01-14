#!/usr/bin/env python3
"""
src/mgc/hash_utils.py

Hash + determinism utilities used across the CLI.

Goals:
- Single, stable implementation of common hashing helpers.
- No heavy dependencies.
- Works on Python 3.11+ (CI runner) and newer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """
    SHA256 of a file's bytes.
    Uses chunked reads to support large files.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files_sorted(root: Path) -> List[Path]:
    r = Path(root).resolve()
    files = [p for p in r.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.relative_to(r).as_posix())
    return files


def sha256_tree(root: Path) -> str:
    """
    Deterministic directory tree hash.

    Algorithm:
      - For each file (sorted by relpath):
          line = "<sha256(file)>  <relpath>"
      - sha256 of the concatenated lines with trailing newline
    """
    r = Path(root).resolve()
    files = _iter_files_sorted(r)

    lines: List[str] = []
    for p in files:
        rel = p.relative_to(r).as_posix()
        lines.append(f"{sha256_file(p)}  {rel}")

    joined = ("\n".join(lines) + "\n").encode("utf-8")
    return sha256_bytes(joined)


def sha256_manifest_lines(lines: Iterable[str]) -> str:
    """
    Deterministic hash of manifest-style lines.
    Caller is responsible for sorting if needed.
    """
    joined = ("\n".join(list(lines)) + "\n").encode("utf-8")
    return sha256_bytes(joined)


__all__ = [
    "sha256_bytes",
    "sha256_file",
    "sha256_tree",
    "sha256_manifest_lines",
]