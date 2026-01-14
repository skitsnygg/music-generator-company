#!/usr/bin/env python3
"""
src/mgc/hash_utils.py

Small deterministic hashing helpers used across the MGC CLI.

Contract:
- sha256_file(path): hex sha256 of file bytes
- sha256_tree(root): deterministic "tree hash" of a directory, stable across OS/filesystems:
    * enumerate files in sorted relpath order
    * for each file: sha256_file(bytes)
    * build lines: "<file_sha256>  <relpath_posix>"
    * tree hash = sha256(concat(lines + trailing newline))
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Union


PathLike = Union[str, Path]


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_text(s: str, *, encoding: str = "utf-8") -> str:
    return sha256_bytes(s.encode(encoding))


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """
    Hash a file's raw bytes. Chunked for large files.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    # Only files, stable order by posix relpath.
    files = [p for p in root.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files


def sha256_tree(root: PathLike) -> str:
    """
    Deterministic directory tree hash.

    The output changes if:
    - any file contents change
    - any file is added/removed
    - any file's relative path changes
    """
    r = Path(root).resolve()
    lines = []
    for p in _iter_files(r):
        rel = p.relative_to(r).as_posix()
        lines.append(f"{sha256_file(p)}  {rel}")
    joined = ("\n".join(lines) + "\n").encode("utf-8")
    return sha256_bytes(joined)


__all__ = [
    "sha256_bytes",
    "sha256_text",
    "sha256_file",
    "sha256_tree",
]
