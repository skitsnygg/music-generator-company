from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ManifestFile:
    path: str  # posix path, relative to root_dir
    sha256: str
    bytes: int


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


def write_manifest(
    *,
    root_dir: Path,
    out_path: Path,
    generated_at: str,
    algorithm: str = "sha256",
    include_suffixes: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Deterministic directory manifest.

    - Enumerates files under root_dir
    - Hashes each file with sha256
    - Sorts by relative path
    - Computes a "tree" hash over the sorted lines:
        "<sha256>  <path>\\n"
    - Writes the manifest JSON to out_path
    """
    root_dir = root_dir.resolve()
    out_path = out_path.resolve()

    include_suffixes = include_suffixes or [".json"]
    exclude = set(exclude_names or [])
    exclude.add(out_path.name)  # never hash the manifest itself

    files: List[ManifestFile] = []

    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name in exclude:
            continue
        if include_suffixes and p.suffix not in include_suffixes:
            continue

        rel = p.relative_to(root_dir).as_posix()
        digest, nbytes = _sha256_file(p)
        files.append(ManifestFile(path=rel, sha256=digest, bytes=nbytes))

    files.sort(key=lambda f: f.path)

    tree_lines = "".join([f"{f.sha256}  {f.path}\n" for f in files]).encode("utf-8")
    tree_sha256 = _sha256_bytes(tree_lines)

    manifest: Dict[str, Any] = {
        "version": 1,
        "algorithm": algorithm,
        "root": str(root_dir),
        "generated_at": generated_at,
        "file_count": len(files),
        "files": [f.__dict__ for f in files],
        "tree_sha256": tree_sha256,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    return manifest
