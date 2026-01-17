from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


@dataclass(frozen=True)
class ManifestFile:
    """A single file entry in a deterministic manifest."""

    # POSIX path, relative to root_dir
    path: str
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


def _to_posix_relpath(value: str, *, base_dir: Path) -> str:
    """Convert an absolute path string to a POSIX relative path from base_dir.

    If value is not absolute, it is returned unchanged.
    """
    try:
        # Quick reject: most non-path strings
        if not value or "\x00" in value:
            return value
        # Windows drive paths count as absolute too.
        is_abs = os.path.isabs(value)
        if not is_abs and len(value) >= 2 and value[1] == ":":
            is_abs = True
        if not is_abs:
            return value

        rel = os.path.relpath(value, start=str(base_dir))
        return rel.replace("\\", "/")
    except Exception:
        # Never raise in a determinism/provenance utility.
        return value


def _scrub_absolute_paths(obj: JSONType, *, base_dir: Path) -> JSONType:
    """Recursively replace absolute path strings inside obj with paths relative to base_dir."""
    if isinstance(obj, str):
        return _to_posix_relpath(obj, base_dir=base_dir)
    if isinstance(obj, list):
        return [_scrub_absolute_paths(x, base_dir=base_dir) for x in obj]
    if isinstance(obj, dict):
        # Preserve key order as provided.
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[k] = _scrub_absolute_paths(v, base_dir=base_dir)
        return out
    return obj


def write_manifest(
    *,
    root_dir: Path,
    out_path: Path,
    generated_at: str,
    algorithm: str = "sha256",
    include_suffixes: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministic directory manifest.

    Goals:
    - Deterministic outputs (stable ordering, stable JSON)
    - Portability: never serialize absolute paths in JSON

    Behavior:
    - Enumerates files under root_dir
    - Hashes each file with sha256
    - Sorts by relative path
    - Computes a "tree" hash over sorted lines: "<sha256>  <path>\n"
    - Writes the manifest JSON to out_path

    Notes:
    - The manifest's "root" field is always "." to avoid leaking host paths.
    - As a final safety net, any absolute path strings anywhere in the JSON are scrubbed
      to be relative to the directory containing out_path.
    """
    root_dir = root_dir.resolve()
    out_path = out_path.resolve()
    out_dir = out_path.parent

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
        # Never serialize absolute paths.
        "root": ".",
        "generated_at": generated_at,
        "file_count": len(files),
        "files": [f.__dict__ for f in files],
        "tree_sha256": tree_sha256,
    }

    manifest = _scrub_absolute_paths(manifest, base_dir=out_dir)  # type: ignore[assignment]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest
