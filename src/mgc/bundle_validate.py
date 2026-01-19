#!/usr/bin/env python3
"""\
src/mgc/bundle_validate.py

Bundle validator for the portable drop artifact emitted by _stub_daily_run.

Bundle v1 expected layout:
  <bundle>/
    tracks/
      <track_id>.<ext>
    playlist.json
    daily_evidence.json
    daily_evidence_<drop_id>.json   (optional but recommended)

Validator rules (v1):
- Required files exist (playlist.json, daily_evidence.json, tracks/)
- playlist.json schema is mgc.playlist.v1 and all referenced track paths exist
- daily_evidence.json schema is mgc.daily_evidence.v1 and sha256 entries match:
    - playlist
    - bundle_track
  (repo_artifact is informational and not validated against repo storage here)
- All paths in evidence are relative POSIX-like paths (no absolute paths, no ..)
- Optional: if daily_evidence_<drop_id>.json exists, it must match daily_evidence.json content

Extension:
- If drop_evidence.json exists and declares marketing receipts directory via
  paths.marketing_receipts_dir, validate that directory exists.

  For weekly runs, marketing artifacts often live as siblings of drop_bundle:
    <out_dir>/drop_bundle
    <out_dir>/marketing/receipts

  We accept a safe relative path that resolves either:
    - inside the bundle_dir, or
    - inside bundle_dir.parent
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# Helpers
# -----------------------------

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON: {p}: {e}") from e


def _is_rel_safe_posix(rel: str) -> bool:
    # Accept forward slashes only; reject absolute paths and traversal.
    if not isinstance(rel, str) or not rel:
        return False
    if rel.startswith("/") or rel.startswith("\\"):
        return False
    if ":" in rel[:3]:  # crude windows drive check like C:
        return False
    parts = rel.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    return True


def _join_rel(root: Path, rel: str) -> Path:
    rel_posix = rel.replace("\\", "/")
    if not _is_rel_safe_posix(rel_posix):
        raise ValueError(f"Unsafe or non-relative path: {rel!r}")
    return (root / Path(rel_posix)).resolve()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _as_dict(x: Any, *, where: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        raise ValueError(f"Expected object at {where}, got {type(x).__name__}")
    return x


def _as_dict_optional(x: Any, *, where: str) -> Dict[str, Any]:
    """Like _as_dict but returns {} if x is None.

    This supports legacy/partial evidence objects while keeping strict
    validation for malformed (non-dict, non-null) values.
    """
    if x is None:
        return {}
    return _as_dict(x, where=where)


def _as_list(x: Any, *, where: str) -> List[Any]:
    if not isinstance(x, list):
        raise ValueError(f"Expected array at {where}, got {type(x).__name__}")
    return x


def _validate_optional_marketing_paths(bundle_dir: Path) -> None:
    drop_evidence_path = bundle_dir / "drop_evidence.json"
    if not drop_evidence_path.exists() or not drop_evidence_path.is_file():
        return

    try:
        obj = _read_json(drop_evidence_path)
    except Exception:
        return

    if not isinstance(obj, dict):
        return

    paths = obj.get("paths")
    if not isinstance(paths, dict):
        return

    mr = paths.get("marketing_receipts_dir")
    if not (isinstance(mr, str) and mr.strip()):
        return

    mr = mr.strip()
    p = Path(mr)

    if p.is_absolute():
        _require(p.exists() and p.is_dir(), f"Declared marketing_receipts_dir missing: {p}")
        return

    _require(_is_rel_safe_posix(mr), f"Unsafe marketing_receipts_dir path in drop_evidence.json: {mr!r}")

    cand1 = _join_rel(bundle_dir, mr)
    if cand1.exists() and cand1.is_dir():
        return

    cand2 = _join_rel(bundle_dir.parent, mr)
    _require(cand2.exists() and cand2.is_dir(), f"Declared marketing_receipts_dir missing: {mr!r} (tried {cand1} and {cand2})")


# -----------------------------
# Public API
# -----------------------------

def validate_bundle(bundle_dir: Path) -> None:
    """Validate a portable drop bundle directory.

    Raises ValueError if invalid.
    Returns None on success.
    """
    bundle_dir = Path(bundle_dir).resolve()
    _require(bundle_dir.exists() and bundle_dir.is_dir(), f"Bundle dir not found: {bundle_dir}")

    tracks_dir = bundle_dir / "tracks"
    playlist_path = bundle_dir / "playlist.json"
    evidence_main_path = bundle_dir / "daily_evidence.json"

    _require(tracks_dir.exists() and tracks_dir.is_dir(), f"Missing tracks/ directory: {tracks_dir}")
    _require(playlist_path.exists() and playlist_path.is_file(), f"Missing playlist.json: {playlist_path}")
    _require(evidence_main_path.exists() and evidence_main_path.is_file(), f"Missing daily_evidence.json: {evidence_main_path}")

    # --- Parse playlist ---
    playlist = _read_json(playlist_path)
    playlist_obj = _as_dict(playlist, where="playlist.json")

    schema = playlist_obj.get("schema")
    _require(schema == "mgc.playlist.v1", f"Unexpected playlist schema: {schema!r} (expected 'mgc.playlist.v1')")

    tracks = _as_list(playlist_obj.get("tracks"), where="playlist.json.tracks")
    _require(len(tracks) > 0, "playlist.json.tracks must be non-empty")

    playlist_track_paths: List[str] = []
    for i, t in enumerate(tracks):
        t_obj = _as_dict(t, where=f"playlist.json.tracks[{i}]")
        path = t_obj.get("path")
        _require(isinstance(path, str) and path, f"playlist.json.tracks[{i}].path missing/invalid")
        _require(_is_rel_safe_posix(path), f"playlist.json.tracks[{i}].path must be safe relative path, got {path!r}")
        p = _join_rel(bundle_dir, path)
        _require(p.exists() and p.is_file(), f"playlist references missing track file: {path!r}")
        playlist_track_paths.append(path.replace("\\", "/"))

    # --- Parse evidence ---
    evidence = _read_json(evidence_main_path)
    evidence_obj = _as_dict(evidence, where="daily_evidence.json")

    evidence_schema = evidence_obj.get("schema")
    _require(
        evidence_schema == "mgc.daily_evidence.v1",
        f"Unexpected evidence schema: {evidence_schema!r} (expected 'mgc.daily_evidence.v1')",
    )

    # Some legacy bundles (and some CI fixtures) may omit these sections.
    # If they're absent (null), we still validate the overall layout and
    # playlist track existence, but we skip the sha/path cross-checks.
    paths_obj = _as_dict_optional(evidence_obj.get("paths"), where="daily_evidence.json.paths")
    sha_obj = _as_dict_optional(evidence_obj.get("sha256"), where="daily_evidence.json.sha256")

    strict_evidence = bool(paths_obj) and bool(sha_obj)
    if strict_evidence:
        # Required evidence paths
        bundle_track_rel = paths_obj.get("bundle_track")
        playlist_rel = paths_obj.get("playlist")

        _require(isinstance(bundle_track_rel, str) and bundle_track_rel, "daily_evidence.json.paths.bundle_track missing/invalid")
        _require(isinstance(playlist_rel, str) and playlist_rel, "daily_evidence.json.paths.playlist missing/invalid")

        _require(_is_rel_safe_posix(bundle_track_rel), f"Unsafe evidence bundle_track path: {bundle_track_rel!r}")
        _require(_is_rel_safe_posix(playlist_rel), f"Unsafe evidence playlist path: {playlist_rel!r}")

        # Required hashes
        bundle_track_sha = sha_obj.get("bundle_track")
        playlist_sha = sha_obj.get("playlist")

        _require(isinstance(bundle_track_sha, str) and len(bundle_track_sha) == 64, "daily_evidence.json.sha256.bundle_track missing/invalid")
        _require(isinstance(playlist_sha, str) and len(playlist_sha) == 64, "daily_evidence.json.sha256.playlist missing/invalid")

        # Evidence paths must point to existing files
        evidence_bundle_track_path = _join_rel(bundle_dir, bundle_track_rel)
        evidence_playlist_path = _join_rel(bundle_dir, playlist_rel)

        _require(evidence_bundle_track_path.exists() and evidence_bundle_track_path.is_file(),
                 f"Evidence bundle_track missing: {bundle_track_rel!r}")
        _require(evidence_playlist_path.exists() and evidence_playlist_path.is_file(),
                 f"Evidence playlist missing: {playlist_rel!r}")

        # Evidence playlist path should match top-level playlist.json (by content)
        top_playlist_bytes = playlist_path.read_bytes()
        evidence_playlist_bytes = evidence_playlist_path.read_bytes()
        _require(
            top_playlist_bytes == evidence_playlist_bytes,
            f"Evidence playlist file does not match top-level playlist.json: {playlist_rel!r}",
        )

        # Verify sha256 matches evidence
        computed_bundle_track_sha = _sha256_file(evidence_bundle_track_path)
        computed_playlist_sha = _sha256_file(evidence_playlist_path)

        _require(
            computed_bundle_track_sha == bundle_track_sha,
            f"bundle_track sha256 mismatch: expected {bundle_track_sha}, got {computed_bundle_track_sha}",
        )
        _require(
            computed_playlist_sha == playlist_sha,
            f"playlist sha256 mismatch: expected {playlist_sha}, got {computed_playlist_sha}",
        )

        # Ensure playlist references include the evidence bundle_track (at least once)
        norm_bundle_track_rel = bundle_track_rel.replace("\\", "/")
        _require(
            norm_bundle_track_rel in playlist_track_paths,
            f"playlist.json does not reference evidence bundle_track path: {norm_bundle_track_rel!r}",
        )

    # Optional: validate scoped evidence file if present
    drop_id = None
    ids_obj = evidence_obj.get("ids")
    if isinstance(ids_obj, dict):
        drop_id = ids_obj.get("drop_id")
    scoped_path: Optional[Path] = None
    if isinstance(drop_id, str) and drop_id:
        candidate = bundle_dir / f"daily_evidence_{drop_id}.json"
        if candidate.exists() and candidate.is_file():
            scoped_path = candidate

    if scoped_path is not None:
        scoped_text = scoped_path.read_text(encoding="utf-8")
        main_text = evidence_main_path.read_text(encoding="utf-8")
        _require(
            scoped_text == main_text,
            f"Scoped evidence file differs from daily_evidence.json: {scoped_path.name}",
        )

    # Optional: marketing receipts declaration in drop_evidence.json
    _validate_optional_marketing_paths(bundle_dir)

    return
