#!/usr/bin/env python3
"""
src/mgc/evidence_normalize.py

Clean, explicit normalization for evidence JSON.

Why:
- Evidence often contains unstable fields (timestamps, absolute paths, host/pid,
  durations, wall times, random IDs, etc.).
- Weekly determinism should compare stable artifacts only.

Design:
- remove known-unstable keys
- canonicalize ordering (dict keys sorted; lists stabilized when possible)
- normalize paths (absolute -> basename or repo-relative)
- stabilize floats (rounded)
- ensure JSON-serializable output

Usage:
  from mgc.evidence_normalize import normalize_evidence

  stable = normalize_evidence(obj)
  json.dump(stable, f, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

You should call this *right before writing* daily_evidence.json / drop_evidence.json /
weekly_evidence.json (or equivalent).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# Keys that are almost always unstable across runs.
# Add/remove based on what your evidence schema actually emits.
UNSTABLE_KEYS = {
    # time-ish
    "ts",
    "timestamp",
    "created_ts",
    "updated_ts",
    "started_ts",
    "finished_ts",
    "published_ts",
    "generated_ts",
    "now",
    "utc_now",
    # runtime/perf-ish
    "elapsed_ms",
    "elapsed_sec",
    "duration_ms",
    "duration_sec",
    "wall_time_sec",
    "cpu_time_sec",
    # env/system-ish
    "pid",
    "ppid",
    "hostname",
    "host",
    "user",
    "cwd",
    "python",
    "python_version",
    "platform",
    "uname",
    "git_sha",      # if you want evidence independent of git state
    "git_branch",   # same
    # volatile IDs (often regenerated even when content is same)
    "run_id",
    "drop_id",
    "marketing_batch_id",
    "batch_id",
    # file system specifics
    "tmp_dir",
    "temp_dir",
}

# Keys that often hold absolute paths.
PATH_KEYS = {
    "path",
    "paths",
    "artifact_path",
    "evidence_path",
    "manifest_path",
    "out_dir",
    "bundle_dir",
    "root",
    "dir",
    "file",
    "filename",
}

# If you have keys where the value is a dict of paths, normalize that too
PATH_DICT_KEYS = {"paths"}

# Float rounding for stability
FLOAT_DECIMALS = 6


def _is_pathlike_key(k: str) -> bool:
    lk = k.lower()
    if lk in PATH_KEYS:
        return True
    # Heuristic: anything ending with _path/_dir tends to be a path
    return lk.endswith("_path") or lk.endswith("_dir") or lk.endswith("_file")


def _normalize_path_value(v: Any, *, repo_root: str | None) -> Any:
    if isinstance(v, str):
        # If it's an absolute path, try to rewrite to repo-relative.
        # Otherwise, keep as-is.
        p = Path(v)
        if p.is_absolute():
            if repo_root:
                try:
                    rp = p.relative_to(Path(repo_root))
                    return str(rp.as_posix())
                except Exception:
                    # Fall back to basename to avoid machine-specific paths
                    return p.name
            return p.name
        return v
    if isinstance(v, list):
        return [_normalize_path_value(x, repo_root=repo_root) for x in v]
    if isinstance(v, dict):
        return {str(k): _normalize_path_value(val, repo_root=repo_root) for k, val in v.items()}
    return v


def _round_float(x: float) -> float:
    return float(f"{x:.{FLOAT_DECIMALS}f}")


def _normalize_scalar(v: Any) -> Any:
    # Keep ints/bools/None/strings as-is.
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return _round_float(v)
    return v


def _try_stabilize_list(items: List[Any]) -> List[Any]:
    """
    Attempt to stabilize list ordering when items are dicts with a stable key.
    If we can't safely sort, keep original order.
    """
    if not items:
        return items
    # If all dicts and share a common stable ID, sort by it
    if all(isinstance(x, dict) for x in items):
        for key in ("track_id", "id", "artifact_path", "path", "name", "key"):
            if all(key in x for x in items):  # type: ignore[operator]
                try:
                    return sorted(items, key=lambda d: str(d.get(key, "")))  # type: ignore[arg-type]
                except Exception:
                    return items
    # If all scalars, sort strings/ints deterministically
    if all(isinstance(x, (str, int, float, bool)) or x is None for x in items):
        try:
            return sorted(items, key=lambda x: (str(type(x)), str(x)))
        except Exception:
            return items
    return items


def normalize_evidence(obj: Any, *, repo_root: str | None = None) -> Any:
    """
    Normalize an evidence JSON-like object for determinism.

    repo_root:
      - If provided, absolute paths under repo_root are rewritten to repo-relative.
      - Otherwise absolute paths are reduced to basename.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in UNSTABLE_KEYS:
                continue

            # Normalize nested dict-of-paths
            if k in PATH_DICT_KEYS and isinstance(v, dict):
                v = _normalize_path_value(v, repo_root=repo_root)

            # Normalize direct path-like keys
            if _is_pathlike_key(str(k)):
                v = _normalize_path_value(v, repo_root=repo_root)

            out[str(k)] = normalize_evidence(v, repo_root=repo_root)

        # Return dict with stable key ordering (actual JSON dump should also sort_keys=True)
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    if isinstance(obj, list):
        norm = [normalize_evidence(x, repo_root=repo_root) for x in obj]
        norm = _try_stabilize_list(norm)
        return norm

    return _normalize_scalar(obj)


def normalize_daily_evidence(d: Dict[str, Any], *, repo_root: str | None = None) -> Dict[str, Any]:
    """
    Convenience wrapper if you want to tailor daily-specific rules later.
    """
    return normalize_evidence(d, repo_root=repo_root)


def normalize_drop_evidence(d: Dict[str, Any], *, repo_root: str | None = None) -> Dict[str, Any]:
    """
    Convenience wrapper if you want to tailor drop-specific rules later.
    """
    return normalize_evidence(d, repo_root=repo_root)
