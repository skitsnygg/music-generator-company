from __future__ import annotations

import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Tuple


DEFAULT_GENERATED = "data/playlists/_manifest.playlists.json"
DEFAULT_COMMITTED = "git:HEAD"

VOLATILE_KEYS = {"generated_at", "exported_at", "built_at", "timestamp", "host"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _die(msg: str, code: int = 2) -> "NoReturn":
    _eprint(msg)
    raise SystemExit(code)


def _has_git() -> bool:
    try:
        subprocess.check_output(["git", "--version"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _die(f"[manifest diff] ERROR: missing file: {path}", 2)
    except json.JSONDecodeError as e:
        _die(f"[manifest diff] ERROR: invalid JSON in {path}: {e}", 2)


def _load_json_from_git(rev: str, repo_rel_path: str) -> Any:
    """
    Load JSON blob from git, e.g. rev='HEAD', path='data/playlists/_manifest.playlists.json'
    """
    if not repo_rel_path or repo_rel_path.startswith("/"):
        _die("[manifest diff] ERROR: git path must be repo-relative (no leading slash)", 2)

    if not _has_git():
        _die("[manifest diff] ERROR: git is not available; cannot read committed manifest from git", 2)

    spec = f"{rev}:{repo_rel_path}"
    try:
        out = subprocess.check_output(["git", "show", spec], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = (e.output or b"").decode("utf-8", errors="replace").strip()
        _die(f"[manifest diff] ERROR: git show failed for {spec}\n{msg}", 2)

    try:
        return json.loads(out.decode("utf-8"))
    except json.JSONDecodeError as je:
        _die(f"[manifest diff] ERROR: committed manifest in git is not valid JSON: {spec}: {je}", 2)


def _normalize(x: Any) -> Any:
    """
    Normalize to reduce meaningless diffs:
      - drop volatile keys
      - sort dict keys
      - produce stable list ordering (best-effort)
    """
    if isinstance(x, dict):
        out: dict[str, Any] = {}
        for k, v in sorted(x.items(), key=lambda kv: str(kv[0])):
            ks = str(k)
            if ks in VOLATILE_KEYS:
                continue
            out[ks] = _normalize(v)
        return out

    if isinstance(x, list):
        norm = [_normalize(v) for v in x]

        # If list elements are dicts/lists, sort by JSON string to stabilize ordering.
        # If they're primitives, preserve original order (often meaningful).
        if all(isinstance(v, (dict, list)) for v in norm):
            return sorted(norm, key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))

        return norm

    return x


def _resolve_committed_and_generated(
    committed_path: str, generated_path: str
) -> Tuple[Any, Any, str, str]:
    """
    Returns: committed_obj, generated_obj, committed_label, generated_label
    """
    committed_path = (committed_path or DEFAULT_COMMITTED).strip()
    generated_path = (generated_path or DEFAULT_GENERATED).strip()

    if committed_path.startswith("git:"):
        rev = committed_path.split(":", 1)[1].strip() or "HEAD"
        repo_rel = generated_path.strip().lstrip("./")
        committed_obj = _normalize(_load_json_from_git(rev, repo_rel))
        generated_obj = _normalize(_load_json_file(Path(generated_path)))
        committed_label = f"{rev}:{repo_rel}"
        generated_label = generated_path
        return committed_obj, generated_obj, committed_label, generated_label

    committed_obj = _normalize(_load_json_file(Path(committed_path)))
    generated_obj = _normalize(_load_json_file(Path(generated_path)))
    return committed_obj, generated_obj, committed_path, generated_path


def diff_manifest(committed_path: str = DEFAULT_COMMITTED, generated_path: str = DEFAULT_GENERATED) -> int:
    """
    Compare committed vs generated manifests.

    committed_path:
      - normal file path (JSON), OR
      - "git:REV" to read committed version from git, where generated_path is the repo-relative path.

    Examples:
      diff_manifest()  # git:HEAD vs data/playlists/_manifest.playlists.json
      diff_manifest("git:HEAD~1", "data/playlists/_manifest.playlists.json")
      diff_manifest("data/playlists/_manifest.playlists.json", "data/playlists/_manifest.playlists.json")
    """
    print("[manifest diff] comparing generated manifest vs committed")

    committed_obj, generated_obj, committed_label, generated_label = _resolve_committed_and_generated(
        committed_path, generated_path
    )

    print(f"[manifest diff] committed: {committed_label}")
    print(f"[manifest diff] generated: {generated_label}")

    if committed_obj == generated_obj:
        print("[manifest diff] OK — no differences")
        return 0

    print("[manifest diff] DIFF — committed vs generated\n")

    a = json.dumps(committed_obj, indent=2, sort_keys=True, ensure_ascii=False).splitlines()
    b = json.dumps(generated_obj, indent=2, sort_keys=True, ensure_ascii=False).splitlines()

    for line in difflib.unified_diff(
        a,
        b,
        fromfile=committed_label,
        tofile=generated_label,
        lineterm="",
    ):
        print(line)

    print("\n[manifest diff] Fix: regenerate and commit the manifest, or chase nondeterminism.")
    return 1
