from __future__ import annotations

import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


VOLATILE_KEYS = {"generated_at", "exported_at", "built_at", "timestamp", "host"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _eprint(f"[manifest diff] ERROR: missing file: {path}")
        raise SystemExit(2)
    except json.JSONDecodeError as e:
        _eprint(f"[manifest diff] ERROR: invalid JSON in {path}: {e}")
        raise SystemExit(2)


def _load_json_from_git(rev: str, repo_rel_path: str) -> Any:
    """
    Load JSON blob from git, e.g. rev='HEAD', path='data/playlists/_manifest.playlists.json'
    """
    if not repo_rel_path or repo_rel_path.startswith("/"):
        _eprint("[manifest diff] ERROR: git path must be repo-relative (no leading slash)")
        raise SystemExit(2)

    if not _has_git():
        _eprint("[manifest diff] ERROR: git is not available; cannot read committed manifest from git")
        raise SystemExit(2)

    spec = f"{rev}:{repo_rel_path}"
    try:
        out = subprocess.check_output(["git", "show", spec], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        msg = (e.output or b"").decode("utf-8", errors="replace")
        _eprint(f"[manifest diff] ERROR: git show failed for {spec}\n{msg.strip()}")
        raise SystemExit(2)

    try:
        return json.loads(out.decode("utf-8"))
    except json.JSONDecodeError as je:
        _eprint(f"[manifest diff] ERROR: committed manifest in git is not valid JSON: {spec}: {je}")
        raise SystemExit(2)


def _has_git() -> bool:
    try:
        subprocess.check_output(["git", "--version"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _normalize(x: Any) -> Any:
    if isinstance(x, dict):
        out = {}
        for k, v in sorted(x.items(), key=lambda kv: str(kv[0])):
            ks = str(k)
            if ks in VOLATILE_KEYS:
                continue
            out[ks] = _normalize(v)
        return out
    if isinstance(x, list):
        norm = [_normalize(v) for v in x]
        # stable ordering for diff readability/determinism
        return sorted(norm, key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
    return x


def diff_manifest(committed_path: str, generated_path: str) -> int:
    """
    Compare committed vs generated manifests.

    committed_path:
      - a normal file path (JSON), OR
      - "git:REV" to read from git, where generated_path is the repo-relative file path to read.

        Example:
          committed_path = "git:HEAD"
          generated_path = "data/playlists/_manifest.playlists.json"
    """
    print("[manifest diff] comparing generated manifest vs committed")

    committed_obj: Optional[Any] = None

    if committed_path.startswith("git:"):
        rev = committed_path.split(":", 1)[1].strip() or "HEAD"
        repo_rel = generated_path.strip().lstrip("./")
        committed_obj = _normalize(_load_json_from_git(rev, repo_rel))
        generated_obj = _normalize(_load_json_file(Path(generated_path)))
        committed_label = f"{rev}:{repo_rel}"
        generated_label = generated_path
    else:
        committed_obj = _normalize(_load_json_file(Path(committed_path)))
        generated_obj = _normalize(_load_json_file(Path(generated_path)))
        committed_label = committed_path
        generated_label = generated_path

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
