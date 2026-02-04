#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ALWAYS_EXCLUDE_NAMES = {
    "run",
    "submission",
    "web",          # container dir
    "bundle",
    "marketing",
    "tracks",
    "evidence",
    "drop_bundle",
    "receipts",
}

def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return "1970-01-01T00:00:00Z"

def track_count_from_playlist(dirpath: Path) -> int:
    p = dirpath / "playlist.json"
    obj = read_json(p)
    if not obj:
        return 0
    tracks = obj.get("tracks") or []
    return len(tracks) if isinstance(tracks, list) else 0

def looks_like_web_context_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    return any((d / fn).exists() for fn in ("index.html", "web_manifest.json", "playlist.json"))

def is_context_name_allowed(name: str, *, include_backups: bool) -> bool:
    if name in ALWAYS_EXCLUDE_NAMES:
        return False
    if not include_backups and ".bak." in name:
        return False
    if name.startswith("."):
        return False
    return True

def make_url(base_url: str, path: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    return f"{base_url}{path}" if base_url else path

@dataclass(frozen=True)
class ContextEntry:
    context: str
    kind: str
    mtime: str
    track_count: int
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "kind": self.kind,
            "mtime": self.mtime,
            "track_count": self.track_count,
            "url": self.url,
        }

def discover_web_contexts(web_root: Path, *, include_backups: bool) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not web_root.exists() or not web_root.is_dir():
        return out

    for child in web_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not is_context_name_allowed(name, include_backups=include_backups):
            continue
        if looks_like_web_context_dir(child):
            out.append((name, child))

    out.sort(key=lambda x: x[0])
    return out

def latest_section(latest_dir: Path, *, base_url: str, include_backups: bool) -> Dict[str, Any]:
    web_root = latest_dir / "web"
    contexts: List[Dict[str, Any]] = []

    for ctx, ctx_dir in discover_web_contexts(web_root, include_backups=include_backups):
        contexts.append(
            ContextEntry(
                context=ctx,
                kind="web",
                mtime=mtime_iso(ctx_dir),
                track_count=track_count_from_playlist(ctx_dir),
                url=make_url(base_url, f"/latest/web/{ctx}/"),
            ).to_dict()
        )

    return {"contexts": contexts}

def is_release_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if p.name.startswith("."):
        return False
    if p.name in ("latest", "feed.json"):
        return False
    return True

def releases_section(root_dir: Path, *, base_url: str, max_items: int, include_backups: bool) -> List[Dict[str, Any]]:
    releases: List[Dict[str, Any]] = []
    candidates = [p for p in root_dir.iterdir() if is_release_dir(p)]
    candidates.sort(key=lambda p: p.name)

    for rel in candidates:
        web_root = rel / "web"
        ctxs: List[Dict[str, Any]] = []

        for ctx, ctx_dir in discover_web_contexts(web_root, include_backups=include_backups):
            ctxs.append(
                ContextEntry(
                    context=ctx,
                    kind="web",
                    mtime=mtime_iso(ctx_dir),
                    track_count=track_count_from_playlist(ctx_dir),
                    url=make_url(base_url, f"/releases/{rel.name}/web/{ctx}/"),
                ).to_dict()
            )

        if not ctxs:
            continue

        rel_mtime = "1970-01-01T00:00:00Z"
        for c in ctxs:
            rel_mtime = max(rel_mtime, c["mtime"])

        releases.append(
            {
                "release_id": rel.name,
                "mtime": rel_mtime,
                "contexts": ctxs,
                "url": make_url(base_url, f"/releases/{rel.name}/"),
            }
        )

    releases.sort(key=lambda r: (r["mtime"], r["release_id"]), reverse=True)
    if max_items > 0:
        releases = releases[:max_items]
    return releases

def canonical_content(obj: Dict[str, Any]) -> bytes:
    """
    Canonical bytes for content hashing:
    - remove generated_at (timestamp)
    - remove content_sha256 itself
    - stable JSON with sort_keys and compact separators
    """
    clone = dict(obj)
    clone.pop("generated_at", None)
    clone.pop("content_sha256", None)
    return json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")

def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate MGC internal release feed.json")
    ap.add_argument("--root-dir", default="/var/lib/mgc/releases")
    ap.add_argument("--out", default="/var/lib/mgc/releases/feed.json")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--include-backups", action="store_true")
    ap.add_argument("--stable", action="store_true", help="Omit generated_at for byte-stable output")
    args = ap.parse_args()

    root = Path(args.root_dir)
    out = Path(args.out)

    feed: Dict[str, Any] = {
        "schema_version": 1,
        "latest": latest_section(root / "latest", base_url=args.base_url, include_backups=args.include_backups),
        "releases": releases_section(root, base_url=args.base_url, max_items=args.max_items, include_backups=args.include_backups),
    }

    if not args.stable:
        feed["generated_at"] = iso_now()

    feed["content_sha256"] = hashlib.sha256(canonical_content(feed)).hexdigest()

    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, feed)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
