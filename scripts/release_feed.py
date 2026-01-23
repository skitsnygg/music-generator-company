#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_tree(root: Path, *, ignore_dirs: Tuple[str, ...] = (".git",)) -> str:
    """
    Stable tree hash:
    - Walk all files under root
    - Sort by relative posix path
    - Hash each file's bytes and path into a combined digest
    """
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        if any(part in ignore_dirs for part in rel.parts):
            continue
        files.append(p)

    files.sort(key=lambda x: x.relative_to(root).as_posix())

    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        try:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        except Exception:
            # If something is unreadable, still keep deterministic output by hashing marker.
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


def file_mtime_iso(p: Path) -> str:
    try:
        ts = p.stat().st_mtime
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return "1970-01-01T00:00:00Z"


@dataclass(frozen=True)
class WebBundleInfo:
    context: str
    path: Path
    web_manifest: Optional[Path]
    playlist_json: Optional[Path]
    tree_sha256: str
    mtime: str

    def to_dict(self, base_url: str, kind: str, release_id: str) -> Dict[str, Any]:
        # kind: "latest" or "releases"
        # URL shape mirrors nginx: /latest/web/{context} and /releases/{release_id}/web/{context}
        if base_url:
            base_url = base_url.rstrip("/")
        if kind == "latest":
            url_path = f"/latest/web/{self.context}/"
        else:
            url_path = f"/releases/{release_id}/web/{self.context}/"

        return {
            "context": self.context,
            "mtime": self.mtime,
            "tree_sha256": self.tree_sha256,
            "has_web_manifest": bool(self.web_manifest),
            "has_playlist_json": bool(self.playlist_json),
            "url": f"{base_url}{url_path}" if base_url else url_path,
        }


def discover_context_dirs(base: Path) -> List[Tuple[str, Path]]:
    """
    Return [(context, dirpath)] for immediate children that are directories.
    """
    out: List[Tuple[str, Path]] = []
    if not base.exists():
        return out
    try:
        for child in base.iterdir():
            if child.is_dir():
                out.append((child.name, child))
    except Exception:
        return out
    out.sort(key=lambda x: x[0])
    return out


def collect_web_bundle(dirpath: Path, context: str) -> Optional[WebBundleInfo]:
    """
    Accepts a directory that should represent the web root for a given context.
    We consider it "valid enough" if it contains index.html OR web_manifest.json OR playlist.json.
    """
    if not dirpath.exists() or not dirpath.is_dir():
        return None

    index_html = dirpath / "index.html"
    web_manifest = dirpath / "web_manifest.json"
    playlist_json = dirpath / "playlist.json"

    if not (index_html.exists() or web_manifest.exists() or playlist_json.exists()):
        return None

    # Use tree hash over the directory to give an integrity fingerprint.
    tree_hash = sha256_of_tree(dirpath)
    mtime = max(
        file_mtime_iso(index_html) if index_html.exists() else "1970-01-01T00:00:00Z",
        file_mtime_iso(web_manifest) if web_manifest.exists() else "1970-01-01T00:00:00Z",
        file_mtime_iso(playlist_json) if playlist_json.exists() else "1970-01-01T00:00:00Z",
    )

    return WebBundleInfo(
        context=context,
        path=dirpath,
        web_manifest=web_manifest if web_manifest.exists() else None,
        playlist_json=playlist_json if playlist_json.exists() else None,
        tree_sha256=tree_hash,
        mtime=mtime,
    )


def collect_latest(latest_dir: Path, base_url: str) -> Dict[str, Any]:
    """
    latest_dir is expected: .../latest/web
    with children contexts: .../latest/web/<context>
    """
    latest: Dict[str, Any] = {"contexts": []}

    contexts = discover_context_dirs(latest_dir)
    items: List[Dict[str, Any]] = []

    for ctx, ctx_dir in contexts:
        info = collect_web_bundle(ctx_dir, ctx)
        if not info:
            continue
        items.append(info.to_dict(base_url, kind="latest", release_id="latest"))

    # deterministic ordering: context asc
    items.sort(key=lambda x: x["context"])
    latest["contexts"] = items
    return latest


def is_release_id_dir(p: Path) -> bool:
    # Relaxed: accept any dir that isn't hidden and isn't "latest"
    if not p.is_dir():
        return False
    name = p.name
    if name.startswith("."):
        return False
    if name == "latest":
        return False
    return True


def collect_releases(releases_dir: Path, base_url: str, max_items: int) -> List[Dict[str, Any]]:
    """
    releases_dir expected: .../releases
    containing: <release_id>/web/<context>/
    """
    if not releases_dir.exists():
        return []

    candidates = [p for p in releases_dir.iterdir() if is_release_id_dir(p)]
    # deterministic ordering for scan
    candidates.sort(key=lambda p: p.name)

    entries: List[Dict[str, Any]] = []

    for release_dir in candidates:
        release_id = release_dir.name
        web_root = release_dir / "web"
        if not web_root.exists():
            continue

        contexts = discover_context_dirs(web_root)
        ctx_items: List[Dict[str, Any]] = []
        # compute release-level mtime as max of context mtimes
        release_mtime = "1970-01-01T00:00:00Z"

        for ctx, ctx_dir in contexts:
            info = collect_web_bundle(ctx_dir, ctx)
            if not info:
                continue
            d = info.to_dict(base_url, kind="releases", release_id=release_id)
            ctx_items.append(d)
            release_mtime = max(release_mtime, d["mtime"])

        if not ctx_items:
            # allow partial release directories to exist without poisoning the feed
            continue

        # Deterministic ordering: context asc
        ctx_items.sort(key=lambda x: x["context"])

        # Optional: if a release.json exists, include it
        release_meta_path = release_dir / "release.json"
        release_meta = safe_read_json(release_meta_path) if release_meta_path.exists() else None

        entries.append(
            {
                "release_id": release_id,
                "mtime": release_mtime,
                "contexts": ctx_items,
                "has_release_json": bool(release_meta),
                "release_json": release_meta if release_meta else None,
                "url": (f"{base_url.rstrip('/')}/releases/{release_id}/" if base_url else f"/releases/{release_id}/"),
            }
        )

    # Sort newest-first by mtime then by release_id (deterministic)
    entries.sort(key=lambda e: (e["mtime"], e["release_id"]), reverse=True)

    if max_items > 0:
        entries = entries[:max_items]

    return entries


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(obj, sort_keys=True, indent=2) + "\n"
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate internal MGC release feed.json")
    ap.add_argument("--releases-dir", required=True, help="Path to .../releases directory")
    ap.add_argument("--latest-dir", required=True, help="Path to .../latest/web directory")
    ap.add_argument("--out", required=True, help="Output feed.json path")
    ap.add_argument("--base-url", default="", help="Optional base URL (e.g. https://example.com)")
    ap.add_argument("--max-items", type=int, default=200, help="Max release entries")
    args = ap.parse_args()

    releases_dir = Path(args.releases_dir)
    latest_dir = Path(args.latest_dir)
    out_path = Path(args.out)

    base_url = (args.base_url or "").strip()

    feed: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": iso_utc_now(),
        "latest": collect_latest(latest_dir, base_url),
        "releases": collect_releases(releases_dir, base_url, args.max_items),
    }

    # Convenience: "latest_release_id" per context if it can be inferred from the releases list
    # (Not required, but nice in an interview)
    latest_by_context: Dict[str, str] = {}
    for rel in feed["releases"]:
        for ctx in rel["contexts"]:
            c = ctx["context"]
            if c not in latest_by_context:
                latest_by_context[c] = rel["release_id"]
    feed["latest_release_id_by_context"] = latest_by_context

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, feed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
