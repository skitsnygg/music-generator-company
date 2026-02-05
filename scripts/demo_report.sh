#!/usr/bin/env bash
set -euo pipefail

echo "[demo_report] starting demo report"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MGC_DEMO_NO_SUDO="${MGC_DEMO_NO_SUDO:-0}"
MGC_PYTHON="${MGC_PYTHON:-}"

if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  export MGC_OUT_BASE="${MGC_OUT_BASE:-${REPO_ROOT}/data/local_demo_evidence}"
  export MGC_WEB_LATEST_ROOT="${MGC_WEB_LATEST_ROOT:-${REPO_ROOT}/data/releases/latest/web}"
  export MGC_RELEASE_ROOT="${MGC_RELEASE_ROOT:-${REPO_ROOT}/data/releases}"
  export MGC_RELEASE_FEED_OUT="${MGC_RELEASE_FEED_OUT:-${REPO_ROOT}/data/releases/feed.json}"
fi

if [[ -z "${MGC_RELEASE_ROOT:-}" ]]; then
  export MGC_RELEASE_ROOT="/var/lib/mgc/releases"
fi
if [[ -z "${MGC_WEB_LATEST_ROOT:-}" ]]; then
  export MGC_WEB_LATEST_ROOT="${MGC_RELEASE_ROOT}/latest/web"
fi
if [[ -z "${MGC_RELEASE_FEED_OUT:-}" ]]; then
  export MGC_RELEASE_FEED_OUT="${MGC_RELEASE_ROOT}/feed.json"
fi

if [[ -z "${MGC_PYTHON}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    MGC_PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    MGC_PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi

if [[ -z "${MGC_PYTHON}" ]]; then
  echo "[demo_report] ERROR: python not found (set MGC_PYTHON)" >&2
  exit 2
fi

FEED_PATH="${MGC_FEED_PATH:-${MGC_RELEASE_FEED_OUT}}"
OUT_BASE="${MGC_OUT_BASE:-}"

"${MGC_PYTHON}" - "${FEED_PATH}" "${MGC_WEB_LATEST_ROOT}" "${OUT_BASE}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

feed_path = Path(sys.argv[1])
web_root = Path(sys.argv[2])
out_base_raw = sys.argv[3].strip()
out_base = Path(out_base_raw) if out_base_raw else None

def iso_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return "n/a"

def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def count_audio(root: Path) -> int:
    if not root.exists():
        return 0
    count = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".mp3", ".wav"):
            count += 1
    return count

def maybe_get(d, key, default="n/a"):
    if isinstance(d, dict) and key in d and d[key] is not None:
        return d[key]
    return default

def report_text(feed):
    if feed is None:
        print("[demo_report] feed: missing")
        return []

    latest = feed.get("latest") or {}
    contexts = latest.get("contexts") or []
    releases = feed.get("releases") or []

    size = feed_path.stat().st_size if feed_path.exists() else 0
    print(f"[demo_report] feed: {feed_path} size={size} mtime={iso_mtime(feed_path)}")
    print(f"[demo_report] feed.generated_at: {maybe_get(feed, 'generated_at')}")
    print(f"[demo_report] feed.content_sha256: {maybe_get(feed, 'content_sha256')}")
    print(f"[demo_report] feed.latest.contexts: {len(contexts)}")
    print(f"[demo_report] feed.releases: {len(releases)}")
    if out_base:
        print(f"[demo_report] evidence_root: {out_base}")

    return contexts

def report_context(ctx_obj):
    if not isinstance(ctx_obj, dict):
        return
    ctx = str(ctx_obj.get("context") or "")
    if not ctx:
        return

    ctx_dir = web_root / ctx
    manifest = read_json(ctx_dir / "web_manifest.json") if ctx_dir.exists() else None
    playlist = read_json(ctx_dir / "playlist.json") if ctx_dir.exists() else None

    manifest_tracks = manifest.get("tracks") if isinstance(manifest, dict) else None
    manifest_track_count = len(manifest_tracks) if isinstance(manifest_tracks, list) else "n/a"

    audio_count = count_audio(ctx_dir)

    # Evidence (if present under out_base)
    drop_id = "n/a"
    run_id = "n/a"
    if out_base and out_base.exists():
        drop_ev = read_json(out_base / ctx / "drop_evidence.json")
        daily_ev = read_json(out_base / ctx / "drop_bundle" / "daily_evidence.json")
        drop_id = maybe_get(drop_ev, "drop_id")
        run_id = maybe_get(daily_ev, "run_id")

    print(f"[demo_report] context={ctx} feed_track_count={maybe_get(ctx_obj, 'track_count')} feed_mtime={maybe_get(ctx_obj, 'mtime')}")
    print(f"[demo_report]  web_dir={ctx_dir} audio_files={audio_count} manifest_version={maybe_get(manifest, 'version')} bundle_audio={maybe_get(manifest, 'bundle_audio')}")
    print(f"[demo_report]  manifest_tracks={manifest_track_count} playlist_sha256={maybe_get(manifest, 'playlist_sha256')} web_tree_sha256={maybe_get(manifest, 'web_tree_sha256')}")
    print(f"[demo_report]  drop_id={drop_id} run_id={run_id}")

feed = read_json(feed_path) if feed_path.exists() else None
contexts = report_text(feed)

if not contexts:
    print("[demo_report] WARN: no contexts in feed")
else:
    for ctx in contexts:
        report_context(ctx)

if os.environ.get("MGC_REPORT_JSON") == "1":
    summary = {
        "feed_path": str(feed_path),
        "feed": feed,
        "web_root": str(web_root),
        "out_base": str(out_base) if out_base else "",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
PY
