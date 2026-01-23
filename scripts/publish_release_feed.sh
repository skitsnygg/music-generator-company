#!/usr/bin/env bash
set -euo pipefail

# Publish a per-context release folder + update global feed.
#
# Inputs:
#   --context focus|workout|sleep
#   --period-key YYYY-MM-DD or YYYY-W## (optional; defaults to UTC date)
#   --src-latest-root /var/lib/mgc/releases/latest/web  (optional)
#   --dest-releases-root /var/lib/mgc/releases         (optional)
#   --out-dir <run output dir> (optional; used only to locate teaser + marketing files)
#
# Produces:
#   /var/lib/mgc/releases/<period>/<context>/index.html
#   /var/lib/mgc/releases/<period>/<context>/release.json
#   /var/lib/mgc/releases/index.json  (feed)

usage() {
  echo "usage: $0 --context CONTEXT [--period-key KEY] [--out-dir DIR] [--src-latest-root DIR] [--dest-releases-root DIR]" >&2
  exit 2
}

CONTEXT=""
PERIOD_KEY=""
OUT_DIR=""
SRC_LATEST_ROOT="${MGC_WEB_LATEST_ROOT:-/var/lib/mgc/releases/latest/web}"
DEST_RELEASES_ROOT="${MGC_RELEASES_ROOT:-/var/lib/mgc/releases}"

now_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
date_utc() { date -u +"%Y-%m-%d"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context) CONTEXT="${2:-}"; shift 2;;
    --period-key) PERIOD_KEY="${2:-}"; shift 2;;
    --out-dir) OUT_DIR="${2:-}"; shift 2;;
    --src-latest-root) SRC_LATEST_ROOT="${2:-}"; shift 2;;
    --dest-releases-root) DEST_RELEASES_ROOT="${2:-}"; shift 2;;
    -h|--help) usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

[[ -n "$CONTEXT" ]] || usage
[[ -n "$PERIOD_KEY" ]] || PERIOD_KEY="$(date_utc)"

SRC_BUNDLE="${SRC_LATEST_ROOT%/}/${CONTEXT}"
DEST_DIR="${DEST_RELEASES_ROOT%/}/${PERIOD_KEY}/${CONTEXT}"
FEED_PATH="${DEST_RELEASES_ROOT%/}/index.json"

if [[ ! -d "$SRC_BUNDLE" ]]; then
  echo "[publish_release_feed] missing source bundle: $SRC_BUNDLE" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

# Copy the already-built web bundle (index.html + tracks/ + web_manifest.json + playlist.json)
# Use rsync for clean overwrite.
rsync -a --delete "${SRC_BUNDLE%/}/" "${DEST_DIR%/}/"

# Optional: include teaser + marketing plan if they exist in OUT_DIR
if [[ -n "$OUT_DIR" ]]; then
  if [[ -f "${OUT_DIR%/}/marketing/teaser.wav" ]]; then
    mkdir -p "${DEST_DIR%/}/marketing"
    cp -f "${OUT_DIR%/}/marketing/teaser.wav" "${DEST_DIR%/}/marketing/teaser.wav"
  fi
  if [[ -f "${OUT_DIR%/}/marketing/marketing_plan.json" ]]; then
    mkdir -p "${DEST_DIR%/}/marketing"
    cp -f "${OUT_DIR%/}/marketing/marketing_plan.json" "${DEST_DIR%/}/marketing/marketing_plan.json"
  fi
  if [[ -f "${OUT_DIR%/}/drop_bundle/playlist.json" ]]; then
    mkdir -p "${DEST_DIR%/}/bundle"
    cp -f "${OUT_DIR%/}/drop_bundle/playlist.json" "${DEST_DIR%/}/bundle/playlist.json"
  fi
fi

# Build release.json from the bundle’s playlist.json (keep it simple + robust).
python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

context = ${CONTEXT!r}
period_key = ${PERIOD_KEY!r}
dest_dir = Path(${DEST_DIR!r})
playlist_path = dest_dir / "playlist.json"  # publish_latest writes this
wm_path = dest_dir / "web_manifest.json"

playlist = {}
if playlist_path.exists():
    playlist = json.loads(playlist_path.read_text(encoding="utf-8"))

release = {
    "ok": True,
    "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
    "period_key": period_key,
    "context": context,
    "paths": {
        "web": f"/releases/{period_key}/{context}/",
        "playlist_json": f"/releases/{period_key}/{context}/playlist.json",
        "web_manifest_json": f"/releases/{period_key}/{context}/web_manifest.json",
    },
    "playlist": {
        "track_count": len(playlist.get("tracks") or []),
        "tracks": playlist.get("tracks") or [],
    },
}

(dest_dir / "release.json").write_text(json.dumps(release, indent=2, sort_keys=True), encoding="utf-8")
print("wrote", dest_dir / "release.json")
PY

# Update /var/lib/mgc/releases/index.json (append newest first, de-dupe by period+context)
python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

feed_path = Path(${FEED_PATH!r})
period_key = ${PERIOD_KEY!r}
context = ${CONTEXT!r}
release_rel = f"{period_key}/{context}/release.json"
release_abs = Path(${DEST_DIR!r}) / "release.json"

release = json.loads(release_abs.read_text(encoding="utf-8"))

feed = {"ok": True, "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "items": []}
if feed_path.exists():
    try:
        feed = json.loads(feed_path.read_text(encoding="utf-8"))
    except Exception:
        pass

items = feed.get("items") or []
key = f"{period_key}:{context}"

def item_key(it):
    return f"{it.get('period_key','')}:{it.get('context','')}"

# Remove existing
items = [it for it in items if item_key(it) != key]

# New item (small)
item = {
    "period_key": period_key,
    "context": context,
    "ts_utc": release.get("ts_utc"),
    "track_count": (release.get("playlist") or {}).get("track_count", 0),
    "web": (release.get("paths") or {}).get("web"),
    "release_json": f"/releases/{release_rel}",
}

items.insert(0, item)

# Cap feed size
items = items[:200]

feed["items"] = items
feed_path.parent.mkdir(parents=True, exist_ok=True)
feed_path.write_text(json.dumps(feed, indent=2, sort_keys=True), encoding="utf-8")
print("updated", feed_path, "items=", len(items))
PY

echo "[publish_release_feed] OK context=${CONTEXT} period_key=${PERIOD_KEY} dest=${DEST_DIR}"
