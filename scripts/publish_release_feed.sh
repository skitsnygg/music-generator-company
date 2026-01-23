cat > scripts/publish_release_feed.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: ./scripts/publish_release_feed.sh --context CONTEXT [--period-key KEY] [--out-dir DIR] [--src-latest-root DIR] [--dest-releases-root DIR]

Publishes an internal "release" snapshot by copying the already-built latest web bundle
into a versioned releases directory, and updates a simple feed.json.

Defaults:
  --src-latest-root      /var/lib/mgc/releases/latest/web
  --dest-releases-root   /var/lib/mgc/releases

Notes:
  - Expects that latest bundle exists at: <src-latest-root>/<context>/
    (Normally created by scripts/publish_latest.sh which run_daily/run_weekly call.)
  - Writes:
      <dest-releases-root>/<period-key>/<context>/(index.html, web_manifest.json, playlist.json, tracks/...)
      <dest-releases-root>/feed.json
USAGE
}

CONTEXT=""
PERIOD_KEY="$(date -u +%Y-%m-%d)"
OUT_DIR=""
SRC_LATEST_ROOT="/var/lib/mgc/releases/latest/web"
DEST_RELEASES_ROOT="/var/lib/mgc/releases"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context) CONTEXT="${2:-}"; shift 2 ;;
    --period-key) PERIOD_KEY="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --src-latest-root) SRC_LATEST_ROOT="${2:-}"; shift 2 ;;
    --dest-releases-root) DEST_RELEASES_ROOT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[publish_release_feed] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CONTEXT" ]]; then
  echo "[publish_release_feed] missing --context" >&2
  usage
  exit 2
fi

log() { echo "[publish_release_feed] $*"; }

src_dir="${SRC_LATEST_ROOT%/}/${CONTEXT}"
dest_dir="${DEST_RELEASES_ROOT%/}/${PERIOD_KEY}/${CONTEXT}"
feed_path="${DEST_RELEASES_ROOT%/}/feed.json"

if [[ ! -d "$src_dir" ]]; then
  echo "[publish_release_feed] missing source bundle: $src_dir" >&2
  echo "[publish_release_feed] hint: run ./scripts/run_daily.sh (or publish_latest) first on this machine" >&2
  exit 1
fi

mkdir -p "$dest_dir"

# Copy snapshot (atomic-ish via temp dir)
tmp_root="$(mktemp -d "${DEST_RELEASES_ROOT%/}/.tmp_release_${PERIOD_KEY}_${CONTEXT}_XXXXXX")"
tmp_dir="${tmp_root}/${CONTEXT}"
mkdir -p "$tmp_dir"

# Copy everything (including tracks/)
cp -a "$src_dir/." "$tmp_dir/"

# Move into place
rm -rf "$dest_dir"
mkdir -p "$(dirname "$dest_dir")"
mv "$tmp_dir" "$dest_dir"
rm -rf "$tmp_root"

log "published release snapshot: $dest_dir"

# Build/update feed.json (append newest first; keep last 50)
python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

context = ${CONTEXT@Q}
period_key = ${PERIOD_KEY@Q}
dest_releases_root = Path(${DEST_RELEASES_ROOT@Q})
feed_path = dest_releases_root / "feed.json"
dest_dir = dest_releases_root / period_key / context

def read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

playlist = read_json(dest_dir / "playlist.json", {})
wm = read_json(dest_dir / "web_manifest.json", {})

track_count = len((playlist.get("tracks") or []))

item = {
    "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
    "period_key": period_key,
    "context": context,
    "paths": {
        "web": f"/releases/{period_key}/{context}/",
        "playlist_json": f"/releases/{period_key}/{context}/playlist.json",
        "web_manifest_json": f"/releases/{period_key}/{context}/web_manifest.json",
    },
    "track_count": track_count,
}

feed = read_json(feed_path, {"ok": True, "items": []})
items = feed.get("items") or []

# Drop duplicates for same period+context
items = [x for x in items if not (x.get("period_key")==period_key and x.get("context")==context)]
items.insert(0, item)
items = items[:50]

feed = {"ok": True, "items": items}
feed_path.write_text(json.dumps(feed, indent=2, sort_keys=True), encoding="utf-8")
print("wrote", feed_path)
PY

log "updated feed: $feed_path"
EOF

chmod +x scripts/publish_release_feed.sh
