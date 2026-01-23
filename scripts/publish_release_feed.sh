#!/usr/bin/env bash
set -euo pipefail

# publish_release_feed.sh
#
# Interview-ready internal release feed generator.
# Writes /var/lib/mgc/releases/feed.json by scanning the release root.
#
# IMPORTANT:
# - This script does NOT publish/copy snapshots. It only generates the feed.
# - It delegates all feed logic to scripts/release_feed.py (single source of truth).
#
# Usage examples:
#   scripts/publish_release_feed.sh
#   sudo -E scripts/publish_release_feed.sh
#   BASE_URL="https://example.com" scripts/publish_release_feed.sh
#   scripts/publish_release_feed.sh --include-backups

usage() {
  cat <<'USAGE'
usage: scripts/publish_release_feed.sh [--root-dir DIR] [--out FILE] [--max-items N] [--base-url URL] [--include-backups]

Defaults:
  --root-dir   /var/lib/mgc/releases
  --out        /var/lib/mgc/releases/feed.json
  --max-items  200
  --base-url   (empty => relative URLs)

Notes:
  - Delegates generation to scripts/release_feed.py
  - Uses flock + atomic write + JSON validation
  - By default excludes *.bak.* contexts and non-web dirs (run/submission/etc).
USAGE
}

log() { printf '%s %s\n' "[release_feed]" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

ROOT_DIR="${ROOT_DIR:-/var/lib/mgc/releases}"
OUT_JSON="${OUT_JSON:-$ROOT_DIR/feed.json}"
LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/.feed.lock}"
MAX_ITEMS="${MAX_ITEMS:-200}"
BASE_URL="${BASE_URL:-}"
INCLUDE_BACKUPS="false"

# Back-compat: accept --context but ignore it (older callers may still pass it)
CONTEXT_IGNORED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-dir) ROOT_DIR="${2:-}"; shift 2 ;;
    --out) OUT_JSON="${2:-}"; shift 2 ;;
    --max-items) MAX_ITEMS="${2:-}"; shift 2 ;;
    --base-url) BASE_URL="${2:-}"; shift 2 ;;
    --include-backups) INCLUDE_BACKUPS="true"; shift 1 ;;
    --context) CONTEXT_IGNORED="${2:-}"; shift 2 ;; # ignored
    -h|--help) usage; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v flock >/dev/null 2>&1 || die "flock not found"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$SCRIPT_DIR/release_feed.py"
[[ -f "$GENERATOR" ]] || die "missing generator: $GENERATOR"

[[ -d "$ROOT_DIR" ]] || die "root dir not found: $ROOT_DIR"
mkdir -p "$(dirname "$OUT_JSON")"

# Lock
exec 9>"$LOCK_FILE"
flock -n 9 || die "another feed run is in progress (lock: $LOCK_FILE)"

tmp="$(mktemp "${OUT_JSON}.tmp.XXXXXX")"
cleanup() { rm -f "$tmp" || true; }
trap cleanup EXIT

log "ROOT_DIR=$ROOT_DIR"
log "OUT_JSON=$OUT_JSON"
log "MAX_ITEMS=$MAX_ITEMS"
log "BASE_URL=${BASE_URL:-<empty>}"
if [[ -n "$CONTEXT_IGNORED" ]]; then
  log "NOTE: --context '$CONTEXT_IGNORED' is ignored (feed is global)"
fi

args=( "--root-dir" "$ROOT_DIR" "--out" "$tmp" "--max-items" "$MAX_ITEMS" )
if [[ -n "${BASE_URL:-}" ]]; then
  args+=( "--base-url" "$BASE_URL" )
fi
if [[ "$INCLUDE_BACKUPS" == "true" ]]; then
  args+=( "--include-backups" )
fi

python3 "$GENERATOR" "${args[@]}"

# Validate JSON before replacing existing feed
python3 -m json.tool "$tmp" >/dev/null 2>&1 || die "generated feed.json is invalid JSON"

# Atomic replace
mv -f "$tmp" "$OUT_JSON"
chmod 0644 "$OUT_JSON" || true

log "OK wrote $OUT_JSON ($(wc -c <"$OUT_JSON" | tr -d ' ') bytes)"
