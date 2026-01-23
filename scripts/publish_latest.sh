#!/usr/bin/env bash
set -euo pipefail

# publish_latest.sh
#
# Build+publish a "latest" web bundle for a given context from an evidence out_dir.
#
# Usage:
#   ./scripts/publish_latest.sh --context focus --src-out-dir data/evidence/focus --db data/db.sqlite
#
# Output:
#   <MGC_WEB_LATEST_ROOT>/<context>/   (default: data/web/latest/<context>/)
#
# Notes:
# - Builds into a temp dir then atomically swaps the destination.
# - Assumes `mgc web build` can read playlist.json under drop_bundle/.
# - This is "static publish": no API calls, no external services.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"

CONTEXT=""
SRC_OUT_DIR=""
DB_PATH="data/db.sqlite"

WEB_ROOT="${MGC_WEB_LATEST_ROOT:-data/web/latest}"
WEB_BUILD_ARGS="${MGC_WEB_BUILD_ARGS:-}"

die() { echo "[publish_latest] ERROR: $*" >&2; exit 2; }
log() { echo "[publish_latest] $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context) CONTEXT="${2:-}"; shift 2 ;;
    --src-out-dir) SRC_OUT_DIR="${2:-}"; shift 2 ;;
    --db) DB_PATH="${2:-}"; shift 2 ;;
    *)
      die "Unknown arg: $1"
      ;;
  esac
done

[[ -n "${CONTEXT}" ]] || die "--context is required"
[[ -n "${SRC_OUT_DIR}" ]] || die "--src-out-dir is required"

PLAYLIST_BUNDLE="${SRC_OUT_DIR}/drop_bundle/playlist.json"
[[ -f "${PLAYLIST_BUNDLE}" ]] || die "Missing bundle playlist: ${PLAYLIST_BUNDLE}"

DEST_DIR="${WEB_ROOT}/${CONTEXT}"

# Build into a temp dir then swap to DEST_DIR atomically.
TMP_BASE="${ROOT}/.tmp_publish"
mkdir -p "${TMP_BASE}"
TMP_DIR="$(mktemp -d "${TMP_BASE}/latest_${CONTEXT}_XXXXXX")"
trap 'rm -rf "${TMP_DIR}" 2>/dev/null || true' EXIT INT TERM

log "context=${CONTEXT}"
log "src_out_dir=${SRC_OUT_DIR}"
log "db=${DB_PATH}"
log "playlist=${PLAYLIST_BUNDLE}"
log "dest=${DEST_DIR}"
log "tmp=${TMP_DIR}"

# Prefer repo venv if present
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

command -v "${PY}" >/dev/null 2>&1 || die "Python not found: ${PY}"

# Build web bundle
# web.build resolves tracks relative to the playlist directory; it can also stage tracks to out_dir/tracks.
# We build into TMP_DIR, then swap it into DEST_DIR.
set +e
"${PY}" -m mgc.main --db "${DB_PATH}" web build   --playlist "${PLAYLIST_BUNDLE}"   --out-dir "${TMP_DIR}"   --clean   --fail-if-empty   ${WEB_BUILD_ARGS} >/dev/null 2>&1
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  die "web build failed (rc=${RC}). Re-run without redirect to see error."
fi

# Atomic swap: DEST_DIR -> backup, TMP_DIR -> DEST_DIR
PARENT="$(dirname "${DEST_DIR}")"
mkdir -p "${PARENT}"

BACKUP=""
if [[ -d "${DEST_DIR}" ]]; then
  BACKUP="${DEST_DIR}.bak.$(date -u +%Y%m%dT%H%M%SZ)"
  mv -f "${DEST_DIR}" "${BACKUP}"
fi
mv -f "${TMP_DIR}" "${DEST_DIR}"
# cancel trap cleanup since TMP_DIR moved
trap - EXIT INT TERM

# Keep only one backup (optional): delete previous backups beyond the newest one
# (commented out for safety)
# ls -1dt "${DEST_DIR}.bak."* 2>/dev/null | tail -n +2 | xargs -r rm -rf

log "OK published: ${DEST_DIR}"
if [[ -n "${BACKUP}" ]]; then
  log "backup: ${BACKUP}"
fi
