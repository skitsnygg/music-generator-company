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
MARKETING_ASSET_COPIED="0"

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

check_root_owned() {
  local base="$1"
  [[ -d "${base}" ]] || return 0
  local hit=""
  hit="$(find "${base}" -type f -user root -print -quit 2>/dev/null || true)"
  if [[ -n "${hit}" ]]; then
    die "Found root-owned files under ${base}. Fix with: sudo chown -R \"${USER}:$(id -gn)\" \"${base}\""
  fi
}

if [[ "${MGC_SKIP_OWNERSHIP_CHECK:-0}" != "1" ]]; then
  check_root_owned "${SRC_OUT_DIR}"
  check_root_owned "${WEB_ROOT}"
fi

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

# Optional: copy marketing assets into the web bundle (for previews/links)
MARKETING_DIR="${SRC_OUT_DIR}/marketing"
if [[ -d "${MARKETING_DIR}" ]]; then
  mkdir -p "${TMP_DIR}/marketing"

  copy_marketing_file() {
    local src="$1"
    local dst="$2"
    if [[ -f "${src}" ]]; then
      mkdir -p "$(dirname "${dst}")"
      cp -a "${src}" "${dst}"
      MARKETING_ASSET_COPIED="1"
    fi
  }

  copy_marketing_file "${MARKETING_DIR}/marketing_plan.json" "${TMP_DIR}/marketing/marketing_plan.json"
  copy_marketing_file "${MARKETING_DIR}/summary.txt" "${TMP_DIR}/marketing/summary.txt"
  copy_marketing_file "${MARKETING_DIR}/hashtags.txt" "${TMP_DIR}/marketing/hashtags.txt"
  copy_marketing_file "${MARKETING_DIR}/teaser.wav" "${TMP_DIR}/marketing/teaser.wav"
  copy_marketing_file "${MARKETING_DIR}/cover.png" "${TMP_DIR}/marketing/cover.png"
  copy_marketing_file "${MARKETING_DIR}/cover.svg" "${TMP_DIR}/marketing/cover.svg"

  shopt -s nullglob
  for f in "${MARKETING_DIR}"/post_*.txt; do
    cp -a "${f}" "${TMP_DIR}/marketing/"
    MARKETING_ASSET_COPIED="1"
  done
  shopt -u nullglob

  if [[ -d "${MARKETING_DIR}/media" ]]; then
    cp -a "${MARKETING_DIR}/media" "${TMP_DIR}/marketing/"
    MARKETING_ASSET_COPIED="1"
  fi
fi

update_manifest_tree() {
  local root_dir="$1"
  local manifest_path="$2"
  if [[ ! -s "${manifest_path}" ]]; then
    log "WARN: web_manifest.json missing; skipping tree hash update"
    return 0
  fi
  "${PY}" - "${root_dir}" "${manifest_path}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])

items = []
for p in sorted(root.rglob("*")):
    if not p.is_file():
        continue
    rel = p.relative_to(root).as_posix()
    if rel == "web_manifest.json":
        continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    items.append((rel, h.hexdigest()))

payload = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
tree_hash = hashlib.sha256(payload).hexdigest()

data = json.loads(manifest_path.read_text(encoding="utf-8"))
data["web_tree_sha256"] = tree_hash
manifest_path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
print(tree_hash)
PY
}

if [[ "${MARKETING_ASSET_COPIED}" == "1" ]]; then
  log "updating web_manifest.json tree hash after marketing asset copy"
  update_manifest_tree "${TMP_DIR}" "${TMP_DIR}/web_manifest.json" >/dev/null
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
