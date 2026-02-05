#!/usr/bin/env bash
set -euo pipefail

echo "[web_health] starting web bundle health check"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

MGC_PYTHON="${MGC_PYTHON:-}"
WEB_ROOT="${MGC_WEB_LATEST_ROOT:-data/web/latest}"
FEED_PATH="${MGC_RELEASE_FEED_OUT:-data/releases/feed.json}"
CONTEXTS_STR="${MGC_CONTEXTS:-focus workout sleep}"

SKIP_FEED="${MGC_WEB_HEALTH_SKIP_FEED:-0}"
SKIP_AUDIO="${MGC_WEB_HEALTH_SKIP_AUDIO:-0}"
SKIP_MARKETING="${MGC_WEB_HEALTH_SKIP_MARKETING:-0}"
SKIP_MANIFEST="${MGC_WEB_HEALTH_SKIP_MANIFEST:-0}"
SKIP_PLAYLIST="${MGC_WEB_HEALTH_SKIP_PLAYLIST:-0}"
ALLOW_MISSING_MARKETING="${MGC_WEB_HEALTH_ALLOW_MISSING_MARKETING:-0}"

usage() {
  cat <<'USAGE'
Usage: scripts/web_health.sh [options]

Options:
  --web-root PATH             Web bundle root (default: MGC_WEB_LATEST_ROOT or data/web/latest)
  --feed PATH                 Feed JSON path (default: MGC_RELEASE_FEED_OUT or data/releases/feed.json)
  --contexts "c1 c2"          Context list (default: MGC_CONTEXTS or "focus workout sleep")
  --python PATH               Python interpreter
  --skip-feed                 Skip feed JSON check
  --skip-audio                Skip audio file count check
  --skip-marketing            Skip marketing asset checks
  --skip-manifest             Skip mgc web validate
  --skip-playlist             Skip playlist track checks
  --allow-missing-marketing   Treat missing marketing assets as warnings
  -h, --help                  Show this help
USAGE
}

die() { echo "[web_health] ERROR: $*" >&2; exit 2; }
log() { echo "[web_health] $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --web-root) WEB_ROOT="${2:-}"; shift 2 ;;
    --feed) FEED_PATH="${2:-}"; shift 2 ;;
    --contexts) CONTEXTS_STR="${2:-}"; shift 2 ;;
    --python) MGC_PYTHON="${2:-}"; shift 2 ;;
    --skip-feed) SKIP_FEED="1"; shift ;;
    --skip-audio) SKIP_AUDIO="1"; shift ;;
    --skip-marketing) SKIP_MARKETING="1"; shift ;;
    --skip-manifest) SKIP_MANIFEST="1"; shift ;;
    --skip-playlist) SKIP_PLAYLIST="1"; shift ;;
    --allow-missing-marketing) ALLOW_MISSING_MARKETING="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

if [[ -z "${MGC_PYTHON}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    MGC_PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    MGC_PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi
[[ -n "${MGC_PYTHON}" ]] || die "python not found (set --python or MGC_PYTHON)"

log "web_root=${WEB_ROOT}"
log "feed_path=${FEED_PATH}"
log "contexts=${CONTEXTS_STR}"

read -r -a CONTEXTS <<< "${CONTEXTS_STR}"
[[ "${#CONTEXTS[@]}" -gt 0 ]] || die "no contexts specified"

if [[ "${SKIP_FEED}" != "1" ]]; then
  log "checking feed JSON..."
  [[ -s "${FEED_PATH}" ]] || die "feed not found: ${FEED_PATH}"
  "${MGC_PYTHON}" -m json.tool "${FEED_PATH}" >/dev/null
  FEED_PATH="${FEED_PATH}" CONTEXTS_STR="${CONTEXTS_STR}" "${MGC_PYTHON}" - <<'PY'
import json
import os
import sys

feed_path = os.environ["FEED_PATH"]
want = [c for c in os.environ.get("CONTEXTS_STR", "").split() if c]
obj = json.load(open(feed_path, "r", encoding="utf-8"))
latest = obj.get("latest", {})
contexts = latest.get("contexts", [])
names = [c.get("context") for c in contexts if isinstance(c, dict) and c.get("context")]
missing = [c for c in want if c not in names]
print("[web_health] feed contexts:", names)
if missing:
    print(f"[web_health] ERROR: feed missing contexts: {missing}", file=sys.stderr)
    sys.exit(2)
PY
  log "feed ok"
else
  log "skipping feed check"
fi

check_marketing_assets() {
  local marketing_dir="$1"
  local plan_path="${marketing_dir}/marketing_plan.json"
  if [[ ! -f "${plan_path}" ]]; then
    log "marketing plan not found; skipping"
    return 0
  fi
  ALLOW_MISSING_MARKETING="${ALLOW_MISSING_MARKETING}" "${MGC_PYTHON}" - "${plan_path}" "${marketing_dir}" <<'PY'
import json
import os
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
base = Path(sys.argv[2])
obj = json.loads(plan_path.read_text(encoding="utf-8"))
paths = obj.get("paths") if isinstance(obj.get("paths"), dict) else {}
missing = []

def add_path(rel, label):
    if not rel or not isinstance(rel, str):
        return
    rel = rel.strip()
    if not rel:
        return
    path = base / rel
    if not path.exists():
        missing.append({"label": label, "path": rel, "resolved": str(path)})

cover = ""
cover_obj = obj.get("cover") if isinstance(obj.get("cover"), dict) else {}
if isinstance(cover_obj, dict):
    cover = cover_obj.get("dst") or cover_obj.get("path") or ""
if not cover:
    cover = paths.get("cover", "")
add_path(cover, "cover")

add_path(paths.get("summary", ""), "summary")
add_path(paths.get("hashtags", ""), "hashtags")
add_path(paths.get("teaser", ""), "teaser")

media = paths.get("media", "")
if not media:
    media_obj = obj.get("media") if isinstance(obj.get("media"), dict) else {}
    if isinstance(media_obj, dict):
        media = media_obj.get("video_path") or media_obj.get("media_path") or ""
add_path(media, "media")

posts = paths.get("posts") if isinstance(paths.get("posts"), list) else []
for idx, rel in enumerate(posts, start=1):
    add_path(rel, f"post_{idx}")

if missing:
    print("[web_health] ERROR: marketing assets missing:", file=sys.stderr)
    for m in missing[:200]:
        print(m, file=sys.stderr)
    if os.environ.get("ALLOW_MISSING_MARKETING", "0") == "1":
        sys.exit(0)
    sys.exit(2)
print("[web_health] marketing assets ok")
PY
}

for ctx in "${CONTEXTS[@]}"; do
  log "context=${ctx}"
  WEB_DIR="${WEB_ROOT}/${ctx}"
  [[ -d "${WEB_DIR}" ]] || die "missing web dir: ${WEB_DIR}"

  [[ -s "${WEB_DIR}/index.html" ]] || die "missing index.html: ${WEB_DIR}/index.html"
  [[ -s "${WEB_DIR}/playlist.json" ]] || die "missing playlist.json: ${WEB_DIR}/playlist.json"
  [[ -s "${WEB_DIR}/web_manifest.json" ]] || die "missing web_manifest.json: ${WEB_DIR}/web_manifest.json"

  if [[ "${SKIP_MANIFEST}" != "1" ]]; then
    log "validating manifest..."
    "${MGC_PYTHON}" -m mgc.main web validate --out-dir "${WEB_DIR}" >/dev/null
  fi

  if [[ "${SKIP_PLAYLIST}" != "1" ]]; then
    log "checking playlist tracks..."
    "${MGC_PYTHON}" scripts/check_playlist_tracks.py "${WEB_DIR}/playlist.json" "${WEB_DIR}"
  fi

  if [[ "${SKIP_AUDIO}" != "1" ]]; then
    AUDIO_COUNT="$(find "${WEB_DIR}" -maxdepth 8 -type f \( -name '*.mp3' -o -name '*.wav' \) | wc -l | tr -d ' ')"
    if [[ "${AUDIO_COUNT}" == "0" ]]; then
      die "no audio files found under ${WEB_DIR} (set --skip-audio to ignore)"
    fi
    log "audio files: ${AUDIO_COUNT}"
  fi

  if [[ "${SKIP_MARKETING}" != "1" ]]; then
    log "checking marketing assets..."
    check_marketing_assets "${WEB_DIR}/marketing"
  fi
done

log "web health ok"
