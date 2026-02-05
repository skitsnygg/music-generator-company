#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/release_local.sh 2020-01-01 focus sleep workout
# If you omit contexts, defaults to: focus sleep workout
#
# Environment (optional):
#   MGC_PROVIDER=riffusion
#   MGC_RIFFUSION_URL=http://127.0.0.1:3013
#   MGC_MP3_QUALITY=v0
#   MGC_RIFFUSION_CONNECT_TIMEOUT=5
#   MGC_RIFFUSION_READ_TIMEOUT=240
#   MGC_RIFFUSION_RETRIES=0

REL_ID="${1:-}"
shift || true

if [[ -z "${REL_ID}" ]]; then
  echo "error: missing REL_ID (e.g. 2020-01-01)" >&2
  exit 2
fi

# Default to riffusion unless explicitly overridden.
if [[ -z "${MGC_PROVIDER:-}" ]]; then
  export MGC_PROVIDER="riffusion"
fi

# Deterministic runs default to 2020-01-01; pin the fixed time to REL_ID for sane dates.
if [[ -z "${MGC_FIXED_TIME:-}" ]]; then
  export MGC_FIXED_TIME="${REL_ID}T00:00:00Z"
fi

if [[ "$#" -gt 0 ]]; then
  CONTEXTS=("$@")
else
  CONTEXTS=(focus sleep workout)
fi

MARKETING="${MGC_RELEASE_MARKETING:-1}"
TEASER_SECONDS="${MGC_TEASER_SECONDS:-15}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${MGC_PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PY="${ROOT_DIR}/.venv/bin/python"
  else
    PY="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "${PY}" ]]; then
  echo "error: python not found (set MGC_PYTHON)" >&2
  exit 2
fi

DB_PATH="${MGC_DB:-${ROOT_DIR}/data/db.sqlite}"

echo "[release] repo_root=${ROOT_DIR}"
echo "[release] rel_id=${REL_ID}"
echo "[release] contexts=${CONTEXTS[*]}"

# Safety: if you’ve been using strict mode, keep it strict for the release run.
: "${MGC_STRICT_TRACK_PATHS:=1}"

# Ensure docs structure exists
mkdir -p "${ROOT_DIR}/docs/latest/web"
mkdir -p "${ROOT_DIR}/docs/releases"

for ctx in "${CONTEXTS[@]}"; do
  OUT="/private/tmp/mgc_release_${ctx}"
  echo
  echo "[release] run pipeline ctx=${ctx} out=${OUT}"

  rm -rf "${OUT}"
  daily_args=(
    -m mgc.main
    --db "${DB_PATH}"
    run daily
    --context "${ctx}"
    --seed 1
    --out-dir "${OUT}"
    --deterministic
  )
  if [[ "${MARKETING}" == "1" ]]; then
    daily_args+=( --marketing --teaser-seconds "${TEASER_SECONDS}" )
  fi
  "${PY}" "${daily_args[@]}"

  # Verify bundle contract (fail fast).
  "${PY}" "${ROOT_DIR}/scripts/verify_drop_contract.py" --out-dir "${OUT}"

  echo "[release] rebuild web bundle (marketing assets)"
  "${PY}" -m mgc.main --db "${DB_PATH}" web build \
    --playlist "${OUT}/playlist.json" \
    --out-dir "${OUT}/web" \
    --clean \
    --fail-if-empty \
    --deterministic

  # Sanity checks (fail fast)
  test -s "${OUT}/web/index.html"
  test -s "${OUT}/web/web_manifest.json"
  test -d "${OUT}/web/tracks"
  test -s "${OUT}/web/playlist.json"

  # Update docs/latest/web/<ctx>
  echo "[release] publish latest ctx=${ctx}"
  rm -rf "${ROOT_DIR}/docs/latest/web/${ctx}"
  mkdir -p "${ROOT_DIR}/docs/latest/web/${ctx}"
  cp -R "${OUT}/web/." "${ROOT_DIR}/docs/latest/web/${ctx}/"

  # Update docs/releases/<REL_ID>/web/<ctx>
  echo "[release] publish release ctx=${ctx}"
  mkdir -p "${ROOT_DIR}/docs/releases/${REL_ID}/web/${ctx}"
  rm -rf "${ROOT_DIR}/docs/releases/${REL_ID}/web/${ctx}"
  mkdir -p "${ROOT_DIR}/docs/releases/${REL_ID}/web/${ctx}"
  cp -R "${OUT}/web/." "${ROOT_DIR}/docs/releases/${REL_ID}/web/${ctx}/"
done

echo
echo "[release] regenerate feed.json"
"${PY}" "${ROOT_DIR}/scripts/release_feed.py" \
  --root-dir "${ROOT_DIR}/docs/releases" \
  --out "${ROOT_DIR}/docs/releases/feed.json" \
  --stable

echo
echo "[release] verify feed has releases"
"${PY}" - <<'PY'
import json
from pathlib import Path
p = Path("docs/releases/feed.json")
o = json.loads(p.read_text("utf-8"))
rels = o.get("releases") or []
print("feed:", p)
print("releases:", len(rels), [r.get("release_id") for r in rels][:5])
assert rels, "feed.json has zero releases"
PY

echo
echo "[release] DONE"
