#!/usr/bin/env bash
set -euo pipefail

# Run the daily pipeline for multiple contexts (focus/workout/sleep),
# generating new tracks, building a daily drop bundle, and generating
# marketing assets (teaser + copy + receipts) for each context.
#
# Source of truth: commit this script in your Mac repo, then let the runner
# `git pull` and execute it.

# Assume this script lives at <repo>/scripts/run_daily.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"

# ---- Defaults (override via env vars) ----
DB_PATH="${MGC_DB:-data/db.sqlite}"

# Where to write evidence outputs. We default to per-context subdirs under data/evidence.
# (Override with MGC_OUT_BASE if your deployment expects a different layout.)
OUT_BASE="${MGC_OUT_BASE:-data/evidence}"

SEED="${MGC_SEED:-1}"
GENERATE_COUNT="${MGC_DAILY_GENERATE_COUNT:-1}"

# Optional generator/provider settings
PROVIDER="${MGC_PROVIDER:-}"
PROMPT="${MGC_PROMPT:-}"

# Contexts to generate daily drops for
CONTEXTS=(${MGC_CONTEXTS:-focus workout sleep})

# Marketing options (enabled by default for daily)
MARKETING="${MGC_MARKETING:-1}"          # 1=on, 0=off
TEASER_SECONDS="${MGC_TEASER_SECONDS:-15}"

# Portable lock (works on macOS + Linux)
LOCK_DIR="${ROOT}/.run_daily.lock"

log() { printf "[run_daily] %s\n" "$*"; }
die() { printf "[run_daily] ERROR: %s\n" "$*" >&2; exit 2; }

cleanup() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  : # acquired
else
  log "Another run_daily appears to be in progress (lock exists: ${LOCK_DIR}). Exiting."
  exit 0
fi

# Prefer repo venv if present (cron environments vary)
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
else
  log "NOTE: .venv not found; using ${PY} from PATH"
fi

if ! command -v "${PY}" >/dev/null 2>&1; then
  die "Python executable not found: ${PY} (set PYTHON=/path/to/python if needed)"
fi

log "Repo: ${ROOT}"
log "DB: ${DB_PATH}"
log "Seed: ${SEED}"
log "Generate count: ${GENERATE_COUNT}"
log "Contexts: ${CONTEXTS[*]}"
log "Marketing: ${MARKETING} (teaser_seconds=${TEASER_SECONDS})"
"${PY}" -V

run_one() {
  local ctx="$1"
  local out_dir="${OUT_BASE}/${ctx}"

  log "Context=${ctx} out_dir=${out_dir}"

  # Build command args safely (bash array)
  args=(
    -m mgc.main
    --db "${DB_PATH}"
    run daily
    --context "${ctx}"
    --seed "${SEED}"
    --out-dir "${out_dir}"
    --generate-count "${GENERATE_COUNT}"
    --json
  )

  if [[ -n "${PROVIDER}" ]]; then
    args+=( --generate-provider "${PROVIDER}" )
  fi
  if [[ -n "${PROMPT}" ]]; then
    args+=( --prompt "${PROMPT}" )
  fi

  if [[ "${MARKETING}" == "1" ]]; then
    args+=( --marketing --teaser-seconds "${TEASER_SECONDS}" )
  fi

  "${PY}" "${args[@]}"
}

for ctx in "${CONTEXTS[@]}"; do
  run_one "${ctx}"
done

log "OK"
