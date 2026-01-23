#!/usr/bin/env bash
set -euo pipefail

# Run the weekly pipeline for multiple contexts (focus/workout/sleep),
# generating new tracks, building a weekly playlist bundle, and generating
# marketing assets (teaser + copy + receipts) for each context.
#
# Source of truth: commit this script in your Mac repo, then let the runner
# `git pull` and execute it.

# Assume this script lives at <repo>/scripts/run_weekly.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"

# ---- Defaults (override via env vars) ----
DB_PATH="${MGC_DB:-data/db.sqlite}"
OUT_BASE="${MGC_OUT_BASE:-data/evidence/weekly}"

# ISO week like 2026-W04
PERIOD_KEY="${MGC_PERIOD_KEY:-$(date -u +%G-W%V)}"

SEED="${MGC_SEED:-1}"
GENERATE_COUNT="${MGC_WEEKLY_GENERATE_COUNT:-2}"

# Optional generator/provider settings
PROVIDER="${MGC_PROVIDER:-}"
PROMPT="${MGC_PROMPT:-}"

# Contexts to generate weekly drops for
CONTEXTS=(${MGC_CONTEXTS:-focus workout sleep})

# Marketing options (enabled by default for weekly)
MARKETING="${MGC_MARKETING:-1}"          # 1=on, 0=off
TEASER_SECONDS="${MGC_TEASER_SECONDS:-15}"

# Portable lock (works on macOS + Linux)
LOCK_DIR="${ROOT}/.run_weekly.lock"

log() { printf "[run_weekly] %s\n" "$*"; }
die() { printf "[run_weekly] ERROR: %s\n" "$*" >&2; exit 2; }

cleanup() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  : # acquired
else
  log "Another run_weekly appears to be in progress (lock exists: ${LOCK_DIR}). Exiting."
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
log "Period: ${PERIOD_KEY}"
log "Seed: ${SEED}"
log "Generate count: ${GENERATE_COUNT}"
log "Contexts: ${CONTEXTS[*]}"
log "Marketing: ${MARKETING} (teaser_seconds=${TEASER_SECONDS})"
"${PY}" -V

run_one() {
  local ctx="$1"
  local out_dir="${OUT_BASE}/${PERIOD_KEY}/${ctx}"

  log "Context=${ctx} out_dir=${out_dir}"

  # Build command args safely (bash array)
  args=(
    -m mgc.main
    --db "${DB_PATH}"
    run weekly
    --context "${ctx}"
    --seed "${SEED}"
    --period-key "${PERIOD_KEY}"
    --out-dir "${out_dir}"
    --deterministic
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
