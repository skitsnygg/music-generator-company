#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"

DB_PATH="${MGC_DB:-data/db.sqlite}"
OUT_BASE="${MGC_OUT_BASE:-data/evidence/weekly}"
SEED="${MGC_SEED:-1}"
GENERATE_COUNT="${MGC_WEEKLY_GENERATE_COUNT:-2}"
PROVIDER="${MGC_PROVIDER:-}"      # optional
PROMPT="${MGC_PROMPT:-}"          # optional
CONTEXTS=(${MGC_CONTEXTS:-focus workout sleep})

# ISO week like 2026-W04
PERIOD_KEY="${MGC_PERIOD_KEY:-$(date -u +%G-W%V)}"

log() { printf "[run_weekly] %s\n" "$*"; }
die() { printf "[run_weekly] ERROR: %s\n" "$*" >&2; exit 2; }

LOCK_DIR="${ROOT}/.run_weekly.lock"
cleanup() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if mkdir "${LOCK_DIR}" 2>/dev/null; then :; else
  log "Another run_weekly appears to be in progress (lock exists: ${LOCK_DIR}). Exiting."
  exit 0
fi

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
else
  log "NOTE: .venv not found; using ${PY} from PATH"
fi

command -v "${PY}" >/dev/null 2>&1 || die "Python not found: ${PY}"

run_one() {
  local ctx="$1"
  local out_dir="${OUT_BASE}/${PERIOD_KEY}/${ctx}"

  log "Context=${ctx} period=${PERIOD_KEY} out_dir=${out_dir} db=${DB_PATH} generate_count=${GENERATE_COUNT}"

  args=( -m mgc.main --db "${DB_PATH}" run weekly
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

  "${PY}" "${args[@]}"
}

for ctx in "${CONTEXTS[@]}"; do
  run_one "${ctx}"
done

log "OK"
