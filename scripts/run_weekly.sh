#!/usr/bin/env bash
set -euo pipefail

# Weekly multi-context run (focus/workout/sleep) + marketing + publish "latest" web bundle per context.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"

DB_PATH="${MGC_DB:-data/db.sqlite}"
OUT_BASE="${MGC_OUT_BASE:-data/evidence/weekly}"

PERIOD_KEY="${MGC_PERIOD_KEY:-$(date -u +%G-W%V)}"

SEED="${MGC_SEED:-1}"
GENERATE_COUNT="${MGC_WEEKLY_GENERATE_COUNT:-2}"

PROVIDER="${MGC_PROVIDER:-}"
PROMPT="${MGC_PROMPT:-}"

CONTEXTS=(${MGC_CONTEXTS:-focus workout sleep})

MARKETING="${MGC_MARKETING:-1}"          # 1=on, 0=off
TEASER_SECONDS="${MGC_TEASER_SECONDS:-15}"

PUBLISH_LATEST="${MGC_PUBLISH_LATEST:-1}"  # 1=on, 0=off

LOCK_DIR="${ROOT}/.run_weekly.lock"

log() { printf "[run_weekly] %s\n" "$*"; }
die() { printf "[run_weekly] ERROR: %s\n" "$*" >&2; exit 2; }

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

command -v "${PY}" >/dev/null 2>&1 || die "Python executable not found: ${PY}"

log "Repo: ${ROOT}"
log "DB: ${DB_PATH}"
log "Period: ${PERIOD_KEY}"
log "Seed: ${SEED}"
log "Generate count: ${GENERATE_COUNT}"
log "Contexts: ${CONTEXTS[*]}"
log "Marketing: ${MARKETING} (teaser_seconds=${TEASER_SECONDS})"
log "Publish latest web: ${PUBLISH_LATEST}"
"${PY}" -V

run_one() {
  local ctx="$1"
  local out_dir="${OUT_BASE}/${PERIOD_KEY}/${ctx}"

  log "Context=${ctx} period=${PERIOD_KEY} out_dir=${out_dir}"

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

  if [[ "${PUBLISH_LATEST}" == "1" ]]; then
    "${ROOT}/scripts/publish_latest.sh" --context "${ctx}" --src-out-dir "${out_dir}" --db "${DB_PATH}"
  fi
}

for ctx in "${CONTEXTS[@]}"; do
  run_one "${ctx}"
done

log "OK"
