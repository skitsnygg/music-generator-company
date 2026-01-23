#!/usr/bin/env bash
set -euo pipefail

# Repo root (assumes this script lives under scripts/ in the repo)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"

# Defaults (override via env vars if desired)
DB_PATH="${MGC_DB:-data/db.sqlite}"
OUT_BASE="${MGC_OUT_BASE:-data/evidence}"
SEED="${MGC_SEED:-1}"
GENERATE_COUNT="${MGC_GENERATE_COUNT:-1}"
PROVIDER="${MGC_PROVIDER:-}"      # optional
PROMPT="${MGC_PROMPT:-}"          # optional
CONTEXTS=(${MGC_CONTEXTS:-focus workout sleep})

log() { printf "[run_daily] %s\n" "$*"; }
die() { printf "[run_daily] ERROR: %s\n" "$*" >&2; exit 2; }

# Prevent overlapping runs (portable: works on macOS + Linux)
LOCK_DIR="${ROOT}/.run_daily.lock"
cleanup() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  : # acquired
else
  log "Another run_daily appears to be in progress (lock exists: ${LOCK_DIR}). Exiting."
  exit 0
fi

# Prefer existing venv if present (cron environments vary)
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
log "Python: ${PY}"
"${PY}" -V

run_one() {
  local ctx="$1"
  local out_dir="${OUT_BASE}/${ctx}"

  log "Context=${ctx} out_dir=${out_dir} db=${DB_PATH} generate_count=${GENERATE_COUNT} seed=${SEED}"

  args=( -m mgc.main --db "${DB_PATH}" run daily
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

  "${PY}" "${args[@]}"
}

for ctx in "${CONTEXTS[@]}"; do
  run_one "${ctx}"
done

log "OK"
