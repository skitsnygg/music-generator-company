#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"

log() { printf "[run_daily] %s\n" "$*"; }
die() { printf "[run_daily] ERROR: %s\n" "$*" >&2; exit 2; }

# Prevent overlapping runs (works on macOS + Linux)
LOCK_DIR="${ROOT}/.run_daily.lock"
cleanup() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  : # acquired
else
  log "Another run_daily appears to be in progress (lock exists: ${LOCK_DIR}). Exiting."
  exit 0
fi

# Prefer existing venv if present, but don't hard-require it (cron environments vary)
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
  # After activation, python should point at venv python unless PYTHON overrides.
  if [[ "${PY}" == "python" ]]; then
    PY="python"
  fi
else
  log "NOTE: .venv not found; using ${PY} from PATH"
fi

if ! command -v "${PY}" >/dev/null 2>&1; then
  die "Python executable not found: ${PY} (set PYTHON=/path/to/python if needed)"
fi

log "Repo: ${ROOT}"
log "Python: ${PY}"
"${PY}" -V

log "Running: python -m mgc.main run-daily"
"${PY}" -m mgc.main run-daily

log "OK"
