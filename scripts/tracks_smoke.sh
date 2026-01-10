#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DB="${1:-data/db.sqlite}"
PY="${PYTHON:-python}"

log() { printf "[tracks_smoke] %s\n" "$*"; }
die() { printf "[tracks_smoke] ERROR: %s\n" "$*" >&2; exit 2; }

# Optional convenience for local runs; CI should usually set up env explicitly.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if ! command -v "${PY}" >/dev/null 2>&1; then
  die "Python executable not found: ${PY} (set PYTHON=/path/to/python if needed)"
fi

log "Repo: ${ROOT}"
log "Python: ${PY}"
"${PY}" -V

# CI runners likely won't have ignored data/. Don't fail the whole gate for that.
if [[ ! -f "${DB}" ]]; then
  log "SKIP: DB file not found: ${DB}"
  log "Hint: data/ is typically gitignored; provide a fixture DB or generate one in CI to enable this smoke test."
  exit 0
fi

log "Using DB: ${DB}"

log "tracks list (limit 10)"
"${PY}" -m mgc.main tracks list --db "${DB}" --limit 10
echo

log "tracks stats"
"${PY}" -m mgc.main tracks stats --db "${DB}"
echo

log "Try showing first track id from list (if any)"

# Grab the first plausible UUID-ish token from the first list line.
# This is intentionally tolerant of output format changes.
LIST_OUT="$("${PY}" -m mgc.main tracks list --db "${DB}" --limit 1 || true)"
FIRST_ID="$(printf "%s\n" "${LIST_OUT}" \
  | head -n 1 \
  | grep -Eo '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' \
  | head -n 1 \
  || true
)"

if [[ -n "${FIRST_ID}" ]]; then
  log "Showing id=${FIRST_ID}"
  "${PY}" -m mgc.main tracks show "${FIRST_ID}" --db "${DB}"
else
  log "No tracks found (or could not parse id); skipping tracks show."
fi

log "OK"
