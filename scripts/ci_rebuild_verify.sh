#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# CI: deterministic rebuild + verify gate
# ---------------------------------------

log() { printf "[ci_rebuild_verify] %s\n" "$*"; }
die() { printf "[ci_rebuild_verify] ERROR: %s\n" "$*" >&2; exit 2; }

# Resolve repo root safely
if command -v git >/dev/null 2>&1; then
  ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
else
  ROOT=""
fi

if [[ -n "${ROOT}" ]]; then
  cd "${ROOT}"
else
  # Fallback: assume script is in scripts/ under repo root
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# Optional convenience for local runs; CI should set env explicitly.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PY="${PYTHON:-python}"

log "repo_root: $(pwd)"
log "python: ${PY}"
if ! command -v "${PY}" >/dev/null 2>&1; then
  die "Python executable not found: ${PY} (set PYTHON=/path/to/python if needed)"
fi

log "python_path: $(command -v "${PY}")"

log "== python =="
"${PY}" -V

log "== rebuild playlists (determinism check + write) =="
"${PY}" -m mgc.main rebuild playlists --determinism-check --write

log "== verify playlists vs manifest =="
"${PY}" -m mgc.main rebuild verify playlists

log "== latest rebuild events =="
# These are informational; do not fail the gate if listing fails.
"${PY}" -m mgc.main events list --type rebuild.completed --limit 3 || true
"${PY}" -m mgc.main events list --type rebuild.verify_completed --limit 3 || true

log "OK"
