#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

log() { printf "[ci_gate] %s\n" "$*"; }
die() { printf "[ci_gate] ERROR: %s\n" "$*" >&2; exit 2; }

py_compile_all() {
  log "py_compile"

  if ! command -v "${PY}" >/dev/null 2>&1; then
    die "Python executable not found: ${PY} (set PYTHON=/path/to/python if needed)"
  fi

  log "py_compile"

if ! command -v "${PY}" >/dev/null 2>&1; then
  die "Python executable not found: ${PY}"
fi

if command -v git >/dev/null 2>&1; then
  PY_FILES="$(git ls-files '*.py' || true)"
else
  PY_FILES=""
fi

if [[ -z "${PY_FILES}" ]]; then
  log "No tracked *.py files found (skipping py_compile)"
else
  # shellcheck disable=SC2086
  "${PY}" -m py_compile ${PY_FILES}
fi
}


main() {
  log "Repo: ${ROOT}"
  cd "${ROOT}"

  py_compile_all

  log "rebuild + verify"
  bash "${ROOT}/scripts/ci_rebuild_verify.sh"

  log "manifest diff"
  "${PY}" -m mgc.main manifest diff

  log "tracks smoke"
  bash "${ROOT}/scripts/tracks_smoke.sh"

  log "git clean check"
if command -v git >/dev/null 2>&1; then
  if [[ -n "$(git status --porcelain)" ]]; then
    git status --porcelain
    die "Working tree is dirty after gates (generated files changed but not committed)"
  fi
else
  log "git not available; skipping clean check"
fi

  log "OK"
}

main "$@"
