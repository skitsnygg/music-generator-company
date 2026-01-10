#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
ART_DIR="${ROOT}/artifacts/ci"

log() { printf "[ci_gate] %s\n" "$*"; }
die() { printf "[ci_gate] ERROR: %s\n" "$*" >&2; exit 2; }

run_step() {
  # Usage: run_step "name" command...
  local name="$1"; shift
  log "$name"
  # tee eats the exit status unless we preserve it; PIPESTATUS works in bash 3.2+
  "$@" 2>&1 | tee "${ART_DIR}/${name}.log"
  local rc="${PIPESTATUS[0]}"
  if [[ "${rc}" -ne 0 ]]; then
    log "${name} failed (exit=${rc})"
    return "${rc}"
  fi
  return 0
}

snapshot() {
  log "snapshot"
  {
    echo "== date =="; date || true
    echo "== pwd =="; pwd || true
    echo "== git status --porcelain =="; git status --porcelain || true
    echo "== git rev-parse HEAD =="; git rev-parse HEAD || true
  } > "${ART_DIR}/snapshot.txt" 2>&1 || true

  ls -la data/playlists > "${ART_DIR}/ls_data_playlists.txt" 2>&1 || true
  "${PY}" -m mgc.main events list --type rebuild.completed --limit 20 > "${ART_DIR}/events_rebuild_completed.txt" 2>&1 || true
  "${PY}" -m mgc.main events list --type rebuild.verify_completed --limit 20 > "${ART_DIR}/events_verify_completed.txt" 2>&1 || true
  cp -f data/playlists/_manifest.playlists.json "${ART_DIR}/_manifest.playlists.json" 2>/dev/null || true
}

py_compile_all() {
  log "py_compile"

  if ! command -v "${PY}" >/dev/null 2>&1; then
    die "Python executable not found: ${PY} (set PYTHON=/path/to/python if needed)"
  fi

  local files
  files="$(git ls-files '*.py' 2>/dev/null || true)"
  if [[ -z "${files}" ]]; then
    log "No tracked *.py files found (skipping)"
    return 0
  fi

  # shellcheck disable=SC2086
  "${PY}" -m py_compile ${files}
}

main() {
  cd "${ROOT}"
  log "Repo: ${ROOT}"

  mkdir -p "${ART_DIR}"
  : > "${ART_DIR}/.keep"
  log "Artifacts: ${ART_DIR}"

  # Always try to snapshot, even on failures.
  trap snapshot EXIT

  py_compile_all

  run_step "rebuild_verify" bash "${ROOT}/scripts/ci_rebuild_verify.sh"
  run_step "manifest_diff" "${PY}" -m mgc.main manifest diff
  run_step "tracks_smoke" bash "${ROOT}/scripts/tracks_smoke.sh"

  log "OK"
}

main "$@"
