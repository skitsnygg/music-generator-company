#!/usr/bin/env bash
set -euo pipefail

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
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

PY="${PYTHON:-python}"
DB="${MGC_DB:-data/db.sqlite}"

log "repo_root: $(pwd)"
log "python: ${PY}"
log "db: ${DB}"

if ! command -v "${PY}" >/dev/null 2>&1; then
  die "Python executable not found: ${PY} (set PYTHON=/path/to/python if needed)"
fi

log "python_path: $(command -v "${PY}")"
log "== python =="
"${PY}" -V

# If the DB file doesn't exist in CI, sqlite will create an empty DB and rebuild will fail.
if [[ ! -f "${DB}" ]]; then
  log "SKIP: DB not found at ${DB}"
  log "Hint: data/ is typically gitignored. Provide a fixture DB or generate one in CI to enable rebuild/verify."
  exit 0
fi

# Detect missing tables (common on CI if DB is empty)
HAS_PLAYLISTS_TABLE="$("${PY}" - <<PY
import sqlite3, sys
db = "${DB}"
con = sqlite3.connect(db)
try:
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='playlists'")
    sys.stdout.write("yes" if cur.fetchone() else "no")
finally:
    con.close()
PY
)"

if [[ "${HAS_PLAYLISTS_TABLE}" != "yes" ]]; then
  log "SKIP: DB exists but has no 'playlists' table (${DB})"
  log "Hint: commit a fixture DB (schema + sample rows) or generate a DB during CI setup."
  exit 0
fi

log "== rebuild playlists (determinism check + write) =="
"${PY}" -m mgc.main rebuild playlists --determinism-check --write --db "${DB}"

log "== verify playlists vs manifest =="
"${PY}" -m mgc.main rebuild verify playlists --db "${DB}"

log "== latest rebuild events =="
"${PY}" -m mgc.main events list --type rebuild.completed --limit 3 --db "${DB}" || true
"${PY}" -m mgc.main events list --type rebuild.verify_completed --limit 3 --db "${DB}" || true

log "OK"
