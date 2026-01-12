#!/usr/bin/env bash
set -euo pipefail

# CI gate: compile + rebuild/verify + determinism checks
#
# Env:
#   MGC_DB         DB path (default: data/db.sqlite)
#   PYTHON         python executable (default: python)
#   ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT   override output root for rebuilds:
#                  - if set to "data", writes to data/playlists + data/tracks
#                  - otherwise writes under $ARTIFACTS_DIR/data/...
#
# Files written:
#   $ARTIFACTS_DIR/ci_gate.log
#   (plus rebuild outputs under chosen output root)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
DB="${MGC_DB:-${DB:-data/db.sqlite}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"

mkdir -p "$ARTIFACTS_DIR"

log_file="$ARTIFACTS_DIR/ci_gate.log"
: >"$log_file"

info() { echo "[ci_gate] $*"; }
err() { echo "[ci_gate] ERROR: $*" >&2; }

run_gate() {
  info "Repo: $repo_root"
  info "MGC_DB=$DB"
  info "ARTIFACTS_DIR=$ARTIFACTS_DIR"
  info "MGC_OUT_ROOT=${MGC_OUT_ROOT:-}"
  info "git_sha: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  info "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  info "python: $("$PYTHON" -V 2>&1)"

  if [[ ! -f "$DB" ]]; then
    err "DB not found: $DB"
    return 2
  fi
  info "DB OK."

  info "drops smoke test"
  MGC_DB="$DB" MGC_DETERMINISTIC=1 MGC_FIXED_TIME="2020-01-01T00:00:00Z" \
    "$PYTHON" -m mgc.main run drop --context focus --seed 1 >/dev/null

  # IMPORTANT: --json is a global flag on the top-level parser, so it must appear
  # before the subcommand unless the subcommand also defines it.
  MGC_DB="$DB" "$PYTHON" -m mgc.main --json drops list --limit 1 >/dev/null

  info "py_compile"
  "$PYTHON" -m py_compile src/mgc/main.py

  info "rebuild + verify"
  MGC_DB="$DB" ARTIFACTS_DIR="$ARTIFACTS_DIR" PYTHON="$PYTHON" MGC_OUT_ROOT="${MGC_OUT_ROOT:-}" \
    bash scripts/ci_rebuild_verify.sh

  info "publish receipts determinism"
  MGC_DB="$DB" PYTHON="$PYTHON" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
    bash scripts/ci_publish_determinism.sh "ci_publish"

  info "OK"
}

run_gate 2>&1 | tee -a "$log_file"
