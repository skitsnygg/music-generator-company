#!/usr/bin/env bash
set -euo pipefail

# CI gate: compile + deterministic checks
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
#   $ARTIFACTS_DIR/ci_rebuild_verify.log
#   (plus rebuild outputs under chosen output root)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
DB="${MGC_DB:-${DB:-data/db.sqlite}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"

mkdir -p "$ARTIFACTS_DIR"

log_file="$ARTIFACTS_DIR/ci_gate.log"
: > "$log_file"

run() {
  echo "[ci_gate] Repo: $repo_root"
  echo "[ci_gate] MGC_DB=$DB"
  echo "[ci_gate] ARTIFACTS_DIR=$ARTIFACTS_DIR"
  echo "[ci_gate] MGC_OUT_ROOT=${MGC_OUT_ROOT:-}"
  echo "[ci_gate] git_sha: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "[ci_gate] git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "[ci_gate] python: $("$PYTHON" -V 2>&1)"

  if [[ ! -f "$DB" ]]; then
    echo "[ci_gate] ERROR: DB not found: $DB" >&2
    return 2
  fi
  echo "[ci_gate] DB OK."

  echo "[ci_gate] py_compile"
  "$PYTHON" -m py_compile src/mgc/main.py

  echo "[ci_gate] rebuild + verify"
  MGC_DB="$DB" ARTIFACTS_DIR="$ARTIFACTS_DIR" PYTHON="$PYTHON" MGC_OUT_ROOT="${MGC_OUT_ROOT:-}" \
    bash scripts/ci_rebuild_verify.sh

  echo "[ci_gate] publish receipts determinism"
  MGC_DB="$DB" PYTHON="$PYTHON" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
    bash scripts/ci_publish_determinism.sh "ci_publish"

  echo "[ci_gate] OK"
}

run 2>&1 | tee -a "$log_file"
