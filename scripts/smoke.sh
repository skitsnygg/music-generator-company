#!/usr/bin/env bash
set -euo pipefail

# Fast sanity smoke. Not a determinism test; just proves the CLI runs.
#
# Env:
#   MGC_DB         DB path (default: data/db.sqlite)
#   PYTHON         python executable (default: python)
#   ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT   if "data", rebuild outputs go to data/...; otherwise to artifacts/ci/data/...
#
# Writes:
#   artifacts/ci/smoke.log
#   (optional) rebuild outputs under chosen output root when we exercise rebuild commands

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
DB="${MGC_DB:-${DB:-data/db.sqlite}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"
OUT_ROOT="${MGC_OUT_ROOT:-}"

mkdir -p "$ARTIFACTS_DIR"
log_file="$ARTIFACTS_DIR/smoke.log"
: > "$log_file"

if [[ "$OUT_ROOT" == "data" ]]; then
  OUT_PLAYLISTS="data/playlists"
  OUT_TRACKS="data/tracks"
else
  OUT_PLAYLISTS="$ARTIFACTS_DIR/data/playlists"
  OUT_TRACKS="$ARTIFACTS_DIR/data/tracks"
fi

mkdir -p "$OUT_PLAYLISTS" "$OUT_TRACKS"

{
  echo "[smoke] Repo: $repo_root"
  echo "[smoke] DB: $DB"
  echo "[smoke] python: $("$PYTHON" -V 2>&1)"
  echo "[smoke] ARTIFACTS_DIR=$ARTIFACTS_DIR"
  echo "[smoke] MGC_OUT_ROOT=${MGC_OUT_ROOT:-}"
  echo "[smoke] out_playlists=$OUT_PLAYLISTS"
  echo "[smoke] out_tracks=$OUT_TRACKS"

  "$PYTHON" -m py_compile src/mgc/main.py

  # Basic CLI sanity
  "$PYTHON" -m mgc.main --db "$DB" rebuild ls --json >/dev/null

  # These may be empty in some DBs; keep smoke non-blocking
  "$PYTHON" -m mgc.main --db "$DB" playlists list --limit 1 >/dev/null || true
  "$PYTHON" -m mgc.main --db "$DB" tracks stats >/dev/null || true

  # Light rebuild exercise (no determinism check, quick write)
  "$PYTHON" -m mgc.main --db "$DB" rebuild playlists --out-dir "$OUT_PLAYLISTS" --stamp smoke --write --json >/dev/null || true
  "$PYTHON" -m mgc.main --db "$DB" rebuild tracks --out-dir "$OUT_TRACKS" --stamp smoke --write --json >/dev/null || true

  echo "[smoke] OK"
} 2>&1 | tee -a "$log_file"
