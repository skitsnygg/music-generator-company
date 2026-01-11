#!/usr/bin/env bash
set -euo pipefail

# Deterministic rebuild + verify runner
#
# Env:
#   MGC_DB         Path to sqlite DB (default: data/db.sqlite)
#   PYTHON         Python executable (default: python)
#   STAMP          Optional stamp string (default: ci)
#   ARTIFACTS_DIR  Where logs go (default: artifacts/ci)
#   MGC_OUT_ROOT   If "data", write outputs to data/{playlists,tracks}
#                 Otherwise write outputs under $ARTIFACTS_DIR/data/{playlists,tracks}
#
# Exit:
#   0 success
#   2 verification strict failure

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
DB="${MGC_DB:-${DB:-data/db.sqlite}}"
STAMP="${STAMP:-ci}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"
OUT_ROOT="${MGC_OUT_ROOT:-}"

mkdir -p "$ARTIFACTS_DIR"
log_file="$ARTIFACTS_DIR/ci_rebuild_verify.log"
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
  echo "[ci_rebuild_verify] repo_root: $repo_root"
  echo "[ci_rebuild_verify] python: $PYTHON"
  echo "[ci_rebuild_verify] db: $DB"
  echo "[ci_rebuild_verify] out_playlists: $OUT_PLAYLISTS"
  echo "[ci_rebuild_verify] out_tracks: $OUT_TRACKS"
  echo "[ci_rebuild_verify] stamp: $STAMP"
  echo "[ci_rebuild_verify] ARTIFACTS_DIR=$ARTIFACTS_DIR"
  echo "[ci_rebuild_verify] MGC_OUT_ROOT=${MGC_OUT_ROOT:-}"

  echo "[ci_rebuild_verify] == python =="
  "$PYTHON" -V

  if [[ ! -f "$DB" ]]; then
    echo "[ci_rebuild_verify] ERROR: DB not found: $DB" >&2
    exit 2
  fi

  echo "[ci_rebuild_verify] == rebuild playlists (determinism check + write) =="
  "$PYTHON" -m mgc.main \
    --db "$DB" \
    rebuild playlists \
    --out-dir "$OUT_PLAYLISTS" \
    --stamp "$STAMP" \
    --determinism-check \
    --write \
    --json \
    >/dev/null

  echo "[ci_rebuild_verify] == rebuild tracks (determinism check + write) =="
  "$PYTHON" -m mgc.main \
    --db "$DB" \
    rebuild tracks \
    --out-dir "$OUT_TRACKS" \
    --stamp "$STAMP" \
    --determinism-check \
    --write \
    --json \
    >/dev/null

  echo "[ci_rebuild_verify] == verify playlists (strict) =="
  "$PYTHON" -m mgc.main \
    --db "$DB" \
    rebuild verify playlists \
    --out-dir "$OUT_PLAYLISTS" \
    --stamp "$STAMP" \
    --strict \
    --json \
    >/dev/null

  echo "[ci_rebuild_verify] == verify tracks (strict) =="
  "$PYTHON" -m mgc.main \
    --db "$DB" \
    rebuild verify tracks \
    --out-dir "$OUT_TRACKS" \
    --stamp "$STAMP" \
    --strict \
    --json \
    >/dev/null

  echo "[ci_rebuild_verify] OK"

  echo "[ci_rebuild_verify] outputs:"
  echo "  playlists: $(ls -1 "$OUT_PLAYLISTS" 2>/dev/null | wc -l | tr -d ' ') files"
  echo "  tracks:    $(ls -1 "$OUT_TRACKS" 2>/dev/null | wc -l | tr -d ' ') files"
} 2>&1 | tee -a "$log_file"
