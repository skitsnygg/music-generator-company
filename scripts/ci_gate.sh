#!/usr/bin/env bash
# scripts/ci_gate.sh
set -euo pipefail

# CI gate: compile + rebuild/verify + determinism checks + autonomous smoke + golden hash (optional)
#
# Env:
#   MGC_DB         DB path (required)
#   MGC_ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT   override output root for rebuilds:
#                  - if set to "data", writes to data/playlists + data/tracks
#                  - otherwise writes under $MGC_ARTIFACTS_DIR/rebuild/...
#   MGC_GOLDEN_STRICT  if "1"/"true"/"yes", fail CI if submission.zip sha not in ci/known_good_submission_sha256.txt

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${MGC_DB:?set MGC_DB}"

PYTHON="${PYTHON:-python}"
ARTIFACTS_DIR="${MGC_ARTIFACTS_DIR:-artifacts/ci}"
OUT_ROOT="${MGC_OUT_ROOT:-}"

mkdir -p "$ARTIFACTS_DIR"

echo "[ci_gate] MGC_ARTIFACTS_DIR=$ARTIFACTS_DIR"
echo "[ci_gate] Repo: $repo_root"
echo "[ci_gate] MGC_DB=$MGC_DB"
echo "[ci_gate] ARTIFACTS_DIR=$ARTIFACTS_DIR"
echo "[ci_gate] MGC_OUT_ROOT=$OUT_ROOT"
echo "[ci_gate] git_sha: $(git rev-parse HEAD)"
echo "[ci_gate] git_branch: $(git rev-parse --abbrev-ref HEAD)"
echo "[ci_gate] python: $($PYTHON --version)"

# quick DB check
$PYTHON - <<PY
import sqlite3, os
con = sqlite3.connect(os.environ["MGC_DB"])
try:
    con.execute("SELECT 1").fetchone()
finally:
    con.close()
PY
echo "[ci_gate] DB OK."

echo "[ci_gate] py_compile"
$PYTHON -m py_compile src/mgc/main.py src/mgc/run_cli.py src/mgc/web_cli.py src/mgc/submission_cli.py

AUTO_OUT="${ARTIFACTS_DIR%/}/auto"
mkdir -p "$AUTO_OUT"

echo "[ci_gate] autonomous smoke test (deterministic) out_dir=$AUTO_OUT"
MGC_DETERMINISTIC=1 \
$PYTHON -m mgc.main --db "$MGC_DB" --json run autonomous \
  --context focus \
  --seed 1 \
  --out-dir "$AUTO_OUT" \
  --repo-root "$repo_root" \
  --no-resume \
  > "${AUTO_OUT%/}/autonomous.json"

echo "[ci_gate] submission artifact check"
bash scripts/ci_submission_determinism.sh --evidence-root "$AUTO_OUT"
echo "[ci_gate] submission artifact check OK"

echo "[ci_gate] determinism gate: submission.zip (evidence-root=$AUTO_OUT)"
bash scripts/ci_submission_determinism.sh --evidence-root "$AUTO_OUT"

echo "[ci_gate] determinism gate: web bundle (evidence-root=$AUTO_OUT)"
bash scripts/ci_web_bundle_determinism.sh --evidence-root "$AUTO_OUT"

echo "[ci_gate] drops list smoke (global --json hoist)"
$PYTHON -m mgc.main --db "$MGC_DB" drops list --json >/dev/null

echo "[ci_gate] rebuild + verify"
# Delegate to your existing rebuild verify script if present; otherwise do direct calls.
if [ -x "scripts/ci_rebuild_verify.sh" ]; then
  bash scripts/ci_rebuild_verify.sh
else
  # Reasonable defaults for rebuild outputs under artifacts
  OUT_PLAYLISTS="${ARTIFACTS_DIR%/}/rebuild/playlists"
  OUT_TRACKS="${ARTIFACTS_DIR%/}/rebuild/tracks"
  STAMP="ci"

  mkdir -p "$OUT_PLAYLISTS" "$OUT_TRACKS"

  $PYTHON -m mgc.main --db "$MGC_DB" rebuild playlists --out-dir "$OUT_PLAYLISTS" --stamp "$STAMP" --determinism-check --write --json >/dev/null
  $PYTHON -m mgc.main --db "$MGC_DB" rebuild tracks    --out-dir "$OUT_TRACKS"    --stamp "$STAMP" --determinism-check --write --json >/dev/null
  $PYTHON -m mgc.main --db "$MGC_DB" rebuild verify playlists --out-dir "$OUT_PLAYLISTS" --stamp "$STAMP" --strict --json >/dev/null
  $PYTHON -m mgc.main --db "$MGC_DB" rebuild verify tracks    --out-dir "$OUT_TRACKS"    --stamp "$STAMP" --strict --json >/dev/null
fi
echo "[ci_gate] rebuild + verify OK"

echo "[ci_gate] publish receipts determinism"
if [ -x "scripts/ci_publish_determinism.sh" ]; then
  bash scripts/ci_publish_determinism.sh
else
  echo "[ci_gate] (skip) scripts/ci_publish_determinism.sh not found"
fi

echo "[ci_gate] manifest diff gate (since-ok, strict JSON)"
$PYTHON -m mgc.main --db "$MGC_DB" run diff --since-ok --fail-on-changes --summary-only --json | $PYTHON -m json.tool

echo "[ci_gate] golden hash gate (warn by default)"
MODE="warn"
v="${MGC_GOLDEN_STRICT:-0}"
if [ "$v" = "1" ] || [ "$v" = "true" ] || [ "$v" = "yes" ]; then
  MODE="strict"
fi
bash scripts/ci_golden_hash_gate.sh \
  --evidence-root "$AUTO_OUT" \
  --known-file "ci/known_good_submission_sha256.txt" \
  --mode "$MODE"

echo "[ci_gate] OK"
