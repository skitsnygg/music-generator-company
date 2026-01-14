#!/usr/bin/env bash
# scripts/ci_gate.sh
set -euo pipefail

# CI gate: compile + rebuild/verify + determinism checks + autonomous smoke + golden hash (optional)
#
# Env:
#   MGC_DB             DB path (required)
#   PYTHON             python executable (default: python)
#   MGC_ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT       override output root for rebuilds:
#                      - if set to "data", writes to data/playlists + data/tracks
#                      - otherwise writes under $MGC_ARTIFACTS_DIR/rebuild/...
#   MGC_GOLDEN_STRICT  if "1"/"true"/"yes", fail CI if submission.zip sha not in ci/known_good_submission_sha256.txt
#
# Optional golden-tree hashes (recommended):
#   fixtures/golden_hashes.json
#   scripts/ci_golden_check.py (uses mgc.hash_tree)
#
# If fixtures/golden_hashes.json exists, we will check:
#   - ci.rebuild.playlists against rebuild playlists output dir
#   - ci.rebuild.tracks    against rebuild tracks output dir
#
# Bless/update via:
#   python scripts/ci_golden_bless.py --golden fixtures/golden_hashes.json --key ci.rebuild.playlists --root <dir>
#   python scripts/ci_golden_bless.py --golden fixtures/golden_hashes.json --key ci.rebuild.tracks    --root <dir>

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

# -----------------------------
# quick DB check
# -----------------------------
$PYTHON - <<'PY'
import os, sqlite3
con = sqlite3.connect(os.environ["MGC_DB"])
try:
    con.execute("SELECT 1").fetchone()
finally:
    con.close()
PY
echo "[ci_gate] DB OK."

# -----------------------------
# compile
# -----------------------------
echo "[ci_gate] py_compile"
$PYTHON -m py_compile \
  src/mgc/main.py \
  src/mgc/run_cli.py \
  src/mgc/web_cli.py \
  src/mgc/submission_cli.py

# -----------------------------
# autonomous deterministic smoke
# -----------------------------
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

# -----------------------------
# submission determinism + artifact check
# -----------------------------
echo "[ci_gate] submission artifact check"
bash scripts/ci_submission_determinism.sh --evidence-root "$AUTO_OUT"
echo "[ci_gate] submission artifact check OK"

echo "[ci_gate] determinism gate: submission.zip (evidence-root=$AUTO_OUT)"
bash scripts/ci_submission_determinism.sh --evidence-root "$AUTO_OUT"

# -----------------------------
# web bundle determinism (optional)
# -----------------------------
echo "[ci_gate] determinism gate: web bundle (evidence-root=$AUTO_OUT)"
bash scripts/ci_web_bundle_determinism.sh --evidence-root "$AUTO_OUT"

# -----------------------------
# drops list smoke (global --json hoist)
# -----------------------------
echo "[ci_gate] drops list smoke (global --json hoist)"
$PYTHON -m mgc.main --db "$MGC_DB" drops list --json >/dev/null

# -----------------------------
# rebuild + verify
# -----------------------------
echo "[ci_gate] rebuild + verify"

# Default paths (also used for golden-tree hashing below)
OUT_PLAYLISTS=""
OUT_TRACKS=""
STAMP="ci"

if [ "$OUT_ROOT" = "data" ]; then
  OUT_PLAYLISTS="data/playlists"
  OUT_TRACKS="data/tracks"
else
  OUT_PLAYLISTS="${ARTIFACTS_DIR%/}/rebuild/playlists"
  OUT_TRACKS="${ARTIFACTS_DIR%/}/rebuild/tracks"
fi

mkdir -p "$OUT_PLAYLISTS" "$OUT_TRACKS"

if [ -x "scripts/ci_rebuild_verify.sh" ]; then
  # Keep compatibility with your existing helper, but ensure it targets the same output dirs when possible.
  # If your helper ignores these env vars, it can still do its own thing; golden checks will use the dirs above.
  MGC_OUT_PLAYLISTS="$OUT_PLAYLISTS" \
  MGC_OUT_TRACKS="$OUT_TRACKS" \
  MGC_STAMP="$STAMP" \
  bash scripts/ci_rebuild_verify.sh
else
  $PYTHON -m mgc.main --db "$MGC_DB" rebuild playlists \
    --out-dir "$OUT_PLAYLISTS" \
    --stamp "$STAMP" \
    --determinism-check \
    --write \
    --json >/dev/null

  $PYTHON -m mgc.main --db "$MGC_DB" rebuild tracks \
    --out-dir "$OUT_TRACKS" \
    --stamp "$STAMP" \
    --determinism-check \
    --write \
    --json >/dev/null

  $PYTHON -m mgc.main --db "$MGC_DB" rebuild verify playlists \
    --out-dir "$OUT_PLAYLISTS" \
    --stamp "$STAMP" \
    --strict \
    --json >/dev/null

  $PYTHON -m mgc.main --db "$MGC_DB" rebuild verify tracks \
    --out-dir "$OUT_TRACKS" \
    --stamp "$STAMP" \
    --strict \
    --json >/dev/null
fi

echo "[ci_gate] rebuild + verify OK"
echo "[ci_gate] rebuild outputs:"
echo "  playlists: $OUT_PLAYLISTS"
echo "  tracks:    $OUT_TRACKS"

# -----------------------------
# golden TREE hash gate (fixtures/golden_hashes.json) - optional
# -----------------------------
GOLDEN_JSON="fixtures/golden_hashes.json"
if [ -f "$GOLDEN_JSON" ] && [ -f "scripts/ci_golden_check.py" ]; then
  echo "[ci_gate] golden tree hash gate (golden_hashes.json)"

  # Playlists
  $PYTHON scripts/ci_golden_check.py \
    --golden "$GOLDEN_JSON" \
    --key "ci.rebuild.playlists" \
    --root "$OUT_PLAYLISTS"

  # Tracks
  $PYTHON scripts/ci_golden_check.py \
    --golden "$GOLDEN_JSON" \
    --key "ci.rebuild.tracks" \
    --root "$OUT_TRACKS"

  echo "[ci_gate] golden tree hash OK"
else
  echo "[ci_gate] golden tree hash SKIP (missing $GOLDEN_JSON or scripts/ci_golden_check.py)"
fi

# -----------------------------
# publish receipts determinism
# -----------------------------
echo "[ci_gate] publish receipts determinism"
if [ -x "scripts/ci_publish_determinism.sh" ]; then
  bash scripts/ci_publish_determinism.sh
else
  echo "[ci_gate] (skip) scripts/ci_publish_determinism.sh not found"
fi

# -----------------------------
# manifest diff gate (since-ok, strict JSON)
# -----------------------------
echo "[ci_gate] manifest diff gate (since-ok, strict JSON)"
$PYTHON -m mgc.main --db "$MGC_DB" run diff --since-ok --fail-on-changes --summary-only --json | $PYTHON -m json.tool

# -----------------------------
# golden SUBMISSION hash gate (known-good list) - warn by default
# -----------------------------
echo "[ci_gate] golden submission hash gate (warn by default)"
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
