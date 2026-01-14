#!/usr/bin/env bash
# scripts/ci_gate.sh
set -euo pipefail

# CI gate with modes:
#   - fast (default): compile + rebuild/verify + optional golden tree hashes (WARN-only by default)
#   - full: everything (autonomous smoke + submission/web determinism + publish determinism + manifest diff + golden checks)
#
# Env:
#   MGC_DB             DB path (required)
#   PYTHON             python executable (default: python)
#   MGC_ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT       override output root for rebuilds:
#                      - if set to "data", writes to data/playlists + data/tracks
#                      - otherwise writes under $MGC_ARTIFACTS_DIR/rebuild/...
#   MGC_CI_MODE        fast|full (default: fast)
#   MGC_GOLDEN_STRICT  if "1"/"true"/"yes":
#                      - full: enforce golden tree hashes + known-good submission hash
#                      - fast: enforce golden tree hashes (otherwise warn-only)
#
# Evidence conventions (robust):
#   - Some code paths write evidence files directly under <out_dir>/ (drop_evidence.json, manifest.json, etc.)
#   - Some code paths may write under <out_dir>/evidence/
#   - CI auto-detects which one is real by locating drop_evidence.json
#
# Full mode extras:
#   - manifest diff: if since-ok not available, SKIP (do not fail CI)
#   - golden submission hash gate: warn-only truly warns (nonzero rc ignored unless strict)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${MGC_DB:?set MGC_DB}"

PYTHON="${PYTHON:-python}"
ARTIFACTS_DIR="${MGC_ARTIFACTS_DIR:-artifacts/ci}"
OUT_ROOT="${MGC_OUT_ROOT:-}"
CI_MODE="${MGC_CI_MODE:-fast}"

if [ "$CI_MODE" != "fast" ] && [ "$CI_MODE" != "full" ]; then
  echo "[ci_gate] ERROR: MGC_CI_MODE must be 'fast' or 'full' (got: $CI_MODE)" >&2
  exit 2
fi

_env_truthy() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  [ "$v" = "1" ] || [ "$v" = "true" ] || [ "$v" = "yes" ] || [ "$v" = "y" ] || [ "$v" = "on" ]
}

mkdir -p "$ARTIFACTS_DIR"

echo "[ci_gate] mode=$CI_MODE"
echo "[ci_gate] Repo: $repo_root"
echo "[ci_gate] MGC_DB=$MGC_DB"
echo "[ci_gate] MGC_ARTIFACTS_DIR=$ARTIFACTS_DIR"
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
# rebuild + verify (fast + full)
# -----------------------------
echo "[ci_gate] rebuild + verify"

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
# golden TREE hash gate (fast + full, optional)
# - fast: WARN-only by default
# - full: STRICT by default
# - MGC_GOLDEN_STRICT forces STRICT in either mode
# -----------------------------
GOLDEN_JSON="fixtures/golden_hashes.json"
if [ -f "$GOLDEN_JSON" ] && [ -f "scripts/ci_golden_check.py" ]; then
  echo "[ci_gate] golden tree hash gate"

  # Optional: bless goldens from the current outputs (useful to align with CI runner).
  # Only does anything if MGC_GOLDEN_BLESS is truthy.
  if _env_truthy "${MGC_GOLDEN_BLESS:-0}"; then
    echo "[ci_gate] blessing golden hashes (MGC_GOLDEN_BLESS=1)"
    "$PYTHON" scripts/ci_golden_bless.py --golden "$GOLDEN_JSON" --key ci.rebuild.playlists --root "$OUT_PLAYLISTS"
    "$PYTHON" scripts/ci_golden_bless.py --golden "$GOLDEN_JSON" --key ci.rebuild.tracks    --root "$OUT_TRACKS"
    echo "[ci_gate] golden hashes blessed"
  fi


    GOLDEN_STRICT=0
  if _env_truthy "${MGC_GOLDEN_STRICT:-0}"; then
    GOLDEN_STRICT=1
  fi

  set +e
  "$PYTHON" scripts/ci_golden_check.py \
    --golden "$GOLDEN_JSON" \
    --key ci.rebuild.playlists \
    --root "$OUT_PLAYLISTS"
  rc_playlists=$?

  "$PYTHON" scripts/ci_golden_check.py \
    --golden "$GOLDEN_JSON" \
    --key ci.rebuild.tracks \
    --root "$OUT_TRACKS"
  rc_tracks=$?
  set -e

  if [ $rc_playlists -ne 0 ] || [ $rc_tracks -ne 0 ]; then
    if [ $GOLDEN_STRICT -eq 1 ]; then
      echo "[ci_gate] FAIL: golden tree hash mismatch (strict mode)"
      exit 3
    fi
    echo "[ci_gate] WARN: golden tree hash mismatch (warn-only mode; continuing)"
  else
    echo "[ci_gate] golden tree hash OK"
  fi
else
  echo "[ci_gate] golden tree hash SKIP (missing fixtures/golden_hashes.json or scripts/ci_golden_check.py)"
fi

# -----------------------------
# helpers: evidence root detection
# -----------------------------
_detect_evidence_root() {
  # Prefer <out_dir>/evidence if it contains drop_evidence.json; otherwise <out_dir>.
  local out_dir="$1"
  if [ -f "${out_dir%/}/evidence/drop_evidence.json" ]; then
    echo "${out_dir%/}/evidence"
    return 0
  fi
  if [ -f "${out_dir%/}/drop_evidence.json" ]; then
    echo "${out_dir%/}"
    return 0
  fi
  # Not found
  echo ""
  return 1
}

# -----------------------------
# full-mode gates
# -----------------------------
if [ "$CI_MODE" = "full" ]; then
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

  EVIDENCE_ROOT="$(_detect_evidence_root "$AUTO_OUT" || true)"
  if [ -z "$EVIDENCE_ROOT" ]; then
    echo "[ci_gate] FAIL: could not locate drop_evidence.json under $AUTO_OUT or $AUTO_OUT/evidence" >&2
    echo "[ci_gate] contents of $AUTO_OUT:" >&2
    ls -la "$AUTO_OUT" >&2 || true
    if [ -d "${AUTO_OUT%/}/evidence" ]; then
      echo "[ci_gate] contents of ${AUTO_OUT%/}/evidence:" >&2
      ls -la "${AUTO_OUT%/}/evidence" >&2 || true
    fi
    exit 4
  fi

  export MGC_EVIDENCE_DIR="$EVIDENCE_ROOT"
  echo "[ci_gate] evidence_root=$EVIDENCE_ROOT"

  echo "[ci_gate] determinism gate: submission.zip (evidence-root=$EVIDENCE_ROOT)"
  bash scripts/ci_submission_determinism.sh --evidence-root "$EVIDENCE_ROOT"

  echo "[ci_gate] determinism gate: web bundle (evidence-root=$EVIDENCE_ROOT)"
  bash scripts/ci_web_bundle_determinism.sh --evidence-root "$EVIDENCE_ROOT"

  echo "[ci_gate] drops list smoke (global --json hoist)"
  $PYTHON -m mgc.main --db "$MGC_DB" drops list --json >/dev/null

  echo "[ci_gate] publish receipts determinism"
  if [ -x "scripts/ci_publish_determinism.sh" ]; then
    bash scripts/ci_publish_determinism.sh
  else
    echo "[ci_gate] (skip) scripts/ci_publish_determinism.sh not found"
  fi

  # -----------------------------
  # manifest diff gate (since-ok, strict JSON)
  # - if since-ok not available in CI workspace, SKIP (do not fail)
  # -----------------------------
  echo "[ci_gate] manifest diff gate (since-ok, strict JSON) evidence_root=$EVIDENCE_ROOT"

  set +e
  DIFF_OUT="$($PYTHON -m mgc.main --db "$MGC_DB" run diff \
    --since-ok --fail-on-changes --summary-only --json \
    --out-dir "$EVIDENCE_ROOT" \
    2>/dev/null)"
  rc_diff=$?
  set -e

  if [ $rc_diff -ne 0 ]; then
    if echo "$DIFF_OUT" | grep -q '"reason"[[:space:]]*:[[:space:]]*"since_ok_not_found"'; then
      echo "$DIFF_OUT" | $PYTHON -m json.tool
      echo "[ci_gate] manifest diff gate SKIP (since-ok not present)"
    else
      echo "$DIFF_OUT" >&2
      echo "[ci_gate] manifest diff gate FAIL"
      exit $rc_diff
    fi
  else
    echo "$DIFF_OUT" | $PYTHON -m json.tool
  fi

  # -----------------------------
  # golden SUBMISSION hash gate (known-good list)
  # - warn by default; strict only if MGC_GOLDEN_STRICT truthy
  # -----------------------------
  echo "[ci_gate] golden submission hash gate (warn by default)"

  SUB_MODE="warn"
  if _env_truthy "${MGC_GOLDEN_STRICT:-0}"; then
    SUB_MODE="strict"
  fi

  set +e
  bash scripts/ci_golden_hash_gate.sh \
    --evidence-root "$EVIDENCE_ROOT" \
    --known-file "ci/known_good_submission_sha256.txt" \
    --mode "$SUB_MODE"
  rc_sub=$?
  set -e

  if [ $rc_sub -ne 0 ]; then
    if [ "$SUB_MODE" = "strict" ]; then
      echo "[ci_gate] FAIL: submission golden hash gate (strict mode)"
      exit $rc_sub
    fi
    echo "[ci_gate] WARN: submission not in known-good list (warn-only mode; continuing)"
  fi
fi

echo "[ci_gate] OK"
