#!/usr/bin/env bash
set -euo pipefail

# CI gate: compile + rebuild/verify + determinism checks
#
# Modes:
#   - fast (default): py_compile + rebuild/verify
#   - full: fast + autonomous smoke + submission/web determinism + publish determinism + weekly determinism
#
# Env:
#   MGC_DB         DB path (default: fixtures/ci_db.sqlite in CI, or data/db.sqlite locally)
#   PYTHON         python executable (default: python)
#   ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT   override output root for rebuilds:
#                  - if set to "data", writes to data/playlists + data/tracks
#                  - otherwise writes under $ARTIFACTS_DIR/data/...
#
# Exit codes:
#   0 ok
#   2 determinism failure / script failures
#   3 golden mismatch / rebuild verify failures
#   4 missing expected evidence outputs

MODE="${1:-fast}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
MGC_DB="${MGC_DB:-fixtures/ci_db.sqlite}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"
MGC_OUT_ROOT="${MGC_OUT_ROOT:-}"

mkdir -p "$ARTIFACTS_DIR"

echo "[ci_gate] mode=$MODE"
echo "[ci_gate] Repo: $repo_root"
echo "[ci_gate] MGC_DB=$MGC_DB"
echo "[ci_gate] MGC_ARTIFACTS_DIR=$ARTIFACTS_DIR"
echo "[ci_gate] MGC_OUT_ROOT=$MGC_OUT_ROOT"
echo "[ci_gate] python: $("$PYTHON" -V 2>&1 || true)"

# -----------------------------
# Helpers
# -----------------------------

_choose_rebuild_out_dirs() {
  local out_root="${1:-}"
  OUT_PLAYLISTS=""
  OUT_TRACKS=""

  if [ -z "$out_root" ]; then
    OUT_PLAYLISTS="$ARTIFACTS_DIR/rebuild/playlists"
    OUT_TRACKS="$ARTIFACTS_DIR/rebuild/tracks"
    return 0
  fi

  if [ "$out_root" = "data" ]; then
    OUT_PLAYLISTS="data/playlists"
    OUT_TRACKS="data/tracks"
    return 0
  fi

  OUT_PLAYLISTS="$out_root/playlists"
  OUT_TRACKS="$out_root/tracks"
}

_detect_evidence_root() {
  # Accept evidence roots that either:
  # - contain drop_evidence.json directly, OR
  # - contain evidence/drop_evidence.json
  local base="${1:-}"
  if [ -z "$base" ]; then
    return 1
  fi
  if [ -f "${base%/}/drop_evidence.json" ]; then
    printf "%s" "${base%/}"
    return 0
  fi
  if [ -f "${base%/}/evidence/drop_evidence.json" ]; then
    printf "%s" "${base%/}/evidence"
    return 0
  fi
  return 1
}

_run_web_bundle_determinism() {
  local evidence_root="${1:?evidence_root}"
  if [ -f "$repo_root/scripts/ci_web_bundle_determinism.sh" ]; then
  ( cd "$evidence_root" && MGC_DB="$MGC_DB" PYTHON="$PYTHON" bash "$repo_root/scripts/ci_web_bundle_determinism.sh" --evidence-root "." )
    return 0
  fi
  echo "[ci_gate] WARN: scripts/ci_web_bundle_determinism.sh missing; skipping web determinism"
  return 0
}

# -----------------------------
# Basic sanity
# -----------------------------

if [ ! -f "$MGC_DB" ]; then
  echo "[ci_gate] FAIL: DB not found: $MGC_DB" >&2
  exit 4
fi

echo "[ci_gate] DB OK."

# -----------------------------
# Compile
# -----------------------------

echo "[ci_gate] py_compile"
"$PYTHON" -m py_compile \
  src/mgc/main.py \
  src/mgc/run_cli.py \
  src/mgc/submission_cli.py \
  src/mgc/web_cli.py >/dev/null

# -----------------------------
# Rebuild + verify (deterministic)
# -----------------------------

STAMP="${STAMP:-ci_root}"
_choose_rebuild_out_dirs "$MGC_OUT_ROOT"

rm -rf "$OUT_PLAYLISTS" "$OUT_TRACKS"
mkdir -p "$OUT_PLAYLISTS" "$OUT_TRACKS"

echo "[ci_gate] rebuild + verify"
"$PYTHON" -m mgc.main --db "$MGC_DB" rebuild playlists \
  --out-dir "$OUT_PLAYLISTS" \
  --stamp "$STAMP" \
  --determinism-check \
  --write \
  --json >/dev/null

"$PYTHON" -m mgc.main --db "$MGC_DB" rebuild tracks \
  --out-dir "$OUT_TRACKS" \
  --stamp "$STAMP" \
  --determinism-check \
  --write \
  --json >/dev/null

"$PYTHON" -m mgc.main --db "$MGC_DB" rebuild verify playlists --out-dir "$OUT_PLAYLISTS" --json >/dev/null
"$PYTHON" -m mgc.main --db "$MGC_DB" rebuild verify tracks --out-dir "$OUT_TRACKS" --json >/dev/null

if [ "$MODE" != "full" ]; then
  echo "[ci_gate] OK"
  exit 0
fi

# -----------------------------
# Full mode: autonomous smoke + determinism
# -----------------------------

AUTO_OUT="${ARTIFACTS_DIR%/}/auto"
rm -rf "$AUTO_OUT"
mkdir -p "$AUTO_OUT"

AUTONOMOUS_JSON="${AUTO_OUT%/}/autonomous.json"
AUTONOMOUS_ERR="${AUTO_OUT%/}/autonomous.err"
rm -f "$AUTONOMOUS_JSON" "$AUTONOMOUS_ERR"

echo "[ci_gate] autonomous smoke test (deterministic) out_dir=$AUTO_OUT"

cmd=(
  "$PYTHON" -m mgc.main
  --db "$MGC_DB"
  --repo-root "$repo_root"
  --seed 1
  --no-resume
  --json
  run autonomous
  --context focus
  --out-dir "$AUTO_OUT"
  --deterministic
)

printf "[ci_gate] autonomous argv:"
for a in "${cmd[@]}"; do printf " %q" "$a"; done
printf "\n"

set +e
"${cmd[@]}" >"$AUTONOMOUS_JSON" 2>"$AUTONOMOUS_ERR"
rc=$?
set -e

if [ "$rc" -ne 0 ]; then
  echo "[ci_gate] FAIL: autonomous returned rc=$rc" >&2
  echo "[ci_gate] autonomous stderr (first 200 lines):" >&2
  sed -n '1,200p' "$AUTONOMOUS_ERR" >&2 || true
  echo "[ci_gate] autonomous stdout (first 200 lines):" >&2
  sed -n '1,200p' "$AUTONOMOUS_JSON" >&2 || true
  exit "$rc"
fi

# Catch a common failure mode: argparse help printed to stdout (non-json)
if grep -Eq 'usage:|positional arguments:|optional arguments:' "$AUTONOMOUS_JSON"; then
  echo "[ci_gate] FAIL: autonomous printed argparse help to stdout" >&2
  echo "[ci_gate] autonomous stderr (first 200 lines):" >&2
  sed -n '1,200p' "$AUTONOMOUS_ERR" >&2 || true
  echo "[ci_gate] autonomous stdout (first 200 lines):" >&2
  sed -n '1,200p' "$AUTONOMOUS_JSON" >&2 || true
  exit 2
fi

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

if [ ! -f "${EVIDENCE_ROOT%/}/drop_evidence.json" ]; then
  echo "[ci_gate] FAIL: missing drop_evidence.json under $EVIDENCE_ROOT" >&2
  ls -la "$EVIDENCE_ROOT" >&2 || true
  exit 4
fi

echo "[ci_gate] determinism gate: submission.zip (evidence-root=$EVIDENCE_ROOT)"
( cd "$EVIDENCE_ROOT" && bash "$repo_root/scripts/ci_submission_determinism.sh" --evidence-root "." )

echo "[ci_gate] determinism gate: web bundle (evidence-root=$EVIDENCE_ROOT)"
_run_web_bundle_determinism "$EVIDENCE_ROOT"

echo "[ci_gate] weekly determinism"
MGC_DB="$MGC_DB" PYTHON="$PYTHON" \
MGC_DETERMINISTIC=1 DETERMINISTIC=1 \
  bash "$repo_root/scripts/ci_weekly_determinism.sh" ci_weekly

else
  echo "[ci_gate] WARN: scripts/ci_weekly_determinism.sh missing; skipping weekly determinism"
fi

echo "[ci_gate] drops list smoke (global --json hoist)"
"$PYTHON" -m mgc.main --db "$MGC_DB" drops list --json >/dev/null

echo "[ci_gate] publish receipts determinism"
if [ -f "$repo_root/scripts/ci_publish_determinism.sh" ]; then
  bash "$repo_root/scripts/ci_publish_determinism.sh" --db "$MGC_DB" --artifacts-dir "$ARTIFACTS_DIR"
else
  echo "[ci_gate] WARN: scripts/ci_publish_determinism.sh missing; skipping publish determinism"
fi

echo "[ci_gate] OK"
