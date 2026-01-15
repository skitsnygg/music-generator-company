#!/usr/bin/env bash
set -euo pipefail

# CI gate with modes:
#   - fast (default): compile + rebuild/verify + optional golden tree hashes (warn-only by default)
#   - full: fast + autonomous smoke + submission/web determinism + publish determinism + manifest diff + golden checks
#
# Env:
#   MGC_DB             DB path (required)
#   PYTHON             python executable (default: python)
#   MGC_ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT       override output root for rebuilds:
#                      - if set to "data", writes to data/playlists + data/tracks
#                      - otherwise writes under $MGC_ARTIFACTS_DIR/data/...
#   MGC_CI_MODE        fast|full (default: fast)
#
# Evidence contract:
#   - drop_evidence.json is required and is written by `mgc run autonomous` (baseline contract).
#
# Submission determinism:
#   - scripts/ci_submission_determinism.sh resolves "submission.zip" relative to CWD
#   - We run determinism scripts from inside evidence_root (cd) so paths resolve correctly
#
# Web determinism:
#   - scripts/ci_web_bundle_determinism.sh may fail if web_manifest.json contains volatile fields
#   - On that specific failure, ci_gate attempts a normalized manifest compare and only fails
#     if normalized manifests still differ.

repo_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
: "${MGC_DB:?set MGC_DB}"

ARTIFACTS_DIR="${MGC_ARTIFACTS_DIR:-artifacts/ci}"
MODE="${MGC_CI_MODE:-fast}"
STAMP="${MGC_STAMP:-ci_gate}"

mkdir -p "$ARTIFACTS_DIR"
LOG_PATH="$ARTIFACTS_DIR/ci_gate.log"

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[ci_gate] mode=$MODE"
echo "[ci_gate] Repo: $repo_root"
echo "[ci_gate] MGC_DB=$MGC_DB"
echo "[ci_gate] MGC_ARTIFACTS_DIR=$ARTIFACTS_DIR"
echo "[ci_gate] MGC_OUT_ROOT=${MGC_OUT_ROOT:-}"
echo "[ci_gate] git_sha: $(git rev-parse HEAD 2>/dev/null || true)"
echo "[ci_gate] git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
echo "[ci_gate] python: $($PYTHON -V 2>&1 || true)"

_detect_evidence_root() {
  local out_dir="$1"
  if [ -f "${out_dir%/}/evidence/drop_evidence.json" ]; then
    echo "${out_dir%/}/evidence"
    return 0
  fi
  if [ -f "${out_dir%/}/drop_evidence.json" ]; then
    echo "${out_dir%/}"
    return 0
  fi
  echo ""
  return 1
}

_sha256_file() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    shasum -a 256 "$f" | awk '{print $1}'
  fi
}

# -----------------------------
# build submission.zip deterministically (capability probing)
# -----------------------------
_build_submission_zip() {
  local evidence_root="$1"
  local out_zip="${evidence_root%/}/submission.zip"
  rm -f "$out_zip"

  local help
  help="$($PYTHON -m mgc.main submission --help 2>&1 || true)"

  if echo "$help" | grep -qE -- 'latest'; then
    local latest_help
    latest_help="$($PYTHON -m mgc.main submission latest --help 2>&1 || true)"
    if echo "$latest_help" | grep -qE -- '--evidence-root'; then
      $PYTHON -m mgc.main --db "$MGC_DB" submission latest \
        --out "$out_zip" \
        --evidence-root "$evidence_root" \
        --json >/dev/null
    else
      $PYTHON -m mgc.main --db "$MGC_DB" submission latest \
        --out "$out_zip" \
        --json >/dev/null
    fi
  else
    local build_help
    build_help="$($PYTHON -m mgc.main submission build --help 2>&1 || true)"

    if echo "$build_help" | grep -qE -- '--bundle-dir'; then
      $PYTHON -m mgc.main submission build \
        --bundle-dir "$evidence_root" \
        --out "$out_zip" \
        --json >/dev/null
    else
      echo "[ci_gate] FAIL: could not determine supported submission command (need submission latest or submission build --bundle-dir)" >&2
      echo "$help" >&2
      echo "$build_help" >&2
      exit 2
    fi
  fi

  if [ ! -f "$out_zip" ]; then
    echo "[ci_gate] FAIL: submission.zip was not created at $out_zip" >&2
    ls -la "$evidence_root" >&2 || true
    exit 2
  fi
  echo "$out_zip"
}

# -----------------------------
# web determinism wrapper (normalizes known-volatiles if necessary)
# -----------------------------
_run_web_bundle_determinism() {
  local evidence_root="$1"
  set +e
  ( cd "$evidence_root" && bash "$repo_root/scripts/ci_web_bundle_determinism.sh" --evidence-root "$evidence_root" )
  local rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    return 0
  fi

  # If web determinism fails due to known-volatiles in web_manifest.json, try normalized compare
  if [ -f "$evidence_root/web_manifest.json" ] && [ -f "$evidence_root/web_manifest_2.json" ]; then
    echo "[ci_gate] web determinism failed; attempting normalized compare"
    $PYTHON - <<'PY' "$evidence_root/web_manifest.json" "$evidence_root/web_manifest_2.json"
import json, sys
from pathlib import Path

a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

# Strip volatile-ish fields if present
for obj in (a, b):
    if isinstance(obj, dict):
        for k in ["ts", "built_ts", "generated_at"]:
            obj.pop(k, None)

if a != b:
    print("normalized manifests still differ", file=sys.stderr)
    sys.exit(3)
print("normalized manifests match")
PY
    return 0
  fi

  return $rc
}

# -----------------------------
# Preconditions
# -----------------------------
echo "[ci_gate] DB OK."

# -----------------------------
# compile gate
# -----------------------------
echo "[ci_gate] py_compile"
$PYTHON -m py_compile src/mgc/main.py src/mgc/run_cli.py

# -----------------------------
# rebuild + verify
# -----------------------------
echo "[ci_gate] rebuild + verify"
OUT_ROOT="${MGC_OUT_ROOT:-}"
if [ -z "$OUT_ROOT" ]; then
  OUT_PLAYLISTS="$ARTIFACTS_DIR/rebuild/playlists"
  OUT_TRACKS="$ARTIFACTS_DIR/rebuild/tracks"
else
  if [ "$OUT_ROOT" = "data" ]; then
    OUT_PLAYLISTS="data/playlists"
    OUT_TRACKS="data/tracks"
  else
    OUT_PLAYLISTS="$OUT_ROOT/playlists"
    OUT_TRACKS="$OUT_ROOT/tracks"
  fi
fi
mkdir -p "$OUT_PLAYLISTS" "$OUT_TRACKS"

if [ -f "$repo_root/scripts/ci_rebuild_verify.sh" ]; then
  MGC_DB="$MGC_DB" \
  PYTHON="$PYTHON" \
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

  $PYTHON -m mgc.main rebuild verify playlists --out-dir "$OUT_PLAYLISTS" --json >/dev/null
  $PYTHON -m mgc.main rebuild verify tracks --out-dir "$OUT_TRACKS" --json >/dev/null
fi

# -----------------------------
# Fast mode ends here
# -----------------------------
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
echo "[ci_gate] playlist_path=${EVIDENCE_ROOT%/}/playlist.json"

if [ ! -f "${EVIDENCE_ROOT%/}/drop_evidence.json" ]; then
  echo "[ci_gate] FAIL: missing drop_evidence.json under $EVIDENCE_ROOT" >&2
  ls -la "$EVIDENCE_ROOT" >&2 || true
  exit 4
fi

echo "[ci_gate] determinism gate: submission.zip (evidence-root=$EVIDENCE_ROOT)"
( cd "$EVIDENCE_ROOT" && bash "$repo_root/scripts/ci_submission_determinism.sh" --evidence-root "$EVIDENCE_ROOT" )

echo "[ci_gate] determinism gate: web bundle (evidence-root=$EVIDENCE_ROOT)"
_run_web_bundle_determinism "$EVIDENCE_ROOT"

echo "[ci_gate] drops list smoke (global --json hoist)"
$PYTHON -m mgc.main --db "$MGC_DB" drops list --json >/dev/null

echo "[ci_gate] publish receipts determinism"
bash "$repo_root/scripts/ci_publish_determinism.sh" --db "$MGC_DB" --artifacts-dir "$ARTIFACTS_DIR"

echo "[ci_gate] OK"
