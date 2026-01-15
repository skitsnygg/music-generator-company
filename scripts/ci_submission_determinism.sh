#!/usr/bin/env bash
set -euo pipefail

# scripts/ci_submission_determinism.sh
#
# Determinism gate for submission packaging.
# We build the submission zip twice from the SAME evidence root and compare sha256.
#
# Usage:
#   bash scripts/ci_submission_determinism.sh --evidence-root <dir>
#
# Notes:
# - In ci_gate.sh we often run this as:
#     (cd "$EVIDENCE_ROOT" && bash scripts/ci_submission_determinism.sh --evidence-root ".")
#   so DO NOT assume $EVIDENCE_ROOT env var exists.

PYTHON="${PYTHON:-python}"

EVIDENCE_ROOT=""

usage() {
  echo "usage: $0 --evidence-root <dir>" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence-root)
      EVIDENCE_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ci_submission_determinism] unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$EVIDENCE_ROOT" ]]; then
  echo "[ci_submission_determinism] FAIL: --evidence-root is required" >&2
  usage
  exit 2
fi

EVIDENCE_ROOT="$(cd "$EVIDENCE_ROOT" && pwd)"
ZIP1="$EVIDENCE_ROOT/submission.zip"
ZIP2="$EVIDENCE_ROOT/submission_2.zip"

rm -f "$ZIP1" "$ZIP2"

# Build twice from the same evidence root (packaging must be deterministic)
"$PYTHON" -m mgc.main submission build --evidence-root "$EVIDENCE_ROOT" --out "$ZIP1" >/dev/null || true
"$PYTHON" -m mgc.main submission build --evidence-root "$EVIDENCE_ROOT" --out "$ZIP2" >/dev/null || true

if [[ ! -f "$ZIP1" || ! -f "$ZIP2" ]]; then
  echo "[ci_submission_determinism] FAIL: submission zips not created" >&2
  echo "[ci_submission_determinism] expected: $ZIP1 and $ZIP2" >&2
  ls -la "$EVIDENCE_ROOT" >&2 || true
  exit 2
fi

S1="$(shasum -a 256 "$ZIP1" | awk '{print $1}')"
S2="$(shasum -a 256 "$ZIP2" | awk '{print $1}')"

if [[ "$S1" != "$S2" ]]; then
  echo "[ci_submission_determinism] FAIL: submission.zip sha256 mismatch" >&2
  echo "[ci_submission_determinism] zip1=$ZIP1 sha256=$S1" >&2
  echo "[ci_submission_determinism] zip2=$ZIP2 sha256=$S2" >&2
  exit 2
fi

# Optional: verify (permissive verify should pass)
"$PYTHON" -m mgc.main submission verify --zip "$ZIP1" >/dev/null || {
  echo "[ci_submission_determinism] FAIL: submission verify failed for $ZIP1" >&2
  exit 2
}

echo "[ci_submission_determinism] OK sha256=$S1"
