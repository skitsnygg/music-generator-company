#!/usr/bin/env bash
set -euo pipefail

# Determinism gate for submission.zip
#
# Env:
#   PYTHON         python executable (default: python)
#   BUNDLE_DIR     bundle dir to package (required)
#   ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#
# Output:
#   $ARTIFACTS_DIR/submission_determinism/run1/submission.zip
#   $ARTIFACTS_DIR/submission_determinism/run2/submission.zip
#   and a hash comparison

PYTHON="${PYTHON:-python}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"
: "${BUNDLE_DIR:?set BUNDLE_DIR to the drop bundle directory}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_root="$ARTIFACTS_DIR/submission_determinism"
run1_dir="$out_root/run1"
run2_dir="$out_root/run2"
mkdir -p "$run1_dir" "$run2_dir"

zip1="$run1_dir/submission.zip"
zip2="$run2_dir/submission.zip"

echo "[ci_submission_determinism] repo_root=$repo_root"
echo "[ci_submission_determinism] BUNDLE_DIR=$BUNDLE_DIR"
echo "[ci_submission_determinism] ARTIFACTS_DIR=$ARTIFACTS_DIR"

build_once () {
  local out_zip="$1"
  rm -f "$out_zip"
  # If your command differs, change it here:
  "$PYTHON" -m mgc.main submission build \
    --bundle-dir "$BUNDLE_DIR" \
    --out "$out_zip"
  test -f "$out_zip"
}

hash_zip_bytes () {
  local f="$1"
  shasum -a 256 "$f" | awk '{print $1}'
}

echo "[ci_submission_determinism] build run1 -> $zip1"
build_once "$zip1"
h1="$(hash_zip_bytes "$zip1")"
echo "[ci_submission_determinism] run1_sha256=$h1"

echo "[ci_submission_determinism] build run2 -> $zip2"
build_once "$zip2"
h2="$(hash_zip_bytes "$zip2")"
echo "[ci_submission_determinism] run2_sha256=$h2"

if [[ "$h1" != "$h2" ]]; then
  echo "[ci_submission_determinism] FAIL: submission.zip sha256 mismatch"
  echo "[ci_submission_determinism] run1=$h1"
  echo "[ci_submission_determinism] run2=$h2"
  echo "[ci_submission_determinism] hint: run 'diff -u <(unzip -l $zip1) <(unzip -l $zip2)' to see ordering/timestamp drift"
  exit 1
fi

echo "[ci_submission_determinism] OK"
