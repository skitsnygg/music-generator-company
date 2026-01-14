#!/usr/bin/env bash
set -euo pipefail

# Determinism gate for web bundle directory output.
#
# Env:
#   PYTHON         python executable (default: python)
#   PLAYLIST       playlist json path (required)
#   ARTIFACTS_DIR  default: artifacts/ci
#
# It hashes the *directory tree* deterministically:
# - sort file list
# - sha256 each file
# - sha256 the combined listing (path + filehash)

PYTHON="${PYTHON:-python}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"
: "${PLAYLIST:?set PLAYLIST to a playlist JSON path}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_root="$ARTIFACTS_DIR/web_bundle_determinism"
run1_dir="$out_root/run1"
run2_dir="$out_root/run2"
mkdir -p "$out_root"

echo "[ci_web_bundle_determinism] repo_root=$repo_root"
echo "[ci_web_bundle_determinism] PLAYLIST=$PLAYLIST"
echo "[ci_web_bundle_determinism] ARTIFACTS_DIR=$ARTIFACTS_DIR"

build_web () {
  local out_dir="$1"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  # If your command differs, change it here:
  "$PYTHON" -m mgc.main web build \
    --playlist "$PLAYLIST" \
    --out-dir "$out_dir" \
    --clean
  test -d "$out_dir"
}

tree_hash () {
  local root="$1"
  (
    cd "$root"
    # portable-ish file listing (mac + linux)
    find . -type f -print | LC_ALL=C sort | while IFS= read -r p; do
      # strip leading "./" for stable output
      pp="${p#./}"
      h="$(shasum -a 256 "$pp" | awk '{print $1}')"
      printf "%s  %s\n" "$h" "$pp"
    done
  ) | shasum -a 256 | awk '{print $1}'
}

echo "[ci_web_bundle_determinism] build run1 -> $run1_dir"
build_web "$run1_dir"
h1="$(tree_hash "$run1_dir")"
echo "[ci_web_bundle_determinism] run1_tree_sha256=$h1"

echo "[ci_web_bundle_determinism] build run2 -> $run2_dir"
build_web "$run2_dir"
h2="$(tree_hash "$run2_dir")"
echo "[ci_web_bundle_determinism] run2_tree_sha256=$h2"

if [[ "$h1" != "$h2" ]]; then
  echo "[ci_web_bundle_determinism] FAIL: web bundle tree hash mismatch"
  echo "[ci_web_bundle_determinism] run1=$h1"
  echo "[ci_web_bundle_determinism] run2=$h2"
  echo "[ci_web_bundle_determinism] hint: compare file hashes:"
  echo "  (cd $run1_dir && find . -type f -print | sort | xargs -I{} shasum -a 256 \"{}\") > $out_root/run1_files.sha"
  echo "  (cd $run2_dir && find . -type f -print | sort | xargs -I{} shasum -a 256 \"{}\") > $out_root/run2_files.sha"
  echo "  diff -u $out_root/run1_files.sha $out_root/run2_files.sha || true"
  exit 1
fi

echo "[ci_web_bundle_determinism] OK"
