#!/usr/bin/env bash
set -euo pipefail

# scripts/ci_web_bundle_determinism.sh
#
# Build the static web bundle twice (deterministic) and verify outputs match.
#
# Usage:
#   MGC_DB=fixtures/ci_db.sqlite bash scripts/ci_web_bundle_determinism.sh --evidence-root <dir> [--tag <name>]
#
# Notes:
# - This script MUST NOT pass --out-dir as a global flag to mgc.main.
#   It must be passed after the subcommand (e.g., `web build --out-dir ...`).
# - Prefer playlist from the portable bundle:
#     <evidence-root>/drop_bundle/playlist.json
#   Fallback:
#     <evidence-root>/playlist.json

PYTHON="${PYTHON:-python}"
: "${MGC_DB:?set MGC_DB}"

EVIDENCE_ROOT="."
TAG="web"

while [ $# -gt 0 ]; do
  case "$1" in
    --evidence-root)
      EVIDENCE_ROOT="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-web}"
      shift 2
      ;;
    *)
      echo "[ci_web_bundle_determinism] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

echo "[ci_web_bundle_determinism] evidence_root=$EVIDENCE_ROOT"
echo "[ci_web_bundle_determinism] MGC_DB=$MGC_DB"
echo "[ci_web_bundle_determinism] python=$($PYTHON -V 2>&1 || true)"

root="$(cd "$EVIDENCE_ROOT" && pwd)"

playlist=""
if [ -f "$root/drop_bundle/playlist.json" ]; then
  playlist="$root/drop_bundle/playlist.json"
elif [ -f "$root/playlist.json" ]; then
  playlist="$root/playlist.json"
else
  echo "[ci_web_bundle_determinism] FAIL: playlist.json not found." >&2
  echo "[ci_web_bundle_determinism] Expected:" >&2
  echo "  - $root/drop_bundle/playlist.json" >&2
  echo "  - $root/playlist.json" >&2
  echo "[ci_web_bundle_determinism] Contents of evidence root:" >&2
  ls -la "$root" >&2 || true
  if [ -d "$root/drop_bundle" ]; then
    echo "[ci_web_bundle_determinism] Contents of drop_bundle:" >&2
    ls -la "$root/drop_bundle" >&2 || true
  fi
  exit 3
fi

echo "[ci_web_bundle_determinism] playlist=$playlist"

out1="/tmp/mgc_${TAG}_bundle_1"
out2="/tmp/mgc_${TAG}_bundle_2"
rm -rf "$out1" "$out2"
mkdir -p "$out1" "$out2"

build_one() {
  local out_dir="$1"
  # IMPORTANT: --out-dir must come AFTER `web build`
  MGC_DETERMINISTIC=1 \
  "$PYTHON" -m mgc.main \
    --db "$MGC_DB" \
    web build \
    --playlist "$playlist" \
    --out-dir "$out_dir" \
    --clean \
    --json >/dev/null
}

echo "[ci_web_bundle_determinism] build #1 -> $out1"
build_one "$out1"

echo "[ci_web_bundle_determinism] build #2 -> $out2"
build_one "$out2"

m1="$out1/web_manifest.json"
m2="$out2/web_manifest.json"

if [ ! -f "$m1" ] || [ ! -f "$m2" ]; then
  echo "[ci_web_bundle_determinism] FAIL: missing web_manifest.json" >&2
  echo "  m1=$m1 exists? $([ -f "$m1" ] && echo yes || echo no)" >&2
  echo "  m2=$m2 exists? $([ -f "$m2" ] && echo yes || echo no)" >&2
  echo "[ci_web_bundle_determinism] out1 listing:" >&2
  ls -la "$out1" >&2 || true
  echo "[ci_web_bundle_determinism] out2 listing:" >&2
  ls -la "$out2" >&2 || true
  exit 4
fi

# Normalize known volatile keys before comparing manifests.
norm1="$out1/web_manifest_norm.json"
norm2="$out2/web_manifest_norm.json"

"$PYTHON" - <<'PY' "$m1" "$m2" "$norm1" "$norm2"
import json, sys
from pathlib import Path

m1, m2, o1, o2 = map(Path, sys.argv[1:5])

a = json.loads(m1.read_text(encoding="utf-8"))
b = json.loads(m2.read_text(encoding="utf-8"))

VOLATILE = {"ts", "built_ts", "generated_at", "created_at", "updated_at"}

def scrub(x):
    if isinstance(x, dict):
        for k in list(x.keys()):
            if k in VOLATILE:
                x.pop(k, None)
            else:
                scrub(x[k])
    elif isinstance(x, list):
        for it in x:
            scrub(it)

scrub(a)
scrub(b)

o1.write_text(json.dumps(a, indent=2, sort_keys=True) + "\n", encoding="utf-8")
o2.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if ! diff -u "$norm1" "$norm2" >/dev/null; then
  echo "[ci_web_bundle_determinism] FAIL: normalized web_manifest.json differs" >&2
  diff -u "$norm1" "$norm2" | sed -n '1,200p' >&2 || true
  exit 5
fi

# Optional: ensure the directory tree (excluding the manifest itself) matches by hashing file contents.
tree_hash() {
  local dir="$1"
  ( cd "$dir" && \
    find . -type f \
      ! -name 'web_manifest.json' \
      ! -name 'web_manifest_norm.json' \
      -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}' )
}

h1="$(tree_hash "$out1")"
h2="$(tree_hash "$out2")"

if [ "$h1" != "$h2" ]; then
  echo "[ci_web_bundle_determinism] FAIL: web bundle file tree differs" >&2
  echo "  hash1=$h1" >&2
  echo "  hash2=$h2" >&2
  echo "[ci_web_bundle_determinism] showing first 200 differing files (by sha listing)" >&2
  ( cd "$out1" && find . -type f ! -name 'web_manifest.json' -print0 | sort -z | xargs -0 shasum -a 256 ) >"/tmp/mgc_${TAG}_sha_1.txt"
  ( cd "$out2" && find . -type f ! -name 'web_manifest.json' -print0 | sort -z | xargs -0 shasum -a 256 ) >"/tmp/mgc_${TAG}_sha_2.txt"
  diff -u "/tmp/mgc_${TAG}_sha_1.txt" "/tmp/mgc_${TAG}_sha_2.txt" | sed -n '1,200p' >&2 || true
  exit 6
fi

echo "[ci_web_bundle_determinism] OK hash=$h1"
