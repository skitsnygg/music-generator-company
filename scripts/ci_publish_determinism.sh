#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DB="${MGC_DB:-${DB:-data/db.sqlite}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"

STAMP="${1:-ci_publish}"
LIMIT="${LIMIT:-50}"

mkdir -p "$ARTIFACTS_DIR"

info() { echo "[ci_publish_determinism] $*"; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

info "run publish-marketing twice (deterministic) and compare sha256"

MGC_DB="$DB" MGC_DETERMINISTIC=1 MGC_FIXED_TIME="2020-01-01T00:00:00Z" \
  "$PYTHON" -m mgc.main run publish-marketing \
    --deterministic \
    --limit "$LIMIT" \
    --dry-run \
    >"$tmp/pub1.json"

MGC_DB="$DB" MGC_DETERMINISTIC=1 MGC_FIXED_TIME="2020-01-01T00:00:00Z" \
  "$PYTHON" -m mgc.main run publish-marketing \
    --deterministic \
    --limit "$LIMIT" \
    --dry-run \
    >"$tmp/pub2.json"

H1="$("$PYTHON" - "$tmp/pub1.json" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
)"

H2="$("$PYTHON" - "$tmp/pub2.json" <<'PY'
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1])
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
)"

if [[ "$H1" != "$H2" ]]; then
  echo "[ci_publish_determinism] FAIL: output sha256 mismatch"
  echo "[ci_publish_determinism] run1=${H1}"
  echo "[ci_publish_determinism] run2=${H2}"
  exit 1
fi

echo "[ci_publish_determinism] OK: stamp=${STAMP} sha256=${H1}"
