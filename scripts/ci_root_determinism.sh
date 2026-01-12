#!/usr/bin/env bash
set -euo pipefail

: "${MGC_DB:?set MGC_DB}"
PYTHON="${PYTHON:-python}"

STAMP="${1:-ci_root}"
CONTEXT="${MGC_CONTEXT:-focus}"
SEED="${MGC_SEED:-1}"
LIMIT="${MGC_PUBLISH_LIMIT:-200}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"
mkdir -p "$ARTIFACTS_DIR"

info() { echo "[ci_root_determinism] $*"; }

hash_once() {
  rm -rf "$ARTIFACTS_DIR/runs/${STAMP}" "$ARTIFACTS_DIR/receipts/${STAMP}" || true

  local out_json="$ARTIFACTS_DIR/root_${STAMP}_out.json"

  MGC_DETERMINISTIC=1 MGC_FIXED_TIME="2020-01-01T00:00:00Z" MGC_DB="${MGC_DB}" \
    "$PYTHON" -m mgc.main run drop \
      --db "${MGC_DB}" \
      --context "${CONTEXT}" \
      --seed "${SEED}" \
      --limit "${LIMIT}" \
      --deterministic \
    >"$out_json"

  "$PYTHON" - <<PY
import json, pathlib, sys
p = pathlib.Path("${out_json}")
obj = json.loads(p.read_text(encoding="utf-8"))
h = (obj.get("paths") or {}).get("manifest_sha256") or ""
if not h:
    print("missing paths.manifest_sha256", file=sys.stderr)
    sys.exit(2)
print(h)
PY
}

info "compute root hash twice and compare (deterministic)"
H1="$(hash_once)"
H2="$(hash_once)"

if [[ "$H1" != "$H2" ]]; then
  echo "[ci_root_determinism] FAIL: root sha256 mismatch"
  echo "[ci_root_determinism] run1=$H1"
  echo "[ci_root_determinism] run2=$H2"
  exit 1
fi

echo "[ci_root_determinism] OK: stamp=${STAMP} root_sha256=${H1}"
