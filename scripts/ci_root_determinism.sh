#!/usr/bin/env bash
set -euo pipefail

: "${MGC_DB:?set MGC_DB}"
PYTHON="${PYTHON:-python}"

STAMP="${1:-ci_root}"
CONTEXT="${MGC_CONTEXT:-focus}"
SEED="${MGC_SEED:-1}"
TARGET_MINUTES="${MGC_TARGET_MINUTES:-20}"
LIMIT="${MGC_PUBLISH_LIMIT:-200}"

hash_once() {
  rm -rf "artifacts/runs/${STAMP}" "artifacts/receipts/${STAMP}"

  # Build run evidence bundle deterministically
  MGC_DETERMINISTIC=1 "$PYTHON" -m mgc.main run daily \
    --db "${MGC_DB}" \
    --stamp "${STAMP}" \
    --context "${CONTEXT}" \
    --seed "${SEED}" \
    --target-minutes "${TARGET_MINUTES}" \
    --deterministic \
    >/dev/null

  # Build deterministic receipts (still simulated)
  MGC_DETERMINISTIC=1 "$PYTHON" -m mgc.main publish marketing \
    --db "${MGC_DB}" \
    --stamp "${STAMP}" \
    --deterministic \
    --limit "${LIMIT}" \
    >/dev/null

  # Root hash across run evidence + receipts (single line)
  MGC_DETERMINISTIC=1 "$PYTHON" -m mgc.main run manifest \
    --stamp "${STAMP}" \
    --include-receipts \
    --hash-only \
    | head -n 1
}

H1="$(hash_once)"
H2="$(hash_once)"

if [[ "$H1" != "$H2" ]]; then
  echo "[ci_root_determinism] FAIL: root_tree_sha256 mismatch"
  echo "[ci_root_determinism] run1=$H1"
  echo "[ci_root_determinism] run2=$H2"
  exit 1
fi

echo "[ci_root_determinism] OK: stamp=${STAMP} root_tree_sha256=${H1}"
