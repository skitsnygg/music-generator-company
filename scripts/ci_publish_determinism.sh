#!/usr/bin/env bash
set -euo pipefail

: "${MGC_DB:?set MGC_DB}"
STAMP="${1:-ci_publish}"

# Start clean so filesystem state can't affect results
rm -rf "artifacts/receipts/${STAMP}" "artifacts/_tmp_receipts_a" "artifacts/_tmp_receipts_b"
mkdir -p artifacts

run_once () {
  local outdir="$1"
  rm -rf "artifacts/receipts/${STAMP}"
  python -m mgc.main publish marketing --stamp "${STAMP}" --deterministic --limit 200 >/dev/null
  cp -R "artifacts/receipts/${STAMP}" "${outdir}"
}

run_once "artifacts/_tmp_receipts_a"
run_once "artifacts/_tmp_receipts_b"

# Compare byte-for-byte
diff -qr "artifacts/_tmp_receipts_a" "artifacts/_tmp_receipts_b" >/dev/null

echo "[ci_publish_determinism] OK: receipts deterministic for stamp=${STAMP}"
