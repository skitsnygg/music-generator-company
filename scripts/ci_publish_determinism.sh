#!/usr/bin/env bash
set -euo pipefail

: "${MGC_DB:?set MGC_DB}"
PYTHON="${PYTHON:-python}"

STAMP="${1:-ci_publish}"
LIMIT="${MGC_PUBLISH_LIMIT:-200}"

receipts_root="artifacts/receipts/${STAMP}/marketing"
manifest_path="${receipts_root}/_manifest.receipts.json"

clean() {
  rm -rf "artifacts/receipts/${STAMP}"
}

tree_hash() {
  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("${manifest_path}")
if not p.exists():
    raise SystemExit(f"manifest missing: {p}")
m = json.loads(p.read_text(encoding="utf-8"))
print(m["tree_sha256"])
PY
}

clean
"$PYTHON" -m mgc.main publish marketing --stamp "${STAMP}" --deterministic --limit "${LIMIT}" >/dev/null
H1="$(tree_hash)"

clean
"$PYTHON" -m mgc.main publish marketing --stamp "${STAMP}" --deterministic --limit "${LIMIT}" >/dev/null
H2="$(tree_hash)"

if [[ "${H1}" != "${H2}" ]]; then
  echo "[ci_publish_determinism] FAIL: tree_sha256 mismatch"
  echo "[ci_publish_determinism] run1=${H1}"
  echo "[ci_publish_determinism] run2=${H2}"
  exit 1
fi

echo "[ci_publish_determinism] OK: stamp=${STAMP} tree_sha256=${H1}"
