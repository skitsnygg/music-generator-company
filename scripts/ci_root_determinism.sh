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

  # IMPORTANT:
  # --json is a GLOBAL mgc option, so it must come immediately after "-m mgc.main"
  MGC_DETERMINISTIC=1 MGC_FIXED_TIME="2020-01-01T00:00:00Z" MGC_DB="${MGC_DB}" \
    "$PYTHON" -m mgc.main --json run drop \
      --db "${MGC_DB}" \
      --context "${CONTEXT}" \
      --seed "${SEED}" \
      --limit "${LIMIT}" \
      --deterministic \
    >"$out_json"

  "$PYTHON" - <<PY
import json, pathlib, sys
p = pathlib.Path("${out_json}")
raw = p.read_text(encoding="utf-8").strip()
try:
    obj = json.loads(raw)
except Exception:
    print("[ci_root_determinism] ERROR: output is not valid JSON", file=sys.stderr)
    print("----- begin raw -----", file=sys.stderr)
    print(raw[:2000], file=sys.stderr)
    print("----- end raw -----", file=sys.stderr)
    sys.exit(2)

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
