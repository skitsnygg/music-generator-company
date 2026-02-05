#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
DB="${MGC_DB:-${DB:-fixtures/ci_db.sqlite}}"
OUT_DIR="${OUT_DIR:-${MGC_EVIDENCE_DIR:-artifacts/ci/auto}}"

# Deterministic defaults (override if you want live runs)
export MGC_DB="$DB"
export MGC_DETERMINISTIC="${MGC_DETERMINISTIC:-1}"
export MGC_FIXED_TIME="${MGC_FIXED_TIME:-2020-01-01T00:00:00Z}"

# Optional knobs
export MGC_CONTEXT="${MGC_CONTEXT:-focus}"
export MGC_SEED="${MGC_SEED:-1}"
PROVIDER="${MGC_PROVIDER:-}"

mkdir -p "$OUT_DIR"

args=(-m mgc.main run autonomous --out-dir "$OUT_DIR")
if [[ -n "${PROVIDER}" ]]; then
  args+=( --provider "${PROVIDER}" )
fi
"$PYTHON" "${args[@]}"
