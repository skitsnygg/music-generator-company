#!/usr/bin/env bash
set -euo pipefail

# scripts/ci_weekly_determinism.sh
#
# Run the weekly pipeline twice in deterministic mode and compare JSON outputs.
#
# Expected env:
#   MGC_DB   path to sqlite db (required)
# Optional:
#   PYTHON   python executable (default: python)
#   CONTEXT  pipeline context (default: focus)
#   SEED     deterministic seed (default: 1)

: "${MGC_DB:?set MGC_DB}"
PYTHON="${PYTHON:-python}"
CONTEXT="${CONTEXT:-focus}"
SEED="${SEED:-1}"

STAMP="${1:-ci_weekly}"

echo "[ci_weekly_determinism] run weekly twice and compare outputs"
out1="/tmp/mgc_ci_weekly_${STAMP}_1"
out2="/tmp/mgc_ci_weekly_${STAMP}_2"

echo "[ci_weekly_determinism] out1=$out1"
echo "[ci_weekly_determinism] out2=$out2"

rm -rf "$out1" "$out2"
mkdir -p "$out1" "$out2"

# Force deterministic mode in the CLI AND in env (belt + suspenders).
export MGC_DETERMINISTIC=1
export DETERMINISTIC=1

run_weekly() {
  local out_dir="$1"
  "$PYTHON" -m mgc.main \
    --db "$MGC_DB" \
    --seed "$SEED" \
    --no-resume \
    run weekly \
    --context "$CONTEXT" \
    --out-dir "$out_dir" \
    --deterministic \
    >/dev/null
}

run_weekly "$out1"
run_weekly "$out2"

# Compare json sha256s (portable determinism check)
( cd "$out1" && find . -name "*.json" -print0 | sort -z | xargs -0 shasum -a 256 ) > /tmp/mgc_weekly_json_hashes_1.txt
( cd "$out2" && find . -name "*.json" -print0 | sort -z | xargs -0 shasum -a 256 ) > /tmp/mgc_weekly_json_hashes_2.txt

if ! diff -u /tmp/mgc_weekly_json_hashes_1.txt /tmp/mgc_weekly_json_hashes_2.txt >/tmp/mgc_weekly_json_hashes.diff; then
  echo "[ci_weekly_determinism] FAIL: json outputs differ"
  sed -n '1,160p' /tmp/mgc_weekly_json_hashes.diff
  exit 2
fi

echo "[ci_weekly_determinism] OK"
