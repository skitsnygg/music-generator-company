#!/usr/bin/env bash
set -euo pipefail

# scripts/ci_run_determinism.sh
#
# Determinism checks for core pipeline steps:
# - run daily (force generation so it never returns empty)
# - run publish-marketing (dry-run)
# - run manifest
#
# CI MUST provide MGC_DB and we MUST pass it explicitly.

: "${MGC_DB:?set MGC_DB}"

PYTHON="${PYTHON:-python}"

export MGC_DETERMINISTIC=1
export MGC_FIXED_TIME="${MGC_FIXED_TIME:-2020-01-01T00:00:00Z}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tmp="${TMPDIR:-/tmp}/mgc_run_det"
rm -rf "$tmp"
mkdir -p "$tmp"

echo "[ci_run_determinism] repo_root=$repo_root"
echo "[ci_run_determinism] MGC_DB=$MGC_DB"
echo "[ci_run_determinism] MGC_FIXED_TIME=$MGC_FIXED_TIME"

OUT_DIR="$tmp/out"

run_daily_once() {
  local out_json="$1"
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR"

  # IMPORTANT:
  # - --generate-count 1 ensures daily never returns an empty playlist
  # - --deterministic makes output stable across runs
  "$PYTHON" -m mgc.main \
    --db "$MGC_DB" \
    --repo-root "$repo_root" \
    --seed 1 \
    --no-resume \
    --json \
    run daily \
    --context focus \
    --out-dir "$OUT_DIR" \
    --deterministic \
    --generate-count 1 > "$out_json"
}

# Daily determinism
run_daily_once "$tmp/daily1.json"
run_daily_once "$tmp/daily2.json"
diff -u "$tmp/daily1.json" "$tmp/daily2.json" >/dev/null

# Publish marketing determinism (dry-run, no mutation)
"$PYTHON" -m mgc.main --db "$MGC_DB" --seed 1 --no-resume --json \
  run publish-marketing --limit 10 --dry-run > "$tmp/pub1.json"
"$PYTHON" -m mgc.main --db "$MGC_DB" --seed 1 --no-resume --json \
  run publish-marketing --limit 10 --dry-run > "$tmp/pub2.json"
diff -u "$tmp/pub1.json" "$tmp/pub2.json" >/dev/null

# Manifest determinism
"$PYTHON" -m mgc.main --db "$MGC_DB" --seed 1 --no-resume --json \
  run manifest --repo-root "$repo_root" > "$tmp/man1.json"
"$PYTHON" -m mgc.main --db "$MGC_DB" --seed 1 --no-resume --json \
  run manifest --repo-root "$repo_root" > "$tmp/man2.json"
diff -u "$tmp/man1.json" "$tmp/man2.json" >/dev/null

echo "[ci_run_determinism] OK"
