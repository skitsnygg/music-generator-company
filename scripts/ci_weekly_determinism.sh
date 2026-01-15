#!/usr/bin/env bash
set -euo pipefail

# Weekly determinism gate:
# - run weekly twice with different out dirs
# - compare bundle JSONs (should be identical)
# - build submission.zip for each and compare sha256 (should be identical)
#
# Env:
#   MGC_DB (required)
#   PYTHON (optional; default: python)
#   REPO_ROOT (optional; default: .)
#   CONTEXT (optional; default: focus)
#   SEED (optional; default: 1)

: "${MGC_DB:?set MGC_DB}"

PYTHON="${PYTHON:-python}"
REPO_ROOT="${REPO_ROOT:-.}"
CONTEXT="${CONTEXT:-focus}"
SEED="${SEED:-1}"

STAMP="${1:-ci_weekly}"

out1="/tmp/mgc_${STAMP}_weekly_1"
out2="/tmp/mgc_${STAMP}_weekly_2"

rm -rf "$out1" "$out2"
mkdir -p "$out1" "$out2"

echo "[ci_weekly_determinism] run weekly twice and compare outputs"
echo "[ci_weekly_determinism] out1=$out1"
echo "[ci_weekly_determinism] out2=$out2"

"$PYTHON" -m mgc.main --db "$MGC_DB" --repo-root "$REPO_ROOT" --seed "$SEED" --no-resume --json \
  run weekly --context "$CONTEXT" --out-dir "$out1" --deterministic >/dev/null

"$PYTHON" -m mgc.main --db "$MGC_DB" --repo-root "$REPO_ROOT" --seed "$SEED" --no-resume --json \
  run weekly --context "$CONTEXT" --out-dir "$out2" --deterministic >/dev/null

echo "[ci_weekly_determinism] diff bundle jsons"
diff -u "$out1/drop_bundle/daily_evidence.json" "$out2/drop_bundle/daily_evidence.json" >/dev/null
diff -u "$out1/drop_bundle/playlist.json"      "$out2/drop_bundle/playlist.json"      >/dev/null

# receipts should be identical too (JSONL)
if [[ -f "$out1/marketing/receipts.jsonl" && -f "$out2/marketing/receipts.jsonl" ]]; then
  diff -u "$out1/marketing/receipts.jsonl" "$out2/marketing/receipts.jsonl" >/dev/null
fi

echo "[ci_weekly_determinism] build submission zips"
"$PYTHON" -m mgc.main --db "$MGC_DB" submission build \
  --bundle-dir "$out1/drop_bundle" --out "$out1/submission.zip" >/dev/null

"$PYTHON" -m mgc.main --db "$MGC_DB" submission build \
  --bundle-dir "$out2/drop_bundle" --out "$out2/submission.zip" >/dev/null

sha1="$(shasum -a 256 "$out1/submission.zip" | awk '{print $1}')"
sha2="$(shasum -a 256 "$out2/submission.zip" | awk '{print $1}')"

if [[ "$sha1" != "$sha2" ]]; then
  echo "[ci_weekly_determinism] FAIL: submission.zip sha mismatch"
  echo "[ci_weekly_determinism] run1=$sha1"
  echo "[ci_weekly_determinism] run2=$sha2"
  exit 2
fi

echo "[ci_weekly_determinism] OK: weekly bundle + submission.zip deterministic"
echo "[ci_weekly_determinism] sha=$sha1"
