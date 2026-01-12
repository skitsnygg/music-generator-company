#!/usr/bin/env bash
set -euo pipefail

: "${MGC_DB:?set MGC_DB}"

export MGC_DETERMINISTIC=1
export MGC_FIXED_TIME="${MGC_FIXED_TIME:-2020-01-01T00:00:00Z}"

tmp="${TMPDIR:-/tmp}/mgc_run_det"
rm -rf "$tmp"
mkdir -p "$tmp"

python -m mgc.main run daily --context focus --seed 1 > "$tmp/daily1.json"
python -m mgc.main run daily --context focus --seed 1 > "$tmp/daily2.json"
diff -u "$tmp/daily1.json" "$tmp/daily2.json"

python -m mgc.main run publish-marketing --limit 10 --dry-run > "$tmp/pub1.json"
python -m mgc.main run publish-marketing --limit 10 --dry-run > "$tmp/pub2.json"
diff -u "$tmp/pub1.json" "$tmp/pub2.json"

python -m mgc.main run manifest --repo-root . > "$tmp/man1.json"
python -m mgc.main run manifest --repo-root . > "$tmp/man2.json"
diff -u "$tmp/man1.json" "$tmp/man2.json"

echo "[ci_run_determinism] OK"
