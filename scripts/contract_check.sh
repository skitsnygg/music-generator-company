#!/usr/bin/env bash
set -euxo pipefail

DB="${1:-fixtures/ci_db.sqlite}"
OUT="${2:-/tmp/mgc_contract}"

rm -rf "$OUT"
mkdir -p "$OUT"

# Run drop deterministically and capture the JSON it prints
python -m mgc.main --db "$DB" --seed 1 --no-resume run drop \
  --context focus --out-dir "$OUT" --deterministic \
  > "$OUT/drop_stdout.json"

python -m json.tool < "$OUT/drop_stdout.json" >/dev/null

# Portable bundle outputs (new contract)
test -d "$OUT/drop_bundle"
test -d "$OUT/drop_bundle/tracks"
test -s "$OUT/drop_bundle/playlist.json"
test -s "$OUT/drop_bundle/daily_evidence.json"

# Legacy top-level outputs (kept for now)
test -s "$OUT/playlist.json"
test -s "$OUT/drop_evidence.json"
test -s "$OUT/manifest.json"

# Build submission.zip from the portable bundle
python -m mgc.main submission build \
  --bundle-dir "$OUT/drop_bundle" \
  --out "$OUT/submission.zip"
test -s "$OUT/submission.zip"

# Build web bundle from playlist (prefer bundle playlist if present)
PLAYLIST_PATH="$OUT/playlist.json"
if [ -s "$OUT/drop_bundle/playlist.json" ]; then
  PLAYLIST_PATH="$OUT/drop_bundle/playlist.json"
fi

python -m mgc.main web build \
  --playlist "$PLAYLIST_PATH" \
  --out-dir "$OUT/web" \
  --clean \
  --fail-if-none-copied \
  --fail-on-missing
test -s "$OUT/web/web_manifest.json"

echo "OK: contract check passed (out_dir=$OUT)"
