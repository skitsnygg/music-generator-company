#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-docs/fixtures/releases}"
OUT_JSON="${2:-docs/releases/feed.json}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "[pages_demo] ERROR: missing fixtures root: $ROOT_DIR" >&2
  echo "[pages_demo] Expected something like: $ROOT_DIR/latest/web/<context>/index.html" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_JSON")"

echo "[pages_demo] Generating stable feed..."
python3 scripts/release_feed.py \
  --root-dir "$ROOT_DIR" \
  --out "$OUT_JSON" \
  --stable

python3 -m json.tool "$OUT_JSON" >/dev/null
echo "[pages_demo] OK wrote $OUT_JSON ($(wc -c <"$OUT_JSON" | tr -d ' ') bytes)"
