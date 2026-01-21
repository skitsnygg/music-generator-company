#!/usr/bin/env bash
set -euxo pipefail

DB="${1:-fixtures/ci_db.sqlite}"
OUT="${2:-/tmp/mgc_contract}"

rm -rf "$OUT"
mkdir -p "$OUT"

DROP_JSON="$(python -m mgc.main --db "$DB" --seed 1 --no-resume run drop --context focus --out-dir "$OUT" --deterministic)"
export DROP_JSON

# Prefer explicit bundle dir from JSON, but fall back to OUT/drop_bundle for backward compatibility
BUNDLE_DIR_VALUE="$(python - <<'PY'
import json, os
j = json.loads(os.environ["DROP_JSON"])
p = j.get("paths") or {}
print(p.get("bundle_dir_path") or p.get("bundle_dir") or "")
PY
)"

if [ -z "${BUNDLE_DIR_VALUE}" ]; then
  BUNDLE_DIR="${OUT}/drop_bundle"
else
  case "${BUNDLE_DIR_VALUE}" in
    /*) BUNDLE_DIR="${BUNDLE_DIR_VALUE}" ;;
    *)  BUNDLE_DIR="${OUT}/${BUNDLE_DIR_VALUE}" ;;
  esac
fi

test -d "$BUNDLE_DIR"
test -d "$BUNDLE_DIR/tracks"
test -s "$BUNDLE_DIR/playlist.json"
test -s "$BUNDLE_DIR/daily_evidence.json"

# Legacy top-level outputs (keep for now)
test -s "$OUT/playlist.json"
test -s "$OUT/drop_evidence.json"
test -s "$OUT/manifest.json"

python -m mgc.main submission build --bundle-dir "$BUNDLE_DIR" --out "$OUT/submission.zip"
test -s "$OUT/submission.zip"

python -m mgc.main web build --playlist "$BUNDLE_DIR/playlist.json" --out-dir "$OUT/web" --clean --fail-if-none-copied --fail-on-missing
test -s "$OUT/web/web_manifest.json"

echo "OK: contract check passed (bundle_dir=$BUNDLE_DIR out_dir=$OUT)"
