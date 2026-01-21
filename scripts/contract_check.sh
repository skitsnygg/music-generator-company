#!/usr/bin/env bash
set -euxo pipefail

DB="${1:-fixtures/ci_db.sqlite}"
OUT="${2:-/tmp/mgc_contract}"

rm -rf "$OUT"
mkdir -p "$OUT"

DROP_JSON="$(python -m mgc.main --db "$DB" --seed 1 --no-resume run drop --context focus --out-dir "$OUT" --deterministic)"
export DROP_JSON

DROP_ID="$(python - <<'PY'
import json, os
j = json.loads(os.environ["DROP_JSON"])
print(j["drop_id"])
PY
)"

SUBMISSION_DIR="$(python - <<'PY'
import json, os
j = json.loads(os.environ["DROP_JSON"])
print(j["paths"]["submission_dir"])
PY
)"

SUBMISSION_ZIP_SHA256_EXPECTED="$(python - <<'PY'
import json, os
j = json.loads(os.environ["DROP_JSON"])
print(j["artifacts"]["submission_zip_sha256"])
PY
)"

test -s "$OUT/playlist.json"
test -s "$OUT/drop_evidence.json"
test -s "$OUT/manifest.json"

test -d "$SUBMISSION_DIR"
test -s "$SUBMISSION_DIR/submission.json"
test -s "$SUBMISSION_DIR/submission.zip"

# sha256(actual zip) == sha256 reported by run drop
export SUBMISSION_ZIP_PATH="$SUBMISSION_DIR/submission.zip"
SUBMISSION_ZIP_SHA256_ACTUAL="$(python - <<'PY'
import hashlib, os
p = os.environ["SUBMISSION_ZIP_PATH"]
h = hashlib.sha256()
with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(1024*1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"

test "$SUBMISSION_ZIP_SHA256_ACTUAL" = "$SUBMISSION_ZIP_SHA256_EXPECTED"

python -m mgc.main web build --playlist "$OUT/playlist.json" --out-dir "$OUT/web" --clean --fail-if-none-copied --fail-on-missing
test -s "$OUT/web/web_manifest.json"

echo "OK: contract check passed (drop_id=$DROP_ID submission_dir=$SUBMISSION_DIR)"