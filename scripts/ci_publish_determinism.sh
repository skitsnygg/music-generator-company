#!/usr/bin/env bash
set -euo pipefail

# scripts/ci_publish_determinism.sh
#
# Run publish-marketing twice (deterministic) and verify receipts/evidence are stable.
#
# Usage:
#   bash scripts/ci_publish_determinism.sh --db <db> --artifacts-dir <dir>
#
# Notes:
# - Must not pass unsupported flags (no --limit, no --dry-run).
# - Uses `mgc run publish-marketing` and compares:
#     1) evidence/publish_marketing_evidence.json (normalized)
#     2) marketing/receipts.jsonl (exact)
# - Deterministic mode: fixed time + deterministic ids.

PYTHON="${PYTHON:-python}"

DB=""
ARTIFACTS_DIR="artifacts/ci"
TAG="publish"

while [ $# -gt 0 ]; do
  case "$1" in
    --db)
      DB="${2:-}"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="${2:-artifacts/ci}"
      shift 2
      ;;
    --tag)
      TAG="${2:-publish}"
      shift 2
      ;;
    *)
      echo "[ci_publish_determinism] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$DB" ]; then
  echo "[ci_publish_determinism] missing --db" >&2
  exit 2
fi

echo "[ci_publish_determinism] run publish-marketing twice (deterministic) and compare sha256"

out1="/tmp/mgc_${TAG}_pub_1"
out2="/tmp/mgc_${TAG}_pub_2"
rm -rf "$out1" "$out2"
mkdir -p "$out1" "$out2"

run_one() {
  local out_dir="$1"
  # Make sure outputs are isolated but content remains deterministic
  MGC_DETERMINISTIC=1 \
  MGC_FIXED_TIME="2020-01-01T00:00:00Z" \
  "$PYTHON" -m mgc.main \
    --db "$DB" \
    --repo-root "." \
    --seed 1 \
    --no-resume \
    --json \
    run publish-marketing \
    --out-dir "$out_dir" \
    --deterministic \
    >/dev/null
}

run_one "$out1"
run_one "$out2"

ev1="$out1/evidence/publish_marketing_evidence.json"
ev2="$out2/evidence/publish_marketing_evidence.json"
rc1="$out1/marketing/receipts.jsonl"
rc2="$out2/marketing/receipts.jsonl"

if [ ! -f "$ev1" ] || [ ! -f "$ev2" ]; then
  echo "[ci_publish_determinism] FAIL: missing publish_marketing_evidence.json" >&2
  echo "  ev1=$ev1 exists? $([ -f "$ev1" ] && echo yes || echo no)" >&2
  echo "  ev2=$ev2 exists? $([ -f "$ev2" ] && echo yes || echo no)" >&2
  echo "[ci_publish_determinism] out1 listing:" >&2
  ls -la "$out1/evidence" >&2 || true
  echo "[ci_publish_determinism] out2 listing:" >&2
  ls -la "$out2/evidence" >&2 || true
  exit 3
fi

if [ ! -f "$rc1" ] || [ ! -f "$rc2" ]; then
  echo "[ci_publish_determinism] FAIL: missing receipts.jsonl" >&2
  echo "  rc1=$rc1 exists? $([ -f "$rc1" ] && echo yes || echo no)" >&2
  echo "  rc2=$rc2 exists? $([ -f "$rc2" ] && echo yes || echo no)" >&2
  echo "[ci_publish_determinism] out1 marketing listing:" >&2
  ls -la "$out1/marketing" >&2 || true
  echo "[ci_publish_determinism] out2 marketing listing:" >&2
  ls -la "$out2/marketing" >&2 || true
  exit 4
fi

# Evidence json: normalize volatile keys and compare
n1="/tmp/mgc_${TAG}_ev_norm_1.json"
n2="/tmp/mgc_${TAG}_ev_norm_2.json"

"$PYTHON" - <<'PY' "$ev1" "$ev2" "$n1" "$n2"
import json, sys
from pathlib import Path

ev1, ev2, o1, o2 = map(Path, sys.argv[1:5])
a = json.loads(ev1.read_text(encoding="utf-8"))
b = json.loads(ev2.read_text(encoding="utf-8"))

VOL = {"ts","created_at","updated_at","generated_at","built_ts"}
def scrub(x):
    if isinstance(x, dict):
        for k in list(x.keys()):
            if k in VOL:
                x.pop(k, None)
            else:
                scrub(x[k])
    elif isinstance(x, list):
        for it in x:
            scrub(it)

scrub(a); scrub(b)
o1.write_text(json.dumps(a, indent=2, sort_keys=True) + "\n", encoding="utf-8")
o2.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if ! diff -u "$n1" "$n2" >/dev/null; then
  echo "[ci_publish_determinism] FAIL: publish evidence differs" >&2
  diff -u "$n1" "$n2" | sed -n '1,200p' >&2 || true
  exit 5
fi

# Receipts: must be byte-identical (they contain no absolute paths)
if ! diff -u "$rc1" "$rc2" >/dev/null; then
  echo "[ci_publish_determinism] FAIL: receipts.jsonl differs" >&2
  diff -u "$rc1" "$rc2" | sed -n '1,200p' >&2 || true
  exit 6
fi

sha="$("$PYTHON" - <<'PY' "$rc1"
import hashlib, sys
from pathlib import Path
p = Path(sys.argv[1])
h = hashlib.sha256(p.read_bytes()).hexdigest()
print(h)
PY
)"

echo "[ci_publish_determinism] OK receipts_sha256=$sha"
