#!/usr/bin/env bash
# scripts/ci_golden_hash_gate.sh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ci_golden_hash_gate.sh --evidence-root <dir> --known-file <file> [--mode warn|strict]

Reads submission_zip_sha256 from:
  <evidence-root>/drop_evidence.json (artifacts.submission_zip_sha256)
or
  submission.receipt.json referenced by artifacts.submission_receipt_json

Then checks if sha is listed in known file.

Modes:
  warn   (default) -> prints warning, exits 0
  strict          -> exits 2 if sha not found
EOF
}

EVIDENCE_ROOT=""
KNOWN_FILE=""
MODE="warn"

while [ $# -gt 0 ]; do
  case "$1" in
    --evidence-root) EVIDENCE_ROOT="${2:-}"; shift 2;;
    --known-file)    KNOWN_FILE="${2:-}"; shift 2;;
    --mode)          MODE="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[golden] unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [ -z "$EVIDENCE_ROOT" ]; then echo "[golden] missing --evidence-root" >&2; usage; exit 2; fi
if [ -z "$KNOWN_FILE" ]; then echo "[golden] missing --known-file" >&2; usage; exit 2; fi
if [ "$MODE" != "warn" ] && [ "$MODE" != "strict" ]; then
  echo "[golden] invalid --mode: $MODE (expected warn|strict)" >&2
  exit 2
fi

evidence_path="${EVIDENCE_ROOT%/}/drop_evidence.json"
if [ ! -f "$evidence_path" ]; then
  echo "[golden] evidence missing: $evidence_path" >&2
  # This is a CI integration problem; fail in strict mode.
  if [ "$MODE" = "strict" ]; then exit 2; fi
  exit 0
fi

sha="$(
python - <<PY
import json, sys, pathlib
p = pathlib.Path("$evidence_path")
obj = json.loads(p.read_text(encoding="utf-8"))
a = obj.get("artifacts") if isinstance(obj.get("artifacts"), dict) else {}
sha = a.get("submission_zip_sha256")
receipt = a.get("submission_receipt_json")
if not sha and receipt:
    rp = pathlib.Path(str(receipt))
    if rp.exists():
        robj = json.loads(rp.read_text(encoding="utf-8"))
        sha = robj.get("submission_zip_sha256")
print(sha or "")
PY
)"

if [ -z "$sha" ]; then
  echo "[golden] could not read submission_zip_sha256 from evidence/receipt" >&2
  if [ "$MODE" = "strict" ]; then exit 2; fi
  exit 0
fi

if [ ! -f "$KNOWN_FILE" ]; then
  echo "[golden] known file missing: $KNOWN_FILE" >&2
  echo "[golden] sha=$sha" >&2
  if [ "$MODE" = "strict" ]; then exit 2; fi
  exit 0
fi

# strip comments + blank lines
if grep -E -q "^[[:space:]]*$sha[[:space:]]*$" <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$KNOWN_FILE"); then
  echo "[golden] OK: submission_zip_sha256 is known-good: $sha"
  exit 0
fi

msg="[golden] NOT IN known-good list: $sha (file=$KNOWN_FILE)"
if [ "$MODE" = "strict" ]; then
  echo "$msg" >&2
  echo "[golden] Add it to $KNOWN_FILE to bless this artifact." >&2
  exit 2
fi

echo "$msg" >&2
echo "[golden] WARN-ONLY: continuing (mode=warn). To enforce, set MGC_GOLDEN_STRICT=1." >&2
exit 0
