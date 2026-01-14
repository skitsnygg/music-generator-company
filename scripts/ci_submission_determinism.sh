#!/usr/bin/env bash
# scripts/ci_submission_determinism.sh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ci_submission_determinism.sh --evidence-root <dir>

Reads <evidence-root>/drop_evidence.json, finds artifacts.submission_zip_sha256 and paths.submission_zip,
then re-computes sha256 of the zip and compares to recorded value.

Exits:
  0 OK
  2 mismatch / missing
EOF
}

EVIDENCE_ROOT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --evidence-root) EVIDENCE_ROOT="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ci_submission_determinism] unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [ -z "$EVIDENCE_ROOT" ]; then
  echo "[ci_submission_determinism] missing --evidence-root" >&2
  usage
  exit 2
fi

evidence_path="${EVIDENCE_ROOT%/}/drop_evidence.json"
if [ ! -f "$evidence_path" ]; then
  echo "[ci_submission_determinism] missing evidence: $evidence_path" >&2
  exit 2
fi

python - <<PY
import json, hashlib, pathlib, sys

evidence = pathlib.Path("$evidence_path")
obj = json.loads(evidence.read_text(encoding="utf-8"))

paths = obj.get("paths") if isinstance(obj.get("paths"), dict) else {}
arts  = obj.get("artifacts") if isinstance(obj.get("artifacts"), dict) else {}

zip_path = paths.get("submission_zip")
want = arts.get("submission_zip_sha256")

if not zip_path:
    print("[ci_submission_determinism] missing paths.submission_zip", file=sys.stderr)
    raise SystemExit(2)
if not want:
    print("[ci_submission_determinism] missing artifacts.submission_zip_sha256", file=sys.stderr)
    raise SystemExit(2)

zp = pathlib.Path(str(zip_path))
if not zp.exists():
    print(f"[ci_submission_determinism] submission_zip missing: {zp}", file=sys.stderr)
    raise SystemExit(2)

h = hashlib.sha256()
with zp.open("rb") as f:
    for chunk in iter(lambda: f.read(1024*1024), b""):
        h.update(chunk)
got = h.hexdigest()

if got != want:
    print("[ci_submission_determinism] FAIL sha mismatch", file=sys.stderr)
    print(f"[ci_submission_determinism] expected={want}", file=sys.stderr)
    print(f"[ci_submission_determinism] got     ={got}", file=sys.stderr)
    raise SystemExit(2)

print(f"[ci_submission_determinism] OK sha256={got}")
PY
