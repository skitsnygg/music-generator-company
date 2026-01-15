#!/usr/bin/env bash
set -euo pipefail

# Determinism check for submission.zip
# Builds submission twice and compares sha256.
#
# Accepts evidence roots that either:
#   - contain drop_evidence.json directly, OR
#   - contain evidence/drop_evidence.json
#
# Usage:
#   bash scripts/ci_submission_determinism.sh --evidence-root .
#   bash scripts/ci_submission_determinism.sh --evidence-root /abs/path/to/evidence_root
#
# Env:
#   PYTHON (default: python)
#   MGC_DB (optional but recommended; CI sets it)

PYTHON="${PYTHON:-python}"

EVIDENCE_ROOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --evidence-root)
      EVIDENCE_ROOT="${2:-}"; shift 2;;
    *)
      echo "[ci_submission_determinism] unknown arg: $1" >&2
      exit 2;;
  esac
done

if [ -z "$EVIDENCE_ROOT" ]; then
  echo "[ci_submission_determinism] missing --evidence-root" >&2
  exit 2
fi

# Normalize / resolve EVIDENCE_ROOT to an absolute path (without relying on readlink -f)
EVIDENCE_ROOT="${EVIDENCE_ROOT%/}"
EVIDENCE_ROOT_ABS="$(
  cd "$EVIDENCE_ROOT" 2>/dev/null && pwd
)" || {
  echo "[ci_submission_determinism] evidence root does not exist: $EVIDENCE_ROOT" >&2
  exit 2
}

# Resolve drop_evidence.json path in either supported layout
DROP_EVIDENCE=""
if [ -f "${EVIDENCE_ROOT_ABS}/drop_evidence.json" ]; then
  DROP_EVIDENCE="${EVIDENCE_ROOT_ABS}/drop_evidence.json"
elif [ -f "${EVIDENCE_ROOT_ABS}/evidence/drop_evidence.json" ]; then
  DROP_EVIDENCE="${EVIDENCE_ROOT_ABS}/evidence/drop_evidence.json"
else
  echo "[ci_submission_determinism] missing evidence: ${EVIDENCE_ROOT_ABS}/drop_evidence.json" >&2
  echo "[ci_submission_determinism] (also tried: ${EVIDENCE_ROOT_ABS}/evidence/drop_evidence.json)" >&2
  ls -la "$EVIDENCE_ROOT_ABS" >&2 || true
  [ -d "${EVIDENCE_ROOT_ABS}/evidence" ] && ls -la "${EVIDENCE_ROOT_ABS}/evidence" >&2 || true
  exit 2
fi

_sha256_file() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    shasum -a 256 "$f" | awk '{print $1}'
  fi
}

# Work dir is the directory containing drop_evidence.json
# (This ensures relative "submission.zip" path assumptions hold.)
WORK_DIR="$(cd "$(dirname "$DROP_EVIDENCE")" && pwd)"

# Ensure we always create zips in WORK_DIR (and compare those)
ZIP1="${WORK_DIR}/submission.zip"
ZIP2="${WORK_DIR}/submission_2.zip"
rm -f "$ZIP1" "$ZIP2"

# Helper to run mgc.main submission with optional --db
_mgc() {
  if [ -n "${MGC_DB:-}" ]; then
    "$PYTHON" -m mgc.main --db "$MGC_DB" "$@"
  else
    "$PYTHON" -m mgc.main "$@"
  fi
}

# Probe CLI capabilities
SUB_HELP="$(_mgc submission --help 2>&1 || true)"
HAS_LATEST=0
HAS_BUILD=0
echo "$SUB_HELP" | grep -qE '(^|[[:space:]])latest($|[[:space:]])' && HAS_LATEST=1
echo "$SUB_HELP" | grep -qE '(^|[[:space:]])build($|[[:space:]])' && HAS_BUILD=1

_build_once() {
  local out_zip="$1"

  # Prefer "submission latest" if available
  if [ "$HAS_LATEST" -eq 1 ]; then
    local latest_help
    latest_help="$(_mgc submission latest --help 2>&1 || true)"

    if echo "$latest_help" | grep -qE -- '--evidence-root'; then
      # Run from WORK_DIR so any relative assumptions behave
      ( cd "$WORK_DIR" && _mgc submission latest --out "$out_zip" --evidence-root "." --json >/dev/null )
      return 0
    else
      # Evidence-root not supported; still try latest (some impls infer)
      ( cd "$WORK_DIR" && _mgc submission latest --out "$out_zip" --json >/dev/null )
      return 0
    fi
  fi

  # Fall back to "submission build"
  if [ "$HAS_BUILD" -eq 1 ]; then
    local build_help
    build_help="$(_mgc submission build --help 2>&1 || true)"

    if echo "$build_help" | grep -qE -- '--bundle-dir'; then
      ( cd "$WORK_DIR" && _mgc submission build --bundle-dir "." --out "$out_zip" --json >/dev/null )
      return 0
    fi

    # Another common pattern is build --drop-id (with --db)
    if echo "$build_help" | grep -qE -- '--drop-id'; then
      # Extract drop_id from drop_evidence.json
      local drop_id
      drop_id="$("$PYTHON" - <<'PY' "$DROP_EVIDENCE"
import json, sys
p = sys.argv[1]
obj = json.load(open(p, "r", encoding="utf-8"))
# try a few likely shapes
for k in ("drop_id", "id"):
    if isinstance(obj, dict) and k in obj and obj[k]:
        print(obj[k])
        sys.exit(0)
# nested
if isinstance(obj, dict) and "drop" in obj and isinstance(obj["drop"], dict):
    for k in ("drop_id", "id"):
        if k in obj["drop"] and obj["drop"][k]:
            print(obj["drop"][k])
            sys.exit(0)
sys.exit(2)
PY
)" || true

      if [ -n "${drop_id:-}" ]; then
        ( cd "$WORK_DIR" && _mgc submission build --drop-id "$drop_id" --out "$out_zip" --json >/dev/null )
        return 0
      fi
    fi
  fi

  echo "[ci_submission_determinism] FAIL: could not determine supported submission command" >&2
  echo "[ci_submission_determinism] submission --help:" >&2
  echo "$SUB_HELP" >&2
  exit 2
}

# Build twice
set +e
_build_once "$ZIP1"
rc1=$?
_build_once "$ZIP2"
rc2=$?
set -e

if [ $rc1 -ne 0 ] || [ $rc2 -ne 0 ]; then
  echo "[ci_submission_determinism] FAIL: submission build failed (rc1=$rc1 rc2=$rc2)" >&2
  exit 2
fi

if [ ! -f "$ZIP1" ] || [ ! -f "$ZIP2" ]; then
  echo "[ci_submission_determinism] FAIL: submission zips not created" >&2
  echo "[ci_submission_determinism] expected: $ZIP1 and $ZIP2" >&2
  ls -la "$WORK_DIR" >&2 || true
  exit 3
fi

h1="$(_sha256_file "$ZIP1")"
h2="$(_sha256_file "$ZIP2")"

if [ "$h1" != "$h2" ]; then
  echo "[ci_submission_determinism] FAIL: sha256 mismatch" >&2
  echo "[ci_submission_determinism] run1=$h1" >&2
  echo "[ci_submission_determinism] run2=$h2" >&2
  exit 4
fi

echo "[ci_submission_determinism] OK sha256=$h1"
