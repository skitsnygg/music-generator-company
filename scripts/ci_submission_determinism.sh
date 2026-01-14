#!/usr/bin/env bash
set -euo pipefail

# Determinism gate: submission.zip
#
# Usage:
#   EVIDENCE_ROOT=artifacts/ci/auto bash scripts/ci_submission_determinism.sh
#   # (back-compat) BUNDLE_DIR=... bash scripts/ci_submission_determinism.sh
#
# Reads:
#   $EVIDENCE_ROOT/drop_evidence.json  (preferred)
#
# Emits:
#   Logs to stderr, exits nonzero on mismatch.

: "${PYTHON:=python}"

EVIDENCE_ROOT="${EVIDENCE_ROOT:-}"
BUNDLE_DIR="${BUNDLE_DIR:-}"

if [[ -n "${EVIDENCE_ROOT}" ]]; then
  EVIDENCE_JSON="${EVIDENCE_ROOT%/}/drop_evidence.json"
  if [[ ! -f "${EVIDENCE_JSON}" ]]; then
    echo "[ci_submission_determinism] FAIL: missing evidence json: ${EVIDENCE_JSON}" >&2
    exit 2
  fi

  # Prefer bundle_dir from evidence; fallback to evidence root.
  BUNDLE_DIR="$(
    "${PYTHON}" - <<'PY'
import json, os, sys
p = os.environ["EVIDENCE_JSON"]
obj = json.load(open(p, "r", encoding="utf-8"))
paths = obj.get("paths") if isinstance(obj.get("paths"), dict) else {}
v = paths.get("bundle_dir") or ""
print(v)
PY
  )"
  if [[ -z "${BUNDLE_DIR}" ]]; then
    BUNDLE_DIR="${EVIDENCE_ROOT%/}"
  fi
fi

if [[ -z "${BUNDLE_DIR}" ]]; then
  echo "scripts/ci_submission_determinism.sh: BUNDLE_DIR: set BUNDLE_DIR to the drop bundle directory" >&2
  echo "  or set EVIDENCE_ROOT to a run output dir containing drop_evidence.json" >&2
  exit 2
fi

BUNDLE_DIR="$(cd "${BUNDLE_DIR}" && pwd)"
echo "[ci_submission_determinism] bundle_dir=${BUNDLE_DIR}" >&2

# Run submission build twice and compare sha256 of produced submission.zip.
# We call the CLI so we test the real contract.
run_once() {
  local out_json
  out_json="$("${PYTHON}" -m mgc.main submission build --bundle-dir "${BUNDLE_DIR}" --json)"
  "${PYTHON}" - <<'PY' <<<"${out_json}"
import json, sys
obj = json.loads(sys.stdin.read())
if not obj.get("ok", False):
    raise SystemExit(2)
print(obj["zip_sha256"])
PY
}

sha1="$(run_once)"
sha2="$(run_once)"

echo "[ci_submission_determinism] run1_sha256=${sha1}" >&2
echo "[ci_submission_determinism] run2_sha256=${sha2}" >&2

if [[ "${sha1}" != "${sha2}" ]]; then
  echo "[ci_submission_determinism] FAIL: submission.zip sha mismatch" >&2
  exit 2
fi

echo "[ci_submission_determinism] OK" >&2
