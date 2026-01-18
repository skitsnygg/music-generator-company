#!/usr/bin/env bash
set -euo pipefail

# scripts/ci_submission_determinism.sh
#
# Build submission ZIP twice and compare sha256 deterministically.
#
# Primary path: build from <evidence-root>/drop_bundle (portable; no DB needed).
# Fallback path: build via --drop-id (requires MGC_DB).
#
# IMPORTANT: We prefer NOT passing --repo-root because some argv normalization
# paths in mgc.main appear to print top-level help when --repo-root is present.

PYTHON="${PYTHON:-python}"
MGC_DB="${MGC_DB:-}"
EVIDENCE_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence-root) EVIDENCE_ROOT="${2:-}"; shift 2 ;;
    *) echo "usage: $0 --evidence-root <dir>" >&2; exit 2 ;;
  esac
done

if [[ -z "$EVIDENCE_ROOT" || ! -d "$EVIDENCE_ROOT" ]]; then
  echo "[ci_submission_determinism] FAIL: evidence root dir not found: $EVIDENCE_ROOT" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EVIDENCE_ROOT_ABS="$(cd "$EVIDENCE_ROOT" && pwd)"
DROP_EVIDENCE_JSON="${EVIDENCE_ROOT_ABS%/}/drop_evidence.json"
BUNDLE_DIR="${EVIDENCE_ROOT_ABS%/}/drop_bundle"
OUT1="${EVIDENCE_ROOT_ABS%/}/submission.zip"
OUT2="${EVIDENCE_ROOT_ABS%/}/submission_2.zip"

if [[ ! -f "$DROP_EVIDENCE_JSON" ]]; then
  echo "[ci_submission_determinism] FAIL: missing drop_evidence.json at: $DROP_EVIDENCE_JSON" >&2
  ls -la "$EVIDENCE_ROOT_ABS" >&2 || true
  exit 4
fi

rm -f "$OUT1" "$OUT2"

_drop_id_from_evidence() {
  "$PYTHON" - <<PY
import json
from pathlib import Path
obj = json.loads(Path(r"$DROP_EVIDENCE_JSON").read_text(encoding="utf-8"))
print(obj.get("drop_id",""))
PY
}

_sha256_file() {
  local p="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$p" | awk '{print $1}'
  else
    shasum -a 256 "$p" | awk '{print $1}'
  fi
}

# Run mgc.main with a few argv layouts.
# We capture output and print it if all variants fail.
_run_mgc() {
  # args: out_path mode bundle_or_drop_id
  local out_path="$1"
  local mode="$2"         # "bundle" or "drop"
  local drop_id="${3:-}"  # for mode=drop

  local tmp_out rc
  tmp_out="$(mktemp)"
  rc=0

  # Always run from repo root so we can omit --repo-root.
  pushd "$REPO_ROOT" >/dev/null

  # -----------------------------
  # Variant 0 (preferred): NO --repo-root
  # -----------------------------
  if [[ "$mode" == "bundle" ]]; then
    echo "[ci_submission_determinism] build (bundle-dir): $out_path"
    set +e
    "$PYTHON" -m mgc.main \
      submission build \
      --bundle-dir "$BUNDLE_DIR" \
      --out "$out_path" \
      --evidence-root "$EVIDENCE_ROOT_ABS" >"$tmp_out" 2>&1
    rc=$?
    set -e
  else
    echo "[ci_submission_determinism] build (drop-id=$drop_id): $out_path"
    set +e
    "$PYTHON" -m mgc.main \
      --db "$MGC_DB" \
      submission build \
      --drop-id "$drop_id" \
      --out "$out_path" \
      --evidence-root "$EVIDENCE_ROOT_ABS" >"$tmp_out" 2>&1
    rc=$?
    set -e
  fi

  if [[ $rc -eq 0 ]]; then
    popd >/dev/null
    rm -f "$tmp_out"
    return 0
  fi

  # -----------------------------
  # Variant A: --repo-root BEFORE subcommand
  # -----------------------------
  if [[ "$mode" == "bundle" ]]; then
    set +e
    "$PYTHON" -m mgc.main \
      --repo-root "$REPO_ROOT" \
      submission build \
      --bundle-dir "$BUNDLE_DIR" \
      --out "$out_path" \
      --evidence-root "$EVIDENCE_ROOT_ABS" >"$tmp_out" 2>&1
    rc=$?
    set -e
  else
    set +e
    "$PYTHON" -m mgc.main \
      --repo-root "$REPO_ROOT" \
      --db "$MGC_DB" \
      submission build \
      --drop-id "$drop_id" \
      --out "$out_path" \
      --evidence-root "$EVIDENCE_ROOT_ABS" >"$tmp_out" 2>&1
    rc=$?
    set -e
  fi

  if [[ $rc -eq 0 ]]; then
    popd >/dev/null
    rm -f "$tmp_out"
    return 0
  fi

  # -----------------------------
  # Variant B: --repo-root AFTER subcommand
  # -----------------------------
  if [[ "$mode" == "bundle" ]]; then
    set +e
    "$PYTHON" -m mgc.main \
      submission build \
      --bundle-dir "$BUNDLE_DIR" \
      --out "$out_path" \
      --evidence-root "$EVIDENCE_ROOT_ABS" \
      --repo-root "$REPO_ROOT" >"$tmp_out" 2>&1
    rc=$?
    set -e
  else
    set +e
    "$PYTHON" -m mgc.main \
      submission build \
      --drop-id "$drop_id" \
      --out "$out_path" \
      --evidence-root "$EVIDENCE_ROOT_ABS" \
      --repo-root "$REPO_ROOT" \
      --db "$MGC_DB" >"$tmp_out" 2>&1
    rc=$?
    set -e
  fi

  popd >/dev/null

  if [[ $rc -ne 0 ]]; then
    echo "[ci_submission_determinism] FAIL: mgc.main submission build failed (rc=$rc)" >&2
    echo "----- mgc.main output (captured) -----" >&2
    cat "$tmp_out" >&2
    echo "----- end captured output -----" >&2
    rm -f "$tmp_out"
    exit 2
  fi

  rm -f "$tmp_out"
  return 0
}

build_once() {
  local out_path="$1"

  if [[ -d "$BUNDLE_DIR" ]]; then
    _run_mgc "$out_path" "bundle"
    return 0
  fi

  if [[ -z "$MGC_DB" ]]; then
    echo "[ci_submission_determinism] FAIL: drop_bundle missing and MGC_DB not set" >&2
    exit 4
  fi
  if [[ ! -f "$MGC_DB" ]]; then
    echo "[ci_submission_determinism] FAIL: MGC_DB not found: $MGC_DB" >&2
    exit 4
  fi

  local drop_id
  drop_id="$(_drop_id_from_evidence)"
  if [[ -z "$drop_id" ]]; then
    echo "[ci_submission_determinism] FAIL: could not read drop_id from $DROP_EVIDENCE_JSON" >&2
    exit 4
  fi

  _run_mgc "$out_path" "drop" "$drop_id"
}

build_once "$OUT1"
build_once "$OUT2"

if [[ ! -f "$OUT1" || ! -f "$OUT2" ]]; then
  echo "[ci_submission_determinism] FAIL: submission zips not created" >&2
  echo "[ci_submission_determinism] expected: $OUT1 and $OUT2" >&2
  ls -la "$EVIDENCE_ROOT_ABS" >&2 || true
  exit 4
fi

H1="$(_sha256_file "$OUT1")"
H2="$(_sha256_file "$OUT2")"

echo "[ci_submission_determinism] sha256_1=$H1"
echo "[ci_submission_determinism] sha256_2=$H2"

if [[ "$H1" != "$H2" ]]; then
  echo "[ci_submission_determinism] FAIL: submission.zip not deterministic" >&2
  exit 2
fi

echo "[ci_submission_determinism] OK"
