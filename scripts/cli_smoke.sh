#!/usr/bin/env bash
set -euo pipefail

# Fast CLI smoke tests.
# Enforces:
# - commands run without tracebacks
# - JSON mode produces valid JSON (exactly one object)
# - expected exit codes for "gates" are stable
#
# Usage:
#   MGC_DB=fixtures/ci_db.sqlite bash scripts/cli_smoke.sh
#
# Optional:
#   PYTHON=python
#   MGC_BIN="python -m mgc.main" (default)
#
# Notes:
# - Avoids jq; uses python -m json.tool for portability.
# - Keeps stdout clean in JSON mode; logs may go to stderr.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
MGC_BIN="${MGC_BIN:-$PYTHON -m mgc.main}"
DB="${MGC_DB:-fixtures/ci_db.sqlite}"

fail() {
  echo "[cli_smoke] FAIL: $*" >&2
  exit 2
}

run_json() {
  # $1.. = command
  # Prints nothing on success; fails if output is not valid JSON.
  local out
  # Capture stdout only; stderr passes through.
  out="$($MGC_BIN "$@" 2> >(cat >&2))" || return $?
  # Must be non-empty JSON
  if [[ -z "${out}" ]]; then
    fail "empty stdout in JSON mode for: $*"
  fi
  # Validate JSON (exactly one object/array; we expect object)
  echo "$out" | $PYTHON -m json.tool >/dev/null 2>&1 || fail "invalid JSON for: $* ; stdout=$out"
}

echo "[cli_smoke] repo=$REPO_ROOT" >&2
echo "[cli_smoke] db=$DB" >&2
echo "[cli_smoke] mgc=$MGC_BIN" >&2

# Compile sanity
$PYTHON -m py_compile src/mgc/main.py src/mgc/run_cli.py >/dev/null 2>&1 || fail "py_compile failed"

# Top-level status (JSON)
run_json status --db "$DB" --json

# Rebuild ls (JSON) - supports db after subcommand (CI/humans do this)
run_json rebuild ls --db "$DB" --json

# Run status (JSON)
run_json run status --db "$DB" --json

# Run diff in JSON should always be valid, found true/false is ok
run_json run diff --db "$DB" --json

# Run diff fail gate should be stable: exit 0 if no changes; 2 if changes and fail-on-changes.
set +e
$MGC_BIN run diff --db "$DB" --fail-on-changes >/dev/null 2>&1
rc=$?
set -e

if [[ "$rc" -ne 0 && "$rc" -ne 2 ]]; then
  fail "unexpected exit code from: run diff --fail-on-changes (got $rc; expected 0 or 2)"
fi

echo "[cli_smoke] OK" >&2
