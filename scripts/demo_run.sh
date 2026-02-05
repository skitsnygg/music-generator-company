#!/usr/bin/env bash
set -euo pipefail

echo "[demo_run] starting"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODE="${MGC_DEMO_MODE:-smoke}"
if [[ "${MGC_DEMO_FULL:-0}" == "1" ]]; then
  MODE="check"
fi

if [[ "${MGC_DEMO_CLEAN:-0}" == "1" ]]; then
  "${REPO_ROOT}/scripts/demo_clean.sh"
fi

case "${MODE}" in
  smoke)
    "${REPO_ROOT}/scripts/demo_smoke.sh"
    ;;
  check)
    "${REPO_ROOT}/scripts/demo_check.sh"
    ;;
  *)
    echo "[demo_run] ERROR: invalid MGC_DEMO_MODE: ${MODE} (use smoke|check)" >&2
    exit 2
    ;;
esac

if [[ "${MGC_DEMO_REPORT:-1}" == "1" ]]; then
  "${REPO_ROOT}/scripts/demo_report.sh"
fi

echo "[demo_run] OK"
