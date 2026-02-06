#!/usr/bin/env bash
set -euo pipefail

echo "[demo_run] starting"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Demo defaults: short clips + stub fallback when riffusion is unreachable.
: "${MGC_DEMO_FALLBACK_TO_STUB:=1}"
: "${MGC_RIFFUSION_TARGET_SECONDS:=5}"
: "${MGC_RIFFUSION_SEGMENT_SECONDS:=5}"
: "${MGC_RIFFUSION_MAX_SEGMENTS:=1}"
: "${MGC_RIFFUSION_CROSSFADE_SECONDS:=0}"
: "${MGC_STUB_SECONDS:=5}"
export MGC_DEMO_FALLBACK_TO_STUB
export MGC_RIFFUSION_TARGET_SECONDS
export MGC_RIFFUSION_SEGMENT_SECONDS
export MGC_RIFFUSION_MAX_SEGMENTS
export MGC_RIFFUSION_CROSSFADE_SECONDS
export MGC_STUB_SECONDS

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
