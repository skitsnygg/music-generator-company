#!/usr/bin/env bash
set -euo pipefail

echo "[demo_reset] starting"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${EUID}" -eq 0 && "${MGC_DEMO_ALLOW_ROOT:-0}" != "1" ]]; then
  echo "[demo_reset] ERROR: refusing to run as root (set MGC_DEMO_ALLOW_ROOT=1 to override)" >&2
  exit 2
fi

export MGC_DEMO_NO_SUDO="${MGC_DEMO_NO_SUDO:-1}"
export MGC_DEMO_CLEAN="${MGC_DEMO_CLEAN:-1}"
export MGC_CONTEXTS="${MGC_CONTEXTS:-focus}"

export MGC_OUT_BASE="${MGC_OUT_BASE:-${REPO_ROOT}/data/local_demo_evidence}"
export MGC_DB="${MGC_DB:-${REPO_ROOT}/data/local_demo_db.sqlite}"
export MGC_WEB_LATEST_ROOT="${MGC_WEB_LATEST_ROOT:-${REPO_ROOT}/data/releases/latest/web}"
export MGC_RELEASE_ROOT="${MGC_RELEASE_ROOT:-${REPO_ROOT}/data/releases}"
export MGC_RELEASE_FEED_OUT="${MGC_RELEASE_FEED_OUT:-${REPO_ROOT}/data/releases/feed.json}"

if [[ "${MGC_DEMO_RESET_DB:-0}" == "1" ]]; then
  rm -f "${MGC_DB}"
fi

echo "[demo_reset] no-sudo=${MGC_DEMO_NO_SUDO} clean=${MGC_DEMO_CLEAN} contexts=${MGC_CONTEXTS}"

"${REPO_ROOT}/scripts/demo_smoke.sh"
"${REPO_ROOT}/scripts/web_health.sh"

echo "[demo_reset] OK"
