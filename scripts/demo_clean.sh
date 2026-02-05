#!/usr/bin/env bash
set -euo pipefail

echo "[demo_clean] starting cleanup"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MGC_DEMO_NO_SUDO="${MGC_DEMO_NO_SUDO:-0}"
MGC_RELEASE_ROOT="${MGC_RELEASE_ROOT:-}"
MGC_OUT_BASE="${MGC_OUT_BASE:-}"
MGC_WEB_LATEST_ROOT="${MGC_WEB_LATEST_ROOT:-}"
MGC_RELEASE_FEED_OUT="${MGC_RELEASE_FEED_OUT:-}"

if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  MGC_OUT_BASE="${MGC_OUT_BASE:-${REPO_ROOT}/data/local_demo_evidence}"
  MGC_WEB_LATEST_ROOT="${MGC_WEB_LATEST_ROOT:-${REPO_ROOT}/data/releases/latest/web}"
  MGC_RELEASE_ROOT="${MGC_RELEASE_ROOT:-${REPO_ROOT}/data/releases}"
  MGC_RELEASE_FEED_OUT="${MGC_RELEASE_FEED_OUT:-${REPO_ROOT}/data/releases/feed.json}"
fi

if [[ -z "${MGC_RELEASE_ROOT}" ]]; then
  MGC_RELEASE_ROOT="/var/lib/mgc/releases"
fi
if [[ -z "${MGC_OUT_BASE}" ]]; then
  MGC_OUT_BASE="${REPO_ROOT}/data/evidence"
fi
if [[ -z "${MGC_WEB_LATEST_ROOT}" ]]; then
  MGC_WEB_LATEST_ROOT="${MGC_RELEASE_ROOT}/latest/web"
fi
if [[ -z "${MGC_RELEASE_FEED_OUT}" ]]; then
  MGC_RELEASE_FEED_OUT="${MGC_RELEASE_ROOT}/feed.json"
fi

confirm() {
  local msg="$1"
  if [[ "${MGC_DEMO_YES:-0}" == "1" ]]; then
    return 0
  fi
  read -r -p "${msg} [y/N] " ans
  case "${ans}" in
    y|Y|yes|YES) return 0 ;;
  esac
  return 1
}

safe_rm_dir() {
  local path="$1"
  [[ -n "${path}" ]] || return 0
  if [[ "${path}" == "/" || "${path}" == "/var" || "${path}" == "/var/lib" || "${path}" == "/var/lib/mgc" ]]; then
    echo "[demo_clean] ERROR: refusing to remove unsafe path: ${path}" >&2
    exit 2
  fi
  if [[ -d "${path}" ]]; then
    rm -rf "${path}"
  fi
}

safe_rm_file() {
  local path="$1"
  [[ -n "${path}" ]] || return 0
  if [[ -f "${path}" ]]; then
    rm -f "${path}"
  fi
}

if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  echo "[demo_clean] local demo cleanup:"
  echo "  MGC_OUT_BASE=${MGC_OUT_BASE}"
  echo "  MGC_WEB_LATEST_ROOT=${MGC_WEB_LATEST_ROOT}"
  echo "  MGC_RELEASE_ROOT=${MGC_RELEASE_ROOT}"
  echo "  MGC_RELEASE_FEED_OUT=${MGC_RELEASE_FEED_OUT}"
else
  echo "[demo_clean] production-style cleanup:"
  echo "  MGC_OUT_BASE=${MGC_OUT_BASE}"
  echo "  MGC_WEB_LATEST_ROOT=${MGC_WEB_LATEST_ROOT}"
  echo "  MGC_RELEASE_ROOT=${MGC_RELEASE_ROOT}"
  echo "  MGC_RELEASE_FEED_OUT=${MGC_RELEASE_FEED_OUT}"
  echo "  (use MGC_DEMO_NO_SUDO=1 for local-only cleanup)"
fi

if ! confirm "[demo_clean] proceed with cleanup?"; then
  echo "[demo_clean] cancelled"
  exit 0
fi

if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  safe_rm_dir "${MGC_OUT_BASE}"
  safe_rm_dir "${MGC_RELEASE_ROOT}"
  safe_rm_file "${MGC_RELEASE_FEED_OUT}"
  safe_rm_dir "${REPO_ROOT}/.tmp_publish"
else
  if [[ "${EUID}" -ne 0 ]]; then
    exec sudo -E "$0" "$@"
  fi
  safe_rm_dir "${MGC_OUT_BASE}"
  safe_rm_dir "${MGC_RELEASE_ROOT}/latest"
  safe_rm_file "${MGC_RELEASE_FEED_OUT}"
fi

echo "[demo_clean] OK"
