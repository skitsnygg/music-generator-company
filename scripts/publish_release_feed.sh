#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python3}"

# demo_check passes --context; release_feed.py doesn't need it. Ignore if provided.
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --context)
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

has_arg() {
  local needle="$1"
  local arg
  if (( ${#args[@]} == 0 )); then
    return 1
  fi
  for arg in "${args[@]}"; do
    if [[ "${arg}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

RELEASE_ROOT="${MGC_RELEASE_ROOT:-/var/lib/mgc/releases}"
RELEASE_OUT="${MGC_RELEASE_FEED_OUT:-${RELEASE_ROOT}/feed.json}"
RELEASE_BASE_URL="${MGC_RELEASE_BASE_URL:-}"
RELEASE_MAX_ITEMS="${MGC_RELEASE_MAX_ITEMS:-}"

if ! has_arg "--root-dir"; then
  args+=(--root-dir "${RELEASE_ROOT}")
fi
if ! has_arg "--out"; then
  args+=(--out "${RELEASE_OUT}")
fi
if [[ -n "${RELEASE_BASE_URL}" ]] && ! has_arg "--base-url"; then
  args+=(--base-url "${RELEASE_BASE_URL}")
fi
if [[ -n "${RELEASE_MAX_ITEMS}" ]] && ! has_arg "--max-items"; then
  args+=(--max-items "${RELEASE_MAX_ITEMS}")
fi
if [[ "${MGC_RELEASE_INCLUDE_BACKUPS:-0}" == "1" ]] && ! has_arg "--include-backups"; then
  args+=(--include-backups)
fi
if [[ "${MGC_RELEASE_STABLE:-0}" == "1" ]] && ! has_arg "--stable"; then
  args+=(--stable)
fi

exec "${PY}" "${ROOT}/scripts/release_feed.py" "${args[@]}"
