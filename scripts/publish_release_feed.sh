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

if (( ${#args[@]} )); then
  exec "${PY}" "${ROOT}/scripts/release_feed.py" "${args[@]}"
else
  exec "${PY}" "${ROOT}/scripts/release_feed.py"
fi
