#!/usr/bin/env bash
set -euo pipefail

echo "[pages_publish] starting"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

REL_ID="${1:-}"
shift || true

if [[ -z "${REL_ID}" ]]; then
  REL_ID="$(date -u +%Y-%m-%d)"
fi

if [[ "$#" -gt 0 ]]; then
  CONTEXTS=("$@")
else
  CONTEXTS=(focus sleep workout)
fi

echo "[pages_publish] release_id=${REL_ID}"
echo "[pages_publish] contexts=${CONTEXTS[*]}"

"${REPO_ROOT}/scripts/release_local.sh" "${REL_ID}" "${CONTEXTS[@]}"

if [[ -f "${REPO_ROOT}/index.html" ]]; then
  cp "${REPO_ROOT}/index.html" "${REPO_ROOT}/docs/index.html"
fi

touch "${REPO_ROOT}/docs/.nojekyll"

echo "[pages_publish] docs updated"
