#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DB="${1:-data/db.sqlite}"
PY="${PYTHON:-python}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "== Using DB: $DB =="
$PY -m mgc.main tracks list --db "$DB" --limit 10
echo
$PY -m mgc.main tracks stats --db "$DB"
echo

echo "== Try showing first track id from list (if any) =="

FIRST_ID="$($PY -m mgc.main tracks list --db "$DB" --limit 1 \
  | sed -nE 's/.*id=([0-9a-fA-F-]+).*/\1/p' \
  | head -n 1
)"

if [[ -n "${FIRST_ID}" ]]; then
  echo "Showing id=$FIRST_ID"
  $PY -m mgc.main tracks show "$FIRST_ID" --db "$DB"
else
  echo "No tracks found to show."
fi
