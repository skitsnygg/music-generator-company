#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${MGC_DB:=fixtures/ci_db.sqlite}"

case "$MGC_DB" in
  /*) db_path="$MGC_DB" ;;
  *)  db_path="$repo_root/$MGC_DB" ;;
esac
export MGC_DB="$db_path"

echo "[ci_rebuild_verify] repo_root: $repo_root"
echo "[ci_rebuild_verify] python: python"
echo "[ci_rebuild_verify] db: $MGC_DB"
echo "[ci_rebuild_verify] python_path: $(command -v python || true)"
echo "[ci_rebuild_verify] == python =="
python -V

# Ensure DB exists and is non-empty
mkdir -p "$(dirname "$MGC_DB")"
if [[ ! -f "$MGC_DB" || ! -s "$MGC_DB" ]]; then
  echo "[ci_rebuild_verify] Fixture DB missing/empty; generating: $MGC_DB"
  python scripts/make_fixture_db.py
fi

# --- keep your existing rebuild/verify logic below this line ---
echo "[ci_rebuild_verify] == rebuild playlists (determinism check + write) =="

# (leave whatever you already had here; do not change behavior)
# bash scripts/ci_rebuild_verify.sh previously succeeded locally, so keep those commands.

echo "[ci_rebuild_verify] == verify playlists vs manifest =="

# (leave your existing verify commands here)

echo "[ci_rebuild_verify] == rebuild playlists (determinism check + write) =="

# Your existing rebuild command(s) go here. Keep exactly what you already had.
# Example (do NOT change if yours differs):
python -m mgc.main rebuild playlists --limit 1000 --stamp ci

echo "[ci_rebuild_verify] == verify playlists vs manifest =="

test -f data/playlists/_manifest.playlists.json
python -c "import json; json.load(open('data/playlists/_manifest.playlists.json'))"

echo "[ci_rebuild_verify] OK: manifest exists and is valid JSON"

# Your existing verify logic goes here. Keep exactly what you already had.
# Example:
