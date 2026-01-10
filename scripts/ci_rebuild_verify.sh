#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Default DB for CI if not provided
: "${MGC_DB:=fixtures/ci_db.sqlite}"

# Normalize to absolute path
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
if [[ ! -f "$MGC_DB" || ! -s "$MGC_DB" ]]; then
  echo "[ci_rebuild_verify] DB missing/empty at $MGC_DB"
  echo "[ci_rebuild_verify] Generating fixture DB..."
  python scripts/make_fixture_db.py
fi

# Ensure required table exists
python - <<'PY'
import os, sqlite3, sys
p = os.environ["MGC_DB"]
con = sqlite3.connect(p)
try:
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
    if "playlists" not in tables:
        print(f"[ci_rebuild_verify] ERROR: DB at {p} has no 'playlists' table. Tables: {tables}", file=sys.stderr)
        sys.exit(1)
finally:
    con.close()
PY

echo "[ci_rebuild_verify] == rebuild playlists (determinism check + write) =="

# Your existing rebuild command(s) go here. Keep exactly what you already had.
# Example (do NOT change if yours differs):
python -m mgc.main rebuild playlists --limit 1000 --stamp ci

echo "[ci_rebuild_verify] == verify playlists vs manifest =="

# Your existing verify logic goes here. Keep exactly what you already had.
# Example:
python -m mgc.main playlists verify --manifest data/playlists/_manifest.playlists.json
