#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Default DB for CI
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

mkdir -p "$(dirname "$MGC_DB")"

ensure_db_has_playlists () {
  python - <<'PY'
import os, sqlite3, sys
p = os.environ["MGC_DB"]
con = sqlite3.connect(p)
try:
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
    if "playlists" not in tables:
        print(f"[ci_rebuild_verify] ERROR: DB at {p} has no 'playlists'. Tables={tables}", file=sys.stderr)
        sys.exit(1)
finally:
    con.close()
PY
}

# If DB missing/empty, generate it
if [[ ! -f "$MGC_DB" || ! -s "$MGC_DB" ]]; then
  echo "[ci_rebuild_verify] Fixture DB missing/empty; generating: $MGC_DB"
  python scripts/make_fixture_db.py
fi

ensure_db_has_playlists

echo "[ci_rebuild_verify] == rebuild playlists (determinism check + write) =="

# IMPORTANT:
# Replace the command below with the exact rebuild command you already use
# (the one that prints fingerprint_sha256 and writes the manifest).
run_rebuild () {
  MGC_DB="$MGC_DB" python -m mgc.main rebuild playlists --stamp ci --write
}

# Run #1
ensure_db_has_playlists
run_rebuild

echo "[ci_rebuild_verify] == verify playlists vs manifest =="

test -f data/playlists/_manifest.playlists.json
python -c "import json; json.load(open('data/playlists/_manifest.playlists.json'))"
echo "[ci_rebuild_verify] OK: manifest exists and is valid JSON"

# Run #2 (determinism)
echo "[ci_rebuild_verify] == rebuild playlists (determinism check + write) =="
ensure_db_has_playlists
run_rebuild

# Verify again (basic)
echo "[ci_rebuild_verify] == verify playlists vs manifest =="
test -f data/playlists/_manifest.playlists.json
python -c "import json; json.load(open('data/playlists/_manifest.playlists.json'))"
echo "[ci_rebuild_verify] OK: manifest exists and is valid JSON"
