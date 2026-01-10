#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "[ci_gate] Repo: $repo_root"

# If MGC_DB is unset OR set to empty string, default it.
: "${MGC_DB:=fixtures/ci_db.sqlite}"

# Normalize to absolute path to avoid surprises with working directories.
case "$MGC_DB" in
  /*) db_path="$MGC_DB" ;;
  *)  db_path="$repo_root/$MGC_DB" ;;
esac

export MGC_DB="$db_path"
echo "[ci_gate] MGC_DB=$MGC_DB"

# Ensure fixture DB exists and is non-empty (sqlite will create empty DB files otherwise).
mkdir -p "$(dirname "$MGC_DB")"

if [[ ! -f "$MGC_DB" || ! -s "$MGC_DB" ]]; then
  echo "[ci_gate] Fixture DB missing/empty; generating: $MGC_DB"
  python scripts/make_fixture_db.py
fi

# Preflight: ensure playlists table exists (fail fast with clear error)
python - <<'PY'
import os, sqlite3, sys
p = os.environ["MGC_DB"]
con = sqlite3.connect(p)
try:
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()]
    if "playlists" not in tables:
        print(f"[ci_gate] ERROR: DB at {p} has no 'playlists' table. Tables: {tables}", file=sys.stderr)
        sys.exit(1)
    print(f"[ci_gate] DB OK. Tables: {sorted(tables)}")
finally:
    con.close()
PY

echo "[ci_gate] py_compile"
python -m py_compile $(git ls-files '*.py')

echo "[ci_gate] rebuild + verify"
bash scripts/ci_rebuild_verify.sh

echo "[ci_gate] done"
