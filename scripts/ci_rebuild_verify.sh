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

die () {
  echo "[ci_rebuild_verify] ERROR: $*" >&2
  exit 1
}

db_tables () {
  python - <<'PY'
import os, sqlite3, sys
p = os.environ["MGC_DB"]
con = sqlite3.connect(p)
try:
    rows = con.execute("select name from sqlite_master where type='table' order by name").fetchall()
    print(" ".join(r[0] for r in rows))
finally:
    con.close()
PY
}

db_preflight () {
  if [[ ! -f "$MGC_DB" ]]; then
    die "DB file missing: $MGC_DB"
  fi
  if [[ ! -s "$MGC_DB" ]]; then
    die "DB file empty: $MGC_DB"
  fi

  echo "[ci_rebuild_verify] db_file: $(ls -l "$MGC_DB")"

  local tables
  tables="$(db_tables || true)"
  echo "[ci_rebuild_verify] db_tables: ${tables:-<none>}"

  if [[ " $tables " != *" playlists "* ]]; then
    die "DB at $MGC_DB has no 'playlists' table"
  fi
}

maybe_generate_fixture_db () {
  # If DB missing/empty, generate it
  if [[ ! -f "$MGC_DB" || ! -s "$MGC_DB" ]]; then
    echo "[ci_rebuild_verify] Fixture DB missing/empty; generating: $MGC_DB"
    python scripts/make_fixture_db.py
  fi
}

run_rebuild () {
  # Always pass explicit db/out-dir so the CLI cannot wander.
  # --determinism-check ensures two builds inside the CLI match.
  python -m mgc.main rebuild playlists \
    --db "$MGC_DB" \
    --out-dir "data/playlists" \
    --stamp "ci" \
    --determinism-check \
    --write
}

run_verify () {
  python -m mgc.main rebuild verify playlists \
    --db "$MGC_DB" \
    --out-dir "data/playlists"
}

# --- main flow ---

maybe_generate_fixture_db
db_preflight

echo "[ci_rebuild_verify] == rebuild playlists (determinism check + write) =="
run_rebuild

echo "[ci_rebuild_verify] == verify playlists vs manifest + files =="
test -f data/playlists/_manifest.playlists.json
python -c "import json; json.load(open('data/playlists/_manifest.playlists.json'))"
run_verify

echo "[ci_rebuild_verify] OK"
