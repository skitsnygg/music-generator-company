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

mkdir -p "$(dirname "$MGC_DB")"

die () {
  echo "[ci_rebuild_verify] ERROR: $*" >&2
  exit 1
}

db_tables () {
  python - <<'PY'
import os, sqlite3
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
  if [[ " $tables " != *" tracks "* ]]; then
    die "DB at $MGC_DB has no 'tracks' table"
  fi
}

maybe_generate_fixture_db () {
  if [[ ! -f "$MGC_DB" || ! -s "$MGC_DB" ]]; then
    echo "[ci_rebuild_verify] Fixture DB missing/empty; generating: $MGC_DB"
    python scripts/make_fixture_db.py
  fi
}

# Feature-detect CLI flags so CI won't break if flags aren't present yet.
has_flag () {
  # usage: has_flag "<help command>" "<flag>"
  local help_cmd="$1"
  local flag="$2"
  if eval "$help_cmd" 2>/dev/null | grep -q -- "$flag"; then
    return 0
  fi
  return 1
}

PLAYLISTS_REL_FLAG=""
TRACKS_REL_FLAG=""
VERIFY_PLAYLISTS_STRICT_FLAG=""
VERIFY_TRACKS_STRICT_FLAG=""

if has_flag "python -m mgc.main rebuild playlists --help" "--relative-paths"; then
  PLAYLISTS_REL_FLAG="--relative-paths"
fi
if has_flag "python -m mgc.main rebuild tracks --help" "--relative-paths"; then
  TRACKS_REL_FLAG="--relative-paths"
fi
if has_flag "python -m mgc.main rebuild verify playlists --help" "--strict-paths"; then
  VERIFY_PLAYLISTS_STRICT_FLAG="--strict-paths"
fi
if has_flag "python -m mgc.main rebuild verify tracks --help" "--strict-paths"; then
  VERIFY_TRACKS_STRICT_FLAG="--strict-paths"
fi

echo "[ci_rebuild_verify] flags: playlists_relative='${PLAYLISTS_REL_FLAG:-<none>}' tracks_relative='${TRACKS_REL_FLAG:-<none>}' verify_playlists_strict='${VERIFY_PLAYLISTS_STRICT_FLAG:-<none>}' verify_tracks_strict='${VERIFY_TRACKS_STRICT_FLAG:-<none>}'"

run_ls_json () {
  echo "[ci_rebuild_verify] == ls (status) =="
  python -m mgc.main rebuild ls --json || true
}

run_rebuild_playlists () {
  echo "[ci_rebuild_verify] == rebuild playlists (determinism check + write) =="
  python -m mgc.main rebuild playlists \
    --db "$MGC_DB" \
    --out-dir "data/playlists" \
    --stamp "ci" \
    ${PLAYLISTS_REL_FLAG} \
    --determinism-check \
    --write
}

run_verify_playlists () {
  echo "[ci_rebuild_verify] == verify playlists vs manifest + files =="
  test -f data/playlists/_manifest.playlists.json
  python -c "import json; json.load(open('data/playlists/_manifest.playlists.json'))"
  python -m mgc.main rebuild verify playlists \
    --db "$MGC_DB" \
    --out-dir "data/playlists" \
    ${VERIFY_PLAYLISTS_STRICT_FLAG}
}

run_rebuild_tracks () {
  echo "[ci_rebuild_verify] == rebuild tracks (determinism check + write) =="
  python -m mgc.main rebuild tracks \
    --db "$MGC_DB" \
    --out-dir "data/tracks" \
    --stamp "ci" \
    ${TRACKS_REL_FLAG} \
    --determinism-check \
    --write
}

run_verify_tracks () {
  echo "[ci_rebuild_verify] == verify tracks vs manifest + files =="
  test -f data/tracks/_manifest.tracks.json
  python -c "import json; json.load(open('data/tracks/_manifest.tracks.json'))"
  python -m mgc.main rebuild verify tracks \
    --db "$MGC_DB" \
    --out-dir "data/tracks" \
    ${VERIFY_TRACKS_STRICT_FLAG}
}

# --- main flow ---
maybe_generate_fixture_db
db_preflight

run_ls_json
run_rebuild_playlists
run_verify_playlists
run_rebuild_tracks
run_verify_tracks

echo "[ci_rebuild_verify] OK"
