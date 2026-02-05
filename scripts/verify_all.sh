#!/usr/bin/env bash
set -euo pipefail

# scripts/verify_all.sh
# End-to-end local verification (deterministic, offline-friendly).
# Uses fixtures/ci_db.sqlite + stub provider; does NOT require external services.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

# Prefer repo venv if present
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

command -v "$PY" >/dev/null 2>&1 || { echo "python not found: $PY" >&2; exit 2; }

log() { echo "[verify] $*"; }

tmp_root="$(mktemp -d /tmp/mgc_verify_XXXXXX)"
trap 'echo "[verify] cleanup $tmp_root"; rm -rf "$tmp_root"' EXIT INT TERM

DB="$tmp_root/ci_db.sqlite"
cp -f fixtures/ci_db.sqlite "$DB"

export PYTHONPATH=src
export MGC_DB="$DB"
export MGC_DETERMINISTIC=1
export MGC_FIXED_TIME="2020-01-01T00:00:00Z"
export MGC_PROVIDER=stub
export MGC_PLAYLIST_PROVIDER=stub

log "migrating fixture DB"
MGC_DB="$DB" "$PY" scripts/migrate_db.py >/dev/null

log "providers list"
"$PY" -m mgc.main --db "$DB" providers list --json >/dev/null

log "billing smoke"
"$PY" -m mgc.main --db "$DB" billing users add demo --email demo@example.com --json >/dev/null
TOKEN="demo_token_123"
"$PY" -m mgc.main --db "$DB" billing tokens mint demo --token "$TOKEN" --label demo --show-token --json >/dev/null
"$PY" -m mgc.main --db "$DB" billing entitlements grant demo_ent demo pro --starts-ts "2020-01-01T00:00:00Z" --json >/dev/null
"$PY" -m mgc.main --db "$DB" billing check --token "$TOKEN" --json >/dev/null
"$PY" -m mgc.main --db "$DB" billing whoami --token "$TOKEN" --json >/dev/null

for ctx in focus sleep workout; do
  OUT="$tmp_root/daily_$ctx"
  log "run daily ($ctx)"
  "$PY" -m mgc.main --db "$DB" run daily \
    --context "$ctx" \
    --seed 1 \
    --deterministic \
    --generate-count 1 \
    --out-dir "$OUT" \
    >/dev/null
  test -s "$OUT/drop_bundle/playlist.json"
  "$PY" scripts/check_playlist_tracks.py "$OUT/drop_bundle/playlist.json" "$OUT"
done

OUT_WEEKLY="$tmp_root/weekly_focus"
log "run weekly (focus)"
"$PY" -m mgc.main --db "$DB" run weekly \
  --context focus \
  --period-key 2020-W01 \
  --seed 1 \
  --deterministic \
  --out-dir "$OUT_WEEKLY" \
  >/dev/null

test -s "$OUT_WEEKLY/drop_bundle/playlist.json"
"$PY" scripts/check_playlist_tracks.py "$OUT_WEEKLY/drop_bundle/playlist.json" "$OUT_WEEKLY"

DROP_OUT="$tmp_root/drop_focus"
log "run drop (focus)"
mkdir -p "$DROP_OUT"
"$PY" -m mgc.main --db "$DB" run drop \
  --context focus \
  --seed 1 \
  --deterministic \
  --out-dir "$DROP_OUT" \
  > "$DROP_OUT/drop_stdout.json"

"$PY" -m json.tool < "$DROP_OUT/drop_stdout.json" >/dev/null

export DROP_OUT
BUNDLE_DIR="$($PY - <<'PY'
import json, os
p = os.path.join(os.environ['DROP_OUT'], 'drop_stdout.json')
obj = json.load(open(p, 'r', encoding='utf-8'))
paths = obj.get('paths') or {}
val = paths.get('bundle_dir_path') or paths.get('bundle_dir') or ''
if not val:
    raise SystemExit(2)
print(val)
PY
)"

if [[ "$BUNDLE_DIR" != /* ]]; then
  BUNDLE_DIR="$DROP_OUT/$BUNDLE_DIR"
fi

log "web build"
WEB_OUT="$tmp_root/web"
"$PY" -m mgc.main web build \
  --playlist "$BUNDLE_DIR/playlist.json" \
  --out-dir "$WEB_OUT" \
  --prefer-mp3 \
  --clean \
  --fail-if-empty \
  --json \
  | "$PY" -m json.tool >/dev/null
"$PY" -m mgc.main web validate --out-dir "$WEB_OUT" >/dev/null

log "publish-marketing dry-run (file mode)"
"$PY" -m mgc.main run publish-marketing \
  --bundle-dir "$BUNDLE_DIR" \
  --out-dir "$DROP_OUT" \
  --deterministic \
  --dry-run \
  >/dev/null

test -d "$DROP_OUT/marketing/receipts"

log "analytics checks"
"$PY" -m mgc.main analytics overview --db "$DB" >/dev/null
"$PY" -m mgc.main analytics tracks --db "$DB" --top 3 >/dev/null
"$PY" -m mgc.main analytics playlists --db "$DB" --limit 1 >/dev/null
"$PY" -m mgc.main analytics runs --db "$DB" --limit 1 >/dev/null
"$PY" -m mgc.main analytics marketing --db "$DB" --limit 1 >/dev/null
"$PY" -m mgc.main analytics stability --db "$DB" >/dev/null
"$PY" -m mgc.main analytics reuse --db "$DB" >/dev/null
"$PY" -m mgc.main analytics duration --db "$DB" >/dev/null
"$PY" -m mgc.main analytics export overview --db "$DB" --format json --out "$tmp_root/analytics_overview.json" >/dev/null
test -s "$tmp_root/analytics_overview.json"

log "submission build"
"$PY" -m mgc.main submission build --bundle-dir "$BUNDLE_DIR" --out "$tmp_root/submission.zip" >/dev/null

test -s "$tmp_root/submission.zip"

log "OK - all checks passed"
