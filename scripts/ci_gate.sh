#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fast}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
MGC_DB="${MGC_DB:-fixtures/ci_db.sqlite}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/ci}"
MGC_OUT_ROOT="${MGC_OUT_ROOT:-}"

mkdir -p "$ARTIFACTS_DIR"

echo "[ci_gate] mode=$MODE"
echo "[ci_gate] Repo: $repo_root"
echo "[ci_gate] MGC_DB=$MGC_DB"
echo "[ci_gate] MGC_ARTIFACTS_DIR=$ARTIFACTS_DIR"
echo "[ci_gate] MGC_OUT_ROOT=$MGC_OUT_ROOT"
echo "[ci_gate] python: $($PYTHON -V 2>&1 || true)"

# -----------------------------
# Helpers
# -----------------------------

assert_no_absolute_paths() {
  local dir="$1"
  echo "[ci_gate] path-purity scan: $dir"

  local hits
  hits=$(grep -RInE '(^|[^A-Za-z0-9_])/(tmp|private/tmp|var/folders|Users|home)/' "$dir" || true)

  if [[ -n "$hits" ]]; then
    echo "[ci_gate] FAIL: absolute path leak detected under $dir" >&2
    echo "$hits" >&2
    exit 2
  fi
}

# -----------------------------
# Basic sanity
# -----------------------------

if [[ ! -f "$MGC_DB" ]]; then
  echo "[ci_gate] FAIL: DB not found: $MGC_DB" >&2
  exit 4
fi

echo "[ci_gate] DB OK."

# -----------------------------
# Compile
# -----------------------------

echo "[ci_gate] py_compile"
"$PYTHON" -m py_compile \
  src/mgc/main.py \
  src/mgc/run_cli.py \
  src/mgc/submission_cli.py \
  src/mgc/web_cli.py >/dev/null

# -----------------------------
# Rebuild + verify
# -----------------------------

rm -rf "$ARTIFACTS_DIR/rebuild"
mkdir -p "$ARTIFACTS_DIR/rebuild/playlists" "$ARTIFACTS_DIR/rebuild/tracks"

"$PYTHON" -m mgc.main --db "$MGC_DB" rebuild playlists \
  --out-dir "$ARTIFACTS_DIR/rebuild/playlists" \
  --stamp ci_root \
  --determinism-check --write --json >/dev/null

"$PYTHON" -m mgc.main --db "$MGC_DB" rebuild tracks \
  --out-dir "$ARTIFACTS_DIR/rebuild/tracks" \
  --stamp ci_root \
  --determinism-check --write --json >/dev/null

assert_no_absolute_paths "$ARTIFACTS_DIR/rebuild"

if [[ "$MODE" != "full" ]]; then
  echo "[ci_gate] OK"
  exit 0
fi

# -----------------------------
# Autonomous run
# -----------------------------

AUTO_OUT="$ARTIFACTS_DIR/auto"
rm -rf "$AUTO_OUT"
mkdir -p "$AUTO_OUT"

echo "[ci_gate] autonomous smoke test (deterministic) out_dir=$AUTO_OUT"

"$PYTHON" -m mgc.main \
  --db "$MGC_DB" \
  --repo-root "$repo_root" \
  --seed 1 \
  --no-resume \
  --json \
  run autonomous \
  --context focus \
  --out-dir "$AUTO_OUT" \
  --deterministic >"$AUTO_OUT/autonomous.json" 2>"$AUTO_OUT/autonomous.err"

if [[ ! -f "$AUTO_OUT/drop_evidence.json" ]]; then
  echo "[ci_gate] FAIL: missing drop_evidence.json" >&2
  exit 4
fi

echo "[ci_gate] evidence_root=$AUTO_OUT"
assert_no_absolute_paths "$AUTO_OUT"

# -----------------------------
# Submission determinism
# -----------------------------

echo "[ci_gate] determinism gate: submission.zip (evidence-root=$AUTO_OUT)"

MGC_DB="$MGC_DB" PYTHON="$PYTHON" \
  bash "$repo_root/scripts/ci_submission_determinism.sh" \
  --evidence-root "$AUTO_OUT"

echo "[ci_gate] OK"
