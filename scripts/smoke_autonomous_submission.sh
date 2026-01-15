#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DB="${MGC_DB:-fixtures/ci_db.sqlite}"
CONTEXT="${MGC_CONTEXT:-focus}"
SEED="${MGC_SEED:-1}"

OUT_DIR="${1:-/tmp/mgc_smoke_autonomous}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

export MGC_DETERMINISTIC=1
export MGC_SEED="$SEED"

$PYTHON -m mgc.main --db "$DB" --seed "$SEED" --no-resume run autonomous --context "$CONTEXT" --out-dir "$OUT_DIR"

$PYTHON -m mgc.main submission build --evidence-root "$OUT_DIR" --out "$OUT_DIR/submission.zip"
$PYTHON -m mgc.main submission verify --zip "$OUT_DIR/submission.zip"

echo "[smoke] OK out_dir=$OUT_DIR"
