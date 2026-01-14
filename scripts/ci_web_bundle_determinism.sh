#!/usr/bin/env bash
# scripts/ci_web_bundle_determinism.sh
set -euo pipefail

# Determinism check for the static web bundle build.
#
# Usage:
#   bash scripts/ci_web_bundle_determinism.sh --evidence-root <dir>
#
# Evidence root:
#   This directory should contain drop_evidence.json (and related evidence files),
#   or be the out_dir root used by `mgc run autonomous`.
#
# Env:
#   MGC_DB     sqlite db path (required, or provided by CI environment)
#   PYTHON     python executable (default: python)
#
# Output:
#   Writes build artifacts under <evidence-root>/web_bundle_test/
#   Compares deterministically-rebuilt manifests between run1/run2.

PYTHON="${PYTHON:-python}"
: "${MGC_DB:?set MGC_DB}"

EVIDENCE_ROOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --evidence-root)
      EVIDENCE_ROOT="${2:-}"; shift 2;;
    *)
      echo "[ci_web_bundle_determinism] ERROR: unknown arg: $1" >&2
      exit 2;;
  esac
done

if [ -z "$EVIDENCE_ROOT" ]; then
  echo "[ci_web_bundle_determinism] ERROR: --evidence-root is required" >&2
  exit 2
fi

echo "[ci_web_bundle_determinism] evidence_root=$EVIDENCE_ROOT"
echo "[ci_web_bundle_determinism] MGC_DB=$MGC_DB"

DROP_JSON="${EVIDENCE_ROOT%/}/drop_evidence.json"
PLAYLIST_JSON="${EVIDENCE_ROOT%/}/playlist.json"

if [ ! -f "$DROP_JSON" ]; then
  echo "[ci_web_bundle_determinism] ERROR: missing $DROP_JSON" >&2
  exit 3
fi

# Prefer playlist.json if present (autonomous run writes it); otherwise derive a playlist from drop evidence
# by asking mgc to export it (fallback).
if [ ! -f "$PLAYLIST_JSON" ]; then
  echo "[ci_web_bundle_determinism] playlist.json not found; attempting fallback export from drop evidence"
  # This fallback assumes your CLI has a way to export playlist from evidence; if it doesn't,
  # it will fail loudly and you'll see the missing command.
  #
  # If you already always have playlist.json, this path never triggers.
  $PYTHON -m mgc.main --db "$MGC_DB" --json run manifest \
    --out-dir "${EVIDENCE_ROOT%/}" \
    >/dev/null
  if [ ! -f "$PLAYLIST_JSON" ]; then
    echo "[ci_web_bundle_determinism] ERROR: still missing ${PLAYLIST_JSON} after fallback" >&2
    exit 3
  fi
fi

ROOT="${EVIDENCE_ROOT%/}/web_bundle_test"
RUN1_DROP="${ROOT%/}/run1_drop"
RUN2_DROP="${ROOT%/}/run2_drop"
RUN1_WEB="${ROOT%/}/run1_web"
RUN2_WEB="${ROOT%/}/run2_web"

rm -rf "$ROOT"
mkdir -p "$RUN1_DROP" "$RUN2_DROP" "$RUN1_WEB" "$RUN2_WEB"

echo "[ci_web_bundle_determinism] build run1"
# IMPORTANT: --out-dir must come AFTER `web build` (subcommand arg), not at the global mgc level.
$PYTHON -m mgc.main --db "$MGC_DB" --json web build \
  --playlist "$PLAYLIST_JSON" \
  --out-dir "$RUN1_WEB" \
  --clean \
  --fail-if-none-copied \
  --fail-on-missing \
  >/dev/null

echo "[ci_web_bundle_determinism] build run2"
$PYTHON -m mgc.main --db "$MGC_DB" --json web build \
  --playlist "$PLAYLIST_JSON" \
  --out-dir "$RUN2_WEB" \
  --clean \
  --fail-if-none-copied \
  --fail-on-missing \
  >/dev/null

MAN1="${RUN1_WEB%/}/web_manifest.json"
MAN2="${RUN2_WEB%/}/web_manifest.json"

if [ ! -f "$MAN1" ] || [ ! -f "$MAN2" ]; then
  echo "[ci_web_bundle_determinism] ERROR: missing web manifest(s)" >&2
  echo "  expected: $MAN1" >&2
  echo "  expected: $MAN2" >&2
  ls -la "$RUN1_WEB" >&2 || true
  ls -la "$RUN2_WEB" >&2 || true
  exit 4
fi

# Determinism check: exact byte equality.
if cmp -s "$MAN1" "$MAN2"; then
  echo "[ci_web_bundle_determinism] OK web_manifest.json matches"
else
  echo "[ci_web_bundle_determinism] FAIL: web_manifest.json differs between run1 and run2" >&2
  echo "  run1: $MAN1" >&2
  echo "  run2: $MAN2" >&2
  echo "  hint: diff -u '$MAN1' '$MAN2'" >&2
  exit 5
fi
