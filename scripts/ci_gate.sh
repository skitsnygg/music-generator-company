#!/usr/bin/env bash
set -euo pipefail

# scripts/ci_gate.sh
#
# CI gate with modes:
#   - fast (default): compile + rebuild/verify
#   - full: fast + autonomous smoke + submission/web determinism + publish determinism + weekly determinism
#
# Env:
#   MGC_DB             DB path (required)
#   PYTHON             python executable (default: python)
#   MGC_ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT       override output root for rebuilds:
#                      - if set to "data", writes to data/playlists + data/tracks
#                      - otherwise writes under $MGC_OUT_ROOT/{playlists,tracks}
#   MGC_CI_MODE        fast|full (default: fast)
#   MGC_STAMP          stamp string for rebuild outputs (default: ci)
#
# Evidence contract:
#   - drop_evidence.json is required and is written under: <out-dir>/drop_evidence.json
#     (historically sometimes under <out-dir>/evidence/drop_evidence.json; we detect both)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
: "${MGC_DB:?set MGC_DB}"

ARTIFACTS_DIR="${MGC_ARTIFACTS_DIR:-artifacts/ci}"
MODE="${MGC_CI_MODE:-fast}"
STAMP="${MGC_STAMP:-ci}"

mkdir -p "$ARTIFACTS_DIR"
LOG_PATH="$ARTIFACTS_DIR/ci_gate.log"
exec > >(tee "$LOG_PATH") 2>&1

echo "[ci_gate] mode=$MODE"
echo "[ci_gate] Repo: $repo_root"
echo "[ci_gate] MGC_DB=$MGC_DB"
echo "[ci_gate] MGC_ARTIFACTS_DIR=$ARTIFACTS_DIR"
echo "[ci_gate] MGC_OUT_ROOT=${MGC_OUT_ROOT:-}"
echo "[ci_gate] git_sha: $(git rev-parse HEAD 2>/dev/null || true)"
echo "[ci_gate] git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
echo "[ci_gate] python: $($PYTHON -V 2>&1 || true)"

_detect_evidence_root() {
  local out_dir="$1"
  local p="${out_dir%/}"

  if [ -f "$p/drop_evidence.json" ]; then
    echo "$p"
    return 0
  fi
  if [ -f "$p/evidence/drop_evidence.json" ]; then
    echo "$p/evidence"
    return 0
  fi

  echo ""
  return 1
}

_run_web_bundle_determinism() {
  local evidence_root="$1"

  set +e
  ( cd "$evidence_root" && bash "$repo_root/scripts/ci_web_bundle_determinism.sh" --evidence-root "." )
  local rc=$?
  set -e

  if [ $rc -eq 0 ]; then
    return 0
  fi

  # If web determinism fails due to known volatiles in web_manifest.json, try normalized compare
  if [ -f "$evidence_root/web_manifest.json" ] && [ -f "$evidence_root/web_manifest_2.json" ]; then
    echo "[ci_gate] web determinism failed; attempting normalized compare"
    "$PYTHON" - <<'PY' "$evidence_root/web_manifest.json" "$evidence_root/web_manifest_2.json"
import json, sys
from pathlib import Path

a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

for obj in (a, b):
    if isinstance(obj, dict):
        for k in ("ts", "built_ts", "generated_at"):
            obj.pop(k, None)

if a != b:
    print("normalized manifests still differ", file=sys.stderr)
    sys.exit(3)
print("normalized manifests match")
PY
    return 0
  fi

  return $rc
}

_resolve_rebuild_out_dirs() {
  local out_root="${MGC_OUT_ROOT:-}"
  if [ -z "$out_root" ]; then
    OUT_PLAYLISTS="$ARTIFACTS_DIR/rebuild/playlists"
    OUT_TRACKS="$ARTIFACTS_DIR/rebuild/tracks"
    return 0
  fi

  if [ "$out_root" = "data" ]; then
    OUT_PLAYLISTS="data/playlists"
    OUT_TRACKS="data/tracks"
    return 0
  fi

  OUT_PLAYLISTS="$out_root/playlists"
  OUT_TRACKS="$out_root/tracks"
}

echo "[ci_gate] DB OK."

echo "[ci_gate] py_compile"
"$PYTHON" -m py_compile src/mgc/main.py src/mgc/run_cli.py src/mgc/web_cli.py

echo "[ci_gate] rebuild + verify"
_resolve_rebuild_out_dirs
mkdir -p "$OUT_PLAYLISTS" "$OUT_TRACKS"

if [ -f "$repo_root/scripts/ci_rebuild_verify.sh" ]; then
  MGC_DB="$MGC_DB" \
  PYTHON="$PYTHON" \
  MGC_OUT_PLAYLISTS="$OUT_PLAYLISTS" \
  MGC_OUT_TRACKS="$OUT_TRACKS" \
  MGC_STAMP="$STAMP" \
  bash "$repo_root/scripts/ci_rebuild_verify.sh"
else
  "$PYTHON" -m mgc.main --db "$MGC_DB" rebuild playlists \
    --out-dir "$OUT_PLAYLISTS" \
    --stamp "$STAMP" \
    --determinism-check \
    --write \
    --json >/dev/null

  "$PYTHON" -m mgc.main --db "$MGC_DB" rebuild tracks \
    --out-dir "$OUT_TRACKS" \
    --stamp "$STAMP" \
    --determinism-check \
    --write \
    --json >/dev/null

  "$PYTHON" -m mgc.main --db "$MGC_DB" rebuild verify playlists --out-dir "$OUT_PLAYLISTS" --json >/dev/null
  "$PYTHON" -m mgc.main --db "$MGC_DB" rebuild verify tracks --out-dir "$OUT_TRACKS" --json >/dev/null
fi

if [ "$MODE" != "full" ]; then
  echo "[ci_gate] OK"
  exit 0
fi

# -----------------------------
# Full mode: autonomous smoke + determinism
# -----------------------------
AUTO_OUT="${ARTIFACTS_DIR%/}/auto"
rm -rf "$AUTO_OUT"
mkdir -p "$AUTO_OUT"

AUTONOMOUS_JSON="${AUTO_OUT%/}/autonomous.json"
AUTONOMOUS_ERR="${AUTO_OUT%/}/autonomous.err"
rm -f "$AUTONOMOUS_JSON" "$AUTONOMOUS_ERR"

echo "[ci_gate] autonomous smoke test (deterministic) out_dir=$AUTO_OUT"

cmd=(
  "$PYTHON" -m mgc.main
  --db "$MGC_DB"
  --repo-root "$repo_root"
  --seed 1
  --no-resume
  --json
  run autonomous
  --context focus
  --out-dir "$AUTO_OUT"
  --deterministic
)

printf "[ci_gate] autonomous argv:"
for a in "${cmd[@]}"; do printf " %q" "$a"; done
printf "\n"

set +e
MGC_DETERMINISTIC=1 "${cmd[@]}" >"$AUTONOMOUS_JSON" 2>"$AUTONOMOUS_ERR"
rc=$?
set -e

if [ $rc -ne 0 ]; then
  echo "[ci_gate] FAIL: autonomous returned rc=$rc" >&2
  echo "[ci_gate] autonomous stderr (first 200 lines):" >&2
  sed -n '1,200p' "$AUTONOMOUS_ERR" >&2 || true
  echo "[ci_gate] autonomous stdout (first 120 lines):" >&2
  sed -n '1,120p' "$AUTONOMOUS_JSON" >&2 || true
  exit 4
fi

if head -n 1 "$AUTONOMOUS_JSON" | grep -qE '^usage: mgc\b'; then
  echo "[ci_gate] FAIL: autonomous printed argparse help to stdout" >&2
  echo "[ci_gate] autonomous stderr (first 200 lines):" >&2
  sed -n '1,200p' "$AUTONOMOUS_ERR" >&2 || true
  echo "[ci_gate] autonomous stdout (first 120 lines):" >&2
  sed -n '1,120p' "$AUTONOMOUS_JSON" >&2 || true
  exit 4
fi

EVIDENCE_ROOT="$(_detect_evidence_root "$AUTO_OUT" || true)"
if [ -z "$EVIDENCE_ROOT" ]; then
  echo "[ci_gate] FAIL: could not locate drop_evidence.json under $AUTO_OUT or $AUTO_OUT/evidence" >&2
  echo "[ci_gate] contents of $AUTO_OUT:" >&2
  ls -la "$AUTO_OUT" >&2 || true
  if [ -d "${AUTO_OUT%/}/evidence" ]; then
    echo "[ci_gate] contents of ${AUTO_OUT%/}/evidence:" >&2
    ls -la "${AUTO_OUT%/}/evidence" >&2 || true
  fi
  exit 4
fi

export MGC_EVIDENCE_DIR="$EVIDENCE_ROOT"
echo "[ci_gate] evidence_root=$EVIDENCE_ROOT"

if [ ! -f "${EVIDENCE_ROOT%/}/drop_evidence.json" ]; then
  echo "[ci_gate] FAIL: missing drop_evidence.json under $EVIDENCE_ROOT" >&2
  ls -la "$EVIDENCE_ROOT" >&2 || true
  exit 4
fi

echo "[ci_gate] determinism gate: submission.zip (evidence-root=$EVIDENCE_ROOT)"
( cd "$EVIDENCE_ROOT" && bash "$repo_root/scripts/ci_submission_determinism.sh" --evidence-root "." )

echo "[ci_gate] determinism gate: web bundle (evidence-root=$EVIDENCE_ROOT)"
_run_web_bundle_determinism "$EVIDENCE_ROOT"

echo "[ci_gate] weekly determinism"
if [ -f "$repo_root/scripts/ci_weekly_determinism.sh" ]; then
  MGC_DB="$MGC_DB" PYTHON="$PYTHON" REPO_ROOT="$repo_root" CONTEXT="${MGC_CONTEXT:-focus}" SEED="${MGC_SEED:-1}" \
    bash "$repo_root/scripts/ci_weekly_determinism.sh" ci_weekly
else
  echo "[ci_gate] WARN: scripts/ci_weekly_determinism.sh missing; skipping weekly determinism"
fi

echo "[ci_gate] drops list smoke (global --json hoist)"
"$PYTHON" -m mgc.main --db "$MGC_DB" drops list --json >/dev/null

echo "[ci_gate] publish receipts determinism"
bash "$repo_root/scripts/ci_publish_determinism.sh" --db "$MGC_DB" --artifacts-dir "$ARTIFACTS_DIR"

echo "[ci_gate] OK"
