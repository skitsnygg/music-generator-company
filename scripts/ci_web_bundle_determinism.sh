#!/usr/bin/env bash
# scripts/ci_web_bundle_determinism.sh
set -euo pipefail

# Determinism gate: web bundle tree hash
#
# Usage:
#   bash scripts/ci_web_bundle_determinism.sh --evidence-root <dir>
#
# Env:
#   MGC_DB            required (db path)
#   PYTHON            python executable (default: python)
#   MGC_FIXED_TIME    optional (used by pipeline for deterministic timestamps)
#   MGC_DETERMINISTIC optional ("1" recommended)
#
# Behavior:
#   - Builds a deterministic drop twice (focus, seed=1) under evidence-root
#   - Runs `mgc web build` twice from each drop
#   - Computes tree hash of resulting web output dirs using mgc.hash_tree
#   - Fails if hashes differ
#   - Writes web_bundle_determinism.json in evidence root
#   - If drop_evidence.json exists under evidence root, records:
#       artifacts.web_bundle_tree_sha256 = <hash>
#       paths.web_bundle_dir = <dir>

PYTHON="${PYTHON:-python}"

usage() {
  echo "usage: $0 --evidence-root <dir>" >&2
  exit 2
}

EVIDENCE_ROOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --evidence-root)
      EVIDENCE_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage
      ;;
  esac
done

[ -n "$EVIDENCE_ROOT" ] || usage
: "${MGC_DB:?set MGC_DB}"

mkdir -p "$EVIDENCE_ROOT"

echo "[ci_web_bundle_determinism] evidence_root=$EVIDENCE_ROOT"
echo "[ci_web_bundle_determinism] MGC_DB=$MGC_DB"

BASE="${EVIDENCE_ROOT%/}/web_bundle_test"
RUN1_DROP="${BASE}/run1_drop"
RUN1_WEB="${BASE}/run1_web"
RUN2_DROP="${BASE}/run2_drop"
RUN2_WEB="${BASE}/run2_web"

rm -rf "$BASE"
mkdir -p "$RUN1_DROP" "$RUN1_WEB" "$RUN2_DROP" "$RUN2_WEB"

run_drop_and_web_build() {
  local drop_dir="$1"
  local web_dir="$2"

  # Deterministic drop
  MGC_DETERMINISTIC="${MGC_DETERMINISTIC:-1}" \
  "$PYTHON" -m mgc.main --db "$MGC_DB" run drop \
    --context focus \
    --seed 1 \
    --deterministic \
    --out-dir "$drop_dir" \
    > "${drop_dir%/}/drop_stdout.json"

  "$PYTHON" -m json.tool < "${drop_dir%/}/drop_stdout.json" >/dev/null

  test -s "${drop_dir%/}/playlist.json"
  test -d "${drop_dir%/}/tracks"

  # Web build from inside drop dir (relative paths)
  pushd "$drop_dir" >/dev/null
  "$PYTHON" -m mgc.main web build \
    --playlist "playlist.json" \
    --out-dir "$web_dir" \
    --prefer-mp3 \
    --clean \
    --fail-if-empty \
    --json \
    | "$PYTHON" -m json.tool >/dev/null
  popd >/dev/null

  test "$(find "$web_dir" -maxdepth 6 -type f -name 'index.html' | wc -l | tr -d ' ')" -ge 1
  test "$(find "$web_dir" -maxdepth 8 -type f \( -name '*.mp3' -o -name '*.wav' \) | wc -l | tr -d ' ')" -ge 1
}

echo "[ci_web_bundle_determinism] build run1"
run_drop_and_web_build "$RUN1_DROP" "$RUN1_WEB"

echo "[ci_web_bundle_determinism] build run2"
run_drop_and_web_build "$RUN2_DROP" "$RUN2_WEB"

# Tree hashes (requires module mgc.hash_tree)
H1="$("$PYTHON" -m mgc.hash_tree --root "$RUN1_WEB" --print)"
H2="$("$PYTHON" -m mgc.hash_tree --root "$RUN2_WEB" --print)"

if [ "$H1" != "$H2" ]; then
  echo "[ci_web_bundle_determinism] FAIL: web bundle tree hash mismatch"
  echo "  run1=$H1"
  echo "  run2=$H2"
  echo "  run1_web=$RUN1_WEB"
  echo "  run2_web=$RUN2_WEB"
  exit 1
fi

echo "[ci_web_bundle_determinism] OK sha256=$H1"

OUT_JSON="${EVIDENCE_ROOT%/}/web_bundle_determinism.json"
EVIDENCE_JSON="${EVIDENCE_ROOT%/}/drop_evidence.json"

export OUT_JSON EVIDENCE_JSON H1 RUN1_WEB RUN2_WEB

"$PYTHON" - <<'PY'
import json, os
from pathlib import Path

out_json = Path(os.environ["OUT_JSON"])
evidence_json = Path(os.environ["EVIDENCE_JSON"])
h = os.environ["H1"]
run1_web = os.environ["RUN1_WEB"]
run2_web = os.environ["RUN2_WEB"]

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(
    json.dumps(
        {
            "ok": True,
            "web_bundle_tree_sha256": h,
            "run1_web": run1_web,
            "run2_web": run2_web,
        },
        sort_keys=True,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

# Best-effort: record into drop_evidence.json if it exists
if evidence_json.exists() and evidence_json.stat().st_size > 0:
    obj = json.loads(evidence_json.read_text(encoding="utf-8"))

    artifacts = obj.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        obj["artifacts"] = artifacts
    artifacts["web_bundle_tree_sha256"] = h

    paths = obj.get("paths")
    if not isinstance(paths, dict):
        paths = {}
        obj["paths"] = paths
    # store the canonical run1 output dir
    paths["web_bundle_dir"] = run1_web

    evidence_json.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
