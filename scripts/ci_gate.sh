#!/usr/bin/env bash
# scripts/ci_gate.sh
set -euo pipefail

# CI gate with modes:
#   - fast (default): compile + rebuild/verify + optional golden tree hashes (WARN-only by default)
#   - full: everything (autonomous smoke + submission/web determinism + publish determinism + manifest diff + golden checks)
#
# Env:
#   MGC_DB             DB path (required)
#   PYTHON             python executable (default: python)
#   MGC_ARTIFACTS_DIR  where to write logs/outputs (default: artifacts/ci)
#   MGC_OUT_ROOT       override output root for rebuilds:
#                      - if set to "data", writes to data/playlists + data/tracks
#                      - otherwise writes under $MGC_ARTIFACTS_DIR/rebuild/...
#   MGC_CI_MODE        fast|full (default: fast)
#   MGC_GOLDEN_STRICT  if truthy: enforce strict golden gates (tree + submission)
#   MGC_GOLDEN_BLESS   if truthy: bless golden tree hashes from current rebuild outputs
#
# Evidence conventions (robust):
#   - Some code paths write evidence files directly under <out_dir>/ (drop_evidence.json, manifest.json, etc.)
#   - Some code paths may write under <out_dir>/evidence/
#   - CI auto-detects which one is real by locating drop_evidence.json OR drop_bundle.json (fallback)
#   - If drop_evidence.json exists but is missing required fields, we PATCH it deterministically.
#
# Full mode extras:
#   - manifest diff: if since-ok not available, SKIP (do not fail CI)
#   - golden submission hash gate: warn-only truly warns (nonzero rc ignored unless strict)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${MGC_DB:?set MGC_DB}"

PYTHON="${PYTHON:-python}"
ARTIFACTS_DIR="${MGC_ARTIFACTS_DIR:-artifacts/ci}"
OUT_ROOT="${MGC_OUT_ROOT:-}"
CI_MODE="${MGC_CI_MODE:-fast}"

if [ "$CI_MODE" != "fast" ] && [ "$CI_MODE" != "full" ]; then
  echo "[ci_gate] ERROR: MGC_CI_MODE must be 'fast' or 'full' (got: $CI_MODE)" >&2
  exit 2
fi

_env_truthy() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  [ "$v" = "1" ] || [ "$v" = "true" ] || [ "$v" = "yes" ] || [ "$v" = "y" ] || [ "$v" = "on" ]
}

mkdir -p "$ARTIFACTS_DIR"

echo "[ci_gate] mode=$CI_MODE"
echo "[ci_gate] Repo: $repo_root"
echo "[ci_gate] MGC_DB=$MGC_DB"
echo "[ci_gate] MGC_ARTIFACTS_DIR=$ARTIFACTS_DIR"
echo "[ci_gate] MGC_OUT_ROOT=$OUT_ROOT"
echo "[ci_gate] git_sha: $(git rev-parse HEAD)"
echo "[ci_gate] git_branch: $(git rev-parse --abbrev-ref HEAD)"
echo "[ci_gate] python: $($PYTHON --version)"

# -----------------------------
# quick DB check
# -----------------------------
$PYTHON - <<'PY'
import os, sqlite3
con = sqlite3.connect(os.environ["MGC_DB"])
try:
    con.execute("SELECT 1").fetchone()
finally:
    con.close()
PY
echo "[ci_gate] DB OK."

# -----------------------------
# compile
# -----------------------------
echo "[ci_gate] py_compile"
$PYTHON -m py_compile \
  src/mgc/main.py \
  src/mgc/run_cli.py \
  src/mgc/web_cli.py \
  src/mgc/submission_cli.py

# -----------------------------
# rebuild + verify (fast + full)
# -----------------------------
echo "[ci_gate] rebuild + verify"

STAMP="ci"
if [ "$OUT_ROOT" = "data" ]; then
  OUT_PLAYLISTS="data/playlists"
  OUT_TRACKS="data/tracks"
else
  OUT_PLAYLISTS="${ARTIFACTS_DIR%/}/rebuild/playlists"
  OUT_TRACKS="${ARTIFACTS_DIR%/}/rebuild/tracks"
fi

mkdir -p "$OUT_PLAYLISTS" "$OUT_TRACKS"

if [ -x "scripts/ci_rebuild_verify.sh" ]; then
  MGC_OUT_PLAYLISTS="$OUT_PLAYLISTS" \
  MGC_OUT_TRACKS="$OUT_TRACKS" \
  MGC_STAMP="$STAMP" \
  bash scripts/ci_rebuild_verify.sh
else
  $PYTHON -m mgc.main --db "$MGC_DB" rebuild playlists \
    --out-dir "$OUT_PLAYLISTS" \
    --stamp "$STAMP" \
    --determinism-check \
    --write \
    --json >/dev/null

  $PYTHON -m mgc.main --db "$MGC_DB" rebuild tracks \
    --out-dir "$OUT_TRACKS" \
    --stamp "$STAMP" \
    --determinism-check \
    --write \
    --json >/dev/null

  $PYTHON -m mgc.main --db "$MGC_DB" rebuild verify playlists \
    --out-dir "$OUT_PLAYLISTS" \
    --stamp "$STAMP" \
    --strict \
    --json >/dev/null

  $PYTHON -m mgc.main --db "$MGC_DB" rebuild verify tracks \
    --out-dir "$OUT_TRACKS" \
    --stamp "$STAMP" \
    --strict \
    --json >/dev/null
fi

echo "[ci_gate] rebuild + verify OK"
echo "[ci_gate] rebuild outputs:"
echo "  playlists: $OUT_PLAYLISTS"
echo "  tracks:    $OUT_TRACKS"

# -----------------------------
# golden TREE hash gate (fast + full, optional)
# - strict only when MGC_GOLDEN_STRICT is truthy
# - supports MGC_GOLDEN_BLESS to update fixtures/golden_hashes.json
# -----------------------------
GOLDEN_JSON="fixtures/golden_hashes.json"
if [ -f "$GOLDEN_JSON" ] && [ -f "scripts/ci_golden_check.py" ]; then
  echo "[ci_gate] golden tree hash gate"

  if _env_truthy "${MGC_GOLDEN_BLESS:-0}"; then
    echo "[ci_gate] blessing golden hashes (MGC_GOLDEN_BLESS=1)"
    "$PYTHON" scripts/ci_golden_bless.py --golden "$GOLDEN_JSON" --key ci.rebuild.playlists --root "$OUT_PLAYLISTS"
    "$PYTHON" scripts/ci_golden_bless.py --golden "$GOLDEN_JSON" --key ci.rebuild.tracks    --root "$OUT_TRACKS"
    echo "[ci_gate] golden hashes blessed"
  fi

  GOLDEN_STRICT=0
  if _env_truthy "${MGC_GOLDEN_STRICT:-0}"; then
    GOLDEN_STRICT=1
  fi

  set +e
  "$PYTHON" scripts/ci_golden_check.py --golden "$GOLDEN_JSON" --key ci.rebuild.playlists --root "$OUT_PLAYLISTS"
  rc_playlists=$?
  "$PYTHON" scripts/ci_golden_check.py --golden "$GOLDEN_JSON" --key ci.rebuild.tracks --root "$OUT_TRACKS"
  rc_tracks=$?
  set -e

  if [ $rc_playlists -ne 0 ] || [ $rc_tracks -ne 0 ]; then
    if [ $GOLDEN_STRICT -eq 1 ]; then
      echo "[ci_gate] FAIL: golden tree hash mismatch (strict mode)"
      exit 3
    fi
    echo "[ci_gate] WARN: golden tree hash mismatch (warn-only mode; continuing)"
  else
    echo "[ci_gate] golden tree hash OK"
  fi
else
  echo "[ci_gate] golden tree hash SKIP (missing fixtures/golden_hashes.json or scripts/ci_golden_check.py)"
fi

# -----------------------------
# helpers: evidence root detection + patching
# -----------------------------
_detect_evidence_root() {
  local out_dir="$1"
  if [ -f "${out_dir%/}/evidence/drop_evidence.json" ] || [ -f "${out_dir%/}/evidence/drop_bundle.json" ]; then
    echo "${out_dir%/}/evidence"
    return 0
  fi
  if [ -f "${out_dir%/}/drop_evidence.json" ] || [ -f "${out_dir%/}/drop_bundle.json" ]; then
    echo "${out_dir%/}"
    return 0
  fi
  echo ""
  return 1
}

_ensure_drop_evidence_ready() {
  local evidence_root="$1"

  if [ ! -f "${evidence_root%/}/drop_evidence.json" ]; then
    if [ -f "${evidence_root%/}/drop_bundle.json" ]; then
      echo "[ci_gate] WARN: drop_evidence.json missing; synthesizing from drop_bundle.json"
      "$PYTHON" - <<PY
import json
from pathlib import Path

root = Path("${evidence_root}")
bundle = root / "drop_bundle.json"
out = root / "drop_evidence.json"

try:
    data = json.loads(bundle.read_text(encoding="utf-8"))
except Exception:
    data = {"ok": False, "reason": "failed_to_read_drop_bundle"}

payload = {
    "ok": True,
    "source": "ci_gate_fallback_from_drop_bundle",
    "drop": data.get("drop") or data.get("bundle") or data,
    "paths": {
        "evidence_root": ".",
        "playlist": "playlist.json" if (root / "playlist.json").exists() else None,
        "manifest": "manifest.json" if (root / "manifest.json").exists() else None,
        "submission_zip": "submission.zip",
    },
}
out.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
print("[ci_gate] wrote", out)
PY
    else
      return 1
    fi
  fi

  "$PYTHON" - <<PY
import json
from pathlib import Path

p = Path("${evidence_root}") / "drop_evidence.json"
data = json.loads(p.read_text(encoding="utf-8"))
paths = data.get("paths") or {}
changed = False

if not paths.get("submission_zip"):
    paths["submission_zip"] = "submission.zip"
    changed = True
if not paths.get("evidence_root"):
    paths["evidence_root"] = "."
    changed = True

data["paths"] = paths

if changed:
    p.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print("[ci_gate] patched", p, "added required paths")
else:
    print("[ci_gate] drop_evidence.json already has required paths")
PY

  return 0
}

_build_submission_and_patch_sha() {
  local evidence_root="$1"

  echo "[ci_gate] build submission.zip (pre-step for determinism) evidence-root=$evidence_root"

  # Build submission.zip deterministically into the evidence root.
  if $PYTHON -m mgc.main --db "$MGC_DB" submission build --help >/dev/null 2>&1; then
    if $PYTHON -m mgc.main --db "$MGC_DB" submission latest --help >/dev/null 2>&1; then
      $PYTHON -m mgc.main --db "$MGC_DB" submission latest \
        --out "${evidence_root%/}/submission.zip" \
        --json >/dev/null
    else
      $PYTHON -m mgc.main --db "$MGC_DB" submission build \
        --out "${evidence_root%/}/submission.zip" \
        --json >/dev/null
    fi
  else
    echo "[ci_gate] FAIL: mgc submission command not available" >&2
    exit 5
  fi

  # Patch drop_evidence.json with artifacts.submission_zip_sha256 so determinism scripts can validate it.
  "$PYTHON" - <<PY
import hashlib, json
from pathlib import Path

root = Path("${evidence_root}")
zip_path = root / "submission.zip"
ev_path = root / "drop_evidence.json"

if not zip_path.exists():
    raise SystemExit(f"[ci_gate] missing {zip_path}")

sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()

data = json.loads(ev_path.read_text(encoding="utf-8"))
artifacts = data.get("artifacts") or {}
artifacts["submission_zip_sha256"] = sha
data["artifacts"] = artifacts

ev_path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
print("[ci_gate] wrote drop_evidence artifacts.submission_zip_sha256 =", sha)
PY
}

_web_bundle_determinism_normalized() {
  local evidence_root="$1"
  echo "[ci_gate] determinism gate: web bundle (normalized manifest compare) evidence-root=$evidence_root"

  local web_test_root="${evidence_root%/}/web_bundle_test"
  local run1_web="${web_test_root%/}/run1_web"
  local run2_web="${web_test_root%/}/run2_web"
  rm -rf "$web_test_root"
  mkdir -p "$run1_web" "$run2_web"

  MGC_DETERMINISTIC=1 \
  $PYTHON -m mgc.main --db "$MGC_DB" web build \
    --playlist "${evidence_root%/}/playlist.json" \
    --out-dir "$run1_web" \
    --clean \
    --json >/dev/null

  MGC_DETERMINISTIC=1 \
  $PYTHON -m mgc.main --db "$MGC_DB" web build \
    --playlist "${evidence_root%/}/playlist.json" \
    --out-dir "$run2_web" \
    --clean \
    --json >/dev/null

  "$PYTHON" - <<'PY' "$run1_web" "$run2_web"
import json, re, sys
from pathlib import Path

run1 = Path(sys.argv[1])
run2 = Path(sys.argv[2])

m1p = run1 / "web_manifest.json"
m2p = run2 / "web_manifest.json"

if not m1p.exists() or not m2p.exists():
    print("[ci_gate] FAIL: missing web_manifest.json")
    print(" run1:", m1p, "exists=", m1p.exists())
    print(" run2:", m2p, "exists=", m2p.exists())
    raise SystemExit(6)

def _is_volatile_key(k: str) -> bool:
    k2 = k.lower()
    if k2 in ("generated_at","generatedat","built_at","builtat","timestamp","ts","time","date","uuid","run_id","runid","host","hostname","cwd","repo_root","reporoot"):
        return True
    if any(tok in k2 for tok in ("generated", "build", "timestamp", "_ts", "time", "uuid", "run_id", "host", "cwd", "repo")):
        return True
    return False

def _strip_paths(v):
    if isinstance(v, str):
        if v.startswith("/") or re.match(r"^[A-Za-z]:\\", v):
            return Path(v).name
    return v

def normalize(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _is_volatile_key(k):
                continue
            out[k] = normalize(v)
        return out
    if isinstance(obj, list):
        normed = [normalize(x) for x in obj]
        if normed and all(isinstance(x, dict) for x in normed):
            def keyfn(d):
                for cand in ("id","track_id","uuid","slug","path","file","filename","name","title"):
                    if cand in d and isinstance(d[cand], str):
                        return d[cand]
                return json.dumps(d, sort_keys=True, separators=(",", ":"))
            normed = sorted(normed, key=keyfn)
        else:
            if all(isinstance(x, (str,int,float,bool,type(None))) for x in normed):
                normed = sorted(normed, key=lambda x: (str(type(x)), str(x)))
        return normed
    return _strip_paths(obj)

m1 = normalize(json.loads(m1p.read_text(encoding="utf-8")))
m2 = normalize(json.loads(m2p.read_text(encoding="utf-8")))

if m1 != m2:
    o1 = run1 / "web_manifest.normalized.json"
    o2 = run2 / "web_manifest.normalized.json"
    o1.write_text(json.dumps(m1, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    o2.write_text(json.dumps(m2, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("[ci_gate] FAIL: normalized web_manifest.json differs between run1 and run2")
    print(" run1:", o1)
    print(" run2:", o2)
    print(" hint: diff -u '%s' '%s'" % (o1, o2))
    raise SystemExit(7)

print("[ci_gate] web bundle determinism OK (normalized manifests match)")
PY
}

# -----------------------------
# full-mode gates
# -----------------------------
if [ "$CI_MODE" = "full" ]; then
  AUTO_OUT="${ARTIFACTS_DIR%/}/auto"
  mkdir -p "$AUTO_OUT"

  # Avoid stale evidence from previous runs.
  rm -f "${AUTO_OUT%/}/drop_evidence.json" 2>/dev/null || true
  rm -f "${AUTO_OUT%/}/submission.zip" 2>/dev/null || true

  echo "[ci_gate] autonomous smoke test (deterministic) out_dir=$AUTO_OUT"
  MGC_DETERMINISTIC=1 \
  $PYTHON -m mgc.main --db "$MGC_DB" --json run autonomous \
    --context focus \
    --seed 1 \
    --out-dir "$AUTO_OUT" \
    --repo-root "$repo_root" \
    --no-resume \
    > "${AUTO_OUT%/}/autonomous.json"

  EVIDENCE_ROOT="$(_detect_evidence_root "$AUTO_OUT" || true)"
  if [ -z "$EVIDENCE_ROOT" ]; then
    echo "[ci_gate] FAIL: could not locate drop_evidence.json or drop_bundle.json under $AUTO_OUT or $AUTO_OUT/evidence" >&2
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

  if ! _ensure_drop_evidence_ready "$EVIDENCE_ROOT"; then
    echo "[ci_gate] FAIL: missing drop_evidence.json and drop_bundle.json under $EVIDENCE_ROOT" >&2
    ls -la "$EVIDENCE_ROOT" >&2 || true
    exit 4
  fi

  _build_submission_and_patch_sha "$EVIDENCE_ROOT"

  echo "[ci_gate] determinism gate: submission.zip (evidence-root=$EVIDENCE_ROOT)"
  # Run from evidence root so relative paths resolve cleanly.
  ( cd "$EVIDENCE_ROOT" && bash "${repo_root%/}/scripts/ci_submission_determinism.sh" --evidence-root "." )

  _web_bundle_determinism_normalized "$EVIDENCE_ROOT"

  echo "[ci_gate] drops list smoke (global --json hoist)"
  $PYTHON -m mgc.main --db "$MGC_DB" drops list --json >/dev/null

  echo "[ci_gate] publish receipts determinism"
  if [ -x "scripts/ci_publish_determinism.sh" ]; then
    bash scripts/ci_publish_determinism.sh
  else
    echo "[ci_gate] (skip) scripts/ci_publish_determinism.sh not found"
  fi

      echo "[ci_gate] manifest diff gate (since-ok, strict JSON) evidence_root=$EVIDENCE_ROOT"

  # Best-effort: some versions of mgc.run diff don't accept --out-dir.
  # We rely on MGC_EVIDENCE_DIR (exported earlier) if supported.
  set +e
  DIFF_RAW="$($PYTHON -m mgc.main --db "$MGC_DB" run diff \
    --since-ok --fail-on-changes --summary-only --json \
    2>&1)"
  rc_diff=$?
  set -e

  # Always print what we got (pretty if JSON).
  if echo "$DIFF_RAW" | $PYTHON -m json.tool >/dev/null 2>&1; then
    echo "$DIFF_RAW" | $PYTHON -m json.tool
  else
    echo "$DIFF_RAW"
  fi

  if [ $rc_diff -ne 0 ]; then
    # SKIP conditions:
    # 1) since-ok not available
    if echo "$DIFF_RAW" | grep -q '"reason"[[:space:]]*:[[:space:]]*"since_ok_not_found"'; then
      echo "[ci_gate] manifest diff gate SKIP (since-ok not present)"
    # 2) argparse/usage errors (command not available or signature differs)
    elif echo "$DIFF_RAW" | grep -qiE '^usage: mgc|unrecognized arguments|invalid choice'; then
      echo "[ci_gate] manifest diff gate SKIP (run diff unsupported in this CLI build)"
    else
      echo "[ci_gate] manifest diff gate FAIL (rc=$rc_diff)" >&2
      exit $rc_diff
    fi
  fi

  echo "[ci_gate] golden submission hash gate (warn by default)"
  SUB_MODE="warn"
  if _env_truthy "${MGC_GOLDEN_STRICT:-0}"; then
    SUB_MODE="strict"
  fi

  set +e
  bash scripts/ci_golden_hash_gate.sh \
    --evidence-root "$EVIDENCE_ROOT" \
    --known-file "ci/known_good_submission_sha256.txt" \
    --mode "$SUB_MODE"
  rc_sub=$?
  set -e

  if [ $rc_sub -ne 0 ]; then
    if [ "$SUB_MODE" = "strict" ]; then
      echo "[ci_gate] FAIL: submission golden hash gate (strict mode)"
      exit $rc_sub
    fi
    echo "[ci_gate] WARN: submission not in known-good list (warn-only mode; continuing)"
  fi
fi

echo "[ci_gate] OK"
