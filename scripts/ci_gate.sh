#!/usr/bin/env bash
set -euo pipefail

# scripts/ci_gate.sh
#
# CI gate: compile + smoke (autonomous) + rebuild/verify + determinism checks + manifest diff gate
#
# Env:
#   MGC_DB              DB path (default: data/db.sqlite)
#   PYTHON              python executable (default: python)
#   ARTIFACTS_DIR       where to write logs/outputs (default: artifacts/ci)
#   MGC_ARTIFACTS_DIR   preferred override for ARTIFACTS_DIR
#   MGC_OUT_ROOT        override output root for rebuilds:
#                       - if set to "data", writes to data/playlists + data/tracks
#                       - otherwise writes under $ARTIFACTS_DIR/rebuild/...
#
# Notes:
# - Uses `run autonomous` as the primary smoke test (drop + submission verify).
# - In CI, DO NOT assume repo-local data/ exists. Use $MGC_ARTIFACTS_DIR outputs.
# - Keeps stdout clean where possible; writes a ci_gate.log always.
# - Hygiene: if fixtures/ci_db.sqlite is tracked, restores it on exit.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

PYTHON="${PYTHON:-python}"
DB="${MGC_DB:-${DB:-data/db.sqlite}}"
ARTIFACTS_DIR="${MGC_ARTIFACTS_DIR:-${ARTIFACTS_DIR:-artifacts/ci}}"
export MGC_ARTIFACTS_DIR="$ARTIFACTS_DIR"

mkdir -p "$ARTIFACTS_DIR"
log_file="$ARTIFACTS_DIR/ci_gate.log"
: >"$log_file"

info() { echo "[ci_gate] $*"; }
err()  { echo "[ci_gate] ERROR: $*" >&2; }

cleanup() {
  # Restore tracked fixture DB if it exists in the repo index and was modified.
  if git ls-files --error-unmatch fixtures/ci_db.sqlite >/dev/null 2>&1; then
    if ! git diff --quiet -- fixtures/ci_db.sqlite 2>/dev/null; then
      info "restoring tracked fixtures/ci_db.sqlite"
      git checkout -- fixtures/ci_db.sqlite >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

python_compile() {
  info "py_compile"
  "$PYTHON" -m py_compile src/mgc/main.py
  "$PYTHON" -m py_compile src/mgc/run_cli.py
  "$PYTHON" -m py_compile src/mgc/submission_cli.py
  "$PYTHON" -m py_compile src/mgc/bundle_validate.py
  # Optional modules (ignore if missing)
  [[ -f src/mgc/web_cli.py ]] && "$PYTHON" -m py_compile src/mgc/web_cli.py || true
  [[ -f src/mgc/drops_cli.py ]] && "$PYTHON" -m py_compile src/mgc/drops_cli.py || true
}

_dump_tree_quick() {
  # Helpful for CI debugging when something is missing.
  # Kept quiet unless caller wants it (caller can redirect to log_file).
  local root="$1"
  if [[ -d "$root" ]]; then
    info "tree (top) for: $root"
    find "$root" -maxdepth 4 -type f | sed 's#^#  #'
  else
    info "tree skipped (missing dir): $root"
  fi
}

autonomous_smoke() {
  local out_dir="$1"
  info "autonomous smoke test (deterministic) out_dir=$out_dir"

  mkdir -p "$out_dir"

  # Deterministic + fixed time so CI is stable.
  # This should write a portable bundle under out_dir:
  #   - tracks/
  #   - playlist.json
  #   - daily_evidence.json
  MGC_DB="$DB" MGC_DETERMINISTIC=1 MGC_FIXED_TIME="2020-01-01T00:00:00Z" \
    "$PYTHON" -m mgc.main run autonomous --context focus --seed 1 --out-dir "$out_dir" >/dev/null

  info "submission artifact check"

  # CI contract: ensure the portable bundle exists somewhere under out_dir.
  # We DO NOT assume repo-local data/submissions in CI.
  local evidence_path
  evidence_path="$(find "$out_dir" -maxdepth 6 -type f -name "daily_evidence.json" | head -n 1 || true)"
  if [[ -z "${evidence_path}" ]]; then
    err "No daily_evidence.json found under out_dir=$out_dir (autonomous run didn't produce a portable bundle)"
    _dump_tree_quick "$out_dir" >&2 || true
    return 2
  fi

  local bundle_dir
  bundle_dir="$(cd "$(dirname "$evidence_path")" && pwd)"
  if [[ ! -d "$bundle_dir/tracks" ]]; then
    err "Bundle dir missing tracks/: $bundle_dir"
    _dump_tree_quick "$bundle_dir" >&2 || true
    return 2
  fi
  if [[ ! -f "$bundle_dir/playlist.json" ]]; then
    err "Bundle dir missing playlist.json: $bundle_dir"
    _dump_tree_quick "$bundle_dir" >&2 || true
    return 2
  fi

  # Exercise submission "latest" builder (CLI contract)
  #
  # IMPORTANT:
  # - In CI, data/db.sqlite and data/evidence may not exist.
  # - The autonomous smoke writes its portable bundle under out_dir.
  # So we must pass the DB path explicitly and point evidence scanning at out_dir.
  rm -f /tmp/ci_submission.zip /tmp/ci_submission_build.json >/dev/null 2>&1 || true

  "$PYTHON" -m mgc.main submission latest \
    --db "$DB" \
    --evidence-root "$out_dir" \
    --out /tmp/ci_submission.zip \
    --json \
    >/tmp/ci_submission_build.json

  if [[ ! -s /tmp/ci_submission.zip ]]; then
    err "submission latest produced empty /tmp/ci_submission.zip"
    err "submission latest output:"
    cat /tmp/ci_submission_build.json >&2 || true
    return 2
  fi

  # Tight contract: verify the zip we just built (fast)
  "$PYTHON" -m mgc.main submission verify --zip /tmp/ci_submission.zip --json >/dev/null

  info "submission artifact check OK"
}

run_gate() {
  info "MGC_ARTIFACTS_DIR=${MGC_ARTIFACTS_DIR}"
  info "Repo: $repo_root"
  info "MGC_DB=$DB"
  info "ARTIFACTS_DIR=$ARTIFACTS_DIR"
  info "MGC_OUT_ROOT=${MGC_OUT_ROOT:-}"
  info "git_sha: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  info "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  info "python: $("$PYTHON" -V 2>&1)"

  if [[ ! -f "$DB" ]]; then
    err "DB not found: $DB"
    return 2
  fi
  info "DB OK."

  python_compile

  # Primary smoke test: end-to-end autonomous run (portable bundle + submission zip)
  autonomous_smoke "$ARTIFACTS_DIR/auto"

  # Optional: if drops CLI exists, ensure global --json hoisting still works.
  # Keep it non-fatal if drops command isn't present.
  if "$PYTHON" -m mgc.main --help 2>/dev/null | grep -qE '\bdrops\b'; then
    info "drops list smoke (global --json hoist)"
    MGC_DB="$DB" "$PYTHON" -m mgc.main --json drops list --limit 1 >/dev/null
  fi

  info "rebuild + verify"
  MGC_DB="$DB" ARTIFACTS_DIR="$ARTIFACTS_DIR" PYTHON="$PYTHON" MGC_OUT_ROOT="${MGC_OUT_ROOT:-}" \
    bash scripts/ci_rebuild_verify.sh

  info "publish receipts determinism"
  MGC_DB="$DB" PYTHON="$PYTHON" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
    bash scripts/ci_publish_determinism.sh "ci_publish"

  info "manifest diff gate (since-ok, strict JSON)"
  MGC_DB="$DB" "$PYTHON" -m mgc.main --json run diff --since-ok --fail-on-changes --summary-only | "$PYTHON" -m json.tool

  info "OK"
}

run_gate 2>&1 | tee -a "$log_file"
