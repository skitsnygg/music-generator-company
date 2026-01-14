#!/usr/bin/env bash
set -euo pipefail

# Determinism gate: web bundle tree hash
#
# Usage:
#   EVIDENCE_ROOT=artifacts/ci/auto bash scripts/ci_web_bundle_determinism.sh
#   # (back-compat) PLAYLIST=... bash scripts/ci_web_bundle_determinism.sh
#
# Reads:
#   $EVIDENCE_ROOT/drop_evidence.json (preferred)
#
# Emits:
#   Logs to stderr, exits nonzero on mismatch.

: "${PYTHON:=python}"

EVIDENCE_ROOT="${EVIDENCE_ROOT:-}"
PLAYLIST="${PLAYLIST:-}"

if [[ -n "${EVIDENCE_ROOT}" ]]; then
  EVIDENCE_JSON="${EVIDENCE_ROOT%/}/drop_evidence.json"
  if [[ ! -f "${EVIDENCE_JSON}" ]]; then
    echo "[ci_web_bundle_determinism] FAIL: missing evidence json: ${EVIDENCE_JSON}" >&2
    exit 2
  fi

  # Prefer playlist_path from evidence; fallback to evidence_root/playlist.json
  PLAYLIST="$(
    "${PYTHON}" - <<'PY'
import json, os
p = os.environ["EVIDENCE_JSON"]
obj = json.load(open(p, "r", encoding="utf-8"))
paths = obj.get("paths") if isinstance(obj.get("paths"), dict) else {}
v = paths.get("playlist_path") or ""
print(v)
PY
  )"
  if [[ -z "${PLAYLIST}" ]]; then
    PLAYLIST="${EVIDENCE_ROOT%/}/playlist.json"
  fi
fi

if [[ -z "${PLAYLIST}" ]]; then
  echo "scripts/ci_web_bundle_determinism.sh: line 18: PLAYLIST: set PLAYLIST to a playlist JSON path" >&2
  echo "  or set EVIDENCE_ROOT to a run output dir containing drop_evidence.json" >&2
  exit 2
fi

PLAYLIST="$(cd "$(dirname "${PLAYLIST}")" && pwd)/$(basename "${PLAYLIST}")"
echo "[ci_web_bundle_determinism] playlist=${PLAYLIST}" >&2

# Build twice into temp dirs, hash trees, compare.
hash_tree() {
  local dir="$1"
  "${PYTHON}" - <<'PY'
import hashlib, os, pathlib, sys

root = pathlib.Path(sys.argv[1]).resolve()

def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = sorted([p for p in root.rglob("*") if p.is_file()],
               key=lambda p: p.relative_to(root).as_posix())

lines = []
for p in files:
    rel = p.relative_to(root).as_posix()
    lines.append(f"{sha256_file(p)}  {rel}")
joined = ("\n".join(lines) + "\n").encode("utf-8")
print(hashlib.sha256(joined).hexdigest())
PY "$dir"
}

build_once() {
  local out_dir="$1"
  "${PYTHON}" -m mgc.main web build \
    --playlist "${PLAYLIST}" \
    --out-dir "${out_dir}" \
    --clean \
    --fail-if-empty \
    --fail-if-none-copied \
    --json >/dev/null
}

d1="$(mktemp -d -t mgc_web_det_1.XXXXXX)"
d2="$(mktemp -d -t mgc_web_det_2.XXXXXX)"
trap 'rm -rf "${d1}" "${d2}"' EXIT

build_once "${d1}"
build_once "${d2}"

h1="$(hash_tree "${d1}")"
h2="$(hash_tree "${d2}")"

echo "[ci_web_bundle_determinism] run1_tree_sha256=${h1}" >&2
echo "[ci_web_bundle_determinism] run2_tree_sha256=${h2}" >&2

if [[ "${h1}" != "${h2}" ]]; then
  echo "[ci_web_bundle_determinism] FAIL: web bundle tree sha mismatch" >&2
  exit 2
fi

echo "[ci_web_bundle_determinism] OK" >&2
