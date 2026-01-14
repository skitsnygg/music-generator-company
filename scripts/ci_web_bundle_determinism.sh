#!/usr/bin/env bash
# scripts/ci_web_bundle_determinism.sh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/ci_web_bundle_determinism.sh --evidence-root <dir>

Reads <evidence-root>/drop_evidence.json, finds:
  paths.web_bundle_dir
  artifacts.web_bundle_tree_sha256

Then re-computes a deterministic tree hash and compares.

Exits:
  0 OK (or web not present)
  2 mismatch / missing web_dir when hash present
EOF
}

EVIDENCE_ROOT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --evidence-root) EVIDENCE_ROOT="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ci_web_bundle_determinism] unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [ -z "$EVIDENCE_ROOT" ]; then
  echo "[ci_web_bundle_determinism] missing --evidence-root" >&2
  usage
  exit 2
fi

evidence_path="${EVIDENCE_ROOT%/}/drop_evidence.json"
if [ ! -f "$evidence_path" ]; then
  echo "[ci_web_bundle_determinism] missing evidence: $evidence_path" >&2
  exit 2
fi

python - <<PY
import json, hashlib, pathlib, sys

evidence = pathlib.Path("$evidence_path")
obj = json.loads(evidence.read_text(encoding="utf-8"))
paths = obj.get("paths") if isinstance(obj.get("paths"), dict) else {}
arts  = obj.get("artifacts") if isinstance(obj.get("artifacts"), dict) else {}

web_dir = paths.get("web_bundle_dir")
want = arts.get("web_bundle_tree_sha256")

# If no web hash was recorded, treat as "web not included" and succeed.
if not want:
    print("[ci_web_bundle_determinism] SKIP (no artifacts.web_bundle_tree_sha256 recorded)")
    raise SystemExit(0)

if not web_dir:
    print("[ci_web_bundle_determinism] FAIL: hash recorded but paths.web_bundle_dir missing", file=sys.stderr)
    raise SystemExit(2)

root = pathlib.Path(str(web_dir))
if not root.exists() or not root.is_dir():
    print(f"[ci_web_bundle_determinism] FAIL: web_bundle_dir missing: {root}", file=sys.stderr)
    raise SystemExit(2)

def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = sorted([p for p in root.rglob("*") if p.is_file()], key=lambda p: p.relative_to(root).as_posix())
lines = []
for p in files:
    rel = p.relative_to(root).as_posix()
    lines.append(f"{sha256_file(p)}  {rel}")
joined = ("\n".join(lines) + "\n").encode("utf-8")
got = hashlib.sha256(joined).hexdigest()

if got != want:
    print("[ci_web_bundle_determinism] FAIL tree sha mismatch", file=sys.stderr)
    print(f"[ci_web_bundle_determinism] expected={want}", file=sys.stderr)
    print(f"[ci_web_bundle_determinism] got     ={got}", file=sys.stderr)
    raise SystemExit(2)

print(f"[ci_web_bundle_determinism] OK tree_sha256={got}")
PY
