#!/usr/bin/env bash
set -euo pipefail

# CI smoke test: if riffusion is unreachable and fallback is enabled,
# we should auto-switch to stub and record provider_fallback_from.

: "${MGC_DB:?set MGC_DB}"

PYTHON="${PYTHON:-python}"

tmp="${TMPDIR:-/tmp}/mgc_fallback_smoke"
rm -rf "$tmp"
mkdir -p "$tmp"

export MGC_DETERMINISTIC=1
export MGC_FIXED_TIME="${MGC_FIXED_TIME:-2020-01-01T00:00:00Z}"
export MGC_FALLBACK_TO_STUB=1

# Force riffusion selection but make it unreachable + fast-fail.
export MGC_PROVIDER=riffusion
export MGC_RIFFUSION_URL="${MGC_RIFFUSION_URL:-http://127.0.0.1:9/run_inference}"
export MGC_RIFFUSION_PREFLIGHT_TIMEOUT="${MGC_RIFFUSION_PREFLIGHT_TIMEOUT:-0.25}"
export MGC_RIFFUSION_CONNECT_TIMEOUT="${MGC_RIFFUSION_CONNECT_TIMEOUT:-1}"
export MGC_RIFFUSION_READ_TIMEOUT="${MGC_RIFFUSION_READ_TIMEOUT:-1}"
export MGC_RIFFUSION_RETRIES="${MGC_RIFFUSION_RETRIES:-0}"

export MGC_FALLBACK_TMP="$tmp"

"$PYTHON" -m mgc.main --db "$MGC_DB" --json run drop \
  --context focus \
  --seed 1 \
  --deterministic \
  --provider riffusion \
  --out-dir "$tmp" \
  > "$tmp/drop.json"

"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MGC_FALLBACK_TMP"])
ev = root / "daily_evidence.json"
if not ev.exists():
    alt = root / "drop_bundle" / "daily_evidence.json"
    if alt.exists():
        ev = alt
    else:
        raise SystemExit(f"missing daily_evidence.json under {root}")

obj = json.loads(ev.read_text(encoding="utf-8"))
prov = obj.get("provider")
fallback = obj.get("provider_fallback_from")

if prov != "stub":
    raise SystemExit(f"expected provider=stub, got {prov!r}")
if fallback != "riffusion":
    raise SystemExit(f"expected provider_fallback_from=riffusion, got {fallback!r}")

print("[ci_provider_fallback] ok", ev)
PY

gen_root="$tmp/generate"
gen_out="$gen_root/out"
gen_store="$gen_root/store"
gen_db="$gen_root/generate_db.sqlite"
mkdir -p "$gen_out" "$gen_store"

"$PYTHON" -m mgc.main --db "$gen_db" --json run generate \
  --context focus \
  --seed 1 \
  --deterministic \
  --provider riffusion \
  --repo-root "$gen_root" \
  --store-dir "$gen_store" \
  --out-dir "$gen_out" \
  > "$gen_root/generate.json"

"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MGC_FALLBACK_TMP"])
ev = root / "generate" / "out" / "evidence" / "generate_evidence.json"
if not ev.exists():
    raise SystemExit(f"missing generate_evidence.json at {ev}")

obj = json.loads(ev.read_text(encoding="utf-8"))
meta_path = Path(obj.get("meta_path") or "")
if not meta_path.exists():
    raise SystemExit(f"missing meta_path {meta_path}")

meta = json.loads(meta_path.read_text(encoding="utf-8"))
prov = meta.get("provider")
fallback = (meta.get("meta") or {}).get("provider_fallback_from")

if prov != "stub":
    raise SystemExit(f"expected provider=stub, got {prov!r}")
if fallback != "riffusion":
    raise SystemExit(f"expected provider_fallback_from=riffusion, got {fallback!r}")

print("[ci_provider_fallback_generate] ok", meta_path)
PY
