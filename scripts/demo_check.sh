#!/usr/bin/env bash
set -euo pipefail

echo "[demo_check] starting full demo verification"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MGC_DEMO_NO_SUDO="${MGC_DEMO_NO_SUDO:-0}"
MGC_DEMO_CLEAN="${MGC_DEMO_CLEAN:-0}"
MGC_DEMO_FAST="${MGC_DEMO_FAST:-0}"
MGC_SETUP_NGINX="${MGC_SETUP_NGINX:-1}"
MGC_PYTHON="${MGC_PYTHON:-}"
if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  if [[ "${MGC_DEMO_CLEAN}" == "1" ]]; then
    echo "[demo_check] cleaning local demo outputs..."
    rm -rf "${REPO_ROOT}/data/local_demo_evidence" "${REPO_ROOT}/data/releases" "${REPO_ROOT}/.tmp_publish"
  fi
  export MGC_OUT_BASE="${MGC_OUT_BASE:-${REPO_ROOT}/data/local_demo_evidence}"
  export MGC_DB="${MGC_DB:-${REPO_ROOT}/data/local_demo_db.sqlite}"
  if [[ ! -f "${MGC_DB}" && -f "${REPO_ROOT}/data/db.sqlite" ]]; then
    cp -f "${REPO_ROOT}/data/db.sqlite" "${MGC_DB}"
  fi
  export MGC_WEB_LATEST_ROOT="${MGC_WEB_LATEST_ROOT:-${REPO_ROOT}/data/releases/latest/web}"
  export MGC_RELEASE_ROOT="${MGC_RELEASE_ROOT:-${REPO_ROOT}/data/releases}"
  export MGC_RELEASE_FEED_OUT="${MGC_RELEASE_FEED_OUT:-${REPO_ROOT}/data/releases/feed.json}"
  export MGC_SKIP_NGINX="${MGC_SKIP_NGINX:-1}"
elif [[ "${MGC_DEMO_CLEAN}" == "1" ]]; then
  echo "[demo_check] WARN: MGC_DEMO_CLEAN only cleans local demo outputs (set MGC_DEMO_NO_SUDO=1)" >&2
fi

if [[ -z "${MGC_RELEASE_ROOT:-}" ]]; then
  export MGC_RELEASE_ROOT="/var/lib/mgc/releases"
fi
if [[ -z "${MGC_WEB_LATEST_ROOT:-}" ]]; then
  export MGC_WEB_LATEST_ROOT="${MGC_RELEASE_ROOT}/latest/web"
fi
if [[ -z "${MGC_RELEASE_FEED_OUT:-}" ]]; then
  export MGC_RELEASE_FEED_OUT="${MGC_RELEASE_ROOT}/feed.json"
fi

if [[ "${EUID}" -eq 0 && "${MGC_DEMO_NO_SUDO}" != "1" ]]; then
  if [[ -z "${MGC_OUT_BASE:-}" ]]; then
    export MGC_OUT_BASE="/var/lib/mgc/evidence"
  fi
  export MGC_SKIP_OWNERSHIP_CHECK="${MGC_SKIP_OWNERSHIP_CHECK:-1}"
fi

FEED_PATH="${MGC_FEED_PATH:-${MGC_RELEASE_FEED_OUT:-/var/lib/mgc/releases/feed.json}}"
FEED_URL="${MGC_FEED_URL:-http://127.0.0.1/releases/feed.json}"
SKIP_NGINX="${MGC_SKIP_NGINX:-0}"
REQUIRE_NGINX="${MGC_REQUIRE_NGINX:-1}"
PUBLISH_FEED="${MGC_PUBLISH_FEED:-1}"
PUBLISH_LATEST="${MGC_PUBLISH_LATEST:-1}"
DEMO_VALIDATE_AUDIO="${MGC_DEMO_VALIDATE_AUDIO:-1}"
DEMO_VALIDATE_WEB="${MGC_DEMO_VALIDATE_WEB:-1}"

cd "$REPO_ROOT"

if [[ -z "${MGC_PYTHON}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    MGC_PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    MGC_PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi

if [[ -z "${MGC_PYTHON}" ]]; then
  echo "[demo_check] ERROR: python not found (set MGC_PYTHON)" >&2
  exit 2
fi
export PYTHON="${MGC_PYTHON}"

echo "[demo_check] repo: $REPO_ROOT"

if [[ -z "${MGC_PROVIDER:-}" ]]; then
  export MGC_PROVIDER="riffusion"
fi

DEMO_FALLBACK="${MGC_DEMO_FALLBACK_TO_STUB:-${MGC_FALLBACK_TO_STUB:-0}}"
if [[ "${DEMO_FALLBACK}" == "1" ]]; then
  export MGC_FALLBACK_TO_STUB=1
fi

if [[ "${MGC_PROVIDER}" == "riffusion" ]]; then
  if [[ -z "${MGC_RIFFUSION_URL:-}" ]]; then
    export MGC_RIFFUSION_URL="http://127.0.0.1:3013/run_inference"
    echo "[demo_check] WARN: MGC_RIFFUSION_URL not set; defaulting to ${MGC_RIFFUSION_URL}"
  fi

  if command -v curl >/dev/null 2>&1; then
    base="${MGC_RIFFUSION_URL%/run_inference*}"
    if [[ -z "${base}" ]]; then
      base="${MGC_RIFFUSION_URL}"
    fi
    code="$(curl -sS -o /dev/null -w "%{http_code}" --connect-timeout 2 --max-time 5 "${base}" || true)"
    if [[ "${code}" == "000" ]]; then
      if [[ "${DEMO_FALLBACK}" == "1" ]]; then
        echo "[demo_check] WARN: riffusion not reachable at ${base}; falling back to stub"
        export MGC_PROVIDER="stub"
        export MGC_PROVIDER_FALLBACK_FROM="riffusion"
      else
        echo "[demo_check] ERROR: riffusion not reachable at ${base} (set MGC_DEMO_FALLBACK_TO_STUB=1 to continue)" >&2
        exit 2
      fi
    else
      echo "[demo_check] riffusion reachability: ${base} (http ${code})"
    fi
  else
    echo "[demo_check] WARN: curl not found; skipping riffusion reachability check"
  fi
fi

OUT_BASE="${MGC_OUT_BASE:-data/evidence}"
WEB_ROOT="${MGC_WEB_LATEST_ROOT:-data/web/latest}"
CONTEXTS=(${MGC_CONTEXTS:-focus workout sleep})

FAST_READY="0"
if [[ "${MGC_DEMO_FAST}" == "1" ]]; then
  FAST_READY="1"
  for ctx in "${CONTEXTS[@]}"; do
    if [[ ! -s "${OUT_BASE}/${ctx}/drop_bundle/playlist.json" ]]; then
      FAST_READY="0"
      break
    fi
    if [[ "${PUBLISH_LATEST}" == "1" ]] && [[ ! -s "${WEB_ROOT}/${ctx}/web_manifest.json" ]]; then
      FAST_READY="0"
      break
    fi
  done
  if [[ "${PUBLISH_FEED}" == "1" ]] && [[ ! -s "${FEED_PATH}" ]]; then
    FAST_READY="0"
  fi
  if [[ "${FAST_READY}" == "1" ]]; then
    echo "[demo_check] fast mode: reusing existing outputs"
  else
    echo "[demo_check] fast mode: outputs missing; running daily pipeline"
  fi
fi

# 1) Run daily pipeline (this regenerates latest + feed)
RUN_DAILY_CMD=(sudo -E scripts/run_daily.sh)
if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  RUN_DAILY_CMD=(scripts/run_daily.sh)
elif [[ "${EUID}" -eq 0 ]]; then
  RUN_DAILY_CMD=(scripts/run_daily.sh)
fi
if [[ "${MGC_DEMO_FAST}" == "1" && "${FAST_READY}" == "1" ]]; then
  echo "[demo_check] skipping daily pipeline (MGC_DEMO_FAST=1)"
else
  echo "[demo_check] running daily pipeline..."
  "${RUN_DAILY_CMD[@]}"
fi

# 2) Verify playlist track files exist
echo "[demo_check] verifying playlist track files..."
for ctx in "${CONTEXTS[@]}"; do
  PLAYLIST="${OUT_BASE}/${ctx}/drop_bundle/playlist.json"
  test -s "$PLAYLIST"
  "${MGC_PYTHON}" scripts/check_playlist_tracks.py "$PLAYLIST" "${OUT_BASE}/${ctx}"
done

# 3) Verify latest web bundle exists + audio files
if [[ "${PUBLISH_LATEST}" == "1" ]]; then
  echo "[demo_check] verifying latest web bundles..."
  for ctx in "${CONTEXTS[@]}"; do
    WEB_DIR="${WEB_ROOT}/${ctx}"
    test -s "${WEB_DIR}/web_manifest.json"
    if [[ "${DEMO_VALIDATE_AUDIO}" == "1" ]]; then
      AUDIO_COUNT="$(find "${WEB_DIR}" -maxdepth 8 -type f \( -name '*.mp3' -o -name '*.wav' \) | wc -l | tr -d ' ')"
      test "${AUDIO_COUNT}" != "0"
    fi
    if [[ "${DEMO_VALIDATE_WEB}" == "1" ]]; then
      "${MGC_PYTHON}" -m mgc.main web validate --out-dir "${WEB_DIR}"
    fi
  done
else
  echo "[demo_check] skipping web bundle checks (MGC_PUBLISH_LATEST=0)"
fi

# 4) Verify feed exists and is valid JSON (when enabled)
if [[ "${PUBLISH_FEED}" == "1" ]]; then
  echo "[demo_check] checking feed on disk..."
  test -s "$FEED_PATH"
  ls -la "$FEED_PATH"

  echo "[demo_check] validating feed JSON..."
  "${MGC_PYTHON}" -m json.tool "$FEED_PATH" >/dev/null
  echo "[demo_check] feed json ok"
else
  echo "[demo_check] skipping feed checks (MGC_PUBLISH_FEED=0)"
fi

setup_nginx() {
  if [[ "${MGC_SETUP_NGINX}" != "1" ]]; then
    return 1
  fi
  if [[ ! -x "${REPO_ROOT}/scripts/setup_nginx.sh" ]]; then
    echo "[demo_check] WARN: scripts/setup_nginx.sh not found or not executable" >&2
    return 1
  fi
  if [[ "${EUID}" -eq 0 ]]; then
    "${REPO_ROOT}/scripts/setup_nginx.sh"
  else
    sudo -E "${REPO_ROOT}/scripts/setup_nginx.sh"
  fi
}

# 5) Verify nginx serves the feed
if [[ "${PUBLISH_FEED}" == "1" ]]; then
  if [[ "${SKIP_NGINX}" == "1" ]]; then
    echo "[demo_check] skipping nginx check (MGC_SKIP_NGINX=1)"
  else
    echo "[demo_check] fetching feed via nginx..."
    if curl -fsS "$FEED_URL" | "${MGC_PYTHON}" -m json.tool >/dev/null; then
      echo "[demo_check] nginx serving feed ok"
    else
      if [[ "${MGC_SETUP_NGINX}" == "1" ]]; then
        echo "[demo_check] nginx check failed; attempting setup_nginx.sh..."
        if setup_nginx && curl -fsS "$FEED_URL" | "${MGC_PYTHON}" -m json.tool >/dev/null; then
          echo "[demo_check] nginx serving feed ok (after setup)"
        else
          if [[ "${REQUIRE_NGINX}" == "1" ]]; then
            echo "[demo_check] nginx check failed (set MGC_SKIP_NGINX=1 to skip)" >&2
            exit 2
          fi
          echo "[demo_check] WARN: nginx check failed (continuing)"
        fi
      else
        if [[ "${REQUIRE_NGINX}" == "1" ]]; then
          echo "[demo_check] nginx check failed (set MGC_SKIP_NGINX=1 to skip)" >&2
          exit 2
        fi
        echo "[demo_check] WARN: nginx check failed (continuing)"
      fi
    fi
  fi
fi

if [[ "${PUBLISH_FEED}" == "1" ]]; then
  # 6) Verify feed contexts and filtering (no .bak, no run)
  echo "[demo_check] verifying feed contexts..."
  CONTEXTS_STR="${CONTEXTS[*]}"
  FEED_PATH="${FEED_PATH}" CONTEXTS="${CONTEXTS_STR}" "${MGC_PYTHON}" - <<'PY'
import json
import os

p = os.environ["FEED_PATH"]
want = [c for c in os.environ.get("CONTEXTS", "").split() if c]
o = json.load(open(p, "r", encoding="utf-8"))
names = [c["context"] for c in o.get("latest", {}).get("contexts", []) if isinstance(c, dict)]
print("[demo_check] latest contexts:", names)
missing = [c for c in want if c not in names]
if missing:
    raise SystemExit(f"feed missing contexts: {missing}")
assert all(".bak." not in n for n in names), "backup contexts present"
assert "run" not in names, "run context present"
print("[demo_check] feed contexts ok")
PY

  # 7) Determinism proof: regenerate feed and compare content hash
  echo "[demo_check] verifying content determinism..."
  content_hash() {
    FEED_PATH="${FEED_PATH}" "${MGC_PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

p = Path(os.environ["FEED_PATH"])
o = json.loads(p.read_text(encoding="utf-8"))
o.pop("generated_at", None)
o.pop("content_sha256", None)
canon = json.dumps(o, sort_keys=True, separators=(",", ":")).encode("utf-8")
print(hashlib.sha256(canon).hexdigest())
PY
  }

  h1="$(content_hash)"

  PUBLISH_FEED_CMD=(sudo -E scripts/publish_release_feed.sh)
  if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
    PUBLISH_FEED_CMD=(scripts/publish_release_feed.sh)
  fi
  for c in "${CONTEXTS[@]}"; do
    "${PUBLISH_FEED_CMD[@]}" --context "${c}"
  done

  h2="$(content_hash)"
  echo "[demo_check] content_sha256_1: ${h1}"
  echo "[demo_check] content_sha256_2: ${h2}"
  [[ "${h1}" == "${h2}" ]] || { echo "[demo_check] content hash changed" >&2; exit 2; }
  echo "[demo_check] determinism ok"
else
  echo "[demo_check] skipping context/determinism checks (MGC_PUBLISH_FEED=0)"
fi

echo "[demo_check] ALL CHECKS PASSED"
