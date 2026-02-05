#!/usr/bin/env bash
set -euo pipefail

echo "[demo_smoke] starting demo smoke"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MGC_DEMO_NO_SUDO_DEFAULT="0"
if [[ "$(uname -s)" == "Darwin" ]]; then
  MGC_DEMO_NO_SUDO_DEFAULT="1"
fi
MGC_DEMO_NO_SUDO="${MGC_DEMO_NO_SUDO:-${MGC_DEMO_NO_SUDO_DEFAULT}}"
MGC_DEMO_CLEAN="${MGC_DEMO_CLEAN:-0}"
MGC_DEMO_FAST="${MGC_DEMO_FAST:-0}"
MGC_SETUP_NGINX="${MGC_SETUP_NGINX:-1}"
MGC_PYTHON="${MGC_PYTHON:-}"

if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  if [[ "${MGC_DEMO_CLEAN}" == "1" ]]; then
    echo "[demo_smoke] cleaning local demo outputs..."
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
  echo "[demo_smoke] WARN: MGC_DEMO_CLEAN only cleans local demo outputs (set MGC_DEMO_NO_SUDO=1)" >&2
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
  echo "[demo_smoke] ERROR: python not found (set MGC_PYTHON)" >&2
  exit 2
fi
export PYTHON="${MGC_PYTHON}"

if [[ -z "${MGC_PROVIDER:-}" ]]; then
  export MGC_PROVIDER="stub"
fi

DEMO_FALLBACK="${MGC_DEMO_FALLBACK_TO_STUB:-${MGC_FALLBACK_TO_STUB:-0}}"

if [[ "${MGC_PROVIDER}" == "riffusion" ]]; then
  if [[ -z "${MGC_RIFFUSION_URL:-}" ]]; then
    export MGC_RIFFUSION_URL="http://127.0.0.1:3013/run_inference"
    echo "[demo_smoke] WARN: MGC_RIFFUSION_URL not set; defaulting to ${MGC_RIFFUSION_URL}"
  fi

  if command -v curl >/dev/null 2>&1; then
    base="${MGC_RIFFUSION_URL%/run_inference*}"
    if [[ -z "${base}" ]]; then
      base="${MGC_RIFFUSION_URL}"
    fi
    code="$(curl -sS -o /dev/null -w "%{http_code}" --connect-timeout 2 --max-time 5 "${base}" || true)"
    if [[ "${code}" == "000" ]]; then
      if [[ "${DEMO_FALLBACK}" == "1" ]]; then
        echo "[demo_smoke] WARN: riffusion not reachable at ${base}; falling back to stub"
        export MGC_PROVIDER="stub"
        export MGC_PROVIDER_FALLBACK_FROM="riffusion"
      else
        echo "[demo_smoke] ERROR: riffusion not reachable at ${base} (set MGC_DEMO_FALLBACK_TO_STUB=1 to continue)" >&2
        exit 2
      fi
    else
      echo "[demo_smoke] riffusion reachability: ${base} (http ${code})"
    fi
  else
    echo "[demo_smoke] WARN: curl not found; skipping riffusion reachability check"
  fi
fi

export MGC_CONTEXTS="${MGC_CONTEXTS:-focus}"
export MGC_MARKETING="${MGC_MARKETING:-0}"
export MGC_PUBLISH_MARKETING="${MGC_PUBLISH_MARKETING:-0}"
export MGC_PUBLISH_LATEST="${MGC_PUBLISH_LATEST:-1}"
export MGC_PUBLISH_FEED="${MGC_PUBLISH_FEED:-1}"

OUT_BASE="${MGC_OUT_BASE:-data/evidence}"
WEB_ROOT="${MGC_WEB_LATEST_ROOT:-data/web/latest}"
CONTEXTS=(${MGC_CONTEXTS:-focus})

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
    echo "[demo_smoke] fast mode: reusing existing outputs"
  else
    echo "[demo_smoke] fast mode: outputs missing; running daily pipeline"
  fi
fi

RUN_DAILY_CMD=(sudo -E scripts/run_daily.sh)
if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  RUN_DAILY_CMD=(scripts/run_daily.sh)
fi

if [[ "${MGC_DEMO_FAST}" == "1" && "${FAST_READY}" == "1" ]]; then
  echo "[demo_smoke] skipping daily pipeline (MGC_DEMO_FAST=1)"
else
  echo "[demo_smoke] running daily pipeline..."
  "${RUN_DAILY_CMD[@]}"
fi

echo "[demo_smoke] verifying playlist track files..."
for ctx in "${CONTEXTS[@]}"; do
  PLAYLIST="${OUT_BASE}/${ctx}/drop_bundle/playlist.json"
  test -s "$PLAYLIST"
  "${MGC_PYTHON}" scripts/check_playlist_tracks.py "$PLAYLIST" "${OUT_BASE}/${ctx}"
done

if [[ "${PUBLISH_LATEST}" == "1" ]]; then
  echo "[demo_smoke] verifying latest web bundles..."
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
  echo "[demo_smoke] skipping web bundle checks (MGC_PUBLISH_LATEST=0)"
fi

if [[ "${PUBLISH_FEED}" == "1" ]]; then
  echo "[demo_smoke] checking feed on disk..."
  test -s "$FEED_PATH"
  "${MGC_PYTHON}" -m json.tool "$FEED_PATH" >/dev/null
  echo "[demo_smoke] feed json ok"
  echo "[demo_smoke] verifying feed contexts..."
  CONTEXTS_STR="${CONTEXTS[*]}"
  FEED_PATH="${FEED_PATH}" CONTEXTS="${CONTEXTS_STR}" "${MGC_PYTHON}" - <<'PY'
import json
import os

p = os.environ["FEED_PATH"]
want = [c for c in os.environ.get("CONTEXTS", "").split() if c]
o = json.load(open(p, "r", encoding="utf-8"))
names = [c["context"] for c in o.get("latest", {}).get("contexts", []) if isinstance(c, dict)]
print("[demo_smoke] latest contexts:", names)
missing = [c for c in want if c not in names]
if missing:
    raise SystemExit(f"feed missing contexts: {missing}")
print("[demo_smoke] feed contexts ok")
PY
else
  echo "[demo_smoke] skipping feed checks (MGC_PUBLISH_FEED=0)"
fi

setup_nginx() {
  if [[ "${MGC_SETUP_NGINX}" != "1" ]]; then
    return 1
  fi
  if [[ ! -x "${REPO_ROOT}/scripts/setup_nginx.sh" ]]; then
    echo "[demo_smoke] WARN: scripts/setup_nginx.sh not found or not executable" >&2
    return 1
  fi
  if [[ "${EUID}" -eq 0 ]]; then
    "${REPO_ROOT}/scripts/setup_nginx.sh"
  else
    sudo -E "${REPO_ROOT}/scripts/setup_nginx.sh"
  fi
}

if [[ "${PUBLISH_FEED}" != "1" ]]; then
  echo "[demo_smoke] skipping nginx check (MGC_PUBLISH_FEED=0)"
elif [[ "${SKIP_NGINX}" == "1" ]]; then
  echo "[demo_smoke] skipping nginx check (MGC_SKIP_NGINX=1)"
elif command -v curl >/dev/null 2>&1; then
  echo "[demo_smoke] fetching feed via nginx..."
  if curl -fsS "$FEED_URL" | "${MGC_PYTHON}" -m json.tool >/dev/null; then
    echo "[demo_smoke] nginx serving feed ok"
  else
    if [[ "${MGC_SETUP_NGINX}" == "1" ]]; then
      echo "[demo_smoke] nginx check failed; attempting setup_nginx.sh..."
      if setup_nginx && curl -fsS "$FEED_URL" | "${MGC_PYTHON}" -m json.tool >/dev/null; then
        echo "[demo_smoke] nginx serving feed ok (after setup)"
      else
        if [[ "${REQUIRE_NGINX}" == "1" ]]; then
          echo "[demo_smoke] nginx check failed (set MGC_SKIP_NGINX=1 to skip)" >&2
          exit 2
        fi
        echo "[demo_smoke] WARN: nginx check failed (continuing)"
      fi
    else
      if [[ "${REQUIRE_NGINX}" == "1" ]]; then
        echo "[demo_smoke] nginx check failed (set MGC_SKIP_NGINX=1 to skip)" >&2
        exit 2
      fi
      echo "[demo_smoke] WARN: nginx check failed (continuing)"
    fi
  fi
else
  echo "[demo_smoke] WARN: curl not found; skipping nginx check"
fi

echo "[demo_smoke] OK"
