#!/usr/bin/env bash
set -euo pipefail

echo "[demo_smoke] starting demo smoke"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MGC_DEMO_NO_SUDO="${MGC_DEMO_NO_SUDO:-0}"
MGC_DEMO_CLEAN="${MGC_DEMO_CLEAN:-0}"
MGC_SETUP_NGINX="${MGC_SETUP_NGINX:-1}"

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

FEED_PATH="${MGC_FEED_PATH:-${MGC_RELEASE_FEED_OUT:-/var/lib/mgc/releases/feed.json}}"
FEED_URL="${MGC_FEED_URL:-http://127.0.0.1/releases/feed.json}"
SKIP_NGINX="${MGC_SKIP_NGINX:-0}"
REQUIRE_NGINX="${MGC_REQUIRE_NGINX:-1}"
PUBLISH_FEED="${MGC_PUBLISH_FEED:-1}"
PUBLISH_LATEST="${MGC_PUBLISH_LATEST:-1}"
DEMO_VALIDATE_AUDIO="${MGC_DEMO_VALIDATE_AUDIO:-1}"

cd "$REPO_ROOT"

if [[ -z "${MGC_PROVIDER:-}" ]]; then
  export MGC_PROVIDER="stub"
fi

export MGC_CONTEXTS="${MGC_CONTEXTS:-focus}"
export MGC_MARKETING="${MGC_MARKETING:-0}"
export MGC_PUBLISH_MARKETING="${MGC_PUBLISH_MARKETING:-0}"
export MGC_PUBLISH_LATEST="${MGC_PUBLISH_LATEST:-1}"
export MGC_PUBLISH_FEED="${MGC_PUBLISH_FEED:-1}"

RUN_DAILY_CMD=(sudo -E scripts/run_daily.sh)
if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  RUN_DAILY_CMD=(scripts/run_daily.sh)
fi

echo "[demo_smoke] running daily pipeline..."
"${RUN_DAILY_CMD[@]}"

echo "[demo_smoke] verifying playlist track files..."
OUT_BASE="${MGC_OUT_BASE:-data/evidence}"
CONTEXTS=(${MGC_CONTEXTS:-focus})
for ctx in "${CONTEXTS[@]}"; do
  PLAYLIST="${OUT_BASE}/${ctx}/drop_bundle/playlist.json"
  test -s "$PLAYLIST"
  python3 scripts/check_playlist_tracks.py "$PLAYLIST" "${OUT_BASE}/${ctx}"
done

if [[ "${PUBLISH_LATEST}" == "1" ]]; then
  echo "[demo_smoke] verifying latest web bundles..."
  WEB_ROOT="${MGC_WEB_LATEST_ROOT:-data/web/latest}"
  for ctx in "${CONTEXTS[@]}"; do
    WEB_DIR="${WEB_ROOT}/${ctx}"
    test -s "${WEB_DIR}/web_manifest.json"
    if [[ "${DEMO_VALIDATE_AUDIO}" == "1" ]]; then
      AUDIO_COUNT="$(find "${WEB_DIR}" -maxdepth 8 -type f \( -name '*.mp3' -o -name '*.wav' \) | wc -l | tr -d ' ')"
      test "${AUDIO_COUNT}" != "0"
    fi
  done
else
  echo "[demo_smoke] skipping web bundle checks (MGC_PUBLISH_LATEST=0)"
fi

if [[ "${PUBLISH_FEED}" == "1" ]]; then
  echo "[demo_smoke] checking feed on disk..."
  test -s "$FEED_PATH"
  python3 -m json.tool "$FEED_PATH" >/dev/null
  echo "[demo_smoke] feed json ok"
else
  echo "[demo_smoke] skipping feed checks (MGC_PUBLISH_FEED=0)"
fi

WEB_ROOT="${MGC_WEB_LATEST_ROOT:-data/web/latest}"
for ctx in ${MGC_CONTEXTS}; do
  test -s "${WEB_ROOT}/${ctx}/web_manifest.json"
done

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
  if curl -fsS "$FEED_URL" | python3 -m json.tool >/dev/null; then
    echo "[demo_smoke] nginx serving feed ok"
  else
    if [[ "${MGC_SETUP_NGINX}" == "1" ]]; then
      echo "[demo_smoke] nginx check failed; attempting setup_nginx.sh..."
      if setup_nginx && curl -fsS "$FEED_URL" | python3 -m json.tool >/dev/null; then
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
