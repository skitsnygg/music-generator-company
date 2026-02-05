#!/usr/bin/env bash
set -euo pipefail

echo "[demo_check] starting full demo verification"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MGC_DEMO_NO_SUDO="${MGC_DEMO_NO_SUDO:-0}"
MGC_DEMO_CLEAN="${MGC_DEMO_CLEAN:-0}"
MGC_SETUP_NGINX="${MGC_SETUP_NGINX:-1}"
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

FEED_PATH="${MGC_FEED_PATH:-${MGC_RELEASE_FEED_OUT:-/var/lib/mgc/releases/feed.json}}"
FEED_URL="${MGC_FEED_URL:-http://127.0.0.1/releases/feed.json}"
SKIP_NGINX="${MGC_SKIP_NGINX:-0}"
REQUIRE_NGINX="${MGC_REQUIRE_NGINX:-1}"
PUBLISH_FEED="${MGC_PUBLISH_FEED:-1}"
PUBLISH_LATEST="${MGC_PUBLISH_LATEST:-1}"
DEMO_VALIDATE_AUDIO="${MGC_DEMO_VALIDATE_AUDIO:-1}"

cd "$REPO_ROOT"

echo "[demo_check] repo: $REPO_ROOT"

# 1) Run daily pipeline (this regenerates latest + feed)
echo "[demo_check] running daily pipeline..."
RUN_DAILY_CMD=(sudo -E scripts/run_daily.sh)
if [[ "${MGC_DEMO_NO_SUDO}" == "1" ]]; then
  RUN_DAILY_CMD=(scripts/run_daily.sh)
fi
"${RUN_DAILY_CMD[@]}"

# 2) Verify playlist track files exist
echo "[demo_check] verifying playlist track files..."
OUT_BASE="${MGC_OUT_BASE:-data/evidence}"
CONTEXTS=(${MGC_CONTEXTS:-focus workout sleep})
for ctx in "${CONTEXTS[@]}"; do
  PLAYLIST="${OUT_BASE}/${ctx}/drop_bundle/playlist.json"
  test -s "$PLAYLIST"
  python3 scripts/check_playlist_tracks.py "$PLAYLIST" "${OUT_BASE}/${ctx}"
done

# 3) Verify latest web bundle exists + audio files
if [[ "${PUBLISH_LATEST}" == "1" ]]; then
  echo "[demo_check] verifying latest web bundles..."
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
  echo "[demo_check] skipping web bundle checks (MGC_PUBLISH_LATEST=0)"
fi

# 4) Verify feed exists and is valid JSON (when enabled)
if [[ "${PUBLISH_FEED}" == "1" ]]; then
  echo "[demo_check] checking feed on disk..."
  test -s "$FEED_PATH"
  ls -la "$FEED_PATH"

  echo "[demo_check] validating feed JSON..."
  python3 -m json.tool "$FEED_PATH" >/dev/null
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
    if curl -fsS "$FEED_URL" | python3 -m json.tool >/dev/null; then
      echo "[demo_check] nginx serving feed ok"
    else
      if [[ "${MGC_SETUP_NGINX}" == "1" ]]; then
        echo "[demo_check] nginx check failed; attempting setup_nginx.sh..."
        if setup_nginx && curl -fsS "$FEED_URL" | python3 -m json.tool >/dev/null; then
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
  # 6) Verify contexts are filtered (no .bak, no run)
  echo "[demo_check] verifying context filtering..."
  FEED_PATH="${FEED_PATH}" python3 - <<'PY'
import json
import os
p=os.environ["FEED_PATH"]
o=json.load(open(p,"r",encoding="utf-8"))
names=[c["context"] for c in o["latest"]["contexts"]]
print("[demo_check] latest contexts:", names)
assert all(".bak." not in n for n in names), "backup contexts present"
assert "run" not in names, "run context present"
print("[demo_check] context filtering ok")
PY

  # 7) Determinism proof: regenerate feed and compare content hash
  echo "[demo_check] verifying content determinism..."
  content_hash() {
    FEED_PATH="${FEED_PATH}" python3 - <<'PY'
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
  for c in focus sleep workout; do
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
