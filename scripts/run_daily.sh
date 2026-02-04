#!/usr/bin/env bash
set -euo pipefail

# Daily multi-context run (focus/workout/sleep) + marketing + publish "latest" web bundle per context.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python}"

DB_PATH="${MGC_DB:-data/db.sqlite}"
OUT_BASE="${MGC_OUT_BASE:-data/evidence}"

SEED="${MGC_SEED:-1}"
GENERATE_COUNT="${MGC_DAILY_GENERATE_COUNT:-1}"

PROVIDER="${MGC_PROVIDER:-}"
PROMPT="${MGC_PROMPT:-}"

CONTEXTS=(${MGC_CONTEXTS:-focus workout sleep})

MARKETING="${MGC_MARKETING:-1}"          # 1=on, 0=off
TEASER_SECONDS="${MGC_TEASER_SECONDS:-15}"
PUBLISH_MARKETING="${MGC_PUBLISH_MARKETING:-1}"  # 1=on, 0=off (writes marketing receipts)

PUBLISH_LATEST="${MGC_PUBLISH_LATEST:-1}"  # 1=on, 0=off

PUBLISH_FEED="${MGC_PUBLISH_FEED:-1}"      # 1=on, 0=off (generates /var/lib/mgc/releases/feed.json)
REQUIRE_FEED="${MGC_REQUIRE_FEED:-0}"      # 1=fail run if feed generation fails

LOCK_DIR="${ROOT}/.run_daily.lock"

log() { printf "[run_daily] %s\n" "$*"; }
die() { printf "[run_daily] ERROR: %s\n" "$*" >&2; exit 2; }

cleanup() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if mkdir "${LOCK_DIR}" 2>/dev/null; then :; else
  log "Another run_daily appears to be in progress (lock exists: ${LOCK_DIR}). Exiting."
  exit 0
fi

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
else
  log "NOTE: .venv not found; using ${PY} from PATH"
fi

command -v "${PY}" >/dev/null 2>&1 || die "Python executable not found: ${PY}"

log "Repo: ${ROOT}"
log "DB: ${DB_PATH}"
log "Seed: ${SEED}"
log "Generate count: ${GENERATE_COUNT}"
log "Contexts: ${CONTEXTS[*]}"
log "Marketing: ${MARKETING} (teaser_seconds=${TEASER_SECONDS})"
log "Publish marketing: ${PUBLISH_MARKETING}"
log "Publish latest web: ${PUBLISH_LATEST}"
log "Publish release feed: ${PUBLISH_FEED} (require=${REQUIRE_FEED})"
"${PY}" -V

run_one() {
  local ctx="$1"
  local out_dir="${OUT_BASE}/${ctx}"

  log "Context=${ctx} out_dir=${out_dir}"

  args=(
    -m mgc.main
    --db "${DB_PATH}"
    run daily
    --context "${ctx}"
    --seed "${SEED}"
    --out-dir "${out_dir}"
    --generate-count "${GENERATE_COUNT}"
    --json
  )

  if [[ -n "${PROVIDER}" ]]; then
    args+=( --generate-provider "${PROVIDER}" )
  fi
  if [[ -n "${PROMPT}" ]]; then
    args+=( --prompt "${PROMPT}" )
  fi

  if [[ "${MARKETING}" == "1" ]]; then
    args+=( --marketing --teaser-seconds "${TEASER_SECONDS}" )
  fi

  daily_json="$("${PY}" "${args[@]}")"
  printf "%s\n" "${daily_json}"

  if [[ "${PUBLISH_MARKETING}" == "1" ]]; then
    drop_id="$("${PY}" -c 'import json,sys; \
data=sys.stdin.read() or "{}"; \
print(json.loads(data).get("drop_id",""))' <<<"${daily_json}")"

    if [[ -z "${drop_id}" ]]; then
      log "WARN: drop_id missing from daily JSON; skipping publish-marketing to avoid replay"
    else
      pub_args=(
        -m mgc.main
        run publish-marketing
        --bundle-dir "${out_dir}/drop_bundle"
        --out-dir "${out_dir}"
        --drop-id "${drop_id}"
      )
      if [[ "${MGC_DETERMINISTIC:-}" == "1" ]]; then
        pub_args+=( --deterministic )
      fi
      "${PY}" "${pub_args[@]}"
    fi
  fi

  if [[ "${PUBLISH_LATEST}" == "1" ]]; then
    "${ROOT}/scripts/publish_latest.sh" --context "${ctx}" --src-out-dir "${out_dir}" --db "${DB_PATH}"
  fi
}

for ctx in "${CONTEXTS[@]}"; do
  run_one "${ctx}"
done


# Update the internal release feed (served by nginx under /releases/feed.json)
if [[ "${PUBLISH_LATEST}" == "1" && "${PUBLISH_FEED}" == "1" ]]; then
  if [[ -d "/var/lib/mgc/releases" ]]; then
    log "Updating release feed..."
    if ! "${ROOT}/scripts/publish_release_feed.sh"; then
      if [[ "${REQUIRE_FEED}" == "1" ]]; then
        die "publish_release_feed failed"
      fi
      log "WARN: publish_release_feed failed (continuing)"
    fi
  else
    log "NOTE: /var/lib/mgc/releases not present; skipping feed update"
  fi
fi

log "OK"
