#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -E "$0" "$@"
fi

NGINX_BIN="${MGC_NGINX_BIN:-$(command -v nginx || true)}"
if [[ -z "${NGINX_BIN}" ]]; then
  echo "ERROR: nginx not found in PATH (set MGC_NGINX_BIN to override)" >&2
  exit 2
fi

CONF_PATH="${MGC_NGINX_CONF:-}"
if [[ -z "${CONF_PATH}" ]]; then
  CONF_PATH="$("${NGINX_BIN}" -V 2>&1 | tr ' ' '\n' | awk -F= '/^--conf-path=/{print $2; exit}')"
fi
if [[ -z "${CONF_PATH}" ]]; then
  for candidate in /etc/nginx/nginx.conf /opt/homebrew/etc/nginx/nginx.conf; do
    if [[ -f "${candidate}" ]]; then
      CONF_PATH="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CONF_PATH}" || ! -f "${CONF_PATH}" ]]; then
  echo "ERROR: nginx.conf not found (set MGC_NGINX_CONF to override)" >&2
  exit 2
fi

RELEASES_DIR="${MGC_RELEASES_DIR:-/var/lib/mgc/releases}"
mkdir -p "${RELEASES_DIR}"

PYTHON_BIN="${MGC_PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

RESULT="$("${PYTHON_BIN}" - "${CONF_PATH}" "${RELEASES_DIR}" <<'PY'
import re
import sys
import time
from pathlib import Path
import shutil

conf_path = Path(sys.argv[1])
releases_dir = sys.argv[2].rstrip("/") + "/"
text = conf_path.read_text(encoding="utf-8")

if re.search(r"location\s+/releases/\s*{", text):
  print("noop")
  sys.exit(0)

lines = text.splitlines(keepends=True)

def find_block(start_pat, start, end):
  for i in range(start, end):
    if re.match(start_pat, lines[i]):
      depth = 0
      for j in range(i, end):
        depth += lines[j].count("{")
        depth -= lines[j].count("}")
        if j > i and depth == 0:
          return i, j
  return None

http_block = find_block(r"^\s*http\s*{", 0, len(lines))
if not http_block:
  print("error:http block not found")
  sys.exit(1)

http_start, http_end = http_block
server_block = find_block(r"^\s*server\s*{", http_start + 1, http_end)
if not server_block:
  print("error:server block not found in http")
  sys.exit(1)

server_start, server_end = server_block
server_line = lines[server_start]
m = re.match(r"^(\s*)server\s*{", server_line)
server_indent = m.group(1) if m else ""
inner_indent = server_indent + "    "

insert_lines = [
  f"{inner_indent}location /releases/ {{\n",
  f"{inner_indent}    alias {releases_dir};\n",
  f"{inner_indent}}}\n",
]

new_lines = lines[:server_end] + ["\n"] + insert_lines + lines[server_end:]

ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
backup = conf_path.with_name(conf_path.name + f".bak.{ts}")
shutil.copy2(conf_path, backup)
conf_path.write_text("".join(new_lines), encoding="utf-8")

print(f"changed:{backup}")
PY
)"

case "${RESULT}" in
  noop)
    echo "[setup_nginx] nginx.conf already serves /releases/"
    ;;
  changed:*)
    BACKUP_PATH="${RESULT#changed:}"
    echo "[setup_nginx] updated ${CONF_PATH}"
    echo "[setup_nginx] backup: ${BACKUP_PATH}"
    ;;
  error:*)
    echo "[setup_nginx] ${RESULT#error:}" >&2
    exit 2
    ;;
  *)
    echo "[setup_nginx] unexpected result: ${RESULT}" >&2
    exit 2
    ;;
esac

if ! "${NGINX_BIN}" -t -c "${CONF_PATH}"; then
  if [[ "${RESULT}" == changed:* && -n "${BACKUP_PATH:-}" ]]; then
    cp -f "${BACKUP_PATH}" "${CONF_PATH}"
    echo "[setup_nginx] nginx -t failed; restored ${CONF_PATH}" >&2
  fi
  exit 2
fi

if pgrep -x nginx >/dev/null 2>&1; then
  "${NGINX_BIN}" -s reload -c "${CONF_PATH}"
  echo "[setup_nginx] nginx reloaded"
else
  "${NGINX_BIN}" -c "${CONF_PATH}"
  echo "[setup_nginx] nginx started"
fi
