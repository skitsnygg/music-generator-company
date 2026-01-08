#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY="${PYTHON:-python}"

echo "== python =="
$PY -V

echo "== rebuild playlists (determinism check + write) =="
$PY -m mgc.main rebuild playlists --determinism-check --write

echo "== verify playlists vs manifest =="
$PY -m mgc.main rebuild verify playlists

echo "== latest rebuild events =="
$PY -m mgc.main events list --type rebuild.completed --limit 3 || true
$PY -m mgc.main events list --type rebuild.verify_completed --limit 3 || true

echo "OK"
