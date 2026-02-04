#!/usr/bin/env bash
set -euo pipefail

# Usage: source scripts/diffsinger_env.sh [CHECKPOINT_DIR]
# Example: source scripts/diffsinger_env.sh /Users/admin/ai/DiffSinger/checkpoints/lj_ds_beta6_1213

base_dir="${1:-/Users/admin/ai/DiffSinger/checkpoints/lj_ds_beta6_1213}"

if [[ ! -d "$base_dir" ]]; then
  echo "diffsinger_env: base dir not found: $base_dir" >&2
  return 2 2>/dev/null || exit 2
fi

latest_wavs=""
if latest_wavs=$(ls -td "$base_dir"/generated_*/wavs 2>/dev/null | head -n 1); then
  :
fi

if [[ -z "$latest_wavs" || ! -d "$latest_wavs" ]]; then
  echo "diffsinger_env: no generated_* /wavs found under $base_dir" >&2
  return 2 2>/dev/null || exit 2
fi

export MGC_PROVIDER=diffsinger
export MGC_DIFFSINGER_CMD="python3 scripts/diffsinger_local.py"
export MGC_DIFFSINGER_SAMPLE_DIR="$latest_wavs"
export MGC_DIFFSINGER_OUTPUT_FORMAT=wav
export MGC_PLAYLIST_PROVIDER=any

echo "diffsinger_env: MGC_DIFFSINGER_SAMPLE_DIR=$MGC_DIFFSINGER_SAMPLE_DIR" >&2
