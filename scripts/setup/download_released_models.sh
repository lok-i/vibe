#!/usr/bin/env bash
# Download + sha256-verify the released vibe checkpoints (lkrajan/vibe on HF).
#   bash scripts/setup/download_released_models.sh            # all
#   bash scripts/setup/download_released_models.sh <task> ... # just these;  --list, --force
set -euo pipefail
source "$(dirname "$0")/_common.sh"
use_venv
exec python -m vibe.release "$@"
