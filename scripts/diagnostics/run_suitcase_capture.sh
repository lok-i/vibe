#!/usr/bin/env bash
# Run explicitly after review. This script does not fetch, install, or create tmux.
set -euo pipefail
task_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$task_repo"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
task_logs=logs/uolm_suitcase/numerics_launches
mkdir -p "$task_logs"
task_stamp=$(date -u +%Y%m%dT%H%M%SZ)
# SIGTERM requests a capture at 90 min; SIGKILL is a 60 s cleanup fallback.
timeout --signal=TERM --kill-after=60s 90m \
  .venv/bin/python scripts/diagnostics/suitcase_numerics.py train "$@" \
  2>&1 | tee "$task_logs/$task_stamp.log"
