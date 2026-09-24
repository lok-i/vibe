#!/usr/bin/env bash
# Sync the CODE: a uv venv, vibe itself, and every deps.lock row under dependencies/.
#
#   1. the venv — the active one, else ./.venv (created: `uv venv --python 3.11`)
#   2. `uv pip install -e .[dev]` — vibe FIRST: mjlab pins rsl-rl-lib==5.2.0 off PyPI,
#      so the fork below must be the last install, every run
#   3. per row: shallow-fetch the pinned SHA, `uv pip install -e` it
#   4. generate machine-local object XMLs, then VERIFY every HEAD against its pin
#
#   --check   run the verify pass alone — no network, no pip. This is the pre-flight.
#
# Idempotent: skips fetch when HEAD already matches the pinned SHA. Data: sync_data.sh.
# Assumes: git-lfs + uv on PATH (uv fetches Python itself).

set -euo pipefail
. "$(dirname "$0")/_common.sh"

CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $arg (expected --check)"; exit 2 ;;
    esac
done

rows=$(lock_rows dependencies/)

if [ "$CHECK_ONLY" = 0 ]; then
    use_venv create
    echo
    echo "=== vibe (self) ==="
    $PIP_CMD install -e "$REPO_ROOT[dev]"

    sync_rows "$rows"

    echo
    echo "=== generate: assets ==="
    python -m assets.cli generate --quiet
fi

verify_rows "$rows" "dependencies/"
