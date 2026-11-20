#!/usr/bin/env bash
# Create/use this checkout's Python 3.12 .venv, sync pinned sources and install once.
# Usage: bash scripts/setup/sync_deps.sh [--deploy] [--optix] [--check]
# --deploy: ONNX runtime; --optix: optional RTX renderer; --check: verify only.
# An active environment must belong to this checkout. Existing wrong-version
# environments and user edits are never replaced. Data: sync_data.sh.
set -euo pipefail
. "$(dirname "$0")/_common.sh"
CHECK_ONLY=0 EXTRAS=dev,perloco
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --deploy) EXTRAS="$EXTRAS,deploy" ;;
        --optix) EXTRAS="$EXTRAS,optix" ;;
        -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $arg (expected --check | --deploy | --optix)"; exit 2 ;;
    esac
done
rows=$(lock_rows dependencies/)
if [ "$CHECK_ONLY" = 0 ]; then
    use_venv
    sync_rows "$rows"
    install_code "$EXTRAS" "$rows"
    python -m assets.cli generate --quiet
fi
verify_rows "$rows" dependencies/
"$REPO_ROOT/.venv/bin/python" "$LOCK_HELPER" check-patches "$REPO_ROOT"
"$REPO_ROOT/.venv/bin/python" "$LOCK_HELPER" check-install "$REPO_ROOT"
uv pip check --python "$REPO_ROOT/.venv/bin/python"
