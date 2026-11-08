#!/usr/bin/env bash
# Sync the DATA every Vibe-* task reads, into data/. Needs sync_deps.sh first.
#
#   sync_data.sh [MODE...]      default: all
#     inhouse   retargeted_motions (only the clips a task rosters, *.npz only)
#               + the nominal stand       -> Repose, Uolm, Dodge
#     omre      OmniRetarget terrain      -> PerLoco-OmRe
#     grail     GRAIL curb terrain        -> PerLoco-Grail
#     hoiretarget  native suitcase roster -> Uolm-Suitcase
#     all       all of the above
#   --check     verify selected lock rows and native HOI checksums — no network
#
# Not fetched, on purpose: reconstructed_motions and the SMPL-X body models feed only
# orcs's -Smpl tasks; no Vibe-* task reads them.
# Idempotent: every step no-ops when its output is already there.

set -euo pipefail
. "$(dirname "$0")/_common.sh"

CHECK_ONLY=0 modes=()
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        all|inhouse|omre|grail|hoiretarget) modes+=("$arg") ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $arg (expected all|inhouse|omre|grail|hoiretarget|--check)"; exit 2 ;;
    esac
done
[ ${#modes[@]} -gt 0 ] || modes=(all)
has() { [[ " ${modes[*]} " == *" $1 "* || " ${modes[*]} " == *" all "* ]]; }

rows=$(lock_rows data/)
if [ "$CHECK_ONLY" = 1 ]; then
    if has inhouse || has omre || has grail; then verify_rows "$rows" "data/"; fi
    if has hoiretarget; then
        orcs_env=$("$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/src/vibe/core/paths.py" --env)
        eval "$orcs_env"
        "$REPO_ROOT/.venv/bin/python" -m orcs.cli.stage_hoi_motions --check
    fi
    exit
fi

use_venv

if has hoiretarget; then
    # Install only ORCS's staging extra requirements, never re-resolve its
    # simulator dependencies over ViBe's rsl_rl fork.
    mapfile -t extra < <(python - "$REPO_ROOT/dependencies/orcs/pyproject.toml" <<'PYEOF'
import sys, tomllib
print("\n".join(tomllib.load(open(sys.argv[1], "rb"))["project"]["optional-dependencies"]["hoiretarget"]))
PYEOF
)
    echo "[ PIP    ] orcs[hoiretarget] staging deps: ${extra[*]}"
    "${PIP_CMD[@]}" install "${extra[@]}"
    orcs_env=$(python "$REPO_ROOT/src/vibe/core/paths.py" --env)
    eval "$orcs_env"
    python -m orcs.cli.stage_hoi_motions
fi

if has inhouse; then
    sync_rows "$rows"

    # Dodge's reference: a held init_state, GENERATED, not fetched — absent, every
    # Vibe-Dodge-* task SKIPS at registration. --out pins vibe's data/ (no root
    # resolution to trust); --device cpu so a GPU-less login node can run it.
    echo
    echo "=== generate: nominal stand ==="
    if [ -f "$REPO_ROOT/data/nominal_motions/nominal/stand/sample1/motion.npz" ]; then
        echo "[ NOM OK ] already present — skipping"
    else
        orcs-make-nominal --out "$REPO_ROOT/data/nominal_motions" --device cpu
    fi
fi

# PerLoco: orcs owns the staging, vibe owns the roots. The roots come from
# `vibe.core.paths` — the same definition `bind_orcs()` uses at runtime — so setup and
# training can never disagree about where data lives. Run paths.py as a FILE:
# importing vibe before staging can require the datasets we are about to install.
sources=()
has omre && sources+=(omni)
has grail && sources+=(grail)
if [ ${#sources[@]} -gt 0 ]; then
    # orcs's `perloco` extra (staging-only deps), read from ITS pyproject — never
    # respelled here. Not `-e orcs[perloco]`: that re-resolves mjlab, whose
    # rsl-rl-lib==5.2.0 pin would swap out the fork.
    mapfile -t extra < <(python - "$REPO_ROOT/dependencies/orcs/pyproject.toml" <<'PYEOF'
import sys, tomllib
print("\n".join(tomllib.load(open(sys.argv[1], "rb"))["project"]["optional-dependencies"]["perloco"]))
PYEOF
)
    echo "[ PIP    ] orcs[perloco] staging deps: ${extra[*]}"
    "${PIP_CMD[@]}" install "${extra[@]}"
    orcs_env=$(python "$REPO_ROOT/src/vibe/core/paths.py" --env)
    eval "$orcs_env"
    bash "$ORCS_DEPS_ROOT/orcs/scripts/setup/perceptive_locomotion.sh" \
        --sources "$(IFS=,; echo "${sources[*]}")" --no-smpl --yes
fi

if has inhouse; then verify_rows "$rows" "data/"; fi
