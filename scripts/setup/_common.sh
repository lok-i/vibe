# Shared by sync_deps.sh and sync_data.sh — sourced, never run.
#
# Everything both halves need to materialize a deps.lock row: the lock parser, the
# venv, fetch-by-SHA, the LFS guards, and the verify pass.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK_FILE="$REPO_ROOT/deps.lock"

# ── git-lfs on a shared login node ────────────────────────────────────────────
# git-lfs is a Go binary: it fans out `lfs.concurrenttransfers` workers on top of a runtime
# thread pool sized to GOMAXPROCS = every core it can see (64 on CARC's Discovery login
# node). Walking a 12k-file dataset that way aborts mid-transfer with
#   runtime/cgo: pthread_create failed: Resource temporarily unavailable / SIGABRT
# Two guards: cap the fan-out, and don't pull at all when nothing is on a pointer.
# GOMAXPROCS is the guard that matters — the SIGABRT came from a 64-wide runtime pool,
# not from the transfer count — so the fan-out can be useful while staying capped.
# A shared login node that still aborts: LFS_JOBS=2 LFS_GOMAXPROCS=2.
LFS_JOBS=${LFS_JOBS:-8}
LFS_GOMAXPROCS=${LFS_GOMAXPROCS:-4}
LFS_RETRIES=${LFS_RETRIES:-3}

# git with the LFS fan-out capped. Use for anything that transfers or smudges LFS blobs.
lfs_git() { GOMAXPROCS=$LFS_GOMAXPROCS git -c lfs.concurrenttransfers="$LFS_JOBS" "$@"; }

# Paths still on POINTERS ('-' marker in `lfs ls-files`).
# Empty output => `lfs pull` has nothing to do and MUST be skipped, not "run, it's fast".
lfs_missing() {  # lfs_missing <repo> [comma-separated includes]
    local repo="$1" incl="${2:-}"
    lfs_git -C "$repo" lfs ls-files ${incl:+-I "$incl"} 2>/dev/null \
        | sed -n 's/^[0-9a-f]* - //p' || true
}

# lfs_git + backoff. LFS transfers resume, so a retry costs only what didn't land.
lfs_retry() {
    local n=1
    until lfs_git "$@"; do
        if [ "$n" -ge "$LFS_RETRIES" ]; then
            echo "[ERROR] git-lfs failed ${LFS_RETRIES}x."
            echo "        On a pthread_create/SIGABRT abort the node is the wall, not the repo:"
            echo "        retry with LFS_JOBS=1 LFS_GOMAXPROCS=2, or stage from a compute node."
            return 1
        fi
        echo "[ RETRY  ] git-lfs attempt $n/$LFS_RETRIES failed; backing off $((n * 10))s"
        sleep $((n * 10)); n=$((n + 1))
    done
}

# ── fetch one SHA, depth 1 ───────────────────────────────────────────────────
# Hugging Face's git server dies mid-negotiation with
#     fatal: error reading section header 'acknowledgments'
# whenever the client offers a `have` for a commit under HF's PULL-REQUEST namespace. A
# CLONED worktree carries `+refs/heads/*:refs/remotes/origin/*`, so it holds a
# `refs/remotes/origin/pr/N` the server cannot reach from any branch; deleting that ONE ref
# makes the default fetch succeed again (measured). A worktree this script created never has
# it — `init` + `remote add` sets no refspec, so nothing but FETCH_HEAD is ever written,
# which is why CARC has never seen this and a dev box that once ran `git clone` always does.
#
# Not a protocol-version problem: v0 and v1 both get `remote end hung up` (HF is v2-only).
# `noop` sends zero `have` lines — which is also the honest setting, since every fetch here
# is depth=1 at an exact SHA and has no delta to negotiate. Needs git >= 2.32; the retry
# covers older git, where the value is rejected outright (CARC 2.43, dev 2.34 — both fine).
fetch_sha() {
    local repo="$1" sha="$2"
    if git -C "$repo" -c fetch.negotiationAlgorithm=noop \
           fetch --progress --depth 1 origin "$sha"; then
        return 0
    fi
    echo "[ RETRY  ] no-negotiation fetch failed; falling back to git's default"
    git -C "$repo" fetch --progress --depth 1 origin "$sha"
}

# Run a command with a rotating spinner + elapsed counter (stderr). Forwards exit code.
# Usage: spin "label" <cmd> <args...>
spin() {
    local label="$1"; shift
    "$@" &
    local pid=$!
    trap 'kill -- $pid 2>/dev/null; printf "\r\033[K" >&2; exit 130' INT
    local chars='|/-\' i=0 start=$SECONDS
    while kill -0 "$pid" 2>/dev/null; do
        local e=$((SECONDS - start))
        printf "\r[ %-7s] %s  %02d:%02d  " "$label" "${chars:i++%${#chars}:1}" $((e/60)) $((e%60)) >&2
        sleep 0.15
    done
    wait "$pid"; local rc=$?
    printf "\r\033[K" >&2
    trap - INT
    return $rc
}

# ── the venv: uv-only, and the script activates it for itself ────────────────
# conda and vanilla pip are deprecated: CARC already runs a uv venv, orcs documents uv,
# and one installer means one resolution story. DEPS_PIP_CMD stays as an escape hatch.
#
# An ACTIVE venv wins (CARC's flow); otherwise the repo's `.venv`, so a fresh clone needs
# no activation. "Active" means VIRTUAL_ENV holds a `pyvenv.cfg` — a conda prefix has none.
use_venv() {  # use_venv [create]
    if [ -n "${DEPS_PIP_CMD:-}" ]; then
        echo "[ENV] DEPS_PIP_CMD override: $DEPS_PIP_CMD"
        PIP_CMD="$DEPS_PIP_CMD"; return
    fi
    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv not found. This project is uv-only."
        echo "        curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    # Stale, not fatal: an editor injecting its selected interpreter (a conda prefix) is
    # the usual source, and the export below overrides it for every child anyway.
    if [ -n "${VIRTUAL_ENV:-}" ] && [ ! -f "$VIRTUAL_ENV/pyvenv.cfg" ]; then
        echo "[ WARN   ] ignoring VIRTUAL_ENV=$VIRTUAL_ENV — no pyvenv.cfg, not a venv"
        unset VIRTUAL_ENV
    fi
    local venv="${VIRTUAL_ENV:-$REPO_ROOT/.venv}"
    if [ ! -f "$venv/pyvenv.cfg" ]; then
        if [ "${1:-}" != create ]; then
            echo "[ERROR] no venv at $venv — run scripts/setup/sync_deps.sh first"
            exit 1
        fi
        uv venv --python "$(cat "$REPO_ROOT/.python-version")" --prompt vibe "$venv"
    fi
    export VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH"
    PIP_CMD="uv pip"
    echo "[ENV] installer: $PIP_CMD -> $VIRTUAL_ENV"
}

# ── deps.lock ────────────────────────────────────────────────────────────────
# lock_rows <path-prefix> -> `name|url|sha|path|pip_install|lfs|pip_no_deps|sparse` per row
# whose `path` starts with the prefix. `|`, not a tab: IFS=$'\t' collapses consecutive tabs
# (tab is whitespace IFS), which mangles empty fields. `|` doesn't collapse.
lock_rows() {
    [ -f "$LOCK_FILE" ] || { echo "[ERROR] deps.lock not found at $LOCK_FILE" >&2; exit 1; }
    python3 - "$LOCK_FILE" "$1" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    lock = json.load(f)
for name, info in lock.items():
    if not info["path"].startswith(sys.argv[2]):
        continue
    print("|".join([
        name,
        info["url"],
        info["sha"],
        info["path"],
        "1" if info.get("pip_install") else "0",
        "1" if info.get("lfs") else "0",
        "1" if info.get("pip_no_deps") else "0",
        "1" if info.get("sparse") else "0",
    ]))
PYEOF
}

# `sparse` rows fetch only the blobs a vibe task reads. The globs are DERIVED —
# `vibe.core.paths --lfs-include` asks the task rosters, reading only metadata.json, which is
# not LFS and so is real in a pointer-only checkout. The sparse set keeps every *.json (the
# next resolve needs them all) plus those globs; `lfs pull -I` then moves only those bytes.
sparse_include() {  # sparse_include <repo> -> comma-separated lfs globs
    local globs err
    err=$(mktemp)
    # as a FILE, not `-m`: see lfs_include's docstring. Its stderr carries mjlab's caught
    # [WARN] for vibe itself on a pointer tree — expected, so shown only on failure.
    globs=$(python "$REPO_ROOT/src/vibe/core/paths.py" --lfs-include 2>"$err" \
            | sed -n 's/^include //p') || true
    if [ -z "$globs" ]; then
        echo "[ERROR] --lfs-include resolved nothing:" >&2; tail -20 "$err" >&2; rm -f "$err"
        return 1
    fi
    rm -f "$err"
    # shellcheck disable=SC2046
    env GIT_LFS_SKIP_SMUDGE=1 git -C "$1" sparse-checkout set --no-cone \
        '*.json' $(sed 's|^|/|' <<< "$globs") >&2
    paste -sd, <<< "$globs"
}

sync_one() {
    local name="$1" url="$2" sha="$3" rel_path="$4" pip_install="$5" lfs="$6" pip_no_deps="$7" sparse="$8"
    local path="$REPO_ROOT/$rel_path"

    echo
    echo "=== $name @ ${sha:0:12} ==="
    echo "    url:  $url"
    echo "    path: $rel_path"

    if [ ! -d "$path/.git" ]; then
        echo "[ CLONE  ] $name -> $rel_path"
        mkdir -p "$path"
        git -C "$path" init -q
        git -C "$path" remote add origin "$url"
        if [ "$lfs" = "1" ]; then
            git -C "$path" lfs install --local
        fi
    fi

    local current
    current=$(git -C "$path" rev-parse HEAD 2>/dev/null || echo "none")

    if [ "$current" != "$sha" ]; then
        echo "[ FETCH  ] depth=1 $sha"
        fetch_sha "$path" "$sha"
        if [ "$lfs" = "1" ]; then
            # Do NOT smudge during checkout: `git-lfs filter-process` materializes blobs
            # strictly ONE AT A TIME and ignores lfs.concurrenttransfers, so a 14k-object
            # dataset costs 14k sequential round-trips. Write pointers instead (instant)
            # and let the `lfs pull` below move the bytes concurrently. Same files either
            # way. `env` + real git, not lfs_git: a `VAR=x func` assignment leaks past the
            # call for shell functions, which lfs_git is.
            spin "checkout" env GIT_LFS_SKIP_SMUDGE=1 GOMAXPROCS="$LFS_GOMAXPROCS" \
                git -C "$path" -c lfs.concurrenttransfers="$LFS_JOBS" checkout -q "$sha"
        else
            git -C "$path" checkout -q "$sha"
        fi
    else
        echo "[ HEAD OK] already at $sha"
    fi

    # Pull LFS only when something is actually still a pointer — catches a checkout that
    # happened without git-lfs on PATH, without walking every present blob on each sync
    # (that walk is pure overhead AND the thing that trips a login node's pids cap).
    if [ "$lfs" = "1" ]; then
        local lfs_include="" missing
        if [ "$sparse" = "1" ]; then
            lfs_include=$(sparse_include "$path")
            echo "[ SPARSE ] $(tr ',' '\n' <<< "$lfs_include" | wc -l) glob(s) from the task rosters"
        fi
        missing=$(lfs_missing "$path" "$lfs_include")
        if [ -z "$missing" ]; then
            echo "[ LFS OK ] every object present — skipping pull"
        else
            echo "[ LFS    ] $(wc -l <<< "$missing") file(s) on pointers"
            spin "LFS pull" lfs_retry -C "$path" lfs pull ${lfs_include:+-I "$lfs_include"}
        fi
    fi

    if [ "$pip_install" = "1" ]; then
        local no_deps=""
        [ "$pip_no_deps" = "1" ] && no_deps="--no-deps"
        echo "[ PIP    ] $PIP_CMD install ${no_deps:+$no_deps }-e $rel_path"
        $PIP_CMD install $no_deps -e "$path"
    fi
}

sync_rows() {  # sync_rows <rows>
    local name url sha rel_path pip_install lfs pip_no_deps sparse
    while IFS='|' read -r name url sha rel_path pip_install lfs pip_no_deps sparse; do
        [ -z "$name" ] && continue
        sync_one "$name" "$url" "$sha" "$rel_path" "$pip_install" "$lfs" "$pip_no_deps" "$sparse"
    done <<< "$1"
}

# ── verify: every worktree at its pinned SHA ─────────────────────────────────
# The last gate before a GPU is booked. Two ways the tree and the lock disagree, both silent
# everywhere else and both worth hours: a lock bumped to a SHA that was never PUSHED (the
# sync resolves nothing and the host keeps training the old code), and a worktree parked on a
# local branch. Every dep is an editable install, so a matching HEAD *is* a matching import —
# no package-level check needed.
verify_rows() {  # verify_rows <rows> <label>
    local rc=0 name url sha rel_path rest head note status
    echo
    echo "=== verify: $2 HEAD vs deps.lock ==="
    while IFS='|' read -r name url sha rel_path rest; do
        [ -z "$name" ] && continue
        head=$(git -C "$REPO_ROOT/$rel_path" rev-parse -q --verify HEAD 2>/dev/null || echo "")
        if [ -z "$head" ]; then
            status="[ MISSING]"; note="no git worktree at $rel_path"; rc=1
        elif [ "$head" = "$sha" ]; then
            status="[   OK   ]"; note="${sha:0:12}"
        else
            status="[  DRIFT ]"; note="${head:0:12}  != pinned ${sha:0:12}"; rc=1
        fi
        printf '%s %-20s %s\n' "$status" "$name" "$note"
    done <<< "$1"

    if [ "$rc" != 0 ]; then
        echo
        echo "[ERROR] the checkout does not match deps.lock."
        echo "        MISSING/DRIFT before a sync -> run the sync with no --check."
        echo "        DRIFT that SURVIVES a sync  -> the pinned SHA is not on the remote."
        echo "                                       push the dependency branch, then re-sync."
    fi
    return $rc
}
