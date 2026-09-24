# Setup — sharp bits

Failure modes of `scripts/setup/sync_{deps,data}.sh` (helpers shared in `_common.sh`) and the
env they install into.
Everything here is measured, with the machine it was measured on.

| machine | os | git | git-lfs |
|---|---|---|---|
| dev box | Ubuntu | 2.34.1 | — |
| CARC `discovery1` | Rocky 8.10 | 2.43.7 (`module load git/2.51.0` for newer) | 3.7.1 |

---

## 1 · `error reading section header 'acknowledgments'`

```
=== retargeted_motions @ e635ed670162 ===
[ FETCH  ] depth=1 e635ed670162c8b4cc91dd6247d4f3b47368ab3c
fatal: error reading section header 'acknowledgments'
```

**Cause — one ref, not "too much history".** A worktree that was `git clone`d carries the
refspec `+refs/heads/*:refs/remotes/origin/*`, so it holds `refs/remotes/origin/pr/N` — a
commit in Hugging Face's pull-request namespace that no branch reaches. Offer that as a
`have` during protocol-v2 negotiation and HF's server never emits the `acknowledgments`
section git is blocking on.

| variant | result |
|---|---|
| default | `fatal: error reading section header 'acknowledgments'` |
| `protocol.version=0` / `=1` | `fatal: the remote end hung up` — HF is v2-only |
| default, `refs/remotes/origin/pr/12` deleted | ok (one ref, nothing else changed) |
| `fetch.negotiationAlgorithm=noop` | ok |

**Fixed in the script.** `fetch_sha()` passes `noop`, which sends zero `have` lines — also
the honest setting, since every fetch here is `--depth 1` at an exact SHA and has no delta to
negotiate. Needs git ≥ 2.32; both machines above are past it.

**Why CARC never saw it.** Worktrees this script creates are `init` + `remote add` with NO
refspec, so only `FETCH_HEAD` is ever written. Only a directory that predates the script (or
was cloned by hand) is affected.

```bash
git -C data/retargeted_motions config --get remote.origin.fetch   # prints a refspec => cloned => affected
```

Optional, to make *manual* `git fetch/pull` work in that directory too:

```bash
git -C data/retargeted_motions update-ref -d refs/remotes/origin/pr/12
git -C data/retargeted_motions config --unset-all remote.origin.fetch
```

Nothing is lost — a remote-tracking ref is a cached pointer, re-fetchable.

**Why not switch that entry to `hf download`.** Considered and rejected 2026-08-17.
`retargeted_motions` is an HF dataset, and `huggingface_hub` is already in both envs, so the
transport swap looked free. It isn't: CARC's git-lfs path works today (1011 MB staged, git
2.43), while `cdn-lfs.huggingface.co` does not even resolve from the login node — hf-transport
egress there is unproven. Rewriting a working transport against an unverified egress is the
worse trade. Revisit only if git-lfs actually breaks on CARC.

---

## 2 · `retargeted_motions` is sparse — ~0.25 GB, not 7.6

The full dataset is 3.8 GB of worktree + 3.9 GB of LFS cache (`.git/lfs` keeps every blob
alongside its worktree copy; `gc` touches neither). vibe reads ~0.1 GB of it, so the row is
`"sparse": true` in `deps.lock`:

| step | what |
|---|---|
| 1 | checkout with `GIT_LFS_SKIP_SMUDGE=1` — pointers only, instant |
| 2 | `python src/vibe/core/paths.py --lfs-include` — globs from the task ROSTERS (repose roots + orcs's uolm resolver). Reads only `metadata.json`, which is not LFS |
| 3 | `git sparse-checkout set --no-cone '*.json' <globs>` |
| 4 | `lfs pull -I <globs>` — `*.npz` only; the per-clip mp4/csv are ~90% of the bytes and unread |

A roster change reaches the fetch on the next `sync_data.sh` — nothing is listed by hand.
The resolver runs as a FILE, not `-m`: on a pointer tree `import vibe` raises (repose
registration loads a clip), and only mjlab's entry-point scan catches that.

**A checkout from before the switch** shrinks its worktree on the next sync but keeps its
cache: `git -C data/retargeted_motions lfs prune` reclaims it. Not free — a later checkout of
an older SHA re-downloads — so run it when disk binds, not as hygiene.

---

## 3 · Which env the scripts install into

`use_venv()` in `_common.sh`, most specific wins:

| state | picked |
|---|---|
| `DEPS_PIP_CMD` set | that, verbatim — the escape hatch |
| `VIRTUAL_ENV` set, has `pyvenv.cfg` | that venv (CARC's flow) |
| `VIRTUAL_ENV` set, NO `pyvenv.cfg` | warn + ignore → `./.venv` — stale, usually an editor injecting a conda interpreter |
| nothing active | `./.venv`; `sync_deps.sh` creates it (`uv venv --python 3.11`) |

Then the script exports `VIRTUAL_ENV` + `PATH` for itself, so every `python` / console script
it runs is the venv's. uv-only: a uv venv ships no `pip`, and a bare `pip` on PATH resolves
to whatever conda base the shell auto-activated — every dep lands there silently.

**Install order is fixed, every run:** vibe itself first, then the forks. mjlab 1.4 pins
`rsl-rl-lib==5.2.0` off PyPI, so any `pip install` that re-resolves vibe swaps the rsl_rl fork
for that wheel (`SonicWithAdapterModel` gone, every task broken). Adding an extra later?
Re-run `sync_deps.sh` after it.

---

## 4 · git-lfs on a shared login node

git-lfs is a Go binary: it fans `lfs.concurrenttransfers` workers on a runtime pool sized to
GOMAXPROCS = every visible core (32 on `discovery1`), while the user slice caps pids. Walking
a 12k-file dataset that way aborts with:

```
runtime/cgo: pthread_create failed: Resource temporarily unavailable
SIGABRT
```

The scripts cap both (`LFS_JOBS=8`, `LFS_GOMAXPROCS=4` — GOMAXPROCS is the guard that matters) and skips `lfs pull` entirely when
nothing is still on a pointer. If it still aborts, the node is the wall, not the repo:
`LFS_JOBS=1 LFS_GOMAXPROCS=2`, or stage from a compute node.

---

## 5 · Drift between the lock and the checkout

```bash
bash scripts/setup/sync_deps.sh --check   # dependencies/ rows — no network, no pip
bash scripts/setup/sync_data.sh --check   # data/ rows
```

Prints every worktree HEAD against its pinned SHA and exits non-zero on drift. It also runs
as the last pass of every sync. Two causes, opposite fixes:

| when | means | fix |
|---|---|---|
| DRIFT **before** a sync | stale checkout | run the sync |
| DRIFT that **survives** a sync | the pinned SHA is not on the remote | push the dependency branch, then re-sync |

The second one cost 11 GPU-hours on 2026-08-17 — see the SOP in [CLAUDE.md](../../CLAUDE.md).

---

## 6 · Every UOLM task is missing after an assets update

UOLM loads machine-generated object XMLs. The tracked package contains source
meshes and reusable convex parts; `assets generate` writes XMLs with absolute
paths into the per-user cache printed by `assets generated-path`.

The full dependency sync runs this generation step automatically. To repair an
environment that was synced before that hook existed:

```bash
assets generate
python -c "import orcs; print(orcs.tasks.uolm.SKIP_REASON)"
```

An empty mapping means every ORCS UOLM task registered. Vibe's UOLM task uses
the same object lookup and returns with it.
