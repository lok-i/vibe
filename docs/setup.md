# setup

```bash
bash scripts/setup/sync_deps.sh              # creates .venv; vibe + all code rows in one resolution
bash scripts/setup/sync_deps.sh --deploy     # + onnxruntime, for export-agent
bash scripts/setup/sync_deps.sh --deploy --optix  # optional RTX backend + export
source .venv/bin/activate                   # for train/play/export commands
bash scripts/setup/sync_data.sh [MODE...]    # data rows + staged terrain: all | inhouse | omre | grail
bash scripts/setup/sync_deps.sh --check      # source SHAs, patch, editable origins, package compatibility
bash scripts/setup/sync_data.sh --check      # locked data checkout pins, no downloads
```

| mode | fetches | for |
|---|---|---|
| `inhouse` | retargeted motions (only the clips a task uses, `*.npz` only) + the generated nominal stand | Repose, Uolm, Dodge |
| `omre` / `grail` | OmniRetarget / GRAIL terrain, staged by orcs | PerLoco |

Both scripts are idempotent: re-run after a `deps.lock` bump. Code setup creates
Vibe's Python 3.12 `./.venv` if absent. It rejects an unrelated active environment
or an existing environment with the wrong Python version; use a fresh checkout
for this migration. Dependency checkouts share Vibe's environment.
Installs are editable only: vibe finds `data/` by walking up to the repo root.

Trained on RTX 3090, L40S and RTX 5090 (Linux) at 4096 envs.

## dependencies

```
rsl_rl ──► mocke ──► orcs ──► vibe       one direction; a term lives in exactly one repo
```

| repo | owns |
|---|---|
| [rsl_rl](https://github.com/lok-i/rsl_rl) (fork) | PPO + PPOAux, the SONIC/adapter models, the extractor |
| [mocke](https://github.com/lok-i/mocke) | the frozen SONIC base: checkpoint + its obs/action contract |
| [orcs](https://github.com/lok-i/orcs) | the privileged tasks: commands, rewards, terminations, terrain, robustness |
| vibe | everything downstream of the camera |

`deps.lock` pins each at a SHA; bump = edit `sha`, re-run the sync. One venv holds one
editable install per package, so vibe's lock wins over orcs's; `orcs.core.deps` prints the
live-vs-validated diff at import. Simulator sources also live in `deps.lock`:
mjlab 1.6.1 plus ORCS's reviewed compatibility patch, and an immutable
mujoco-warp main snapshot. See [the simulator guide](simulator-upgrade.md).
`sync_deps.sh` passes all editables and requested extras to one uv resolution,
ignoring dependency-owned uv sources. `requirements.lock` freezes the tested
registry dependencies, including optional export/OptiX packages; constraints
do not install unused extras. Python provisioning is pinned to 3.12.12.
Do not independently install extras or
run `uv sync` inside dependencies: that can replace the rsl_rl fork.

The released ViBe 0.1.2 uses ORCS v0.1.1. This candidate retains ViBe's existing
asset and motion-data pins; its ORCS code changes are still local pending review.
ORCS's newer EgoHTR data is not required by the ViBe tasks; advancing the shared
motion-data pin would also add demonstrations to Uolm's training roster.
The ORCS upgrade removes robot reset perturbations during play for Dodge and
PerLoco, matching Repose and Uolm. Training retains its robustness domain.
Task randomness remains in play; use fixed seeds and clips for repeatable evaluation.

## failure modes

| symptom | cause | fix |
|---|---|---|
| OptiX import reports `Unsupported ABI version` | OptiX 9.1 requires an R590+ driver | use Warp on this machine, or validate OptiX on a compatible host |
| source patch check fails | dependency edits differ from the committed patch | inspect and preserve the edits before resyncing; setup will not overwrite them |
| a model class is missing (`SonicWithAdapterModel`, `CrossAttentionExtractor`, …) | a later `pip install` re-resolved vibe, and mjlab's `rsl-rl-lib==5.2.0` pin replaced the fork | re-run `sync_deps.sh`; add extras through its flags (`--deploy`), never `pip install -e .[...]` |
| `--check` DRIFT before a sync | stale checkout | run the sync |
| `--check` DRIFT that survives a sync | the pinned SHA is not on the remote | push the dependency first, then re-sync |
| `fatal: error reading section header 'acknowledgments'` on a manual fetch | a hand-cloned HF worktree offers a `refs/remotes/origin/pr/N` the server cannot reach | `git -C <repo> update-ref -d refs/remotes/origin/pr/<N>` (the scripts already fetch with `negotiationAlgorithm=noop`) |
| git-lfs `pthread_create failed` / `SIGABRT` | a shared login node's process cap | `LFS_JOBS=1 LFS_GOMAXPROCS=2`, or sync from a compute node |
| every `Uolm` task missing | object XMLs not generated on this machine | `assets generate` (the sync runs it) |
| a `Vibe-*` task missing from `list-envs` (train/play: "invalid choice") | its data is not staged; registration skips instead of raising | `sync_data.sh inhouse` (Repose, Uolm, Dodge) / `omre grail` (PerLoco); why: `python -c "import mjlab, vibe; print(vibe.tasks.<family>.config.g1.SKIP_REASON)"` |
| `CCD overflow` on stderr (mesh-object tasks) | the per-world collision scratch cap (`VIBE_NCCDMAX`, default 64) is too low | raise `VIBE_NCCDMAX` |
| a `Vibe-*` task fails to register when a script imports `orcs` first | mjlab's task scan re-enters orcs mid-import | `import mjlab` before `import orcs` |
