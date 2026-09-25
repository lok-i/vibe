# setup

```bash
bash scripts/setup/sync_deps.sh              # .venv + vibe + the code rows of deps.lock
bash scripts/setup/sync_deps.sh --deploy     # + onnxruntime, for export-agent
bash scripts/setup/sync_data.sh [MODE...]    # data rows + staged terrain: all | inhouse | omre | grail
bash scripts/setup/sync_{deps,data}.sh --check   # verify only: every checkout vs its pin, no network
```

| mode | fetches | for |
|---|---|---|
| `inhouse` | retargeted motions (only the clips a task uses, `*.npz` only) + the generated nominal stand | Repose, Uolm, Dodge |
| `omre` / `grail` | OmniRetarget / GRAIL terrain, staged by orcs | PerLoco |

Both scripts are idempotent: re-run after a `deps.lock` bump. The venv is the active uv venv,
else `./.venv` (created at `.python-version`). Installs are editable only: vibe finds `data/`
by walking up to the repo root.

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
live-vs-validated diff at import.

## failure modes

| symptom | cause | fix |
|---|---|---|
| a model class is missing (`SonicWithAdapterModel`, `CrossAttentionExtractor`, …) | a later `pip install` re-resolved vibe, and mjlab's `rsl-rl-lib==5.2.0` pin replaced the fork | re-run `sync_deps.sh`; add extras through its flags (`--deploy`), never `pip install -e .[...]` |
| `--check` DRIFT before a sync | stale checkout | run the sync |
| `--check` DRIFT that survives a sync | the pinned SHA is not on the remote | push the dependency first, then re-sync |
| `fatal: error reading section header 'acknowledgments'` on a manual fetch | a hand-cloned HF worktree offers a `refs/remotes/origin/pr/N` the server cannot reach | `git -C <repo> update-ref -d refs/remotes/origin/pr/<N>` (the scripts already fetch with `negotiationAlgorithm=noop`) |
| git-lfs `pthread_create failed` / `SIGABRT` | a shared login node's process cap | `LFS_JOBS=1 LFS_GOMAXPROCS=2`, or sync from a compute node |
| every `Uolm` task missing | object XMLs not generated on this machine | `assets generate` (the sync runs it) |
| a `Vibe-PerLoco-*` / `-Dodge-*` task missing | its data is not staged; registration skips instead of raising | `sync_data.sh omre grail` / `inhouse`; `python -c "import vibe; print(vibe.tasks.perloco.config.g1.SKIP_REASON)"` |
| `CCD overflow` on stderr (mesh-object tasks) | the per-world collision scratch cap (`VIBE_NCCDMAX`, default 64) is too low | raise `VIBE_NCCDMAX` |
| a `Vibe-*` task fails to register when a script imports `orcs` first | mjlab's task scan re-enters orcs mid-import | `import mjlab` before `import orcs` |
