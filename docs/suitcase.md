# Suitcase visual adaptation

`Vibe-Uolm-Suitcase-ImgFeat-Ext` adapts frozen SONIC over ORCS's 63 native
HOI-Retarget suitcase clips. It reuses the UOLM visual agent: frozen Theia-tiny,
PPO-trained cross-attention, LoRA rank/alpha 28/28, frozen std ×1.3, and a
privileged critic. The actor reads images, proprioception and reference/goal
commands; measured object state and identity remain critic-only.

## Setup

The published ORCS suitcase, benchmark and optional adaptive-sampling code is
pinned in `deps.lock` at
[`b347b6f`](https://github.com/lok-i/orcs/commit/b347b6f0c1e488df619011dd23a869bd7c6f95e2).
Use ViBe's normal repository-local environment and setup:

```bash
source .venv/bin/activate
bash scripts/setup/sync_deps.sh
bash scripts/setup/sync_deps.sh --check
```

ViBe retains its existing asset, legacy motion-data, mocke and rsl_rl pins,
simulator stack, and released checkpoint provenance.
The ORCS upgrade also corrects VOF to use live randomized mass/inertia in
existing assisted UOLM training; unchanged configs do not imply identical
training behavior. The recipe below disables VOF throughout.

## Data and play

```bash
bash scripts/setup/sync_data.sh hoiretarget
bash scripts/setup/sync_data.sh hoiretarget --check
play Vibe-Uolm-Suitcase-ImgFeat-Ext --agent initial --num-envs 1 --viewer viser
```

Setup delegates conversion and checksum verification to ORCS. It reads
ORCS's `hoiretarget` staging extra, stores native output at
`data/hoi_motions/unitree_g1`, and removes temporary source downloads. `all`
includes this mode. `--check` installs nothing and downloads nothing.
`VIBE_DATA_ROOT` relocates the data through ViBe's normal root binding;
explicit `ORCS_*` overrides retain their existing precedence.

An existing verified native export may be deliberately linked at that path
for development; verify it with the same check command. Keep it separate
from `data/retargeted_motions` so the original UOLM roster stays unchanged.
All 63 stable clip IDs, companion files, contact legends, source attribution
and provenance must survive staging. Missing data skips the new row; stale
or invalid native data raises a diagnostic. See [notices](../THIRD_PARTY_NOTICES.md).

The Viser motion selector exposes the full repertoire. `--agent initial`
constructs the visual agent with a zero-initialized adapter, so its actions
are the frozen SONIC base. There is no released suitcase visual checkpoint.
Train fresh visual adaptation; the privileged ORCS adapter is incompatible
with the visual actor's inputs and is an evaluation control only.

The camera-compatible scene retains the full physical object-variant model
and assigns every world to suitcase variant 0. The command roster remains
one suitcase and 63 clips, including for the critic and benchmark. A reduced
variant model fails in the pinned mujoco_warp policy renderer. The wrapper
changes neither the suitcase asset nor ORCS's task mechanics.

## Training recipe

After checking capacity on the deployment GPU:

```bash
WANDB_USERNAME=lkrajan train Vibe-Uolm-Suitcase-ImgFeat-Ext \
  --env.scene.num-envs 4096 \
  --env.commands.motion.init-phase-anneal-iterations 20000 \
  --env.commands.motion.start-from-zero False \
  --env.events.virtual-object-force.params.natural-frequency 0.0 \
  --agent.seed 42 --agent.max-iterations 21000 --agent.save-interval 1500 \
  --agent.logger wandb --agent.wandb-project vibe
```

This is one uninterrupted run with the default sampler. New episodes start
at phase zero during the final 1,000 updates. Setting annealing to zero
disables the schedule; it does not force phase-zero starts. A zero VOF
natural frequency disables both force and torque even while the logged
curriculum multiplier continues its schedule. Runs live under
`logs/rsl_rl/vibe_uolm_suitcase/<timestamp>/`.

Keep actor/critic fp32 on RTX 3090; encoder compute dtype follows the existing
ViBe recipe. 4096 worlds is a capacity target, to be measured on the job host.

The separate adaptive experiment adds:

```bash
  --env.commands.motion.sampling-mode adaptive \
  --env.commands.motion.adaptive-kernel-size 3 \
  --env.commands.motion.adaptive-start-from-zero-ratio 0.25
```

Adaptive statistics are runtime-only and reset on process restart. Clip
weighting continues after phase reaches zero. Keep comparisons with the
default-sampler baseline explicit.

## Evaluation and export

```bash
orcs-bench-motion-repertoire Vibe-Uolm-Suitcase-ImgFeat-Ext \
  --checkpoint-file /path/to/model_20999.pt \
  --num-envs 4096 --trials-per-motion 65 --seed 20261106 --no-serve
export-agent Vibe-Uolm-Suitcase-ImgFeat-Ext --checkpoint-file /path/to/model_20999.pt
```

The benchmark reuses ORCS's collector and HTML report: 4,095 phase-zero
trials, no assistance, deterministic inference, native failures and final
hold. Default evaluation retains training randomization; `--nominal` also
cleans ViBe's query noise and render domain. Fix seed and batching within
comparisons. This measures training-repertoire competence, not held-out
motion generalization.

Export pins `omomo/sub1_suitcase_010/sample1` in both worlds, narrowing both
include and exclude selections on the export config. Training keeps all 63
clips. Deployment parity requires a trained visual checkpoint to pass the
actual two-world check; initial-policy checks establish plumbing only.
