# Capturing a suitcase PPO instability

The opt-in recorder runs the existing PPO implementation and retains enough state
to replay one optimizer update without simulation or rendering. It supports this
development branch's single-GPU, feedforward, FP32 `Vibe-Uolm-Suitcase-ImgFeat-Ext`
agent. Ordinary `train` commands do not import or enable it.

## Bounded continuation

The experiment resumes `model_6000.pt` for at most 1,000 additional updates,
keeping the original 4,096 environments, spacing 2.5, seed 42, 20,000-update phase
anneal, VOF frequency zero, rank/alpha 28/28, camera, critic and PPO settings.
It keeps the failed control's numerical packages, listed in
[`suitcase_runtime.json`](../scripts/diagnostics/suitcase_runtime.json). The
preflight rejects package drift; it never installs or downgrades anything.
ViBe's backend patch enables `torch.set_float32_matmul_precision("high")`.
Actual backend flags, relevant environment overrides, versions and
source hashes are recorded in the capture.

The experiment's explicit differences are instrumentation, a fresh W&B run in
`lkrajan/vibe` / group `suitcase-numerics`, local checkpoints every 100 updates,
and no checkpoint uploads. The disabled VOF alpha is suppressed in this runner's
logger and replaced by `VirtualObjectForce/enabled=0`; the ORCS event and its
zero-wrench behavior are unchanged.

On resume, Adam moments, normalizers and model weights load normally. The runner
also copies Adam's restored rate into PPO's separate adaptive-LR variable and
refreshes the curriculum counter before resetting episodes. This prevents the
first episode reset from using phase-counter zero. The checkpoint does **not**
contain simulation or RNG state, so this continuation cannot recreate the
historical rollout exactly. Passing the historical failure iteration is not
proof that the failure is fixed.
The upstream runner repeats the saved update label: this job logs 6000–6999,
while the restored environment counter starts at 144024 (6001 completed rollouts).

From the repository root, inspect configuration without a checkpoint or GPU:

```bash
.venv/bin/python scripts/diagnostics/suitcase_numerics.py check --config-only
```

On the intended host, preflight checks checkpoint tensors (including Adam),
iteration/counter consistency, the development branch, a clean source checkout,
dependency/data pins and numerical package versions:

```bash
.venv/bin/python scripts/diagnostics/suitcase_numerics.py check
```

After deployment is approved, the proposed launch from `~/drcl_projects/vibe` is:

```bash
~/.local/bin/tmux new-session -s vibe-suitcase-numerics \
  'cd ~/drcl_projects/vibe && bash scripts/diagnostics/run_suitcase_capture.sh'
```

The wrapper sends SIGTERM at 90 minutes, including startup, and allows up to 60
seconds to save diagnostics before SIGKILL. The recorder also checks its time
budget during collection/update. `--updates` and `--minutes` can reduce the
budget; they cannot exceed 1,000 and 90. No script pulls code or installs packages.

## Capture and interpretation

Each run writes `numerics/` under its normal training directory:

- `resume.json`: source checkpoint, restored Adam/PPO rate and curriculum count.
- `updates.jsonl`: completed updates' per-minibatch loss, value/target extremes,
  gradient norms, largest-residual environment/step, normalized-input extremes,
  and realized parameter delta.
- `capture.pt`: the last pre-update actor, critic, normalizers, Adam, learning
  rate, full rollout (including visual features and cached SONIC tokens), RNG
  states and actual permutation. Also includes the pre-rollout model/RNG state,
  collection diagnostics, and the observed state at stop.
- `stop.json`: stop reason, stage and diagnostic measurements, readable without
  loading tensors.

Completed updates also publish `Numerics/value_loss_max`, actor/critic gradient
norm maxima and parameter-delta maxima to W&B. Stop artifacts include the actual
observed gradients and Adam-state extrema. A diagnostic stop exits nonzero and
records its reason in the W&B summary; it is not reported as completed training.

One pre-update rollout is kept in host RAM and replaced on the next rollout.
It is serialized only on failure or at the budget limit. Full rollout captures
are large (several GB); the runner checks available host RAM and disk. Recorder
reductions and CPU copies add overhead; this is not a throughput benchmark.

The finite-value trigger is **per minibatch**: value loss above 100, or above
50 times the median of the last 100 completed minibatches once 20 are available.
These are diagnostic tripwires, not established numerical stability bounds.
The hook inspects the actual upstream loss **after backward, before clipping or
Adam**. A trigger prevents that minibatch's parameter/moment update; preceding
minibatches may already have updated. Nonfinite actions are stopped before
`env.step`; observations, normalizers, returns, losses, gradients, clipping norms
and post-Adam parameters/state are also checked.

The recorder reads loss and sample-index locals at the pinned PPO diagnostic
hook. It does not copy the training loss or change clipping/optimization. Its
contract is covered by actual-update tests, and replay rejects differing rsl_rl
source hashes. This deliberately narrow tool is not compatible with arbitrary
future PPO implementations.

Collection/return failures save a partial rollout with `update_ready=False`.
Those artifacts help locate the failing input or normalizer but cannot be used
for optimizer replay. No artifact promises a full simulator restart. SIGKILL,
a device crash, insufficient disk or a stuck device can prevent capture.

## Paired update replay

Use separate processes and output directories. Both arms start from the same
captured model, Adam, normalizers, old log-probabilities, targets, RNG state and
explicit minibatch permutation. Changing TF32 affects the replayed learner;
the already-collected features and targets stay fixed. Replay uses ViBe's same
unified matmul API: `high` for TF32 permitted, `highest` for IEEE FP32. The
captured cuDNN settings remain fixed; this learner has no convolutional encoder
because the features are already captured. The original random
permutation draw is consumed before substituting the recorded permutation so
subsequent policy sampling retains its RNG position on the same runtime.

```bash
.venv/bin/python scripts/diagnostics/suitcase_numerics.py replay \
  /path/to/numerics/capture.pt --precision tf32 --output logs/numerics/replay_tf32
.venv/bin/python scripts/diagnostics/suitcase_numerics.py replay \
  /path/to/numerics/capture.pt --precision ieee --output logs/numerics/replay_ieee
```

Each arm applies the same tripwires and writes its own capture/report. Compare
the first differing minibatch, values, gradients, Adam state and parameter delta.
A stable replay does not exclude an error during collection, normalization or
return computation. Cross-version/device replay is an experiment, not a claim of
bitwise reproducibility. Source code is held fixed; runtime differences are
recorded. CPU tests establish recorder transparency and replay correctness for
their tested inputs; GPU/full-rollout validation is separate.
Repeat the control replay in another fresh output directory first: if repeated
controls differ, measure that variation before attributing differences to TF32.
