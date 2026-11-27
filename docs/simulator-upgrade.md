# Simulator candidate

`feat/mjlab-upgrade` is a review candidate. Both ViBe and ORCS dependency commits
are available on their corresponding remote branches. No full training result
or measured training speedup is established by the smoke checks.

## Source and installation

| Component | Candidate |
|---|---|
| Python | 3.12.12 |
| mjlab | main `d872bfd2b048da657345c07669172fa284b38abf` plus ORCS's reviewed patch (`1.6.0+vibe.20261126`) |
| MuJoCo | `mujoco-core[rendering]==4.0.1` |
| mujoco-warp | main snapshot `8f5b9f116bb26ee2f629ce483511f50849ca3df3`, reporting 4.0.1 |
| Warp | 1.19.0 |
| Optional OptiX bindings | `warp-optix==0.3.0`, PyOptiX 9.1.1 |
| rsl_rl | ViBe fork `215b6f4ceda569d2ca39df64dd679b9af15375cb`, unchanged |
| Torch / torchvision | 2.15.0 / 0.29.2, retained from the baseline environment |

The main snapshot includes [OptiX support](https://github.com/google-deepmind/mujoco_warp/pull/1784).
Its version string alone does not identify that code: `deps.lock` owns the SHA.
The mjlab patch adapts geometry/site poses to the backend's `wxyz` quaternions,
camera backend selection, graph capture boundaries, viewer transforms and
dependency metadata. ORCS owns the patch; ViBe references it through its lock.

ORCS also carries `patches/mujoco-warp-optix.patch`: it exposes a rendering-only
world extent so mjlab can include its translated environment origins in OptiX's
world cells. The extent includes the largest origin coordinate plus MJWarp's
usual local motion margin (`max(4 * model.stat.extent, 1)`). This leaves physics
coordinates and camera clipping unchanged. Bounded geometry must stay inside
that region; changing terrain origins after context creation requires recreating
the sensor context. Warp does not use this bound.
There is no separately published mjlab fork or package release.

In a fresh checkout, one command installs the complete code environment:

```bash
bash scripts/setup/sync_deps.sh --deploy --optix
source .venv/bin/activate
bash scripts/setup/sync_data.sh
```

Omit `--optix` for Warp only; omit `--deploy` if export is unnecessary. Code
setup creates `.venv`, applies the exact source patch, and resolves all local
editables and extras together. Re-running it verifies source revisions, patch
contents, editable origins and dependency compatibility. It refuses unrelated
active environments, wrong-version existing environments and unreviewed patch
edits. Keep the old checkout/environment for comparison. `requirements.lock`
freezes the tested Linux/Python 3.12.12 registry packages, including optional
extras, while `deps.lock` owns the editable Git sources. Refresh package
constraints from a clean resolved environment when intentionally upgrading;
exclude investigation-only tooling. Installing without these constraints is
not the documented candidate environment.

## G1 and task compatibility

The robot retains its pre-upgrade mechanical configuration, including the absence
of the eight coupled ankle stops added to upstream G1. ORCS owns this preservation;
see [G1 compatibility](https://github.com/lok-i/orcs/blob/32cab22cd17047946e1d1a28a170bab571ad2ebc/docs/g1-compatibility.md) for the exact
scope, upstream reference and compiled-model regression guard. The patch also
preserves the original world-frame body velocity rewards under their saved-config
import paths. See [API migration](simulator-api-migration.md) for `SetConstGroup`,
manual event replay and cache behavior. Solver and reset fixes can still change
trajectories; configuration preservation is not checkpoint parity.

The installed copy of that document is at
`dependencies/orcs/docs/g1-compatibility.md` in the prepared checkout.

## Deterministic execution

Physics determinism defaults off. Enable it before environment construction:

```bash
train Vibe-Repose-BigCubeFloor-ImgFeat-Ext --env.sim.deterministic True
play Vibe-Repose-BigCubeFloor-ImgFeat-Ext --agent release --viewer native \
  --num-envs 1 --env.sim.deterministic True --env.seed 42
```

`--agent release` loads the original released weights. It does not imply those
weights were trained on this stack. Play retains its usual clean render domain;
training randomization is exercised with training configurations in the tests.
The physics flag is recorded in `params/env.yaml` and the logger's environment
configuration. Tracking evaluation and the repeatability harness accept the same
`--env.sim.deterministic True` spelling.

Repeatability is scoped to the same GPU model, environment count, software,
precision/compile settings and seeds. It is not cross-GPU or cross-version parity.
PyTorch policy/encoder/learning repeatability requires separate validation. Buffer
exhaustion invalidates the claim; backend warnings remain enabled. Solver and
line-search iteration limits are reported separately from capacity overflows.

## Rendering

Warp remains the default for every task. Training selects OptiX using
`--env.render-backend optix`; play/export use `VIBE_RENDER_BACKEND=optix`.
The environment variable takes precedence when set. The resolved choice is
recorded in the environment configuration, and initialization reports the actual
backend. Failure to create the requested backend is an error, never a silent fallback.

PerLoco supports **Warp only**, preserving its privileged ray scanner. OptiX
cannot share the current sensor context with that scanner. No sensors are removed.
Physics CUDA graphs remain enabled with either renderer; OptiX sensing runs
outside capture on the same device stream. Warp sensing remains captured.

OptiX needs an RTX GPU, Python 3.12 and a driver compatible with the pinned
bindings. [NVIDIA requires R590+ for OptiX 9.1](https://developer.nvidia.com/designworks/optix/download).
The local R580 driver rejects those bindings with `Unsupported ABI version`.
Actual OptiX images, object-variant updates and speed therefore still require
validation on a compatible host. The build does not change drivers.

The two renderers need not produce pixel-identical frames. The new mjlab also
uses a black fallback background; the sampled ViBe models have no skybox.
Treat camera distribution changes as a new training condition, while keeping
released-checkpoint provenance in `release.json` intact.

## Native visual randomization

| Term | Behavior |
|---|---|
| Camera mount | Existing startup wrapper now delegates to native `cam_pos` / `cam_quat`; same ±2 cm / ±0.035 rad ranges and import path |
| Light direction | Existing native term retained |
| Floor/terrain palettes and cube-face labels | Existing task-aware implementations retained |
| `--env.visual-dr.lighting True` | Opt-in startup ambient/diffuse scale 0.8–1.2 |
| `--env.visual-dr.floor-material True` | Opt-in plane-floor specular 0–0.15 and shininess 0.1–0.5 |
| `--env.visual-dr.floor-textures True` | Opt-in plane-floor flat/checker grayscale textures, native `mat_texid` selection |

Additional options default off and reject play configurations. Floor material
and texture options support plane terrain (Repose/Uolm), and reject rooms and
generated PerLoco terrain. Textures never attach to manipulated objects or
Repose's semantic cube faces. Native RNG consumption may differ from the prior
custom camera function; identical seeds do not imply identical rollouts.

## Repeatability and capacity checks

Use separate processes, with training noise and events enabled. At these large
counts the pinned snapshot needs an explicit dense Jacobian for deterministic
execution; the default sparse workspace limitation is described below:

```bash
.venv/bin/python scripts/repeat_check.py --task Vibe-Repose-BigCubeFloor-ImgFeat-Ext \
  --num-envs 4096 --jacobian dense --steps 1000 --output logs/repose-repeat
.venv/bin/python scripts/repeat_check.py --task Vibe-Dodge-ConeFast-ImgFeat-Ext \
  --num-envs 2048 --jacobian dense --mode ppo --iterations 100 --output logs/dodge-ppo-repeat
```

`--graphs off` disables graph capture, not just conditional solver nodes.
`--deterministic False` is a comparison condition; it is allowed to repeat by
chance. `--jacobian dense` explicitly selects a different solver layout for
qualification; it is recorded and never enabled implicitly. Small environment
counts and shorter iterations are useful local smokes but do not qualify the
production sizes.

Each worker writes hashes, resolved configs and a summary under its output
folder. Comparisons report the first differing step/iteration and field. The
rollout checks all observation groups, actions, physics state, rewards and done
flags, including a partial reset. Capacity monitoring samples physics substeps;
contact capacity is aggregate across worlds. PPO checks include model, optimizer
and Torch RNG state in addition to non-timing scalars. Outputs remain local.

`--mode benchmark` runs one process without hashing or buffer-monitoring hooks.
It reports collection/learning times after warmup and a separate physics-step
measurement. GPU memory is a sampled device-wide high-water mark, including Warp
allocations, plus Torch's allocator peak. Compare identical task counts and
Jacobian layouts with determinism off/on; record the original layout separately.

MuJoCo Warp's `auto` Jacobian selects sparse storage above 32 velocities. The
pinned Warp deterministic sparse-Hessian path can exceed its signed-32-bit
workspace-array limit at large batches during graph capture. Treat this as a
failed configuration, not a renderer fallback or a passing small smoke. Dense
layout is an explicit diagnostic alternative requiring its own qualification;
it is not a promise of identical trajectories to sparse solving.

## Local checks and subsequent training

```bash
.venv/bin/ruff check src tests
.venv/bin/python -m pytest tests/
bash scripts/setup/sync_deps.sh --check
bash scripts/setup/sync_data.sh --check
.venv/bin/python scripts/smoke_simulator.py --num-envs 6 --steps 32
.venv/bin/python scripts/smoke_simulator.py --mode train --agent initial --visual-dr --num-envs 6
.venv/bin/python scripts/smoke_simulator.py --mode train --agent initial --learn --num-envs 4 --steps 2
```

The harness checks all six task configurations, partial resets, finite outputs,
camera images, graph use and PerLoco scanners. At six environments it also
checks every Uolm object variant. It runs inference by default. `--learn` adds
exactly one PPO iteration with eight rollout steps, one epoch/minibatch, and
no logger or checkpoint saves; it checks finite, updated actor parameters.
Use `--backend both` on a compatible host; PerLoco OptiX is explicitly unsupported.
Use `--agent release` with the five released task IDs (SmallCubeTable has no release).
Detailed local results and first-failure logs belong under `logs/`.

For the first training comparison, retain the default Warp backend and visual
DR settings, and use the [existing task training commands](tasks.md#train).
For example, after review:

```bash
train Vibe-Repose-BigCubeFloor-ImgFeat-Ext --env.render-backend warp \
  --env.scene.num-envs 4096 --agent.max-iterations 60000 \
  --env.commands.motion.init-phase-anneal-iterations 50000
```

Measure collection and learning time separately at the target environment count
before estimating total savings. Add native visual DR and OptiX as separate
comparisons so speed and learning changes can be attributed. Small local smokes
do not establish reward parity, checkpoint equivalence or production throughput.

## Replacing the main snapshot with a release

Advance the source SHA and matching core/Warp pins together in an isolated
checkout. Reconcile each ORCS patch hunk against the new API and remove only
upstreamed equivalents. Retain the rsl_rl fork, task contracts and checkpoint
provenance. Repeat setup, train/play/reset, object-variant, released-policy and
export checks on each supported renderer before publishing the dependency
commit and repinning ViBe. Do not turn a moving `main` branch into the lock.
