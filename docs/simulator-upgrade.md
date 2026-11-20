# Simulator candidate

`feat/mjlab-upgrade` is a local review candidate. Dependency commits must be
published before a fresh remote clone can fetch its final ORCS pin. No full
training result or measured training speedup is established by the smoke checks.

## Source and installation

| Component | Candidate |
|---|---|
| Python | 3.12.12 |
| mjlab | 1.6.1, upstream `11ffee9816e1d0644e538854bfb8dd8acb82ebba` plus ORCS's reviewed patch (`1.6.1+vibe.1`) |
| MuJoCo | `mujoco-core[rendering]==4.0.1` |
| mujoco-warp | main snapshot `e0226800bba7cc09f252f6c8e1a5d922c422f4de`, reporting 4.0.1 |
| Warp | 1.18.0 |
| Optional OptiX bindings | `warp-optix==0.3.0`, PyOptiX 9.1.1 |
| rsl_rl | ViBe fork `215b6f4ceda569d2ca39df64dd679b9af15375cb`, unchanged |
| Torch / torchvision | 2.15.0 / 0.29.2, retained from the baseline environment |

The main snapshot includes [OptiX support](https://github.com/google-deepmind/mujoco_warp/pull/1784).
Its version string alone does not identify that code: `deps.lock` owns the SHA.
The mjlab patch adapts geometry/site poses to the backend's `wxyz` quaternions,
camera backend selection, graph capture boundaries, viewer transforms and
dependency metadata. ORCS owns the patch; ViBe references it through its lock.
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
