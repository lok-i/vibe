# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

`vibe` is an mjlab project: **visual behavior adaptation** for whole-body control on a Unitree
**G1** humanoid. It owns everything that makes a task a VISION problem — the frozen encoder, the
extractor, PPOAux, the color task channel, the render domain — and applies it to FOUR task
families. Each is the twin of an orcs task with **exactly ONE obs group swapped**:

| family | privileged twin in orcs | what vibe swaps |
|---|---|---|
| `repose` — colored-cube reorientation (`BigCubeFloor`, `SmallCubeTable`) | `orcs.tasks.uolm` (single-object special case) | object kinematics → image |
| `perloco` — perceptive locomotion over staged terrain (`Grail`, `OmRe`) | `orcs.tasks.perloco` | 187-ray height scan → image |
| `uolm` — the six-object loco-manip roster itself | `orcs.tasks.uolm` | object kinematics **+ `object_id`** → image |
| `dodge` — whole-body evasion of a thrown ball | `orcs.tasks.dodge` | ball kinematics → image |

**`dodge` is the one whose reference does not perform the task.** The other three track a demo
clip and the adapter corrects it; dodge's reference is a NOMINAL STAND, constant for every frame
of every episode, so the evasion exists only as the adapter's departure from it — which is what
makes behavior drift measurable rather than merely claimed.

**Dependency is strictly one-directional: vibe imports FROM orcs, never the reverse.** Motion
commands, contact schedules, VOF curriculum, phase annealing, tracking rewards/terminations,
terrain staging, the robustness domain and the privileged obs groups are all orcs's. **A
duplicate is a bug.** Same rule one level down: anything task-agnostic (encoder, camera, runner
spine, extractor swap) lives in `vibe.core`, and a task imports it.

The scene object entity is named **"object"** everywhere. Tasks are auto-discovered at import.

**Nomenclature.**

| | is | speaks |
|---|---|---|
| **sys0** | frozen SONIC base + trained adapter — the deployable low-level block | obs → actions |
| **sys1** | the motion/command generator above it (SUGAR-style) | **robot-state language ONLY** |

sys1 carries no object language — object refs are critic-only and thrown away post-training,
because multi-object voids any object-language interface. Its stream enters the `augmentation`
group as feedforward obs: `bodywise_contact_cmd` (12 per-body flags, where SUGAR has 1 scalar)
+ `robot_root_{lin,ang}_vel_cmd` (ref-anchor twist in the anchor's own frame, so sys1 can emit
it open-loop). `object_goal_*` is feedforward too but comes from the task level *above* sys1,
fixed per episode.

## SOP

### building — branch, build, test, bump

Due diligence before a run, in this order because 4 is what a training HOST reads:

1. **branch and build, unless specified not to** — vibe, and every dependency you touch. `dependencies/*` ship detached at the
   locked SHA, so a commit there is dangling until you name it.
2. **do unit and smoke tests** —  module-level, ensuring implementaion consistency
3. **ask for user test**  - system-level, behavior analysis by the uer. Provide usage exampled and await approaval / feedback to continue
4. **push the dependency, THEN bump `deps.lock`** — never the reverse. A lock pointing at an
   unpushed SHA is a BROKEN lock, not a pending chore.

**Why (2026-08-17, cost 11 GPU-hours):** the lock was bumped to an rsl_rl SHA that existed
only in the local checkout. The editable install made it live here; the host silently kept the
old extractor, and new-vibe + old-rsl_rl is a VALID combination — the two K/V frames were
averaged instead of distinguished, with nothing raised. Caught only by a missing metric key,
which is why a layout change should ship a new key rather than a silent behaviour change.

## docs — the library

**Rule: no analysis conclusion lives in this file.** A number that came out of a run belongs in
its doc. The analysis notebooks and their conclusion docs live in the dev repo, not here.

| doc | what |
|---|---|
| `infra/naming.md` | task-id grammar — **the authority** on every token |
| `infra/setup.md` | setup sharp bits — the HF fetch failure, LFS cache, which pip, lock drift |
| `infra/deps.md` | the four-repo hierarchy, who owns which question, the lock rule |
| `infra/agents.md` | agent + obs-group diagrams, the wiring contract |
| `perception/encoders.md` | vision stack: why it looks like this, the built extractor, the roadmap |
| `perception/metrics.md` | every `Z*` metric, train-time and offline — **the watch list** |
| `perception/render_domain.md` | what the policy's camera sees, and the 3 walls the renderer imposes |
| `usage/export_onnx.md` | the ONNX deploy path, the two-world test, the per-task export case |

## Setup

```bash
bash scripts/setup/sync_deps.sh               # .venv + vibe + deps.lock rows under dependencies/
bash scripts/setup/sync_data.sh [MODE...]     # deps.lock rows under data/ + generated + staged; all|inhouse|omre|grail
bash scripts/setup/sync_{deps,data}.sh --check  # verify pass alone — no network, no pip
```

Python **3.11**, in a **uv venv** — the active one, else `./.venv`, which `sync_deps.sh`
creates; both scripts export it for themselves, so neither needs activation. uv-only: a
`VIRTUAL_ENV` without a `pyvenv.cfg` (a conda prefix an editor injected) is warned about and ignored.
Shared helpers: `scripts/setup/_common.sh`. `retargeted_motions` is `"sparse"` — only the
clips a task ROSTERS are fetched, globs derived by `vibe.core.paths --lfs-include`.
`reconstructed_motions` and SMPL-X are deliberately absent: only orcs's `-Smpl` tasks read
them. Failure modes: **`docs/infra/setup.md`**.

- **`deps.lock` is the pin, always.** Five non-PyPI deps, each at a SHA; bump = edit `sha` and
  re-run (idempotent). The sync ends with a **verify pass** — every worktree HEAD against its
  pin, non-zero exit on drift. **Never copy a SHA or a branch name into this file.**
- **Diamond dependency — read before bumping.** vibe and orcs both pin `assets` /
  `retargeted_motions` / `mocke` / `rsl_rl`, and pip gives ONE editable install per env, so
  **vibe's lock wins** and orcs's is advisory. `orcs.core.deps` prints the live-vs-validated diff
  at import; **that line is the source of truth for which row drifts, not a table here.** Believe
  it especially for `mocke` — it carries the obs/action contract the ported SONIC checkpoints are
  bit-coupled to.
  **Unmerged work to remember:** the `assets` commits orcs used to pin (`6822f3b`, `46511a8` —
  optimized tire/woodchair2/largetable convex decompositions, ~11x narrowphase on those objects)
  are NOT in the pinned SHA. They live on the assets remote and want merging there; until then
  `Orcs-Uolm-*` trains without them.
- **Data roots when orcs is vendored — ONE rule, and it is orcs's.** `<host>/dependencies/orcs`
  is code-only, so orcs **disqualifies its own root** for `data`/`dependencies` when nested —
  "the consumer's lock wins", applied to datasets. Both sides read the override from ONE
  definition, `vibe.core.paths.ORCS_ENV` (`bind_orcs()` at runtime, `python -m vibe.core.paths
  --env` in shell). **Do NOT re-derive a root anywhere else** — paths never assert, so a wrong
  one surfaces only as a task that stopped registering.
- `data/` and `dependencies/` are gitignored, materialized from `deps.lock`. Don't commit them.

## Common commands

```bash
# Registered task IDs
python -c "import mjlab, vibe; from mjlab.tasks.registry import list_tasks; print('\n'.join(sorted(list_tasks())))"

train <task-id> --env.scene.num-envs 4096
play  <task-id> --agent initial --viewer native   # initial = frozen base, no ckpt; also zero|random|trained

ruff check src [--fix]
pytest tests/                                     # contracts, no GPU, no checkpoint, ~5 s
export-agent <task-id> --check                    # ONNX export + two-world diff
```

`tests/` follows orcs's rule — **pytest tests CONTRACTS, a real run tests behavior.** Five
files: `test_export_cases.py` (every task declares an ONNX export episode still live on disk),
`test_repose_scenes.py` (the repose roster and its two physical scenes),
`test_dodge_cone_fast.py` (the ConeFast room + throw cfg), `test_vision_knobs.py`
(`--env.img-encoder` rebinds both image terms), `test_stage_render.py` (every task films on the
SAME floor, and that colour clears the separability bar). Behavior
stays where it was: `play <task> --agent initial` and `export-agent <task> --check`.

## Tasks

Grammar and every token: **`docs/infra/naming.md`**. The rationale for a given task — why that LoRA
rank, which group left the augmentation, what its twin controls for, which render departures are
deliberate — lives in its registration docstring, `src/vibe/<family>/config/g1/__init__.py`.
Neither is repeated here.

| task | what it is |
|---|---|
| `Vibe-Repose-BigCubeFloor-ObjKin` | privileged object state — the ImgFeat row's twin |
| `Vibe-Repose-BigCubeFloor-ImgFeat{,-Ext,-Sfd,-Lfd}` | vision; the four rows below — a no-extractor floor plus three extractor families |
| `Vibe-Repose-BigCubeFloor-ImgRgb` | the vision-**stack** baseline — trainable CNN over pixels where `-ImgFeat-Ext` has a frozen encoder + attention pool, both at z128+LN. Tests what a frozen trunk buys (a finetuning recipe you keep, visual robustness), **not reward**. 84.7 KB/env/step vs 16.1: smaller `num_envs`, and match the reference to it |
| `Vibe-Repose-SmallCubeTable-ImgFeat-Ext` | the 0.36 m primitive cube on a per-clip table |
| `Vibe-PerLoco-{Grail,OmRe}-ImgFeat-Ext` | height scan → image. Two sources share ONE obs layout and ONE agent cfg; only the orcs factory differs. Needs `sync_data.sh omre|grail` — absent, it SKIPS rather than raising |
| `Vibe-Uolm-ImgFeat-Ext` | the six-object roster through a camera |
| `Vibe-Dodge-ImgFeat-Ext` | a thrown ball through a camera, on a plane |
| `Vibe-Dodge-ConeFast-ImgFeat-Ext` | same ball in an indoor ROOM (`dodge/terrain.py`: floor + 4 walls + ceiling, own light rig), at a shorter throw interval and a longer release distance (`DODGE_CONE_FAST_THROW`) and a 64-px camera. The room is dodge's alone — every other family stands on a plane or on orcs's generated terrain |
| `Orcs-*`, `Mocke-*` | the privileged twins and the pure-WBC sandboxes — registered by the **dependency**, not vibe. From-scratch (`-TaRa`) and sidecar agents live there, never here |

Two invariants a reader needs before running one: **the critic stays privileged in every vision
row** (it keeps the scan / object state + id / ball state), so a delta against the twin is the
cost of seeing through a camera and nothing else; and **both repose vision rows are hardwired to
the COLOR task channel**, which is what makes them the same TASK by construction.

## Architecture

mjlab is **manager-based**: an env is a `ManagerBasedRlEnvCfg` holding dicts of
`{Observation,Reward,Event,Termination,Command}TermCfg`, each pointing at an **mdp function**.
Building a task = compose these dicts.

**The seam with orcs:** every signal a deployable robot could NOT sense is orcs's; everything
downstream of a camera is vibe's. One shape, thinner each time it is applied:

| family | orcs owns | vibe adds |
|---|---|---|
| **perloco**, **uolm**, **dodge** | *everything* — the `env_cfg.py` factories, `mdp/`, `robustness.py`, `sensors.py`, `observation_cfgs.py` (the signal ATOMS + both groups), `orcs.core.rl`'s agent spine | three files, no mdp / no command / no `wbc.py` (dodge adds a fourth, `terrain.py` — the ConeFast room): `env_cfgs.py` (call the orcs factory, attach the head cam, swap ONE group) · `observation_cfgs.py` (that group MINUS the privileged terms, PLUS `kv_tokens` + query rows) · `agent_cfgs.py` (`vibe.core.rl`'s actor + `attach_extractor`, at orcs's LoRA numbers) |
| **repose** | `orcs.tasks.uolm` — `ObjectMotionCommand` (multi-clip loader, object RSI, phase annealing), contact schedule, demo loader, VOF curriculum, events, rewards, terminations, `robustness.py`, the obs ATOMS | the CUBE and task layer: `assets/repose.py` (cube + table specs), `mdp/{commands,cube_faces,rewards,events,metrics,sys1}.py`, `config/g1/{sensors,wbc}.py` (repose assembles its own frozen base), the color-relabel render domain |

The `critic` group is **imported verbatim** everywhere — actor vision-only, value function
privileged. `vibe.tasks.uolm`'s twin is the ORCS task, so it subtracts from orcs's default
augmentation layout; `vibe.tasks.repose` instead pins `augmentation_group(frame="base", identity=False)`,
because its ObjKin row is the privileged twin and must differ in exactly one thing.

**The play invariant is INVERTED for the thin-import families.** orcs runs its play overrides
*inside* its own factory — before vibe has added anything — so a domain knob added afterwards
silently trains the eval. `assert_play_is_clean(cfg, before)` DIFFs the domain against a pre-vibe
snapshot on every play build, failing at cfg-build time instead of quietly reporting optimistic
numbers. (A name list was tried; it false-positived on uolm's `rand_encoder_bias`, a JOINT
encoder.) perloco also forces the height-scan `debug_vis` OFF — the sensor stays for the critic,
but drawing rays this actor cannot see would show a signal the policy does not have.

**The play cfg is also the FILM cfg**, which is the other half of the same seam:
`apply_stage_render(cfg, play=play)` pins every task's ground to ONE colour
(`STAGE_RGBA`, cool slate — ONE constant, every task reads it) so four families collage as one
shoot instead of four rooms —
play-only, called AFTER the guard (a fixed colour is not a domain), and it POPS
`rand_terrain_color` rather than sitting beside it, because repose keeps that event alive under
play on purpose. perloco passes its `ground_rgba` so the CURBS keep their level-tracking hue;
plane tasks and dodge's room paint everything. Why that colour — **measure the RENDER, and measure TWO
things**: the robot against the floor AND the floor against its own cast shadow. They pull
opposite ways (a light floor meets the silver — 0.76 measured a robot ΔL of ZERO; matte black
wins the silhouette and deletes the shadow), and a nominal rgb-distance bar sees neither failure:
`docs/perception/render_domain.md` §5.

### `src/vibe/core/` — the task-agnostic layer

Peer of `orcs.core`. A task imports FROM core, never the reverse, and nothing in core names a
cube or a terrain. The four families live under **`src/vibe/tasks/<family>/`** (folded there
2026-08-25); `vibe.core`, a peer rather than a task, did not move.

| file | holds |
|---|---|
| `core/mdp/observations.py` | `image_feature` — THE frozen encoder. Atomic obs TERMS only |
| `core/mdp/events.py` | `rand_terrain_color`, `set_terrain_color`, `rand_cam_extrinsics` — the render domain, and the play-only STAGE |
| `core/mdp/metrics.py` | `ObjectCamProjection` + `object_in_fov` — the vision duty cycle. Needs a camera and an entity, never a cube |
| `core/env_cfgs.py` | `domain_state()` + `assert_play_is_clean()` — the inverted play invariant, as a diff. Also `apply_stage_render()`/`STAGE_RGBA` (above) and `VibeEnvCfg` — the per-run vision knobs (`--env.img-encoder`) |
| `core/observation_cfgs.py` | the vision ATOMS: `IMG_ENCODER`/`IMG_DTYPE`, `TOKEN_GROUP`/`TOKEN_TERMS`/`CLS_GROUP`/`CAMERA_GROUP`, `CamSpec`, `img_{tokens,cls,flat}_term`, `kv_tokens_group`, `cls_query_group`, `camera_group` |
| `core/sensors.py` | `head_cam_cfg()` — ONE head camera for every vibe task, so a resolution change re-bases every attention metric together. **Dodge re-aims it** (`dodge/config/g1/sensors.py`: 2 deg UP, not 45 down — measured, the sweep is in that module); nothing else overrides it |
| `core/rl.py` | `VibeRunnerCfg` (amp/nan/compile), `runner()`, `adapt_sonic_agent_cfg`, `EXTRACTOR_CFGS`, `attach_extractor()` |
| `core/cnn_encoder.py` | `CnnEncoder` — ImgRgb's trainable pixels→z block. Lives here, not in rsl_rl: `_build_extractors` resolves `class_name` by dotted path, so it satisfies rsl_rl's `from_obs`/`latent_dim`/`input_groups` contract with **zero** rsl_rl change |
| `core/runner.py` | `VibeOnPolicyRunner` (absorbs mjlab's `registry_name`) |
| `core/paths.py` · `core/_mjlab_compat.py` · `core/_legacy_paths.py` | roots · the mjlab patches · `vibe.<family>` → `vibe.tasks.<family>` |

Sibling packages: `vibe.encoders` (`zoo.py` — the backbone roster + loaders), `vibe.export`
(`agent` / `encoder` CLIs), `vibe.viz` (viser attention overlay, camera director,
recorder), `vibe.assets` (G1 render variants, repose cube + table specs).

**Naming split, at both levels:** `observations.py` is ALWAYS atomic mdp terms;
`observation_cfgs.py` is ALWAYS `ObservationTermCfg`/`ObservationGroupCfg` builders. Moved
symbols keep a re-export line — a saved env cfg records a term by its DEFINING module path, so
deleting the old name breaks reloading an earlier run's cfg.

**`<task>/config/g1/` is flat, one question per file:** `env_cfgs.py` (factory) ·
`observation_cfgs.py` (obs groups) · `agent_cfgs.py` (actors) · `sensors.py` · `export_case.py`
(the ONNX episode) · `wbc.py` (frozen-base assembly, repose only) · `runner.py` (re-export).
**There is no `_helpers.py`.**

### Registration, and the import-order trap

Importing `vibe` runs `core.paths.bind_orcs()` + `_legacy_paths.install()`, then
`vibe.tasks.{perloco,uolm,dodge,repose}` →
registration; the `[project.entry-points."mjlab.tasks"]` line in `pyproject.toml` is what makes
mjlab import the package at startup. **Adding a task = add a registration call there.** repose
registers directly (its data always exists); the other three go through
`orcs.core.registry.register_all`, which turns a missing dataset into a `SKIP_REASON` entry
rather than an exception — an unstaged terrain checkout must not cost you the repose tasks.

**`import mjlab` BEFORE `import orcs`, in every script.** mjlab runs its entry-point scan as the
LAST line of its own `__init__`, importing every registered task package — consumers of orcs
included:

| outermost import | what happens |
|---|---|
| `mjlab` (this is `train`/`play`) | scan runs once against a clean slate; orcs, mocke, vibe each import fully. **Safe.** |
| `vibe` | pulls mjlab first, same as above. **Safe.** |
| `orcs` | orcs's `__init__` reaches its first mjlab import mid-way; the scan re-enters and asks for `vibe`, which needs a FINISHED orcs. **Every `Vibe-*` task fails to register** — mjlab catches the ImportError and only warns, so the symptom is a traceback on stderr and a silently short `list_tasks()`. |

Not fixable by lazy imports inside orcs — vibe needs the finished package, so deferring one
import just moves the failure to the next module. The real fix is a re-entrancy guard in mjlab's
scan; **until then the rule IS the fix.** isort keeps `mjlab` above `orcs` in vibe's own scripts,
which is why none of them ever tripped it.

## The vision stack — extractor, PPOAux, the four ImgFeat rows

Vision agents consume frozen encoder features (`img_feat`, computed env-side **outside the
autograd graph** — the backbone is untrainable by construction). Backend swaps via ONE constant,
`vibe.core.observation_cfgs.IMG_ENCODER`, resolved once at init; default **Theia-tiny** (verdict:
`docs/perception/encoders.md`). A trainable
**extractor** projects those features to a 128-d LayerNorm'd latent **z** feeding the adapter
stream; `PPOAux` trains it *after* each PPO update, keeping the adaptive-KL schedule blind to
encoder drift.

| row | what it is |
|---|---|
| `-ImgFeat` | vanilla adapter, NO extractor (raw feats + color one-hot into the adapter stream), plain PPO. The "is a 3-module architecture necessary?" floor |
| `-ImgFeat-Ext` | extractor-only baseline: same `CrossAttentionExtractor`, plain PPO, no predictor. Isolates extractor + z128 from any objective |
| `-ImgFeat-Sfd` | **Supervised** Forward Dynamics (`StateFdAux`) — autoregress `(ŝ,z,cond,a)→ŝ'` over K-step windows, seeded with true s, loss masked at resets. Target is the WIDE vector (object state + up-face color + task-reward rates + per-hand contact force) — hence "Supervised", not "State". Default `K=10, autoregress, not start_with_current_step`; flipping all three IS the retired decodability probe |
| `-ImgFeat-Lfd` | **Latent** Forward Dynamics (`LatentFdAux`, SSL, "explore later") — residual transition `z+mlp([z,a])`, prediction-side projector, MSE vs raw EMA target (tau 0.99), K=10 |

Four rows, **three agent families** — `-ImgFeat` is the no-extractor floor. Each FD variant
recurses in its own space (Sfd: the supervised target; Lfd: latent z) — that distinction *is* the
experiment. The actor cfg is byte-identical across the three extractor rows, so the comparison is
controlled.

**Two axes are per-RUN flags, not task ids** (`docs/infra/naming.md`): `--env.img-encoder
<hf-id>` swaps the frozen backbone — ONE flag because the name lives in TWO obs terms
(`kv_tokens.img_tokens`, `q_cls.img_cls`) and both are already tyro-reachable, so setting
one alone loads two backbones and queries the wrong global token; `--agent.drop-query-rows
q_proprio` drops attention rows, leaving the env untouched (the group is still built, only
unread) and `z` at 128-d, so an ablation measures **routing, not capacity**. Both inert
when unset. `image_feature` speaks `theia`/`clip`/`siglip`/`dino`/`resnet`; every stride-16
row lands on the SAME P=21 grid at the 112x63 head cam, so a swap moves `C` alone.

**Metrics live in `docs/perception/metrics.md`, in full. Do not restate them here.** The three
rules that decide whether a plot means anything:

1. `ZAttention`/`ZCapacity` are extractor-owned — the **only** sections comparable across
   `-Ext`/`-Sfd`/`-Lfd`.
2. `ZPrediction` is target-set-specific — **never cross-read it between variants.**
3. The offline eval ports a metric **iff it is a pure function of (weights, data)**.

**Extractor — query-group cross-attention.** One query GROUP → one attention ROW (own `W_q`).
`EXTRACTOR_CFGS` in `vibe.core.rl` holds one `(obs group it reads, cfg)` pair per architecture —
`cross_attention` on `kv_tokens`, `cnn` on `camera` — so **adding a vision stack never touches
`attach_extractor` or rsl_rl.** The sys0 hierarchy it encodes:

```
base       policy stream                  ─► tracks the motion   (frozen)
adapter    z + robot-motion-cmd           ─► corrects the base   (LoRA)
extractor  kv_tokens ⟨queried by⟩ q_*     ─► z                   (trainable)
```

**Query VALUES never enter z** (pure pooling) — command values reach control via the adapter
stream; a query only steers *where to look*. Diagrams: `docs/infra/agents.md`; design history and
roadmap: `docs/perception/encoders.md`.

**What decides whether a query row keeps its slot: a diffuse row IS a mean-pool row**
(`attn ≈ uniform ⟹ attn @ V ≈ mean(V)`), so any two diffuse rows are the same row and duplicate
`proj` params — keep at most ONE, since a global-pool path into z is otherwise inexpressible.
That, not "is the CLS informative", is why `q_cls` stays, and why dodge carries TWO rows: with no
varying task command, `q_task_cmd` would be a constant query.

**Obs-group wiring contract** — four rules, and they are why adding a vision stack touches no
rsl_rl code:

- a trainable block's port is an obs **GROUP**, never a slice
- roles are **explicit cfg** (`token_terms`/`query_groups`), never shape-inferred
- rl_cfg carries group **NAMES** only
- **rsl_rl never changes** — extractors declare `input_groups`, built via `from_obs`. Exactly why
  `CnnEncoder` could be added vibe-side

Every obs group a TASK owns lives in ONE place, `<task>/config/g1/observation_cfgs.py`, importing
robot/object ATOMS from orcs and VISION atoms from `vibe.core.observation_cfgs` rather than
redefining either — so "what proprio means" changes for the privileged and vision tasks together,
and a backbone swap reaches every family together.

### Aux-target design invariants (do not "upgrade" these)

- **`prediction_target` is ego-observable only** (AnyAdapter-style): no world xy, no yaw, object
  state base-frame. World-frame targets are unidentifiable from (image, proprio, action) and add
  irreducible error to the loss. Same rule sets the contact columns to **HANDS ONLY**
  (`sensors.HAND_BODY_NAMES`), not the 12-node contact graph — a knee↔object or foot↔ground force is
  outside the head cam's FOV, so its residual is gradient noise on the shared encoder.
- **`prediction_conditioning` carries no object state** (robot terms only), so the object part of
  the target is image-reachable only. Never feed object state @ t as a per-step predictor input —
  it would dead-reckon and bypass the image. The autoregressive SEED (true s at window start) is
  the sanctioned exception. Goal COLOR is likewise out, which is what makes the task-reward term
  a routing probe for `q_task_cmd`.
- **Adding a target term is ONE edit** — append it to the `prediction_target` group. That group is
  a **dict** group and `StateFdAux` derives one loss slice per obs TERM from it, name and width
  both, so nothing can be silently dropped and the `ZPrediction/<term>` panel maintains itself.
  The obs term's NAME is the panel's name.
- **Dim count is the only loss weight.** No per-term weighting: a 2-dim term that is 3x harder
  than a 15-dim one claims a disproportionate share of the gradient, and every target added
  re-weights every existing one. Audit the budget as `|T| × ZPrediction/<term>`.
- **Contact force is a target, not a binary flag** (`bodywise_saturated_force`): a hard threshold
  manufactures label chatter at the boundary; raw ‖f‖ is spike-and-slab and its impact tail eats
  the normalizer's std. `tanh(‖f‖/f_max)` is the variance-stabilizing middle. `f_max` is a
  **robot** constant, never per-object — normalizing by object mass injects an unobservable (same
  ban as world yaw) and silently reweights the loss per env. 50 N from the G1 arm actuation bound
  `max_q Jᵀ(q)τ_lim` (stance-pull peak 57.4 N, arXiv:2505.06776): the tightest value that leaves
  the whole arm-achievable range unsaturated.

## Sharp bits (do not trip)

- **mjlab's train/play scripts gate on `isinstance(cmd, MotionCommandCfg)`** and force the
  single-file motion path (train → passes `registry_name`; play → raises). Fixed globally by
  `src/vibe/core/_mjlab_compat.py` (applied in `vibe/__init__.py`): a metaclass sentinel makes the
  multi-clip cfgs report as *not* tracking inside those scripts. **The sentinel is ONE global**
  and orcs patches the same target at ITS import, so vibe (which patches last) must exempt the
  UNION — otherwise `play Orcs-*` breaks inside a vibe env. **Take that union from
  `orcs.MULTI_CLIP_CFGS`, never respell it**: the hand-written copy went stale the moment orcs
  grew perloco's `TerrainMotionCommandCfg`. `apply()` is idempotent for the same reason. The same
  shim adds **`--agent initial`** to `play`: the task's real runner/actor with NO checkpoint
  loaded (adapters → frozen base bit-exact, full obs plumbing live — unlike `--agent zero`).
- **A camera on a VARIANT scene silently aliased EVERY entity to body `-1`** (patched 2026-08-04,
  `_patch_variant_scene_indexing`). mjSpec assigns element ids at COMPILE and drops them to -1 on
  the next structural edit; `Scene.__init__` adds sensors AFTER attaching entities, so the head
  cam leaves the scene spec dirty. The plain path recompiles that spec inside `Simulation` and the
  ids come back — but `build_variant_model` compiles a `spec.copy()`, so on omni-object scenes the
  original stayed dirty and `_compute_indexing` read -1. `data.xpos[:, -1]` is the LAST body, so
  the object's pose BECAME the robot's — in the obs, the rewards and the terminations at once,
  with no error anywhere (`object_in_fov` 0.000, mean reward -0.13; after the fix 0.81 and +2.0).
  Hits any camera + variant task. Upstream fix is mjlab's (compile the original, not a copy).
- **Perf** — the rules, measured: **`fps`
  is the wrong objective** (bigger batches raise `Perf/total_fps` while slowing wall-clock to a
  given reward — train at **4096**, rank runs by reward vs `_runtime`); the ONE env-side lever is
  the camera-lean visual set (`G1_VIS_LEAN_CFG`, the train default — out-of-frame visual meshes to
  geom group 4, while `play` gets `G1_BASE_CFG`); `--agent.amp-dtype bfloat16` is on at reward
  parity. `enable_backface_culling=False` DESTROYS the image (the head cam sits inside the torso
  mesh). `step_profile.py` locates blocks but its FLAG verdicts do not survive a train A/B.
- **`ExtractorSonicAdapterModel.as_onnx` IS implemented** (multi-input wrapper: base ports +
  extractor token/query ports, plus the attention map as a free second output for a deploy-side
  "where is it looking" overlay). `as_jit` is not — ONNX is the deploy path
  (`docs/usage/export_onnx.md`).
- **`vibe.<family>` still imports** after the fold into `vibe.tasks.<family>`
  (`core/_legacy_paths.py`) — a saved env cfg records an mdp term by its DEFINING module path, so
  deleting a path deletes the ability to reload every run that used it. The finder must go
  **FIRST in `sys.meta_path`, not last**: a legacy package resolves its own submodules through
  its `__path__`, which is the NEW directory, so `PathFinder` finds `.../tasks/dodge/config/g1`
  there and loads a SECOND copy under the old name — two copies is two registrations, and mjlab
  raises on the second. Nothing can be shadowed by going first; the finder matches only names
  that no longer exist.
- **rsl_rl is a fork** carrying PPOAux, the round-2 perf work (fp32-head amp seam, frozen-prefix
  token cache, batched aux encodes) and the extractor-owned diagnostics. It ships everywhere
  `sync_deps.sh` runs, CARC included.

## Conventions

- `repose/repose_cube_env_cfg.py` is deliberately **robot-agnostic**; robot-specific wiring
  (sensors, contact bodies, action scale, dataset path) belongs in `config/<robot>/`.
- mdp config dicts use string keys for terms; per-robot configs mutate them in place (e.g.
  `cfg.events["rsi"].params["dataset_dir"] = ...`, `cfg.observations["critic"].terms.pop(...)`).
- **A config value that leaves a module is a COPY, always** — `dict(X)`, never `X`. Consumers
  treat what they are handed as scratch (`distribution_cfg.pop("class_name")` did exactly that,
  and the deletion outlived the model), and no caller should have to know whether they do.
