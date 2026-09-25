# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## What this is

`vibe` is an mjlab project: **visual behavior adaptation** for whole-body control on a Unitree
G1. A frozen SONIC base is adapted by a LoRA adapter that reads a head camera through a frozen
encoder and a cross-attention extractor. Four task families, each the twin of an orcs task with
**exactly one obs group swapped** for vision:

| family | orcs twin | swapped for vision |
|---|---|---|
| `repose` — colored-cube reorientation (`BigCubeFloor`, `SmallCubeTable`) | `orcs.tasks.uolm` (single-object case) | object kinematics |
| `perloco` — perceptive locomotion (`Grail`, `OmRe`) | `orcs.tasks.perloco` | 187-ray height scan |
| `uolm` — six-object loco-manipulation | `orcs.tasks.uolm` | object kinematics + `object_id` |
| `dodge` — evade a thrown ball | `orcs.tasks.dodge` | ball kinematics |

**Dependency is one-directional: vibe imports from orcs, never the reverse. A duplicate is a
bug.** Anything task-agnostic lives in `vibe.core`; a task imports it. The critic stays
privileged in every vision row. The scene object entity is named `"object"` everywhere.

## SOP — branch, build, test, bump

1. **branch and build**, in vibe and every dependency touched. `dependencies/*` ship detached
   at the locked SHA, so a commit there is dangling until a branch names it.
2. **unit + smoke tests**: `ruff check src tests`, `pytest tests/`.
3. **ask for a user test**: system-level behavior, with usage examples; await approval.
4. **push the dependency, THEN bump `deps.lock`.** A lock pointing at an unpushed SHA is a
   broken lock: the host silently keeps the old code (this once cost 11 GPU-hours).

A layout change ships a new metric key rather than a silent behavior change.

## Docs

| doc | what |
|---|---|
| `docs/setup.md` | sync scripts, dependency roles, the lock, failure modes |
| `docs/tasks.md` | task-id grammar (**the authority** on every token), per-run flags |
| `docs/architecture.md` | the policy, obs groups, aux objectives, render domain |
| `docs/metrics.md` | the `Z*` W&B keys and the rules for reading them |
| `docs/record.md` · `docs/export.md` | the take recorder · ONNX export |
| `docs/roadmap.md` | what comes after v0.1.0 |

Docs describe usage and design. No run result or analysis number goes in them or here.

## Commands

```bash
bash scripts/setup/sync_deps.sh [--deploy]        # .venv + vibe + deps.lock code rows (--deploy: + onnxruntime)
bash scripts/setup/sync_data.sh [MODE...]         # data rows + staged terrain; all|inhouse|omre|grail
bash scripts/setup/sync_{deps,data}.sh --check    # verify pass alone — no network, no pip

list-envs
train <task-id> --env.scene.num-envs 4096        # the released runs' flags: docs/tasks.md#train
play  <task-id> --viewer native                   # agent=auto: initial (no ckpt) unless a checkpoint is named
play  <task-id> --agent release --viewer native   # the lkrajan/vibe checkpoint (vibe/release.json)
export-agent <task-id> --release                  # ONNX export + two-world check

ruff check src tests [--fix]
pytest tests/                                     # contracts, no GPU, no checkpoint, ~7 s
```

uv only, Python from `.python-version`. **`deps.lock` is the pin**: bump = edit `sha`, re-run
the sync. vibe and orcs both pin assets/mocke/rsl_rl and one venv holds one editable install,
so **vibe's lock wins**; believe `orcs.core.deps`'s import-time drift line, above all for
`mocke` (the obs/action contract the SONIC checkpoints are bit-coupled to). Never
`pip install -e .[extra]` after a sync: it re-resolves mjlab's `rsl-rl-lib==5.2.0` over the fork.

`pytest tests/` tests CONTRACTS; a real run (`play`, `export-agent --check`) tests behavior.
Four files: `test_release` (manifest, `--agent` union, release routing, export + clip paths),
`test_export_cases`, `test_repose_scenes`, `test_vision_knobs`.

## Layout

| path | holds |
|---|---|
| `core/mdp/` | `image_feature` (THE frozen encoder), render-domain events, `object_in_fov` |
| `core/observation_cfgs.py` | vision atoms: `IMG_ENCODER`, group names, token/cls terms, `kv_tokens_group` |
| `core/env_cfgs.py` | `apply_render_domain`, `assert_play_is_clean`, `apply_stage_render`, `VibeEnvCfg` (`--env.img-encoder`) |
| `core/rl.py` | `VibeRunnerCfg`, `adapt_sonic_agent_cfg`, `EXTRACTOR_CFGS`, `attach_extractor` |
| `core/sensors.py` | `head_cam_cfg` — ONE head cam for every task (dodge re-aims it) |
| `core/_mjlab_compat.py` | every mjlab patch (table in its docstring) |
| `core/paths.py` · `core/_legacy_paths.py` | roots + `ORCS_ENV` · `vibe.<family>` → `vibe.tasks.<family>` |
| `tasks/<family>/config/g1/` | one question per file: `env_cfgs` · `observation_cfgs` · `agent_cfgs` · `sensors` · `export_case` (+ repose's `wbc`) |
| `encoders/` · `export/` · `viz/` · `release.py` | backbone roster · ONNX exporters · viser attention/director/recorder · HF checkpoints |

`observations.py` is ALWAYS atomic mdp terms; `observation_cfgs.py` is ALWAYS term/group
builders. A task's rationale lives in its registration docstring
(`tasks/<family>/config/g1/__init__.py`). There is no `_helpers.py`.

**Registration.** `import vibe` binds the orcs data roots, installs the legacy-path finder,
imports the four families, then applies the mjlab patches. Adding a task = a registration call.
repose registers directly; the others go through `orcs.core.registry.register_all`, which turns
missing data into a `SKIP_REASON` entry instead of an exception.

## Sharp bits

- **`import mjlab` BEFORE `import orcs`, in every script.** mjlab's entry-point scan runs at
  the end of its `__init__`; entered from orcs, it asks for vibe before orcs has finished, and
  every `Vibe-*` task fails to register with only a warning.
- **The play-script patches are ONE global, and vibe patches last.** The multi-clip sentinel
  must exempt the union of cfgs, taken from `orcs.MULTI_CLIP_CFGS`, never respelled. The
  `--agent` choices are the parent's Literal UNION vibe's (`get_args`), never respelled.
  `release` routes by manifest: vibe's tasks via `vibe.release`, the rest to orcs.
- **A camera on a variant scene aliased every entity to body -1** (`_patch_variant_scene_indexing`):
  mjSpec ids go stale after the sensor edit and `build_variant_model` compiles a copy. Symptom:
  the object's pose becomes the robot's, with no error anywhere.
- **The play invariant is inverted for the thin-import families.** orcs applies its play
  overrides inside its factory, before vibe adds anything, so vibe's training-only domain is a
  no-op under `play` and `assert_play_is_clean` diffs against a pre-vibe snapshot. The film
  stage (`apply_stage_render`) runs AFTER the guard and pops `rand_terrain_color`.
- **Saved cfgs record terms by DEFINING module path.** Moved symbols keep a re-export, and
  `_legacy_paths`' finder must be FIRST in `sys.meta_path` (last loads a second copy → mjlab
  raises on the double registration).
- **Aux targets are ego-observable only** (base frame, hands-only contact), the conditioning
  carries no object state, and dim count is the only loss weight. Adding a target = one term
  in the `prediction_target` dict group; its name is the metric's name.
- **Perf:** rank runs by reward vs wall-clock, not `Perf/total_fps`; train at 4096 envs;
  `G1_VIS_LEAN_CFG` (off-frame visuals hidden) trains, `G1_BASE_CFG` plays;
  `enable_backface_culling=False` destroys the image (the head cam sits inside the torso).
- **`vibe.core.observation_cfgs.IMG_ENCODER` and `image_feature` are the training encoder;
  `vibe.encoders.zoo` is the export one.** They load the same towers; keep them bit-exact.
- `mujoco_warp` renders the policy's frames; the interactive viewer is a different renderer
  (`docs/architecture.md`, render domain).

## Conventions

- **A config value that leaves a module is a COPY** — `dict(X)`, never `X`. Consumers treat
  what they are handed as scratch.
- mdp config dicts key terms by string; per-robot configs mutate them in place.
- `repose/repose_cube_env_cfg.py` is robot-agnostic; robot wiring lives in `config/<robot>/`.
