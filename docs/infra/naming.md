# Task naming

```
Vibe-<Task>[-<Source|Scene>]-<Extero>[-<Suffix>]
```

Example: `Vibe-Repose-BigCubeFloor-ObjKin` — large-cube floor repose with
object-kinematic perception. Every Vibe task adapts SONIC, so the one-valued
agent choice is not a naming axis.

Every token maps to one axis of the env factory
(`g1_repose_cube_env_cfg` / `g1_perloco_grail_env_cfg`); the factory arg is
listed with each token.

**A one-valued axis is not an axis.** A task drops a slot when it has nothing to
vary there, rather than carrying a constant token. That rule retired two slots on
2026-08-06: `<RobotMotionRew>` (every task is motion tracking now that DiMa is
gone) and `<Agent>` (every Vibe task adapts SONIC). `<Source>` runs the other
way: it exists only where provenance or the physical scene changes the
environment.

## `<Task>`

| Token | Meaning |
|---|---|
| `Repose` | Single-object special case: reorient the 6-color cube (goal is orientation-only). |
| `PerLoco` | Perceptive locomotion over staged (terrain, motion) pairs — orcs's task, adapted through a camera. |
| `Uolm` | **U**ni-**o**bject **l**oco-**m**anipulation over per-world object variants (suitcase, tire, table, …); each env tracks its own object's clips. Pairs with `Orcs-Uolm-*`, which is why it carries orcs's token and not a vibe-local synonym. |
| `Dodge` | Whole-body evasion of a thrown ball. The one task whose reference does NOT perform it — a nominal stand — so the evasion is the adapter's departure from it. |

## `<Source>` — PerLoco only

Provenance is in the id because it is what changes code; the terrain TYPE is not
(curb and stair are the same reader and the same env, so they are a roster line).

| Token | Meaning |
|---|---|
| `Grail` | NVIDIA GRAIL curb takes. One row, no difficulty axis. |
| `OmRe` | OmniRetarget robot-terrain climb families x z_scale levels. The obstacle height is orcs's `render_z_scale`, forwarded — not a vibe knob. |

## `<Scene>` — Repose only

| Token | Meaning |
|---|---|
| `BigCubeFloor` | Existing 0.6096 m colored cube and the front/side-flip floor datasets. |
| `SmallCubeTable` | Primitive 0.36 m colored cube, `box_manip` dataset, and a per-environment table placed under the sampled clip's goal. |


## Agent architecture (not in task ids)

Every Vibe policy is an adapted WBC: a frozen GEAR-SONIC base
(encoder→FSQ→decoder, 10@5 tokenizer, history-10 proprio) with a LoRA adapter on
the decoder. Because Vibe ships no competing agent architecture, this is not a
task-id token.

From-scratch baselines (`Orcs-*-TaRa`) live in orcs, where privileged and
scratch policies belong.

## `<Extero>` (`extero=`)

| Token | Arg value | Meaning |
|---|---|---|
| `ObjKin` | `"objkin"` | Privileged object kinematic state — base-frame pose 9 + twist 6 (Repose), env-frame pose+twist + `object_id` (Uolm; orcs's task IS this row, so vibe leaves it unregistered). |
| `HtSc` | `"scan"` | Privileged 187-ray terrain height scan (PerLoco; orcs's task IS this row, so vibe leaves it unregistered). |
| `ImgFeat` | `"imgfeat"` | Frozen Theia-tiny features from the head camera — replaces object kin (Repose, Uolm) or the height scan (PerLoco) in the aug stream. |
| `ImgRgb` | `"imgrgb"` | Raw head-camera RGB into a trainable CNN encoder — the baseline for `ImgFeat`: same base, same task, task-specific encoder instead of a frozen one. |

## `[-<Suffix>]`

| Token | Meaning |
|---|---|
| `Ext` | **Ext**ractor-only: cross-attention extractor -> z, trained by PPO gradients alone (no predictor, plain PPO). The floor every aux row is measured against. |
| `Lfd` / `Sfd` / `Reg` | PPOAux objective on the ImgFeat encoder: **L**atent **f**orward-**d**ynamics vs an EMA target (SSL) / supervised **S**tate **f**orward-**d**ynamics (SL) / privileged object **Reg**ression (SL, retired — now an `Sfd` config). |

## Per-run axes (not in task ids)

Same rule as above, one step further: an axis you vary **between runs of one task** is a
CLI flag, not a token. Registering a task per backbone or per query set would make the id
grammar carry an experiment's shape.

| flag | varies | default |
|---|---|---|
| `--env.img-encoder <hf-id>` | the frozen backbone, all `image_feature` terms at once | the task's (`IMG_ENCODER`, Theia-tiny) |
| `--agent.drop-query-rows q_proprio` | which extractor attention rows exist | none dropped |

Both are recorded in the run's saved cfg and in wandb, so the arm's identity survives in
the artifact rather than in the id. Design: `docs/perception/encoders.md` §11.

## Systems shorthand (not in task ids)

- **sys0** — the adapted WBC (frozen base + trained adapter) = every task's product; the deployable low-level block.
- **sys1** — the motion/command generator above sys0 (planned, SUGAR-style). Speaks robot-state language only; streams `bodywise_contact_cmd` + `robot_root_{lin,ang}_vel_cmd` into sys0's feedforward obs.
- **task level** — above sys1; owns `object_goal_*` (fixed per episode).
