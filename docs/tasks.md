# tasks

```
Vibe-<Task>[-<Source|Scene>]-<Extero>[-<Suffix>]
```

Every task adapts the frozen SONIC base, so the base is not a token. A slot with only one
value is dropped. `list-envs` prints the registered ids, including the `Orcs-*` / `Mocke-*`
ids the dependencies register.

| slot | token | meaning |
|---|---|---|
| Task | `Repose` | reorient a 6-colour cube until the commanded colour faces up |
| | `PerLoco` | perceptive locomotion over staged terrain |
| | `Uolm` | uni-object loco-manipulation over a six-object roster |
| | `Dodge` | evade a thrown ball; the reference is a still stand, so all evasion comes from the adapter |
| Source (PerLoco) | `Grail` · `OmRe` | GRAIL curbs · OmniRetarget climbs |
| Scene (Repose) | `BigCubeFloor` · `SmallCubeTable` | 0.61 m cube on the floor · 0.36 m cube onto a per-clip table |
| Scene (Dodge) | `ConeFast` | indoor room, faster throws from farther out, 64-px camera |
| Extero | `ObjKin` | privileged object state: the twin row (Repose only; orcs registers the others) |
| | `ImgFeat` | frozen-encoder features from the head camera |
| | `ImgRgb` | raw RGB into a trainable CNN: the vision-stack baseline |
| Suffix | *(none)* | features go straight into the adapter, no extractor |
| | `Ext` | cross-attention extractor → z, trained by PPO alone |
| | `Sfd` | + supervised forward-dynamics aux loss |
| | `Lfd` | + latent forward-dynamics aux loss, against an EMA target |

## twins

| vibe (vision) | privileged twin | swapped for the camera |
|---|---|---|
| `Vibe-Repose-BigCubeFloor-ImgFeat{,-Ext,-Sfd,-Lfd}` | `Vibe-Repose-BigCubeFloor-ObjKin` | object kinematics |
| `Vibe-Repose-SmallCubeTable-ImgFeat-Ext` | — | (new scene: small cube onto a table) |
| `Vibe-PerLoco-{Grail,OmRe}-ImgFeat-Ext` | `Orcs-PerLoco-{Grail,OmRe}-AdaptSonic` | height scan |
| `Vibe-Uolm-ImgFeat-Ext` | `Orcs-Uolm-AdaptSonic` | object kinematics + id |
| `Vibe-Dodge-{,ConeFast-}ImgFeat-Ext` | `Orcs-Dodge-AdaptSonic` | ball kinematics |
| `Vibe-Repose-BigCubeFloor-ImgRgb` | `…-ImgFeat-Ext` | frozen encoder → trainable CNN |

The critic stays privileged in every row.

## per-run flags

A knob varied between runs of one task is a flag, not a token. Both are recorded in the run cfg.

| flag | varies | default |
|---|---|---|
| `--env.img-encoder <hf-id>` | the frozen backbone, rebinding both image terms at once ([encoders](#encoders)) | Theia-tiny |
| `--agent.drop-query-rows q_proprio` | which extractor attention rows exist (z stays 128-d) | none dropped |
| `--num-envs 27` (play, `OmRe`) | fills all 9 terrain tiles | |

## train

The released runs, flag for flag. The anneal ends before the run does: shorten both together.
`--agent.amp-dtype bfloat16` needs an Ampere or newer GPU; drop it on older cards.

```bash
train Vibe-Repose-BigCubeFloor-ImgFeat-Ext --env.scene.num-envs 4096 \
  --agent.max-iterations 60000 --env.commands.motion.init-phase-anneal-iterations 50000 \
  --agent.amp-dtype bfloat16
train Vibe-Uolm-ImgFeat-Ext --env.scene.num-envs 4096 \
  --agent.max-iterations 25000 --env.commands.motion.init-phase-anneal-iterations 20000 \
  --agent.amp-dtype bfloat16
train Vibe-PerLoco-Grail-ImgFeat-Ext --env.scene.num-envs 4096 \
  --agent.max-iterations 15000 --env.commands.motion.init-phase-anneal-iterations 10000
train Vibe-PerLoco-OmRe-ImgFeat-Ext --env.scene.num-envs 4096 \
  --agent.max-iterations 15000 --env.commands.motion.init-phase-anneal-iterations 10000
train Vibe-Dodge-ConeFast-ImgFeat-Ext --env.scene.num-envs 4096 \
  --agent.max-iterations 15000
```

## encoders

```bash
train <ImgFeat-task> --env.img-encoder facebook/dinov3-vits16plus-pretrain-lvd1689m
```

| `--env.img-encoder` | C | note |
|---|---|---|
| `theia-tiny-patch16-224-cddsv` | 192 | default, released checkpoints |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 | HF-gated |
| `facebook/dinov2-small` | 384 | |
| `google/siglip2-base-patch16-224` | 768 | |
| `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` | 512 | |
| `openai/clip-vit-base-patch32` | 768 | fewer tokens (patch 32) |

`play` · `export-agent`: Theia-tiny only. `export-encoder --tag <tag>`: all six ([export](export.md)).

Each task's rationale (LoRA rank, query rows, what its twin controls for) is in its
registration docstring: `src/vibe/tasks/<family>/config/g1/__init__.py`.
