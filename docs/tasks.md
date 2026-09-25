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

## per-run flags

A knob varied between runs of one task is a flag, not a token. Both are recorded in the run cfg.

| flag | varies | default |
|---|---|---|
| `--env.img-encoder <hf-id>` | the frozen backbone, rebinding both image terms at once ([encoders](#encoders)) | Theia-tiny |
| `--agent.drop-query-rows q_proprio` | which extractor attention rows exist (z stays 128-d) | none dropped |

## encoders

Any `ImgFeat` task trains on another frozen backbone with one flag. The run cfg records it,
but `play` does not read it back: **repeat the flag at play**. A backbone of a different width
C fails at load with a size mismatch; one of the same width (DINOv2 ↔ DINOv3) loads silently.

```bash
ENC=facebook/dinov3-vits16plus-pretrain-lvd1689m
train Vibe-Repose-BigCubeFloor-ImgFeat-Ext --env.scene.num-envs 4096 --env.img-encoder $ENC
play  Vibe-Repose-BigCubeFloor-ImgFeat-Ext --env.img-encoder $ENC --viewer native \
      --agent trained --wandb-run-path <entity>/<project>/<run-id>      # or --checkpoint-file <path.pt>
```

| `--env.img-encoder` | C | note |
|---|---|---|
| `theia-tiny-patch16-224-cddsv` | 192 | the default and the released checkpoints; bare id, the org is prefixed at load |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 | HF-gated: accept the license once on its model page |
| `facebook/dinov2-small` | 384 | patch 14 |
| `google/siglip2-base-patch16-224` | 768 | no CLS token; the global token is the MAP-head pool |
| `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` | 512 | |
| `openai/clip-vit-base-patch32` | 768 | patch 32: fewer tokens per frame |

C is the token width the extractor reads. The backend is picked from the id
(`image_feature._detect_backend`), so the id must contain `theia`, `dino`, `siglip` or `clip`.
`--agent release` and `export-agent` build the default encoder only; `export-encoder --tag <tag>`
exports any of the six ([export](export.md)).

Each task's rationale (LoRA rank, query rows, what its twin controls for) is in its
registration docstring: `src/vibe/tasks/<family>/config/g1/__init__.py`.
