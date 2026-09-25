# vibe

[![arXiv](https://img.shields.io/badge/arXiv-2609.09918-red)](https://arxiv.org/abs/2609.09918)
[![Project](https://img.shields.io/badge/Project-Page-brightgreen)](https://lok-i.github.io/vibe-control/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-vibe-blue)](https://lok-i.github.io/vibe-control/#demo)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/lkrajan/vibe)
[![License](https://img.shields.io/badge/License-BSD--3-blue)](LICENSE)

implementation accompanying *ViBe: **Vi**sual **Be**havior Adaptation for Perceptive Humanoid Whole-Body Control*.


## install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git LFS; uv fetches
Python itself.

```bash
git clone https://github.com/lok-i/vibe && cd vibe
bash scripts/setup/sync_deps.sh     # .venv + vibe + the code pinned in deps.lock
bash scripts/setup/sync_data.sh     # all (default) |  inhouse | omre | grail
source .venv/bin/activate
```

modes, dependencies, failure modes: [docs/setup.md](docs/setup.md).

## usage

```bash
list-envs                                          # every task id
train <task-id> --env.scene.num-envs 4096
play  <task-id> --viewer native                    # no checkpoint: the untrained policy (= the frozen base)
play  <task-id> --agent release --viewer native    # the released checkpoint, fetched + sha256-verified once
```

| arg | options |
|---|---|
| `--agent` | `auto` · `initial` · `release` · `trained` · `zero` · `random` |
| `--checkpoint-file` · `--wandb-run-path` | for `trained` |
| `--viewer` | `auto` · `native` · `viser` |
| `--env.img-encoder` (train) | [backbones](docs/tasks.md#encoders); default Theia-tiny |

## released checkpoints

[`lkrajan/vibe`](https://huggingface.co/lkrajan/vibe) `v0.1.0`, one per task family:

```bash
bash scripts/setup/download_released_models.sh    # all up front; or --list | <task-id>...
play Vibe-Repose-BigCubeFloor-ImgFeat-Ext --agent release --viewer native
play Vibe-PerLoco-Grail-ImgFeat-Ext       --agent release --viewer native
play Vibe-PerLoco-OmRe-ImgFeat-Ext        --agent release --viewer native --num-envs 27  # 27 fills all 9 tiles
play Vibe-Uolm-ImgFeat-Ext                --agent release --viewer native
play Vibe-Dodge-ImgFeat-Ext               --agent release --viewer native
```

- cache: `~/.cache/vibe/releases`, override with `VIBE_RELEASE_ROOT`
- license: NVIDIA Open Model License, as they embed SONIC base weights (see the model card)

## tasks

| vibe (vision) | privileged twin | swapped for the camera |
|---|---|---|
| `Vibe-Repose-BigCubeFloor-ImgFeat{,-Ext,-Sfd,-Lfd}` | `Vibe-Repose-BigCubeFloor-ObjKin` | object kinematics |
| `Vibe-Repose-SmallCubeTable-ImgFeat-Ext` | — | (new scene: small cube onto a table) |
| `Vibe-PerLoco-{Grail,OmRe}-ImgFeat-Ext` | `Orcs-PerLoco-{Grail,OmRe}-AdaptSonic` | height scan |
| `Vibe-Uolm-ImgFeat-Ext` | `Orcs-Uolm-AdaptSonic` | object kinematics + id |
| `Vibe-Dodge-{,ConeFast-}ImgFeat-Ext` | `Orcs-Dodge-AdaptSonic` | ball kinematics |
| `Vibe-Repose-BigCubeFloor-ImgRgb` | `…-ImgFeat-Ext` | frozen encoder → trainable CNN |

The critic stays privileged in every row. Id grammar and flags: [docs/tasks.md](docs/tasks.md).

## docs

| doc | for |
|---|---|
| [setup](docs/setup.md) | install modes, dependencies, failure modes |
| [tasks](docs/tasks.md) | task-id grammar, per-run flags |
| [architecture](docs/architecture.md) | the policy, obs groups, aux objectives, render domain |
| [metrics](docs/metrics.md) | the `Z*` W&B keys and how to read them |
| [record](docs/record.md) | film clips from `play --viewer viser`, with the policy's view + attention |
| [export](docs/export.md) | ONNX export of the policy and the vision encoder |

## license

Code: [BSD-3-Clause](LICENSE). Third-party models, data and the released checkpoints keep their
own terms: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
