# Third-party notices

The BSD-3-Clause license at the repository root applies only to original VIBE
material. Third-party material retains its upstream terms.

## Frozen vision backbones

VIBE trains no vision backbone. Every encoder is loaded frozen and is
**downloaded at runtime** from the Hugging Face Hub — no weights are bundled in
this repository or in its Python distribution. The roster lives in
[`src/vibe/encoders/zoo.py`](src/vibe/encoders/zoo.py); the default is
`theia-tiny`.

| tag | model | upstream |
|---|---|---|
| `theia-tiny` | `theaiinstitute/theia-tiny-patch16-224-cddsv` | [Theia](https://github.com/bdaiinstitute/theia), The AI Institute |
| `tinyclip-39m` | `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` | [TinyCLIP](https://github.com/wkcn/TinyCLIP), Microsoft |
| `clip-b32` | `openai/clip-vit-base-patch32` | [CLIP](https://github.com/openai/CLIP), OpenAI |
| `siglip2-b16` | `google/siglip2-base-patch16-224` | [SigLIP 2](https://github.com/google-research/big_vision), Google |
| `dinov2-s` | `facebook/dinov2-small` | [DINOv2](https://github.com/facebookresearch/dinov2), Meta |
| `dinov3-splus` | `facebook/dinov3-vits16plus-pretrain-lvd1689m` | [DINOv3](https://github.com/facebookresearch/dinov3), Meta — **HF-gated**, accept the license once with your HF token |

Each carries its own license on its model card. Users are responsible for
accepting those terms and for complying with them; a citation or an
acknowledgement does not replace a license.

## Released checkpoints

The policies at [`lkrajan/vibe`](https://huggingface.co/lkrajan/vibe), fetched by
`play --agent release` and `export-agent --release`, embed the frozen
[SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) base weights. They are
distributed under the NVIDIA Open Model License, not BSD-3-Clause; the model card
and its `LICENSES/` carry the terms. They are downloaded at runtime and not
bundled here.

## External dependencies and data

Resolved separately by `deps.lock` or by pip, and neither relicensed nor
bundled in the VIBE Python distribution:

| Dependency or data | Use | Upstream terms |
|---|---|---|
| [ORCS](https://github.com/lok-i/orcs) | Privileged task mechanics — commands, rewards, terminations, terrain staging, the robustness domain | BSD-3-Clause |
| [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) via [Mocke](https://github.com/lok-i/mocke) | Frozen motion prior and the obs/action contract the ported checkpoints are bit-coupled to | Apache-2.0 code; NVIDIA Open Model License weights |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl), via the [fork](https://github.com/lok-i/rsl_rl) carrying the SONIC/extractor/PPOAux stack | Learning framework | BSD-3-Clause; the fork's delta is documented in its `FORK.md` |
| [mjlab](https://github.com/mujocolab/mjlab) | Manager-based simulation framework | Its own distribution and license |
| [OmniRetarget Dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset) | Retargeted terrain and object motions | MIT as declared by its dataset card |
| [GRAIL](https://github.com/NVlabs/GRAIL) | Terrain-motion source data | NVIDIA's upstream repository and data terms |
| `transformers`, `einops`, `omegaconf` | Backbone loading and configuration | Apache-2.0 / MIT as published |
| The installed `assets` package | Robot and object models | Its own distribution |

Users are responsible for obtaining restricted datasets and model weights and
for complying with their terms.
