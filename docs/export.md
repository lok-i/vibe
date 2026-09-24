# export

Two self-describing ONNX artifacts for a deploy node. Each carries its manifest in the ONNX
metadata (key `manifest`), so the `.onnx` alone is enough.

| CLI | artifact | schema |
|---|---|---|
| `export-agent` | the policy: extractor + adapter + frozen base | `vibe.onnx.v1` |
| `export-encoder` | the frozen vision backbone, preprocessing baked in | `vision.onnx.v1` |

```bash
bash scripts/setup/sync_deps.sh --deploy   # onnxruntime; never `pip install -e .[deploy]` (it swaps out the rsl_rl fork)
```

## agent

```bash
export-agent <task-id> --release                    # the released checkpoint -> exports/agent/<task-id>/
export-agent <task-id> --checkpoint-file <ckpt.pt>  # -> next to the checkpoint
export-agent <task-id> --wandb-run-path <e/p/run> --viewer native
```

| flag | default | effect |
|---|---|---|
| `--check` / `--no-check` | on | the two-world test; nothing is written if it fails |
| `--tolerance` | 1e-5 | open-loop gate |
| `--check-steps` · `--closed-loop-min-steps` · `--seed` | the task's export case | override the pinned episode |
| `--provider` | `auto` | onnxruntime EP for world 1: TensorRT → CUDA → CPU |
| `--viewer` | `none` | `native` / `viser`: watch both worlds after the gate |
| `--dedupe-inputs` · `--with-attn` | on | merge identical input groups · attention map as a 2nd output |
| `--output-dir` | see above | where the `.onnx` + `.manifest.json` go |

**Two-world test.** One env, two worlds, on the episode the task pins
(`<task>/config/g1/export_case.py`):

| world | driven by | checks |
|---|---|---|
| 0 | the checkpoint (torch) | the exported graph's action on world 0's obs, discarded: open-loop parity, gated at `--tolerance` on CPU fp32 |
| 1 | the exported graph only | closed-loop: survives `min_steps` without a reset |

Read the verdict, not the digits: mujoco_warp is not bitwise reproducible and the base's FSQ
quantizer turns ~1e-7 into a token flip within a few steps.

**Inputs** (`-ImgFeat-*`, measured on repose):

| port | shape | contents |
|---|---|---|
| `tokenizer` | 640 | 10 future reference frames |
| `policy` | 930 | proprio, history 10, flattened: **the node keeps the ring buffer** |
| `augmentation` | 18 | per-body contact command (12) + root twist command (6) |
| `kv_tokens__img_tokens` | 21 x 192 | encoder patch tokens |
| `q_task_cmd` · `q_proprio` · `q_cls` | 6 · 64 · 192 | goal colour · proprio · encoder global token |
| out `actions` | 29 | `q_des = default_joint_pos + scale ⊙ a` |
| out `attn` | 3 x 21 | one attention row per query group |

The node must fail hard when:

1. a manifest term is one it cannot produce;
2. a port's term dims do not sum to the ONNX input width;
3. `action.joint_names` differs from its own joint order (it is baked into the checkpoint).

`vibe.export.agent.onnx_agent.OnnxAgent` is the Python reference for that node. It reads
only the `.onnx` and assembles every port term by term from the manifest.

Adapters are merged into the base weights (on CPU fp32: TF32 costs ~1e-4). The tokens and
`q_cls` come from one encoder forward; P is baked from the camera at export.

## encoder

```bash
export-encoder                          # every tag in export_enc.yaml -> exports/enc/
export-encoder --tag theia-tiny         # one backbone
export-encoder --config src/vibe/export/encoder/export_enc_dodge.yaml   # ConeFast's 64-px camera
```

Input: a BGR HWC uint8 frame. Cast, /255, BGR→RGB and normalization are baked in. Outputs:
`patch_tokens` (+ `cls_token`) in fp32, gated torch-vs-onnxruntime on random frames.
The yaml is the single source of truth: resolution, dtype, opset, tags, tolerance.

| tag | P @ 112x63 | D | cls |
|---|---|---|---|
| theia-tiny | 21 | 192 | ✓ |
| siglip2-b16 | 21 | 768 | ✗ |
| dinov2-s | 32 | 384 | ✓ |
| dinov3-splus | 21 | 384 | ✓ |
| tinyclip-39m | 21 | 512 | ✓ |
| clip-b32 | 3 | 768 | ✓ |
