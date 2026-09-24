# ONNX export path (sys0 → C++ ROS2)

Two exporters, one philosophy — the exporter owns all model knowledge, the C++ node executes a
self-describing artifact:

| CLI | code | artifact | schema |
|---|---|---|---|
| `export-agent` | `vibe.export.agent` | policy graph (extractor + adapter + frozen base) | `vibe.onnx.v1` |
| `export-encoder` | `vibe.export.encoder` | frozen vision backbone (preprocessing baked in) | `vision.onnx.v1` |

## Agent export

One `.onnx` = one agent. Extractor + adapter + frozen base all live inside the graph; the only
thing outside it is the vision backbone (env-side in training, its own C++ package in deploy).

```
    ROS2 topics                     policy.onnx                          robot
 ─────────────────────  ┌──────────────────────────────────────┐   ─────────────────
  tokenizer ──────────► │ encoder ─► FSQ ─┐                    │
  policy (proprio H=10)─┤                 ├─► decoder ─► action├─► q_des = default
  augmentation ────────►│                 │   (adapters MERGED)│      + scale ⊙ a
  kv_tokens__img_tokens►│ cross-attn ─► z ┘                    │
  q_task_cmd / q_proprio│    ▲                                 ├─► attn (B,Q,P) [viz]
  q_cls ───────────────►│────┘                                 │
 ─────────────────────  └──────────────────────────────────────┘
        ▲
        │
   vision C++ pkg: image ─► frozen encoder ─► (1+P, C) sequence
```

### Export

ONNX tooling is deliberately absent from the training install. Install the
optional deploy dependencies only on machines that export or execute graphs:

```bash
pip install -e ".[deploy]"
```

```bash
export-agent Vibe-Repose-BigCubeFloor-ImgFeat-Sfd --wandb-run-path vbp/repose/swp7ha7b
export-agent <task-id> --wandb-run-path <e/p/run> --viewer native
export-agent <task-id> --checkpoint-file logs/rsl_rl/<exp>/<run>/model_20000.pt
```

Writes `<ckpt>.onnx` + `<ckpt>.manifest.json` next to the checkpoint (`--output-dir` to move
them). The manifest is also embedded in the ONNX `metadata_props` under key `manifest`, so the
`.onnx` alone is sufficient.

| flag | default | effect |
|---|---|---|
| `--check` / `--no-check` | on | the two-world test below; **nothing is written on failure** |
| `--tolerance` | 1e-5 | open-loop gate budget |
| `--check-steps` / `--closed-loop-min-steps` / `--seed` | the task's **case** | override the declared episode budget |
| `--provider` | `auto` | ORT EP for world 1: TensorRT → CUDA → CPU. The gate is always CPU |
| `--viewer` | `none` | `native` / `viser`, launched after the gate passes |
| `--dedupe-inputs` | on | merge content-identical groups onto one input |
| `--with-attn` | on | second output = attention map (free, already computed) |

### The two-world test

One env, `num_envs=2`, on the episode the **task's export case** declares.

| world | driven by | measures |
|---|---|---|
| **0** | checkpoint agent (torch) | the exported graph is asked for an action on world 0's obs each step and it is **thrown away** → open-loop numerics, **gated** |
| **1** | exported ONNX agent only | closed-loop stability — a pure rollout of the artifact |

```
[export] case: seed 0, 128 steps, episode 6 clips (one per bucket)
[export] onnxruntime providers: ['CPUExecutionProvider']
[export] open-loop  gate  : max|delta a| = 6.199e-06 over 128 steps (cpu fp32 both sides)
[export] device gap       : 8 FSQ token flip(s)
[export] closed-loop onnx : 128/128 steps to first reset, 0 reset(s) [none], r̄ = 0.163
[export] reference  torch : 128/128 steps to first reset, 0 reset(s) [none], r̄ = 0.156
```

**Only the gate compares like with like** — one observation, two backends, exact. World 1 is a
stability check and never a twin of world 0: on a multi-instance task the two worlds are two
different instances by construction (uolm's variant table hands world 0 a suitcase and world 1 a
trashcan), and even a single instance diverges — mujoco-warp is not bitwise reproducible across
processes (~1e-7/step, measured) and FSQ turns that into a token flip within ~3 steps. Read the
verdict, never the last decimal: a repeat run of the same case moved the gate 5.7e-6 → 4.8e-6 and
the flip count 7 → 2, with both verdicts unchanged.

### The export case — the task declares its own episode

`vibe.export.agent.case`. The exporter knows nothing about clips, objects or terrain; each task
declares an `ExportCase` beside its `register_mjlab_task` (episode + why, in
`<task>/config/g1/export_case.py`). What a case pins is that task's **episode identity**, and its
arity is the task's, not the exporter's:

| task | identity | clips declared |
|---|---|---|
| repose | clip | **1** — both worlds track it |
| uolm | object variant | **6** — one per roster object (`world_to_variant` is fixed at sim init) |
| perloco | tile | **0** — the staged dataset is already one clip per tile (9/9) |
| dodge | throw (event RNG) | **0** — one nominal stand; `seed` is the whole identity |

`clips` is a **keep** list applied as its complement through `exclude_motions` (the public
exclusion grammar). Never through `dataset_dir` — rewriting the root to a motion folder is valid
only for a flat scan, and every bucketed scan (`scan_grouped`) takes a `str` at a fixed depth.
That rewrite is what broke every task but repose.

Three constraints, each measured, each enforced by `tests/test_export_cases.py`:

1. **Cover every bucket.** A roster object or grid tile with no surviving clip fails at env
   build — `_clip_allowance` hands `multinomial` an all-zero row.
2. **One clip in the loaded set must carry a `contact_matrix.npz`** where a `ContactSchedule` is
   loaded — the legend comes from the set, not from every clip (uolm's `plasticbox` ships none,
   which is fine and matches training).
3. **Do not shrink uolm's roster.** `object_names=("suitcase",)` — or any subset — plus the head
   camera dies in `sim.sense()` with a CUDA illegal memory access (unfilled variant slots at
   `dataid -1`); the same subset without a camera is fine, and the full roster with a camera is
   fine. Pin clips, never the roster.

A task with no declared case still exports — it prints a warning and runs unpinned.

### `OnnxAgent` — the C++ reference implementation

[`src/vibe/export/agent/onnx_agent.py`](../../src/vibe/export/agent/onnx_agent.py) drives world 1 reading **only
the `.onnx`** — manifest out of `session.get_modelmeta().custom_metadata_map`, never the torch
model, never rsl_rl. It is the Python twin of the ROS2 node; if it flies in sim, the node has
everything it needs.

`assemble()` builds each port **term by term**, slicing at the manifest offset and concatenating
in manifest order. Against a real env that is the identity — which is the point: it proves the
offsets tile every group with no gap and no overlap, the one thing a hand-written node gets
wrong. `joint_targets()` shows the `default + scale ⊙ a` step the node must not forget.

### I/O (`-ImgFeat-*`, adapted SONIC, measured)

| input | shape | terms |
|---|---|---|
| `tokenizer` | (1, 640) | `g1_tokenizer` — 10 future ref frames |
| `policy` | (1, 930) | `base_ang_vel(30) joint_pos(290) joint_vel(290) actions(290) gravity_dir(30)`, **H=10 flattened** |
| `augmentation` | (1, 18) | `bodywise_contact_cmd(12) robot_root_lin_vel_cmd(3) robot_root_ang_vel_cmd(3)` — also feeds `q_motion_cmd` |
| `kv_tokens__img_tokens` | (1, 21, 192) | vision patches |
| `q_task_cmd` | (1, 6) | `object_goal_color` — repose. Uolm is (1, 9): `object_goal_ori(6) object_goal_pos(3)`; dodge has no task-command row at all |
| `q_proprio` | (1, 64) | single-frame proprio |
| `q_cls` | (1, 192) | vision CLS |
| **out** `actions` | (1, 29) | → `q_des = default_joint_pos + scale ⊙ a` |
| **out** `attn` | (1, 3, 21) | one row per LIVE query group, `q_task_cmd q_proprio q_cls` (`q_motion_cmd` is parked) |

`-ObjKin` is the same minus the vision rows: `tokenizer(640) policy(930)
augmentation(42)`.

### Manifest

```jsonc
{ "schema": "vibe.onnx.v1", "task_id": …, "model_class": "ExtractorSonicAdapterModel",
  "control": { "sim_timestep": 0.005, "decimation": 4, "step_dt": 0.02 },
  "action":  { "joint_names": [...29], "scale": [...], "default_joint_pos": [...],
               "stiffness": [...], "damping": [...] },
  "inputs":  [ { "name": "policy", "shape": [930], "groups": ["policy"],
                 "terms": [ { "name": "joint_pos", "dim": 290, "offset": 30,
                              "history_length": 10, "flatten_history_dim": true }, … ] } ],
  "outputs": ["actions", "attn"],
  "versions": { "vibe": "3b8baaf", "rsl_rl": "0261cc7" } }
```

Node contract — **hard throw, never warn**:

1. every `inputs[].terms[].name` is a term the node can produce;
2. `sum(dim)` per input == the ONNX input width;
3. `action.joint_names` == the node's own joint order, elementwise.

### Sharp bits

| # | |
|---|---|
| 1 | **History is the node's job.** `history_length: 10` on a proprio term means the node keeps the ring buffer and hands over the flattened 10 frames. Feeding one frame into a 10-frame slot raises no error anywhere. |
| 2 | **Publish `(1+P, C)`, not `P`.** `kv_tokens__img_tokens` (slice `tokens`) and `q_cls` (slice `cls`) come from ONE encoder forward — the manifest's `source` tag on each vision term says which slice it wants. |
| 3 | **Joint order is checkpoint-baked** (SONIC = MuJoCo order). Read it from the manifest; never assume. |
| 4 | **`augmentation` would feed two slots** — it and `q_motion_cmd` are the same term bundle (`observation_cfgs.robot_motion_cmd_terms`), so dedupe merges them onto one buffer with two consumers (`--no-dedupe` splits them again). `q_motion_cmd` is **parked** in the live rows, so no current export prints a dedupe line. |
| 5 | **`P` is baked at export** from the live camera resolution. A deploy camera of a different resolution fails the manifest assert rather than silently reshaping. |
| 6 | **Adapters are merged, not present.** `W_i += ΔW_i` for hidden layers, `[W_0 \| ΔW_0]` for the input layer — the exported decoder is a plain MLP. Exact in ℝ, ~1e-7 in fp32; measured end-to-end parity is ~3e-6 on actions. |
| 7 | **The merge must run in true fp32.** `Adapter.delta_weight()` is a matmul (`up @ down`); folding it on CUDA under TF32 costs ~1e-4 on the exported actions. `merge_adapters` deep-copies to CPU first — do not "optimize" that away. |

### FSQ makes GPU-vs-deploy discrete, and that is not an export error

The export gate compares CPU fp32 on both sides (~3e-6). The exporter *also* reports the
on-device model vs onnxruntime, which lands at **~3e-2** — 4 orders larger. Measured cause, on
`swp7ha7b`, 64 steps:

| effect | size |
|---|---|
| TF32 matmul noise (train/play default) on actions | ~5e-4 |
| ... which occasionally pushes an encoder latent across an FSQ rounding tie | **1–2 token flips per 64 steps** |
| one flip = one token dim jumping a full grid step, `1 / (levels // 2)` = 1/16 | 6.25e-2 on that token |
| resulting action jump for that step | 2.3e-2 … 3.8e-2 |

Zero flips ever appear CPU-vs-CPU. So the exported graph is exact; the *policy* is piecewise
constant in latent space by construction, and any numeric difference — TF32 vs fp32, GPU vs
CPU, a different BLAS on the robot — can land on the other side of a tie at ~3% of steps.
Action scale is ~0.35 rad, so a flip is ~0.7° on a joint target for one step, and the base was
trained through this quantizer. Treat it as a floor on achievable sim-deploy agreement, not a
bug to chase. The exporter prints the flip count so the number is never mysterious; run
training/play with `configure_torch_backends(allow_tf32=False)` if you want that floor lower.

### Tests

| layer | what | run |
|---|---|---|
| unit | zeros / arange / saturating / random probes, merge-vs-unmerged, alias rewiring, over all 3 model classes; adapters **and** normalizer stats randomized first (zero-init would make it vacuous) | `pytest dependencies/rsl_rl/tests/models/test_sonic_onnx_export.py` |
| contract | every registered task declares a case; every pinned clip still exists, is not on the task's own kill list, feeds every bucket, and leaves a contact legend. No GPU, no checkpoint, ~5 s | `pytest tests/test_export_cases.py` |
| integration | loaded checkpoint vs exported graph over a live rollout, built from the manifest's port table | `export-agent … --check` (on by default) |

The split is orcs's: **pytest tests contracts, a real run tests behavior.** The contract tier
catches what would otherwise blow up minutes into an export, after a wandb download.

## Encoder export (`export-encoder`)

The vision backbone is the one module outside the policy graph — its own artifact, its own C++
package (`vision_encoders`, see its `docs/method.md`). One graph per `vibe.encoders.zoo` tag with
ALL preprocessing baked in (cast, /255, BGR→RGB, HWC→CHW, per-model normalization), so the deploy
node's contract is: resize → memcpy BGR HWC uint8 → run. Outputs are the zoo's `dense`/`cls`
verbatim — parity with training/bench by construction (the exporter wraps `zoo._Enc._fwd`).

```bash
export-encoder                    # every tag in export_enc.yaml
export-encoder --tag theia-tiny   # cherry-pick
export-encoder --out /path/to/vision_encoders/models/
```

Config: `src/vibe/export/encoder/export_enc.yaml` — the single source of truth (resolution
112x63, `compute_dtype` fp32|fp16, opset, tags, gate tolerance). Swap fields there, never in code.

Per tag: `<tag>.onnx` + `<tag>.manifest.json` (vision.onnx.v1, also embedded in
`metadata_props`), gated torch-vs-ORT on random frames — nothing is kept on failure. Grid and
dims are MEASURED from a real forward, never derived arithmetically. Measured @ 112x63 fp32:

| tag | P (grid) | D | cls | gate | size |
|---|---|---|---|---|---|
| theia-tiny | 21 (3x7) | 192 | y | 6.4e-06 | 22 MB |
| siglip2-b16 | 21 (3x7) | 768 | n | 5.2e-05 | 343 MB |
| dinov2-s | 32 (4x8) | 384 | y | 7.1e-05 | 88 MB |
| dinov3-splus | 21 (3x7) | 384 | y | 1.1e-05 | 115 MB |
| tinyclip-39m | 21 (3x7) | 512 | y | 1.5e-05 | 154 MB |
| clip-b32 | 3 (1x3) | 768 | y | 1.2e-05 | 350 MB |

The `.onnx` is architecture-neutral (weights + graph, zero machine code): export on the x86
laptop, `scp` to the G1's aarch64 ORT. Only the ORT runtime library and (later) TRT engine
caches are arch-bound — the latter generated on-device at first load, never shipped.
