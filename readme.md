# vibe

**Vi**sual **Be**havior adaptation for perceptive humanoid control.

## install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git LFS. uv
fetches Python 3.11 itself.

```bash
git clone https://github.com/lok-i/vibe && cd vibe

bash scripts/setup/sync_deps.sh     # .venv + vibe + the code pinned in deps.lock
bash scripts/setup/sync_data.sh     # data for every Vibe-* task (~0.6 GB)
source .venv/bin/activate
```

`sync_data.sh inhouse|omre|grail` fetches a subset — `inhouse` is Repose + Uolm + Dodge,
`omre`/`grail` the two PerLoco sources. Both scripts are idempotent: re-run after a
`deps.lock` bump. uv-only — an active venv is used, else `./.venv`.

failure modes, and why the install order is fixed: **[docs/infra/setup.md](docs/infra/setup.md)**.

## usage

```bash
# list all task-id's
list-envs
# train/play
train <task-id> --env.scene.num-envs 4096
# default rollout out intial policy (zero-init adpater + frozenbase). also zero|random|trained
play  <task-id> --agent initial --viewer native   
```

vibe task differs from its privileged twin env in [orcs](https://github.com/lok-i/orcs) by **ONE** obs group:

| vibe (vision) | privileged twin | swapped |
|---|---|---|
| `Vibe-Repose-BigCubeFloor-ImgFeat{,-Ext,-Sfd,-Lfd}` | `Vibe-Repose-BigCubeFloor-ObjKin` | object kinematics → image |
| `Vibe-Repose-SmallCubeTable-ImgFeat-Ext` | — new physical scene | 0.36 m cube + per-clip table |
| `Vibe-PerLoco-{Grail,OmRe}-ImgFeat-Ext` | `Orcs-PerLoco-{Grail,OmRe}-AdaptSonic` | height scan → image |
| `Vibe-Uolm-ImgFeat-Ext` | `Orcs-Uolm-AdaptSonic` | object kinematics + id → image |
| `Vibe-Dodge-ImgFeat-Ext` | `Orcs-Dodge-AdaptSonic` | ball kinematics → image |
| `Vibe-Repose-BigCubeFloor-ImgRgb` | `…-ImgFeat-Ext` | frozen encoder → trainable CNN |

`Orcs-*` twins come from the dependency and run from this env.

## useful tools:

### recording clips

rides `play --viewer viser`. `VIBE_REC=1` spawns paused, so the shot is set before a step.

```bash
# set the prefered recodinrg cfg via director
VIBE_REC=1 play Vibe-Uolm-ImgFeat-Ext --viewer viser --agent initial --num-envs 5
# post recording:
# stitch, no editor
printf "file '%s'\n" episode_0{0,3,7}.mp4 > list.txt      
ffmpeg -f concat -safe 0 -i list.txt -c copy showreel.mp4
```

Mouse the view → **Director** → `Sync from view` → pick the **Attention** query → `● Record`.
One mp4 per episode plus `take.json` (pose, query, res, checkpoint).

| knob | note |
|---|---|
| `lookat / distance / azimuth / elevation / fovy` | exact round-trip with the mouse view, both ways |
| `track` · `damping` | follow an entity; EMA, because MuJoCo's tracking camera is rigid and a jump shakes the frame |
| `context envs` | neighbours in the SAME frame — **this is how several objects get into one clip** |
| `Save/Load pose` | `videos/<task>/pose.json` — **task-scoped, auto-loaded**, so a task opens on its own view |
| `inset %` · `export panes` | insets are nearest-upscaled 112x63 — that IS the policy input |
| `resolution` · `reset on record` | 1080p default, MuJoCo offscreen (not the viser canvas) |

```
videos/<task>/pose.json           # one angle serves every checkpoint
             /initial/episode_00.mp4 …
             /trained/episode_00.mp4 …
```

`--checkpoint-file` puts clips in `<checkpoint_dir>/videos/<ckpt_stem>/`; the pose stays
task-scoped. `VIBE_REC_DIR` / `VIBE_POSE_DIR` override. Recording runs the viewer slowly (a
1080p render per step) — the mp4 is still real-time correct, captured at exactly `1/step_dt`.

## onnx export ([docs](docs/usage/export_onnx.md))

The deploy path for the C++ ROS2 side — the exporter owns all model knowledge, the
deploy node executes a self-describing artifact.

```bash
export-agent   <task-id>            # policy graph      (vibe.onnx.v1)
export-agent   <task-id> --check    # re-run the export episode, two-world diff
export-encoder                      # frozen backbone   (vision.onnx.v1)
```

## vram (mesh-object tasks)

mujoco_warp sizes GJK/EPA scratch by `nconmax × nworld` — the default swallowed ~21 GB at 12k
envs. Contained by `nccdmax` (64/world, `VIBE_NCCDMAX` overrides), `ccd_iterations=24`,
`nconmax=150`/`njmax=450`. Measured **6.6 → 1.8 GiB @ 2k envs**. Undershoot is loud — mjwarp
prints "CCD overflow" to stderr; raise the knob that overflowed.

**Collision default is `cvx_dcmp` for every object.** Whole-object convex hulls of container
shapes are a narrowphase trap (~11x slower than their decompositions at equal contact counts);
hulls are a per-object override for boxy shapes only.

