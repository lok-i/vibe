# record

Film a policy from `play --viewer viser`: pick the shot in the browser, and MuJoCo renders the
take at up to 1440p, with the policy's own camera and its attention map as insets.

```bash
VIBE_PAUSED=1 play Vibe-Uolm-ImgFeat-Ext --agent release --viewer viser --num-envs 5
```

1. mouse the 3D view → **Director** · `Sync from view` (or set the knobs directly);
2. **Attention** · pick the query row the inset shows;
3. **Record** · `● Record`, then ▶. One mp4 per episode until `■ Stop`.

`VIBE_PAUSED=1` starts paused so the shot is set before the first step. On a machine without
a display, launch with `MUJOCO_GL=egl`.

## where files land

| agent | clips |
|---|---|
| `release` | `videos/<task>/release/` |
| `--checkpoint-file <dir>/<ckpt>.pt` | `<dir>/videos/<ckpt>/` |
| `initial`, or `trained` via W&B | `videos/<task>/<agent>/` |

`VIBE_REC_DIR` overrides the clip folder. A non-empty `take` puts the take in a subfolder.

```
videos/<task>/pose.json                  # the saved shot: task-scoped, auto-loaded on the next play
             /release/episode_00.mp4
                     /episode_00.fpv.mp4  # with `export panes`
                     /episode_00.attn.mp4
                     /take.json           # pose, resolution, fps, task, checkpoint, agent, query
```

`VIBE_POSE_DIR` moves `pose.json`. One pose serves every checkpoint of a task, so two clips of
the same shot are comparable.

## knobs

| panel | knob | does |
|---|---|---|
| Director | `lookat` · `distance` · `azimuth` · `elevation` · `fovy` | the shot; round-trips exactly with the mouse view |
| | `track` · `damping` | follow an entity (EMA-smoothed; MuJoCo's tracking camera shakes) |
| | `context envs` | draw neighbouring envs into the same frame: several objects in one clip |
| | `Save pose` · `Load pose` | `pose.json`; rename the file per shot |
| Record | `resolution` | 720p / 1080p / 1440p |
| | `inset %` · `inset corner` | size and place of the FPV + attention insets; 0 hides them |
| | `export panes` | also write the insets as separate synced mp4s |
| | `debug viz in film` | draw command markers into the take |
| | `reset on record` | start from a fresh episode, or wait for the running one to end |

## notes

- Frames are captured every env step, so the mp4 plays at exactly `1 / step_dt` however
  slowly it renders. Recording slows the viewer, not the clip.
- Insets are nearest-upscaled from 112x63: what you see is the policy's input.
- Stitch clips without an editor:

```bash
printf "file '%s'\n" episode_0{0,3,7}.mp4 > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy showreel.mp4
```
