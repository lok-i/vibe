# Render domain — what the policy's camera actually sees

The visual DR that ships, the three things the renderer **will not** let you randomize, and the
bugs that cost us a week. Everything here is measured on `mujoco_warp`'s rasterizer — **the
thing that produces the POLICY's frames.**

> **The interactive viewer is a DIFFERENT renderer (OpenGL) and obeys none of this.**
> That is what makes every wall below a trap: it looks right in the viewer and trains wrong.

| what | where |
|---|---|
| camera + light + floor variation, every task | [`apply_render_domain`](../../src/vibe/core/env_cfgs.py) — **no-op under `play`**, by design |
| the colour task channel (face→colour perm + terrain palette) | [`apply_color_relabel`](../../src/vibe/tasks/repose/config/g1/env_cfgs.py) — repose vision rows |
| the event terms | [`vibe/core/mdp/events.py`](../../src/vibe/core/mdp/events.py) · [`vibe/tasks/repose/mdp/events.py`](../../src/vibe/tasks/repose/mdp/events.py) |
| flat matte floor | [`flat_floor`](../../src/vibe/core/env_cfgs.py) — plane terrains only |
| the film STAGE, every task | [`apply_stage_render`](../../src/vibe/core/env_cfgs.py) — **play only**, §5 |

## 1 · the three walls

| you want | verdict | why |
|---|---|---|
| per-env `geom_rgba` / `mat_rgba` | **works** | per-world arrays — the whole `rand_*` family |
| translucency (`alpha < 1`) | **no-op** | shading reads `vec3(color[0..2])`, never `color[3]` |
| sky / background colour | settable **ONCE**, never randomized | one global, baked into a captured CUDA graph |
| per-env sky | needs new geometry | no per-world background array exists |

### alpha is not rendered

`render.py` shades a hit with `base_color = vec3(color[0], color[1], color[2])`. No blending, no
second-hit traversal ⇒ **a translucent material renders fully opaque in every obs frame.**

Setting `alpha=0.5` to model a clear plastic bin buys nothing and actively misleads, because the
viewer shows it translucent. Pick the colour a camera actually *sees* — for a clear bin that is
a light neutral, not a hue — and leave alpha at 1.

**Full precedence chain**, and texture MULTIPLIES rather than replaces
(`base_color = cw_mul(base_color, tex_color)`):

```
texture  >  mat_rgba  >  geom_rgba
```

So a flat colour needs the texture **GONE**, not merely a colour set under it. That is what
`assets`' `metadata.json → "rgba"` does (emits a material with no texture) and what `flat_floor`
does for a plane's groundplane.

### the sky is one global, baked into a CUDA graph

```
create_render_context(render_skybox=False)      # mjlab's call — there is no skybox
  └─ every ray that hits nothing ─► RenderContext.background_color = (0.1, 0.1, 0.2)
       └─ render(): rgb_data.fill_(rc.background_color)   ── INSIDE the captured sense_graph
```

Measured in a dodge obs frame: sky pixels read exactly `(25, 25, 51)`.

`background_color` is a plain dataclass field, so it is settable — **but only before graph
capture.** Verified: writing it at runtime changes nothing, the image is byte-identical. A
working set has to wrap `create_render_context` itself.

⇒ **A day/night domain is not available at any price worth paying.** Per-env would need a
per-world background array the renderer lacks; per-step would need a graph re-capture per
sample. The route that *would* work is a visual-only sky DOME geom per env, whose `geom_rgba` is
per-world exactly like the terrain's and drops straight into `rand_terrain_color` — at the cost
of turning every previously-missed ray into a hit test.

> **Not done, and the reason is the important part:** repose transferred to hardware on this
> renderer's stock background. A domain that the one datapoint we have says was never needed is
> not a domain, it is a variable added on speculation. If a task ever *does* look up enough for
> the sky to dominate its frames (dodge is the only candidate — the shared head cam aims 45°
> DOWN), the sky dome is the design, justified by a measurement rather than by how the viewer
> looks.

## 2 · what works — the per-world write

```
[reset e] ─► geom_rgba[e, cube_face_*]    = cube_pal[ci]       per-world
          ─► mat_rgba [e, groundplane]    = ground_pal[gi]     material beats geom
          ─► skip 1 frame                                      the loop-top frame predates the repaint
```

- `sim.expand_model_fields(("geom_rgba", "mat_rgba"))`, then
  `wp.to_torch(sim.wp_model.<field>)[env_id, id] = color`. Worlds render isolated — no
  cross-env leakage, no shared-scene probability hacks.
- **palette hygiene:** ground colours are chosen >100 rgb-dist from all 6 cube colours, so a
  colour-threshold cube mask can never alias the floor.

## 3 · the two stacked bugs (each hid the other)

| # | bug | symptom | fix |
|---|---|---|---|
| 1 | CUDA graphs captured at sim init pin the OLD `geom_rgba` array; `expand_model_fields` swaps in a new allocation the render graph never sees | recolor writes land, **frames unchanged — silent** | `sim.create_graph()` right after expanding (documented contract in mjlab `sim.py`) |
| 2 | terrain geom carries the `groundplane` material (rgba 1,1,1); `mat_rgba` beats `geom_rgba` at render | cubes recolor (faces have no material), **ground stays white** | repaint `mat_rgba[e, geom_matid[ground]]`, not (only) `geom_rgba` |

Corollary on #2: the camera's `use_textures=False` kills the **texture** only — the material's
flat rgba still wins. Hence the precedence chain in §1.

## 4 · rules for the in-env version

1. **Don't hand-roll.** `mjlab.envs.mdp.dr.{geom_rgba, mat_rgba}` already exist as event terms
   with `env_ids` + per-channel `axes`. Registering them in `cfg.events` makes the EventManager
   pre-declare the fields, so expansion happens BEFORE graph capture ⇒ **bug #1 cannot occur on
   the env path.** `create_graph()` is only needed for script-side manual expansion.
2. **Reset-mode DR has one-frame staleness** — the render inside the resetting step predates the
   event write. Irrelevant for RL (one obs frame), **fatal for labeled analysis** — which is why
   `collect_frames.py` drops the first frame per episode.
3. **Verify on the npz frames, not the viewer.** Viewers do sync `geom_rgba`/`mat_rgba` per
   selected world (`viewer/model_sync.py::VIEWER_MODEL_FIELDS`), so eyeballing DR there is
   valid — but the head_cam frames are the artifact the metrics consume
   (`eyeball_frames.py` contact sheet).
4. **Analysis-mask footgun** (the notebook-side twin of bug #2): shading pulls rendered cube
   pixels **26–67 rgb-dist** off their nominal rgba, so a "safe-looking" threshold of 60 matched
   nothing. Measure the in/out distributions before picking a threshold.
5. **Light INTENSITY is not randomizable and cannot be.** mjwarp's batched `Model` carries
   `light_{type,castshadow,active,pos,dir}` and no diffuse/ambient field. Direction is the half
   that survives — and it is the half that moves shading and shadows.

## 5 · the STAGE — one floor for every task, under `play`

The mirror image of `apply_render_domain`: that varies the floor and is a no-op under play, this
one PINS it and is a no-op under train. Four families that each randomize their own ground
cut together as four different rooms; the stage is what makes a collage one shoot.

| | |
|---|---|
| colour | `STAGE_RGBA = (0.30, 0.30, 0.32, 1.0)` — cool slate |
| term | `set_terrain_color` — the deterministic twin of `rand_terrain_color`, same nominal-colour split |
| reach | plane tasks + dodge's room paint EVERYTHING; perloco passes `ground_rgba` so **curbs keep their level-tracking hue** |
| ordering | called AFTER `assert_play_is_clean` (and after repose's `_play_overrides`) — a fixed colour is not a domain, and the call site is what says so |
| replaces | pops `rand_terrain_color`; repose keeps that event alive under play on purpose (its colour relabel is a task CHANNEL), so beside it the stage would lose |
| two writes | the per-world `geom_rgba`/`mat_rgba` (the event — what mjwarp gives the POLICY) **and** the plane's MATERIAL, via `flat_floor(cfg, rgba=STAGE_RGBA)` — what the viewer and the TPV recorder read |
| one place | the constant, and `apply_stage_render`'s default. Every task calls it with `play=` alone (perloco adds `ground_rgba`, the SPLIT key, not a colour) |

### 5.1 · picking the colour — measure the RENDER, and measure TWO things

Four iterations (off-white → teal-gray → light grey → slate), every one overturned by a
measurement rather than an argument. What made each earlier pick wrong is the same mistake, and
avoiding it is the only durable content of this section.

**Nominal rgb-dist is the wrong instrument once a light source is involved.** Three effects it
cannot see:

1. the sun's **~1.2 gain** lifts a flat-lit floor above its nominal, so every tone over ~0.80
   **clips to the same white** — 0.94 and 0.84 render bit-identically;
2. the G1's **curved, specular silver** renders far from its 0.7 nominal, and where depends on
   the FRAMING (0.59 at a close shot, ~0.80 at a far one);
3. a floor is not only a backdrop — it also **carries the cast shadow**, which is the only cue
   that the robot stands on it.

One trajectory, four floors, repainted between renders so the panels are the same motion
frame-for-frame (`az 135 / el -20 / d 3.5`, the framing a clip is cut at):

| stage | floor L | clipped | **robot ΔL** | **shadow ΔL** |
|---|---|---|---|---|
| light grey 0.76 | 1.00 | 83% | **0.00** | 0.51 |
| **slate 0.30 (this)** | 0.42 | 0% | 0.42 | **0.22** |
| graphite 0.18 | 0.25 | 0% | 0.58 | 0.13 |
| matte black 0.02 | 0.04 | 0% | **0.80** | **0.02** |

**The two columns pull in opposite directions.** A light floor meets the silver at the top of the
range — 0.76 measured a robot ΔL of exactly **zero**, the silhouette carried by shadow alone. A
black floor wins the silhouette outright and **deletes the shadow**, leaving a robot that floats
in front of a surface with no gradient. Slate is the darkest tone that still holds a visible
shadow — a joint optimum, not a split difference. **Move it and re-measure BOTH columns.**

### 5.2 · the two bars, and why they differ

| | bar | why |
|---|---|---|
| TASK objects (cube faces, ball) | **100 nominal**, hard | a thing the policy must SEE cannot alias the floor — that deletes the exteroception. Slate: cube 152, ball 150 |
| the SCENE (robot, props) | rendered ΔL + shadow, not a radius | a radius around silver's nominal would have passed 0.76 (robot ΔL 0.00) and matte black (shadow 0.02). It sees neither failure |

**One hard constraint survives all four iterations:** play FPV feeds the frozen encoder, so the
stage must stay inside the training palette's SPAN — an interpolation of what the encoder saw,
never an extrapolation past it (`test_stage_sits_inside_the_palette_span`, per channel).
Alternatives, all one-constant swaps: graphite 0.18 (moodier, shadow at the edge of legibility),
off-white 0.94 (paper-native, blends into a white page), charcoal / steel blue (palette entries).

**The material half is not optional, and repose is the proof.** `texture > mat_rgba > geom_rgba`
(§1), a plane ships mjlab's checker, and repose never called `flat_floor` — its head cam's
`use_textures=False` hid the checker from the FPV **and only from it**. So the first cut of the
stage painted a floor that `play --viewer native` never showed. A generated terrain (perloco,
dodge's room) carries no material at all (`geom_matid == -1` on every terrain geom), which is why
the event alone is enough there.

**Why off-white and not something softer.** The bar is `_MIN_PAIR_DIST` = 100 rgb-dist (0-255),
the same one the ground palette already enforces, measured against everything that stands on the
floor. Off-white is the only candidate that clears it on all of them:

| stage candidate | G1 silver | G1 black | ball red | uolm wood | cube (nearest face) | **worst** |
|---|---|---|---|---|---|---|
| **off-white 0.94** | 106 | 327 | 274 | 245 | 189 (yellow) | **106** |
| steel blue .30/.40/.50 | 137 | 95 | 175 | 90 | 117 | 90 |
| porcelain 0.86 | 69 | 290 | 242 | 208 | 164 | 69 |
| sage / mint / putty | 33-48 | — | — | — | — | 33-48 |
| mid-grey 0.50 | 88 | 133 | 146 | 70 | 115 | 47 |

Every softer tone dies on the ROBOT, not on the cube — silver is 0.7, and a "nice light palette"
sits right on top of it. 0.90 white is already at 88.

**It films CLIPPED, and that is the trade.** At 112x63 with the sun on, 90-94% of stage pixels
land at exactly 255 — a blown-out white with no shading gradient left in the floor. That is what
buys the 106 against silver: rendered separation is maximized precisely where the floor
saturates. If a clip needs a stage with visible shading instead, `STAGE_RGBA = steel blue` is a
one-constant swap at worst-pair 90 — read that number before making the swap, not after.
