"""Env-cfg helpers every thin-import vision task shares.

Two jobs, and they are the two halves of one rule — vibe adds a domain to a cfg
orcs has already finished with:

  `apply_render_domain`   what vibe adds. It is a NO-OP under play, rather than
                          applied-then-subtracted, so there is no subtract list
                          to forget and the guard below stays load-bearing.
  `assert_play_is_clean`  proves it. A vibe task built ON TOP of an orcs factory
                          inverts repose's play invariant: orcs runs its play
                          overrides INSIDE its own factory, i.e. before vibe has
                          added anything, so a domain knob added afterwards
                          silently trains the eval.

The guard is a DIFF, not a name list: it cannot be outrun by a knob nobody
thought to list, and it cannot false-positive on a domain that was orcs's all
along (one of orcs's is literally called `rand_encoder_bias` — a JOINT encoder,
not this one).
"""

from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.spec_config import MaterialCfg
from orcs.core.obs import apply_obs_noise as _apply_obs_noise

from vibe.core import mdp
from vibe.core.mdp.observations import image_feature
from vibe.core.sensors import HEAD_CAM_NAME

__all__ = ["DomainState", "domain_state", "assert_play_is_clean",
           "apply_render_domain", "apply_query_noise", "QUERY_NOISY_GROUPS",
           "flat_floor", "FLOOR_RGBA", "apply_stage_render", "STAGE_RGBA",
           "VibeEnvCfg", "with_vision_knobs"]

QUERY_NOISY_GROUPS = ("q_proprio",)
"""The deployed streams VIBE owns, on top of the `policy` stream orcs stamps.

An extractor query row built from `proprio_terms` is a MEASUREMENT and must
carry the same sensor noise as the base's own stream — the table is orcs's
(`orcs.core.obs.OBS_NOISE`, one set of numbers), only the group list is ours.
`kv_tokens`/`q_cls` are excluded: image noise is a separate axis, not this one.
"""

DomainState = tuple[frozenset[str], frozenset[str]]
"""(event names, obs groups with corruption on) — what a domain looks like."""


def domain_state(cfg: ManagerBasedRlEnvCfg) -> DomainState:
    """Snapshot the domain surface. Take one BEFORE vibe touches an orcs cfg."""
    return (
        frozenset(cfg.events or ()),
        frozenset(n for n, g in cfg.observations.items() if g.enable_corruption),
    )


def assert_play_is_clean(cfg: ManagerBasedRlEnvCfg, before: DomainState) -> None:
    """Fail if vibe ADDED any domain to a play cfg (orcs's own domain is its call)."""
    events, corrupt = domain_state(cfg)
    added_events = sorted(events - before[0])
    added_noise = sorted(corrupt - before[1])
    assert not added_events and not added_noise, (
        f"play cfg carries training domain vibe added — events {added_events}, "
        f"corrupted obs {added_noise}. orcs's play overrides ran BEFORE these "
        f"existed, so subtract them under `play` (see this module's docstring).")


# ---------------------------------------------------------------------------
# The render domain — sim2real for the PIXELS
# ---------------------------------------------------------------------------

FLOOR_RGBA = (0.6, 0.6, 0.6, 1.0)
"""Nominal matte mid-grey floor, replacing mjlab's checkered groundplane.

Two things go, and the second is the one that is hard to find:

  the CHECKER   A high-frequency pattern sampled at 112x63 ALIASES, and the
                camera is head-mounted, so the moire it produces moves whenever
                the robot does. That is synthetic motion in exactly the channel
                a looming cue has to live in — worse than a missing texture,
                because it is a confusing one. It also BEATS `mat_rgba`, so a
                textured floor cannot be recoloured at all: killing it is what
                makes `rand_terrain_color` visible on a plane. Killed at the
                source (`texture=None`) rather than with the camera's
                `use_textures` flag, which is a blunter instrument that flattens
                every OTHER material in the scene too — including the object
                textures uolm's roster is told apart by.
  REFLECTANCE   mjlab's groundplane ships `reflectance=0.2`, and a reflective
                MuJoCo plane renders a MIRRORED copy of anything above it. On
                dodge that is a second, fake, converging ball in frame.

0.6 is the nominal only — `rand_terrain_color` overwrites it per env. It sits
clear of both robot materials (silver 0.7, black 0.2) so the limbs stay
separable from the floor in the one env that draws it.
"""


def flat_floor(cfg: ManagerBasedRlEnvCfg, rgba=FLOOR_RGBA) -> None:
    """Replace a plane terrain's checkered, reflective groundplane with matte flat.

    Plane terrains only (uolm, dodge, repose); a GENERATED terrain (perloco)
    carries per-geom `rgba` and no texture, so there is nothing here to undo.
    """
    terrain = cfg.scene.terrain
    assert terrain is not None, "flat_floor needs a scene terrain"
    cfg.scene.terrain = dataclasses.replace(terrain, materials=(
        MaterialCfg(name="groundplane", rgba=rgba, texture=None,
                    reflectance=0.0, geom_names_expr=("terrain$",)),
    ))


def apply_render_domain(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool = False,
    sensor_name: str = HEAD_CAM_NAME,
    terrain_color: bool = True,
    terrain_ground_rgba: tuple[float, float, float, float] | None = None,
) -> None:
    """Per-env-constant camera + light + floor variation. Bit-identical to repose.

    NO-OP under `play` — that is the design, not a shortcut. A vibe task adds
    this AFTER orcs has run its own play overrides, so the alternative is an
    apply-then-subtract pair whose subtract half is a list someone forgets;
    `assert_play_is_clean` then has nothing to catch and stays honest.

    Light INTENSITY is not here and cannot be: mjwarp's batched `Model` carries
    `light_{type,castshadow,active,pos,dir}` and no diffuse/ambient field, so
    fcrl's "intensity 1000-4000" has no per-world equivalent. Direction is the
    half that survives — and it is the half that moves shading and shadows.

    `terrain_ground_rgba` forwards to `rand_terrain_color`: a task whose terrain
    has RAISED geometry passes its ground's nominal colour, and everything that
    is not the ground draws a colour guaranteed separable from it.
    """
    if play:
        return
    cfg.events["rand_cam_extrinsics"] = EventTermCfg(
        func=mdp.rand_cam_extrinsics,
        mode="startup",
        params={"sensor_name": sensor_name,
                "pos_range": (-0.02, 0.02),      # m
                "rot_range": (-0.035, 0.035)},   # rad, ~2deg
    )
    # sun tilt: +-0.25 on the xy components of a (0,0,-1) directional light is a
    # ~14deg cone. mjlab's own term, per-world like every other `dr.*`.
    cfg.events["rand_light_dir"] = EventTermCfg(
        func=dr.light_dir,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("terrain", light_names="sun"),
                "operation": "add",
                "ranges": {0: (-0.25, 0.25), 1: (-0.25, 0.25)}},
    )
    if terrain_color:
        cfg.events["rand_terrain_color"] = EventTermCfg(
            func=mdp.rand_terrain_color,
            mode="startup",
            params={"ground_rgba": terrain_ground_rgba},
        )


# ---------------------------------------------------------------------------
# The STAGE — one floor for every task, so four families collage as one shoot
# ---------------------------------------------------------------------------

STAGE_RGBA = (0.30, 0.30, 0.32, 1.0)
"""The play-only ground colour, shared by every vibe task. Cool slate.

**Measure the RENDER, never the nominal, and measure TWO things** — the robot
against the floor AND the floor against its own cast shadow. One trajectory,
four floors, repainted between renders so the panels are the same motion
frame-for-frame (`az 135 / el -20 / d 3.5`, the framing a clip is cut at):

    stage              floor L   clipped   robot dL   SHADOW dL
    light grey 0.76      1.00       83%      0.00        0.51
    slate 0.30 THIS      0.42        0%      0.42        0.22
    graphite 0.18        0.25        0%      0.58        0.13
    matte black 0.02     0.04        0%      0.80        0.02

The two columns pull in OPPOSITE directions and that is the whole design:

  a LIGHT floor rides above its nominal under the sun's ~1.2 gain and clips, so
  it meets the G1's specular silver at the top of the range — 0.76 measured a
  robot dL of exactly ZERO, i.e. the silhouette was carried by the shadow alone.
  a BLACK floor wins the silhouette outright (0.80) and deletes the shadow
  (0.02), which is the only cue that the robot STANDS on the floor rather than
  floating in front of it; it also has no gradient left, so the ground stops
  reading as a surface at all.

Slate is the joint optimum, not a compromise: the darkest tone that still holds
a visible shadow. Move it and re-measure BOTH columns — a silhouette number
alone will happily walk you into a floating robot.

Neighbours, one-constant swaps: graphite 0.18 (moodier, shadow at the edge of
legibility) and off-white 0.94 (paper-native, blends into a white page, robot
carried by shadow only).

**What it costs:** play FPV feeds the frozen encoder, so a stage inside
`GROUND_RGBAS` would guarantee an in-distribution eval floor. This one sits
BETWEEN entries — an interpolation of what the encoder saw, never an
extrapolation past it, which is what `test_stage_sits_inside_the_palette_span`
holds. The task-object bar is untouched and still hard: 152 to the nearest cube
face, 150 to the dodge ball.
"""


def apply_stage_render(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool,
    ground_rgba: tuple[float, float, float, float] | None = None,
    rgba: tuple[float, float, float, float] = STAGE_RGBA,
) -> None:
    """Pin the ground to ONE colour under play. NO-OP while training.

    The mirror image of `apply_render_domain`: that one varies the floor and is
    a no-op under play, this one fixes it and is a no-op under train. Filming is
    the whole reason — `play=True` is what every recording and the viser session
    load, so every recording path inherits this for free.

    Call it AFTER `assert_play_is_clean` (and after repose's `_play_overrides`).
    A fixed colour is not a domain, and the ordering is what says so — the guard
    still sees the cfg vibe handed it, with nothing added.

    It REPLACES `rand_terrain_color` rather than sitting beside it: repose keeps
    that event alive under play on purpose (the colour relabel is its task
    channel, not a domain), so without the pop every repose clip would still
    draw its own random floor.

    `ground_rgba` is the nominal colour the terrain's GROUND draws — raised
    geometry keeps its own (perloco's curbs, whose hue tracks the level). `None`
    paints the whole terrain, which is what a plane and dodge's room want.

    Two writes, because a floor has two renderers to satisfy: the MATERIAL (via
    `flat_floor`, texture-less, what the viewer and the TPV recorder read) and
    the per-world `geom_rgba`/`mat_rgba` (the event, what mjwarp's rasterizer
    gives the policy). A generated terrain has no material and needs only the
    second.
    """
    if not play:
        return
    # The TEXTURE beats both rgba fields (docs/perception/render_domain.md §1),
    # and a plane ships mjlab's CHECKER — so a per-world write alone paints a
    # floor nobody sees. repose is where this bites: it never called
    # `flat_floor` (the head cam's `use_textures=False` hid the checker from the
    # FPV, and only from it), so the viewer and every TPV frame still drew it.
    if getattr(cfg.scene.terrain, "terrain_type", None) == "plane":
        flat_floor(cfg, rgba=rgba)
    cfg.events.pop("rand_terrain_color", None)
    cfg.events["set_terrain_color"] = EventTermCfg(
        func=mdp.set_terrain_color,
        mode="startup",
        params={"rgba": rgba, "ground_rgba": ground_rgba},
    )


def apply_query_noise(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool = False,
    groups: tuple[str, ...] = QUERY_NOISY_GROUPS,
) -> None:
    """Stamp orcs's sensor-noise table on the deployed streams vibe added.

    orcs stamps `policy` inside its own factory; the extractor's query rows
    exist only after vibe has run, so they are stamped here. Same table, two
    owners, each covering what it owns. NO-OP under play, same reason as
    `apply_render_domain`.
    """
    if play:
        return
    _apply_obs_noise(cfg, groups)


# ---------------------------------------------------------------------------
# The vision knobs a run — not a task — varies
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VibeEnvCfg(ManagerBasedRlEnvCfg):
    """mjlab's env cfg + the vision knobs that belong to a RUN, not to a task id.

    A backbone is not a naming axis (`docs/infra/naming.md`: a one-valued axis is
    not an axis) — it is a flag, so a bake-off is six lines of a manifest rather
    than six registrations.
    """

    img_encoder: str | None = None
    """Frozen-encoder override, applied to EVERY `image_feature` term at once.

        --env.img-encoder facebook/dinov3-vits16plus-pretrain-lvd1689m

    `None` keeps whatever the task built (`vibe.core.observation_cfgs.IMG_ENCODER`).

    ONE flag on purpose. The backbone name lives in two obs terms — `kv_tokens`'s
    `img_tokens` and `q_cls`'s `img_cls` — and tyro already exposes both
    `...terms.img-tokens.params.model-name` and `...terms.img-cls.params.model-name`.
    Setting one and not the other is silent and expensive: `image_feature._MODELS`
    keys on the name, so TWO backbones load, and `q_cls` then queries a different
    model's global token than the one producing K/V. Rebinding both here makes that
    unspellable.

    Token GEOMETRY is the camera's, not the backbone's — every stride-16 encoder in
    the roster yields P = 7x3 = 21 at the 112x63 head cam — so a swap moves the
    channel dim and leaves every attention metric on its existing normalizer.
    """

    def __post_init__(self) -> None:
        """Rebind the encoder. Runs at BUILD time and again after a tyro override
        (tyro reconstructs the dataclass), so it must stay idempotent — it is: the
        name is written, never derived from the previous one."""
        if not self.img_encoder:
            return
        for group in (self.observations or {}).values():
            for term in group.terms.values():
                if term.func is image_feature:
                    # copy, never mutate: a params dict that leaves a module is a COPY
                    term.params = {**term.params, "model_name": self.img_encoder}


def with_vision_knobs(cfg: ManagerBasedRlEnvCfg) -> VibeEnvCfg:
    """Re-type a finished env cfg as `VibeEnvCfg`, field for field. Idempotent.

    Called LAST in a task factory, after `assert_play_is_clean` — this adds a
    field, never a term, so it cannot change what the guard sees. tyro reads its
    flags off the DEFAULT INSTANCE rather than the `TrainConfig.env` annotation,
    which is why a subclass reaches the CLI without touching mjlab.
    """
    if isinstance(cfg, VibeEnvCfg):
        return cfg
    return VibeEnvCfg(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)})
