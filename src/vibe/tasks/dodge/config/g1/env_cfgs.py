"""G1 dodgeball env config — a THIN import over orcs's.

    g1_dodge_env_cfg(play, ...)

orcs owns the task: the ball, the throw distribution, the nominal-stand
reference, the reward pair, the hit/fall terminations, the frozen SONIC base,
the critic. vibe changes exactly one thing — what the ADAPTER reads: the ball's
kinematics become frozen encoder features from the head camera. That is what
makes this row comparable to `Orcs-Dodge-AdaptSonic` rather than merely similar
to it.

Three departures from the other vibe seams, all forced by the task:

  camera   aimed 2 deg UP instead of the shared 45 DOWN (`sensors.py`) — a ball
           arrives through the air, and the shared aim is a GROUND aim. Measured,
           not chosen; the sweep is in that module.
  floor    matte and untextured, where the other tasks only need it recolourable
           — a reflective plane renders a MIRRORED ball (note below).
  the ask  every other vibe task hands the adapter a reference that already
           performs the task AND a sys1 command to condition on. Here the
           reference says "stand still" and there is no sys1, so the adapter
           reads **z alone** — see `observation_cfgs`. That makes dodge the
           minimal statement of task-optimal behaviour adaptation in this repo:
           nothing but vision can author the evasion.

**This row is built to transfer** (2026-08-06): the full render domain plus
orcs's robot robustness, the set repose transferred on. Run 1's nominal domain
answered the behaviour question; this one asks the hardware question.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from orcs.assets import BALL_BODY_NAME
from orcs.tasks.dodge.env_cfg import dodge_env_cfg

from vibe.assets.g1 import G1_BASE_CFG, G1_VIS_LEAN_CFG
from vibe.core.env_cfgs import (
    apply_query_noise,
    apply_render_domain,
    apply_stage_render,
    assert_play_is_clean,
    domain_state,
    flat_floor,
)
from vibe.core.mdp.metrics import object_in_fov
from vibe.core.sensors import HEAD_CAM_NAME
from vibe.tasks.dodge.config.g1 import observation_cfgs as oc
from vibe.tasks.dodge.config.g1.sensors import attach_dodge_cam
from vibe.tasks.dodge.terrain import apply_room_light_domain, room_terrain_cfg

DODGE_CONE_FAST_THROW = {
    "interval_range_s": (1.2, 2.2),
    "dist_range": (2.0, 3.5),
}
"""ConeFast changes only throw duty cycle and release distance.

The reaction-time distribution remains orcs's default. Releasing farther raises
speed to roughly 5.6 m/s without shortening that reaction window; the 1.2-2.2 s
interval roughly doubles the fraction of training frames containing a throw.
"""

# The flat floor is `vibe.core.env_cfgs.flat_floor` since 2026-08-06 — every
# vision task on a plane needs it, for the reason recorded there (the checker
# ALIASES at 112x63, and it BEATS mat_rgba so nothing else can recolour the
# ground). Dodge found it, so the dodge-specific half of the finding lives here:
#
#   REFLECTANCE  mjlab's groundplane ships `reflectance=0.2`, and a reflective
#                MuJoCo plane renders a MIRRORED BALL below the floor line. A
#                second, fake, converging ball in frame is a worse distractor
#                than the checker ever was.
#   PARALLAX     against "a checker gives parallax the policy could read
#                velocity from": it cannot today — the extractor sees ONE frame
#                and z carries no temporal channel, so a single image has no
#                velocity in it whatever the floor looks like. If a token/z
#                history lands, the answer is a texture ablation, not a guess.
#                Even then the geometry is unhelpful: after the re-pitch the
#                ball's median elevation sits near the camera axis with much of
#                the flight ABOVE the horizon, projected against background.
#
# The floor COLOUR is now randomized per env like every other vibe task, so the
# 0.6 grey is a nominal rather than the value. What still holds after the swap
# is the ball's separability: an 0.85/0.25/0.12 red is separated from the whole
# ground palette in CHROMA, which is what survives a low-resolution ViT patch
# more reliably than luminance.


def g1_dodge_env_cfg(
    *,
    room: bool = False,
    camera_height: int | None = None,
    play: bool = False,
    **orcs_kw,
) -> ManagerBasedRlEnvCfg:
    """Whole-body evasion, adapted through a camera instead of ball state.

    Args:
        room:   Replace the legacy plane with Dodge's indoor room.
        camera_height: Task-local camera-height override in pixels.
        play:   Forwarded to orcs, which applies its own play overrides.
        **orcs_kw: passed through to `dodge_env_cfg` (ball_radius, station_std,
                and the whole threat model via its `**throw_kw`).

    The privileged ball state stays wired for the CRITIC — a value function that
    still sees the ball is the whole reason the actor can afford not to.
    """
    # `robot_cfg` is already an orcs parameter, so the camera-lean visual set
    # costs no orcs change. play gets the whole robot so the viewer shows one.
    cfg = dodge_env_cfg(
        play=play,
        robot_cfg=G1_BASE_CFG if play else G1_VIS_LEAN_CFG,
        **orcs_kw,
    )
    before = domain_state(cfg)

    camera_kw = {} if camera_height is None else {"height": camera_height}
    attach_dodge_cam(cfg, **camera_kw)
    if room:
        cfg.scene.terrain = room_terrain_cfg()
    else:
        flat_floor(cfg)
    # Episode_Metrics/object_in_fov — the vision duty cycle, and on THIS task the
    # first number to read: a ball is in frame for a fraction of a 0.6 s flight,
    # so a row that cannot dodge may simply never have seen the throw. That is a
    # camera-aim result, not an extractor one, and the two are only separable
    # because this is measured rather than assumed.
    cfg.metrics = dict(cfg.metrics or {})
    cfg.metrics["ball_in_fov"] = MetricsTermCfg(
        func=object_in_fov,
        params={"sensor_name": HEAD_CAM_NAME,
                "object_cfg": SceneEntityCfg(BALL_BODY_NAME)},
    )

    # deletes `augmentation`: this adapter reads z alone
    oc.attach_ext_obs(cfg, ctx=oc.ObsCtx(p={"command_name": "motion"}))

    # ── the render domain: camera mount, sun angle, floor palette ──
    # The BALL is not randomized — it is one fixed red ball on the other side of
    # this transfer, so its colour is a known constant, not a nuisance. The floor
    # is whatever room you are standing in, so it is.
    apply_render_domain(cfg, play=play)
    if room:
        apply_room_light_domain(cfg, play=play)
    apply_query_noise(cfg, play=play)

    if play:
        # orcs runs its play overrides INSIDE `dodge_env_cfg`, i.e. before any of
        # the above existed — so a domain knob added here would silently train
        # the eval. The guard diffs against `before` rather than matching names
        # (vibe.core.env_cfgs).
        assert_play_is_clean(cfg, before)
    # after the guard on purpose: a fixed film stage is not a domain. No
    # `ground_rgba` — the room's WALLS take the stage colour too, which is
    # what makes a ConeFast clip cut against a plane clip.
    apply_stage_render(cfg, play=play)
    return cfg


def g1_dodge_cone_fast_env_cfg(
    *, play: bool = False, **orcs_kw
) -> ManagerBasedRlEnvCfg:
    """Dodge in the indoor room: faster throws, farther releases, a 64-px camera."""
    return g1_dodge_env_cfg(
        room=True,
        camera_height=64,
        play=play,
        **(DODGE_CONE_FAST_THROW | orcs_kw),
    )
