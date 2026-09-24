"""G1 perceptive-locomotion env configs — THIN imports over orcs's.

    g1_perloco_omre_env_cfg(play, ...)     OmniRetarget climb
    g1_perloco_grail_env_cfg(play, ...)    GRAIL curb

One factory per source, side by side over a shared `_vision` — orcs's own shape,
for orcs's own reason: the sources differ in the things a `source=` switch would
hide (grid axes, tile size, anchor tubes, the render-height knob), and every one
of those stays orcs's to declare. vibe changes exactly one thing on either — what
the ADAPTER reads: orcs's 187-ray height scan becomes frozen encoder features
from the head camera. That is what makes these rows comparable to
`Orcs-PerLoco-{OmRe,Grail}-AdaptSonic` rather than merely similar to them.

**This row is built to transfer** (2026-08-06), which changed what "one thing
moves" means. Run 1 held everything at orcs parity — no domain, root state in
the adapter stream — to ask whether a frozen WBC could read terrain through a
camera at all. It can, so the question is now hardware, and two things follow:
the adapter loses the privileged root state (odometry + estimated base velocity)
and gains the full render domain, the one repose transferred on.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from orcs.tasks.perloco.env_cfg import grail_env_cfg, omni_env_cfg
from orcs.tasks.perloco.sensors import TERRAIN_SCAN_SENSOR_NAME
from orcs.tasks.perloco.terrain import FLOOR_RGBA as TILE_FLOOR_RGBA

from vibe.assets.g1 import G1_BASE_CFG, G1_VIS_LEAN_CFG
from vibe.core import sensors
from vibe.core.env_cfgs import (
    apply_query_noise,
    apply_render_domain,
    apply_stage_render,
    assert_play_is_clean,
    domain_state,
)
from vibe.tasks.perloco.config.g1 import observation_cfgs as oc


def g1_perloco_omre_env_cfg(*, play: bool = False, **orcs_kw) -> ManagerBasedRlEnvCfg:
    """OmniRetarget climb, adapted through a camera instead of a height scan.

    Everything that makes the grid — climb families, the roster, and
    `render_z_scale` (orcs pins the obstacle height while the motions stay
    per-row) — is orcs's and forwarded, not re-declared here.
    """
    return _vision(omni_env_cfg, play=play, **orcs_kw)


def g1_perloco_grail_env_cfg(*, play: bool = False, **orcs_kw) -> ManagerBasedRlEnvCfg:
    """GRAIL curb, adapted through a camera instead of a height scan."""
    return _vision(grail_env_cfg, play=play, **orcs_kw)


def _vision(orcs_factory, *, play: bool, **orcs_kw) -> ManagerBasedRlEnvCfg:
    """THE swap, shared by both sources — it does not depend on which terrain.

    Args:
        play:   Forwarded to orcs, which applies its own play overrides.
        **orcs_kw: passed through (roster, scan_frame, tile_size, anchor
                thresholds, command_space, render_z_scale, ...).

    The height-scan SENSOR stays wired: the critic reads it, and a privileged
    value function is the whole reason the actor can afford to lose it.
    """
    # `robot_cfg` is already an orcs parameter, so the camera-lean visual set
    # costs no orcs change. play gets the whole robot so the viewer shows one.
    cfg = orcs_factory(
        agent="sonic", play=play,
        robot_cfg=G1_BASE_CFG if play else G1_VIS_LEAN_CFG,
        **orcs_kw,
    )
    before = domain_state(cfg)

    sensors.attach_head_cam(cfg)
    oc.attach_ext_obs(cfg, ctx=oc.ObsCtx(p={"command_name": "motion"}))

    # ── the render domain: camera mount, sun angle, terrain palette ──
    # The tile FLOOR and the thing standing on it draw independent colours,
    # split on orcs's own nominal ground colour — a curb the same colour as its
    # ground is not a harder domain, it is an invisible curb.
    #
    # It also removes a measured shortcut. Obstacle hue is interpolated across
    # the LEVEL rows (`terrain._BOX_RGBA_{LO,HI}`), so on OmRe the built model
    # carries three obstacle colours for three z_scale rows — while
    # `render_z_scale` pins every one of them to the same GEOMETRY. Hue was
    # therefore the only rendered signal that still tracked the level.
    apply_render_domain(cfg, play=play, terrain_ground_rgba=TILE_FLOOR_RGBA)
    apply_query_noise(cfg, play=play)

    # Scan rays off. orcs draws them under `play` because there they ARE the
    # adapter's exteroception; here the adapter cannot see them, so leaving them
    # on would draw a signal the policy does not have. The SENSOR stays — the
    # critic reads it.
    for s in cfg.scene.sensors or ():
        if getattr(s, "name", None) == TERRAIN_SCAN_SENSOR_NAME:
            s.debug_vis = False

    if play:
        assert_play_is_clean(cfg, before)
    # after the guard on purpose: a fixed film stage is not a domain. The
    # CURBS keep their level-tracking hue — only the flat tile under them has
    # to match the other families.
    apply_stage_render(cfg, play=play, ground_rgba=TILE_FLOOR_RGBA)
    return cfg
