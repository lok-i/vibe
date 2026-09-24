"""G1 uni-object loco-manipulation env config — a THIN import over orcs's.

    g1_uolm_env_cfg(play, ...)

orcs owns the task: the six-object roster and its per-world variants, the demo
clips, VOF, rewards, terminations, RSI, the robustness domain, the frozen SONIC
base, the critic. vibe changes exactly one thing — what the ADAPTER reads: the
object kinematics (and the object_id one-hot) become frozen encoder features
from the head camera. That is what makes this row comparable to
`Orcs-Uolm-AdaptSonic` rather than merely similar to it.

**This row is built to transfer** (2026-08-06). Run 1 held everything at orcs
parity to isolate the exteroception swap; it worked, so the question is now
hardware. Two changes follow: the adapter loses every privileged term it had
left (root state out, goal moved to a query row — `observation_cfgs`), and vibe
adds the render domain repose transferred on. orcs's physical robustness domain
still rides along verbatim underneath.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from orcs.assets import OBJECT_BODY_NAME
from orcs.tasks.uolm.env_cfg import uolm_env_cfg

from vibe.assets.g1 import G1_BASE_CFG, G1_VIS_LEAN_CFG
from vibe.core import sensors
from vibe.core.env_cfgs import (
    apply_query_noise,
    apply_render_domain,
    apply_stage_render,
    assert_play_is_clean,
    domain_state,
    flat_floor,
)
from vibe.core.mdp.metrics import object_in_fov
from vibe.tasks.uolm.config.g1 import observation_cfgs as oc

ENV_SPACING = 2.5
"""Metres between env origins — vibe's, tighter than orcs's 6.0 for uolm.

Layout only: worlds do not interact, so spacing moves nothing the policy senses
and nothing the physics computes. It shows in exactly one place — a viewer or a
recorder drawing NEIGHBOUR envs (`ShotPose.context_envs`), where 6.0 spreads the
roster across a field and 2.5 reads as one crowded shoot.

orcs keeps 6.0 for its own reason (a largetable is metres wide and wants
clearance in that same multi-env draw), which is why this is set here and not
there. At 2.5 the biggest roster objects DO overlap their neighbour's footprint
on screen — the cost, taken deliberately, and the reason to raise this number if
a largetable shot looks cluttered.
"""


def g1_uolm_env_cfg(*, play: bool = False, **orcs_kw) -> ManagerBasedRlEnvCfg:
    """Omni-object loco-manip, adapted through a camera instead of object state.

    Args:
        play:   Forwarded to orcs, which applies its own play overrides.
        **orcs_kw: passed through to `uolm_env_cfg` (object_names, collision,
                kill_bodies, command_space, ...).

    The privileged object state stays wired for the CRITIC — a value function
    that still sees the object is the whole reason the actor can afford not to.
    """
    # `robot_cfg` is already an orcs parameter, so the camera-lean visual set
    # costs no orcs change. play gets the whole robot so the viewer shows one.
    cfg = uolm_env_cfg(
        agent="sonic", play=play,
        robot_cfg=G1_BASE_CFG if play else G1_VIS_LEAN_CFG,
        **orcs_kw,
    )
    before = domain_state(cfg)

    cfg.scene.env_spacing = ENV_SPACING

    sensors.attach_head_cam(cfg)
    # Episode_Metrics/object_in_fov — the vision duty cycle. THE first number to
    # read on a roster this size: a largetable and a tire do not sit in the same
    # part of the frame, and a row that never sees its object is a camera-aim
    # problem, not an extractor one.
    cfg.metrics = dict(cfg.metrics or {})
    cfg.metrics["object_in_fov"] = MetricsTermCfg(
        func=object_in_fov,
        params={"sensor_name": sensors.HEAD_CAM_NAME,
                "object_cfg": SceneEntityCfg(OBJECT_BODY_NAME)},
    )

    ctx = oc.ObsCtx(obj=SceneEntityCfg(OBJECT_BODY_NAME), p={"command_name": "motion"})
    oc.attach_ext_obs(cfg, ctx=ctx)

    # ── the render domain: camera mount, sun angle, floor palette ──
    # The FLOOR is randomized; the OBJECTS are not, and that asymmetry is the
    # whole colour policy. The roster's appearance is a real, fixed property of
    # the physical objects this transfers onto (a black tire, a white trashcan,
    # wood), so randomizing it would train an invariance we do not need against
    # a cue we do have. A floor is whatever room you are standing in.
    flat_floor(cfg)  # the checker BEATS mat_rgba — kill it or nothing recolours
    apply_render_domain(cfg, play=play)
    apply_query_noise(cfg, play=play)

    if play:
        # orcs runs its play overrides INSIDE `uolm_env_cfg`, i.e. before any of
        # the above existed — so a domain knob added here would silently train
        # the eval. The guard diffs against `before` rather than matching names
        # (vibe.core.env_cfgs).
        assert_play_is_clean(cfg, before)
    # after the guard on purpose: a fixed film stage is not a domain
    apply_stage_render(cfg, play=play)
    return cfg
