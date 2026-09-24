"""observation_cfgs.py — THE vibe signal library.

One place for every vibe-owned observation group + the atomic term bundles they
share, so an experiment is a re-pick of groups, never a re-plumb. The groups read
as the sys0 hierarchy (docs/perception/encoders.md):

    [base]      policy stream (robot-motion-cmd + state)      — mocke WBC contract
    [adapter]   ADAPTER_STREAM (robot-motion-cmd) + z          — this file
    [extractor] KV_TOKENS  ⟨queried by⟩  QUERY_GROUPS -> z     — this file

The frozen-base obs (policy / tokenizer streams) live in mocke and the privileged
object-kinematics groups in orcs (dependency direction: vibe imports both, never
the reverse); everything VISION — the part that is vibe's reason to exist — is
here. DRY is load-bearing: `robot_motion_cmd_terms` feeds BOTH
the adapter stream AND the `q_motion_cmd` query (one signal, two roles, one def);
`proprio_terms` feeds BOTH `q_proprio` AND the aux `prediction_conditioning`.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg
from orcs.tasks.uolm.observation_cfgs import (
    _T,
    _grp,
    critic_group,
    object_goal_terms,
    object_state_terms,
    proprio_terms,
    robot_motion_cmd_terms,
)
from orcs.tasks.uolm.observation_cfgs import ObsCtx as _ObjKinCtx
from orcs.tasks.uolm.observation_cfgs import (
    augmentation_group as _orcs_augmentation_group,
)

from vibe.core.observation_cfgs import (
    CAMERA_GROUP,
    CLS_GROUP,
    IMG_DTYPE,
    IMG_ENCODER,
    TOKEN_GROUP,
    TOKEN_TERMS,
    CamSpec,
    camera_group,
    cls_query_group,
    img_flat_term,
    kv_tokens_group,
)
from vibe.tasks.repose import mdp

# The signal ATOMS (proprio, robot-motion-cmd, object state) are orcs's — one
# definition, so a change to what "proprio" means reaches the privileged task and
# the vision task together. The VISION atoms (encoder choice, token group naming,
# the token/cls terms) are `vibe.core.observation_cfgs` — shared with every other
# vibe task, so a backbone swap reaches them together. What is left here, and only
# here, is what repose means by a query row and a prediction target.
__all__ = [
    "IMG_ENCODER", "IMG_DTYPE", "TOKEN_GROUP", "TOKEN_TERMS", "CLS_GROUP",
    "CAMERA_GROUP", "DEFAULT_QUERY_GROUPS", "QUERY_GROUPS", "ObsCtx",
    "kv_tokens_group", "camera_group", "adapter_stream_group",
    "prediction_target_group", "prediction_conditioning_group",
    # re-exported unchanged from orcs: the privileged groups are shared
    "objkin_augmentation_group", "critic_group",
    # vibe's own
    "vision_augmentation_group", "attach_aux_obs",
]

# BACKBONE SWAP is TWO coupled lines and they are NOT both here: `IMG_ENCODER`
# lives in `vibe.core.observation_cfgs`, `DEFAULT_QUERY_GROUPS` below is the
# other half (q_cls belongs to a CLIP-family CLS only). Swap them as a pair or
# the extractor spends an attention row on an untrained token.

# DEFAULT_QUERY_GROUPS = ("q_task_cmd", "q_motion_cmd", "q_proprio")
# DEFAULT_QUERY_GROUPS = ("q_task_cmd", "q_motion_cmd", "q_proprio", CLS_GROUP)
DEFAULT_QUERY_GROUPS = ("q_task_cmd", "q_proprio", CLS_GROUP)


@dataclass(frozen=True)
class ObsCtx(_ObjKinCtx, CamSpec):
    """orcs's assembly context + which camera and encoder to read it through."""


# ---------------------------------------------------------------------------
# Extractor query channels — each group is ONE (B, d) vector -> ONE attn row
# ---------------------------------------------------------------------------

QUERY_GROUPS = {
    # task command: what to achieve (static/episode) — becomes LANGUAGE later (swap this def)
    "q_task_cmd": lambda c: _grp({"object_goal_ori": _T(mdp.object_goal_ori_mat6d, c.p)}),
    # robot-motion command: what motion now (time-varying) — the strongest "where to look" cue
    "q_motion_cmd": lambda c: _grp(robot_motion_cmd_terms(c.p)),
    # proprio feedback: where am I now (posture, hands)
    "q_proprio": lambda c: _grp(proprio_terms()),
    # encoder global token: meaningful for CLIP/TinyCLIP (Theia CLS ~ noise). Defined
    # always, but OFF by default — only in DEFAULT_QUERY_GROUPS on the [CLIP] branch.
    CLS_GROUP: cls_query_group,
}


# ---------------------------------------------------------------------------
# The named groups (adapter-stream / aux predictor)
# ---------------------------------------------------------------------------

def adapter_stream_group(c: ObsCtx) -> ObservationGroupCfg:
    """Adapter's direct stream: robot-motion-cmd VALUE (sys0 dynamics signal).

    Same terms as `q_motion_cmd` — the query decides *where to look*, this stream
    carries the command *value* the adapter acts on."""
    return _grp(robot_motion_cmd_terms(c.p))


def objkin_augmentation_group(c: _ObjKinCtx) -> ObservationGroupCfg:
    """orcs's adapter stream, pinned to the BASE-frame, identity-free layout.

    uolm moved its augmentation to env frame + object_id (2026-08-01) — right
    there, wrong here. In repose this group is the privileged TWIN of
    `imgfeat_augmentation_group`: the two must differ in exactly one thing, the
    object-state pair, or the ObjKin-vs-ImgFeat comparison stops being
    controlled. Odometry and object identity have no image-side twin to swap
    against, and repose's roster is one cube, so both are excluded here.
    """
    return _orcs_augmentation_group(c, frame="base", identity=False)


def prediction_target_group(c: ObsCtx) -> ObservationGroupCfg:
    """Aux target: what z must explain (ego-observable object state).

    DICT group (concat=False) deliberately: StateFdAux derives one loss slice PER TERM
    from it — name and width both — so the wandb `ZPrediction/<term>` split maintains
    itself and a term added here can never be silently sliced off the loss. Adding a
    target is ONE edit: append a term. The term's NAME is the metric's name, so rename
    here to rename the panel.
    """
    return _grp(object_state_terms(c.obj), concat=False)


def prediction_conditioning_group(c: ObsCtx) -> ObservationGroupCfg:
    """Aux conditioning r: robot self-state (no object state — that's image-reachable only)."""
    return _grp(proprio_terms())


def vision_augmentation_group(c: ObsCtx, *, feat: bool = True) -> ObservationGroupCfg:
    """The adapter's stream with VISION in place of object kinematics.

    Term-for-term `objkin_augmentation_group` with the object-state pair swapped
    for encoder features — same goal + sys1 command feedforward, so the two
    exteroception modes differ in exactly one thing, which is the experiment.

    `feat=False` is ImgRgb: pixels reach the actor through the `camera` group
    instead, so both vision rows share this stream term for term. Flat when
    present — `attach_aux_obs` replaces the group entirely when an extractor is
    in play (tokens go to `kv_tokens`, and the adapter reads z instead).
    """
    return _grp({
        **({"feat": img_flat_term(c.sensor, c.model, c.model_dtype)} if feat else {}),
        # "base_lin_vel": _T(mdp.base_lin_vel),  #NOTE (lok-i) 1Aug2026: found insensitive
        **object_goal_terms(c.p),
        **robot_motion_cmd_terms(c.p),
    })


def attach_aux_obs(
    cfg,
    *,
    query_channels: tuple[str, ...] = DEFAULT_QUERY_GROUPS,
) -> None:
    """Rewire imgfeat obs into the sys0 extractor hierarchy (docs/perception/encoders.md).

    Pure assembly — every signal comes from the groups above:

      [base]      policy stream (untouched)                      -> tracks the motion
      [adapter]   ADAPTER_STREAM (robot-motion-cmd) + z          -> corrects the base
      [extractor] KV_TOKENS  ⟨queried by⟩  query_channels -> z   -> task-relevant vision

    The adapter reads z ONLY for task/vision signal; robot-motion-cmd reaches it
    directly (its VALUE, not just a query). The goal is query-only
    (vision-grounded). `query_channels` selects the active rows.
    """
    feat = cfg.observations["augmentation"].terms["feat"]  # inherit sensor + encoder
    ctx = ObsCtx(sensor=feat.params["sensor_name"], model=feat.params["model_name"],
                 model_dtype=feat.params["model_dtype"])

    # extractor K/V + one query row per active channel
    cfg.observations[TOKEN_GROUP] = kv_tokens_group(ctx)
    for name in query_channels:
        cfg.observations[name] = QUERY_GROUPS[name](ctx)
    # adapter stream: robot-motion-cmd value (replaces the vision/goal-laden aug)
    cfg.observations["augmentation"] = adapter_stream_group(ctx)
    # aux predictor groups
    cfg.observations["prediction_target"] = prediction_target_group(ctx)
    cfg.observations["prediction_conditioning"] = prediction_conditioning_group(ctx)
