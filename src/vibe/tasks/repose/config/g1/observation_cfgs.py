"""observation_cfgs.py — repose's obs groups (docs/architecture.md).

    [base]      policy + tokenizer streams           — mocke (frozen SONIC contract)
    [adapter]   the motion command + z               — `adapter_stream_group`
    [extractor] kv_tokens ⟨queried by⟩ QUERY_GROUPS -> z

Signal atoms (proprio, motion command, object state) are orcs's and vision atoms are
`vibe.core.observation_cfgs`; what is repose's is its query rows and its aux target.
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

DEFAULT_QUERY_GROUPS = ("q_task_cmd", "q_proprio", CLS_GROUP)


@dataclass(frozen=True)
class ObsCtx(_ObjKinCtx, CamSpec):
    """orcs's assembly context + which camera and encoder to read it through."""


# ---------------------------------------------------------------------------
# Extractor query channels — each group is ONE (B, d) vector -> ONE attn row
# ---------------------------------------------------------------------------

QUERY_GROUPS = {
    # task command, fixed per episode. The vision rows swap the goal quat for the goal
    # COLOUR one-hot (`env_cfgs`), which is what the actor actually receives.
    "q_task_cmd": lambda c: _grp({"object_goal_ori": _T(mdp.object_goal_ori_mat6d, c.p)}),
    # the motion command; off by default — its value reaches the adapter stream
    "q_motion_cmd": lambda c: _grp(robot_motion_cmd_terms(c.p)),
    # proprio feedback: where am I now (posture, hands)
    "q_proprio": lambda c: _grp(proprio_terms()),
    # encoder global token — the one global-pool row (`cls_query_group`), on by default
    CLS_GROUP: cls_query_group,
}


# ---------------------------------------------------------------------------
# The named groups (adapter-stream / aux predictor)
# ---------------------------------------------------------------------------

def adapter_stream_group(c: ObsCtx) -> ObservationGroupCfg:
    """The adapter's direct stream: the motion command's VALUE.

    Same terms as `q_motion_cmd`: a query decides where to look, this stream carries
    the value the adapter acts on."""
    return _grp(robot_motion_cmd_terms(c.p))


def objkin_augmentation_group(c: _ObjKinCtx) -> ObservationGroupCfg:
    """orcs's adapter stream in the BASE frame, without object identity.

    The privileged twin of `vision_augmentation_group`: the two differ only in the
    object-state pair. Odometry and identity have no image-side twin, and repose's
    roster is one cube, so both are left out.
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
    for encoder features: same goal, same motion command.

    `feat=False` is ImgRgb: pixels reach the actor through the `camera` group
    instead, so both vision rows share this stream term for term. Flat when
    present — `attach_aux_obs` replaces the group entirely when an extractor is
    in play (tokens go to `kv_tokens`, and the adapter reads z instead).
    """
    return _grp({
        **({"feat": img_flat_term(c.sensor, c.model, c.model_dtype)} if feat else {}),
        **object_goal_terms(c.p),
        **robot_motion_cmd_terms(c.p),
    })


def attach_aux_obs(
    cfg,
    *,
    query_channels: tuple[str, ...] = DEFAULT_QUERY_GROUPS,
) -> None:
    """Wire the extractor + PPOAux groups onto an imgfeat cfg.

      [base]      policy stream (untouched)                  -> tracks the motion
      [adapter]   motion command + z                         -> corrects the base
      [extractor] kv_tokens ⟨queried by⟩ query_channels -> z
      [aux]       prediction_target · prediction_conditioning (train only)

    The goal becomes query-only. `query_channels` selects the active rows.
    """
    feat = cfg.observations["augmentation"].terms["feat"]  # inherit sensor + encoder
    ctx = ObsCtx(sensor=feat.params["sensor_name"], model=feat.params["model_name"],
                 model_dtype=feat.params["model_dtype"])

    # extractor K/V + one query row per active channel
    cfg.observations[TOKEN_GROUP] = kv_tokens_group(ctx)
    for name in query_channels:
        cfg.observations[name] = QUERY_GROUPS[name](ctx)
    # adapter stream: the motion command (replaces the flat-feature stream)
    cfg.observations["augmentation"] = adapter_stream_group(ctx)
    # aux predictor groups
    cfg.observations["prediction_target"] = prediction_target_group(ctx)
    cfg.observations["prediction_conditioning"] = prediction_conditioning_group(ctx)
