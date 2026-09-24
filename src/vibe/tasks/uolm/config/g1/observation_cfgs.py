"""observation_cfgs.py — uolm's VISION groups.

Read this against `orcs.tasks.uolm.observation_cfgs`: the two differ in the
object half of ONE group, and that is the whole experiment.

    [base]      policy + tokenizer streams   — mocke (frozen SONIC contract)
    [adapter]   AUGMENTATION                 — orcs:  object kin + id + root
                                                      state + goal + sys1 cmd
                                               vibe:  z, and sys1's cmd bundle
    [critic]    privileged full state        — orcs's, VERBATIM (object state
                                               AND object_id included)

**Everything the actor reads is now measurable on hardware** (2026-08-06). Four
things left the adapter stream, and only the first is the vision experiment:

  `object_state_w_*`   the swap — object kinematics become pixels.
  `object_id`          no image-side twin AND no need for one: six objects that
                       differ in geometry are exactly what a camera is supposed
                       to disambiguate, so handing the one-hot over for free
                       would answer a different question.
  root state           `robot_root_pos_env` is odometry, `robot_root_lin_vel_b`
                       wants a state estimator. Neither exists on hardware.
  `object_goal_*`      not deleted — moved to `q_task_cmd`, query-only. It stays
                       in the ENV frame precisely because a query needs no
                       frame closure (see below).

What remains is repose's stream exactly: `bodywise_contact_cmd` + the root
twist. Two tasks, one frozen base, one adapter interface.

Every non-vision atom is orcs's — `proprio_terms`, `object_goal_terms`,
`robot_motion_cmd_terms` — and every vision atom is
`vibe.core.observation_cfgs`. Nothing is defined here that either already owns;
what IS here is what uolm means by a query row.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg
from orcs.core.obs import grp as _grp
from orcs.core.obs import proprio_terms
from orcs.tasks.uolm.observation_cfgs import ObsCtx as _ObjKinCtx
from orcs.tasks.uolm.observation_cfgs import object_goal_terms, robot_motion_cmd_terms

from vibe.core.observation_cfgs import (
    CLS_GROUP,
    TOKEN_GROUP,
    CamSpec,
    cls_query_group,
    kv_tokens_group,
)

__all__ = [
    "ObsCtx", "DEFAULT_QUERY_GROUPS", "QUERY_GROUPS",
    "adapter_stream_group", "attach_ext_obs",
]

# The goal pose is uolm's task channel — "where should this object end up" is
# the strongest available "where should I look". `q_motion_cmd` is parked, as in
# repose: its VALUE still reaches control through the adapter stream.
DEFAULT_QUERY_GROUPS = ("q_task_cmd", "q_proprio", CLS_GROUP)

# The goal stays in the ENV frame and reaches the actor as a QUERY ONLY
# (2026-08-06) — repose's arrangement, and for repose's reason. As a stream
# value an env-frame goal is unusable once the odometry that placed it is gone;
# as a query it never needs to be placed, because `CrossAttentionExtractor` is
# pure pooling and the query's VALUE is not carried into z — it only steers
# where to look. What closes the loop is then the camera, which is the claim.


@dataclass(frozen=True)
class ObsCtx(_ObjKinCtx, CamSpec):
    """orcs's uolm context + which camera and encoder to read it through."""


# ---------------------------------------------------------------------------
# Extractor query channels — each group is ONE (B, d) vector -> ONE attn row
# ---------------------------------------------------------------------------

QUERY_GROUPS = {
    # task command: the object's goal pose, fixed per episode (from the level
    # ABOVE sys1). Not image-derivable, and query-ONLY since 2026-08-06 — the
    # note above `DEFAULT_QUERY_GROUPS` is why.
    "q_task_cmd": lambda c: _grp(object_goal_terms(c.p)),
    # sys1's command stream: per-body contact schedule + root twist. Parked —
    # the row it would buy is only worth an attention head once the goal row
    # earns its keep (docs/repose_representation.md).
    "q_motion_cmd": lambda c: _grp(robot_motion_cmd_terms(c.p)),
    # proprio feedback: where am I now (posture, hands)
    "q_proprio": lambda c: _grp(proprio_terms()),
    # encoder global token — the one sanctioned global-pool row (see core).
    CLS_GROUP: cls_query_group,
}


# ---------------------------------------------------------------------------
# The named groups
# ---------------------------------------------------------------------------

def adapter_stream_group(c: ObsCtx) -> ObservationGroupCfg:
    """The adapter's direct stream: sys1's command bundle, and nothing else.

    Identical to repose's, term for term — `bodywise_contact_cmd` + the root
    twist — which is the point: two tasks that share a frozen base and a
    contact-scheduled reference should present that base the same interface.

    **Two things left this group** (2026-08-06). Robot ROOT STATE, because
    `robot_root_pos_env` is odometry and `robot_root_lin_vel_b` wants a state
    estimator. And the GOAL, which is now query-only — see `QUERY_GROUPS`.
    """
    return _grp(robot_motion_cmd_terms(c.p))


def attach_ext_obs(
    cfg,
    *,
    query_channels: tuple[str, ...] = DEFAULT_QUERY_GROUPS,
    ctx: ObsCtx | None = None,
) -> None:
    """Rewire the obs into the sys0 extractor hierarchy (docs/infra/agents.md §3).

      [base]      policy stream (untouched)                   -> tracks the motion
      [adapter]   sys1 cmd bundle + z                         -> corrects the base
      [extractor] KV_TOKENS ⟨queried by⟩ query_channels -> z  -> task-relevant vision

    Command VALUES reach control through the adapter stream; the query only
    steers *where to look* (`CrossAttentionExtractor` is pure pooling). No
    `prediction_*` groups: `-Ext` is plain PPO, and a predictor is a train-time
    entity that a target set makes task-specific.
    """
    c = ctx or ObsCtx()
    cfg.observations[TOKEN_GROUP] = kv_tokens_group(c)
    for name in query_channels:
        cfg.observations[name] = QUERY_GROUPS[name](c)
    cfg.observations["augmentation"] = adapter_stream_group(c)
