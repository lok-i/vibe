"""observation_cfgs.py — uolm's VISION groups.

Read against `orcs.tasks.uolm.observation_cfgs`: the two differ in the adapter's stream.

    [base]      policy + tokenizer streams   — mocke (frozen SONIC contract)
    [adapter]   orcs: object state + object_id + root state + goal + motion command
                vibe: z + the motion command
    [critic]    privileged full state        — orcs's, verbatim (object state and id included)

What left the actor, and why:

  `object_state_w_*`   the swap: object kinematics become pixels.
  `object_id`          six objects that differ in geometry are what the camera is for.
  root state           odometry + a state estimator, neither on hardware.
  `object_goal_*`      moved to the `q_task_cmd` query, in the env frame: a query only
                       steers where to look, so it needs no odometry to be placed.

What remains is repose's stream exactly: `bodywise_contact_cmd` + the root twist.
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

# The goal pose is uolm's task query. `q_motion_cmd` stays off, as in repose: the
# command's value already reaches control through the adapter stream.
DEFAULT_QUERY_GROUPS = ("q_task_cmd", "q_proprio", CLS_GROUP)


@dataclass(frozen=True)
class ObsCtx(_ObjKinCtx, CamSpec):
    """orcs's uolm context + which camera and encoder to read it through."""


# ---------------------------------------------------------------------------
# Extractor query channels — each group is ONE (B, d) vector -> ONE attn row
# ---------------------------------------------------------------------------

QUERY_GROUPS = {
    # task command: the object's goal pose, fixed per episode
    "q_task_cmd": lambda c: _grp(object_goal_terms(c.p)),
    # the motion command (per-body contact schedule + root twist); off by default
    "q_motion_cmd": lambda c: _grp(robot_motion_cmd_terms(c.p)),
    # proprio feedback: where am I now (posture, hands)
    "q_proprio": lambda c: _grp(proprio_terms()),
    # encoder global token — the one global-pool row (`cls_query_group`)
    CLS_GROUP: cls_query_group,
}


def adapter_stream_group(c: ObsCtx) -> ObservationGroupCfg:
    """The adapter's direct stream: the motion command, identical to repose's."""
    return _grp(robot_motion_cmd_terms(c.p))


def attach_ext_obs(
    cfg,
    *,
    query_channels: tuple[str, ...] = DEFAULT_QUERY_GROUPS,
    ctx: ObsCtx | None = None,
) -> None:
    """Wire the extractor: `kv_tokens` + one group per query row; the adapter stream.

      [base]      policy stream (untouched)                  -> tracks the motion
      [adapter]   motion command + z                         -> corrects the base
      [extractor] kv_tokens ⟨queried by⟩ query_channels -> z
    """
    c = ctx or ObsCtx()
    cfg.observations[TOKEN_GROUP] = kv_tokens_group(c)
    for name in query_channels:
        cfg.observations[name] = QUERY_GROUPS[name](c)
    cfg.observations["augmentation"] = adapter_stream_group(c)
