"""observation_cfgs.py — perloco's VISION groups.

Read this against `orcs.tasks.perloco.observation_cfgs`: the two differ in ONE
term, and that term is the whole experiment.

    [base]      policy + tokenizer streams   — mocke (frozen SONIC contract)
    [adapter]   AUGMENTATION                 — orcs:  TERRAIN height scan
                                                      + robot root state
                                               vibe:  z, and sys1's twist cmd
    [critic]    privileged full state        — orcs's, VERBATIM (scan included)

The critic keeping the height scan is the point, not an oversight: the actor
goes vision-only while the value function keeps the terrain oracle.

**Everything the actor reads is now measurable on hardware.** Two things left,
not one: the scan (swapped for pixels) and the robot ROOT STATE (deleted —
`robot_root_pos_env` is odometry, `robot_root_lin_vel_b` wants a state
estimator). The pair is no longer a one-term A/B against
`Orcs-PerLoco-*-AdaptSonic`; it is a deployable policy, which is the thing this
revision is for. SONIC's history-10 proprio stream carries whatever root
estimate the base has learned to infer, and that one IS deployable.

Every non-vision atom is orcs's — `proprio_terms`, `robot_motion_cmd_terms` —
and every vision atom is `vibe.core.observation_cfgs`.
Nothing is defined here that either already owns; what IS here is what perloco
means by a query row.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg
from orcs.core.obs import grp as _grp
from orcs.core.obs import proprio_terms
from orcs.tasks.perloco.observation_cfgs import ObsCtx as _ScanCtx
from orcs.tasks.perloco.observation_cfgs import robot_motion_cmd_terms

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

# perloco's task command IS the root twist — there is no object, no goal pose,
# nothing else to aim the attention with. So `q_task_cmd` here is what repose
# parks as `q_motion_cmd`, promoted: for terrain, "where am I going" is the
# strongest available "where should I look".
DEFAULT_QUERY_GROUPS = ("q_task_cmd", "q_proprio", CLS_GROUP)


@dataclass(frozen=True)
class ObsCtx(_ScanCtx, CamSpec):
    """orcs's perloco context + which camera and encoder to read it through."""


# ---------------------------------------------------------------------------
# Extractor query channels — each group is ONE (B, d) vector -> ONE attn row
# ---------------------------------------------------------------------------

QUERY_GROUPS = {
    # task command: the sys1 root twist, {v,w}_cmd_t. Time-varying, and the only
    # command this task has. NOT uolm's bundle — the per-body contact schedule is
    # derived from an object contact graph, and there is no object here.
    "q_task_cmd": lambda c: _grp(robot_motion_cmd_terms(c.p)),
    # proprio feedback: where am I now (posture, stance)
    "q_proprio": lambda c: _grp(proprio_terms()),
    # encoder global token — the one sanctioned global-pool row (see core).
    CLS_GROUP: cls_query_group,
}


# ---------------------------------------------------------------------------
# The named groups
# ---------------------------------------------------------------------------

def adapter_stream_group(c: ObsCtx) -> ObservationGroupCfg:
    """The adapter's direct stream: sys1's root-twist command, and nothing else.

    **Robot root state is GONE** (2026-08-06), which is the whole point of this
    revision: `robot_root_pos_env` is odometry and `robot_root_lin_vel_b` needs a
    state estimator, so neither exists on hardware. What is left is what sys1
    emits — a command, exact on hardware because it is not sensed at all — plus
    z. That makes the deployed input a function of (camera, proprio, command)
    and nothing the sim privately knows.

    The scan made root state look load-bearing (it was the odometry that placed
    the rays); with the scan already gone, so is the reason.
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
      [adapter]   root-twist cmd + z                          -> corrects the base
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
