"""observation_cfgs.py — perloco's VISION groups.

Read against `orcs.tasks.perloco.observation_cfgs`: the two differ in the adapter's stream.

    [base]      policy + tokenizer streams   — mocke (frozen SONIC contract)
    [adapter]   orcs: height scan + robot root state + root-twist command
                vibe: z + the root-twist command
    [critic]    privileged full state        — orcs's, verbatim (scan included)

Two terms left the actor: the scan (swapped for pixels) and the root state
(`robot_root_pos_env` is odometry, `robot_root_lin_vel_b` wants a state estimator). What
remains is measurable on hardware; SONIC's proprio history carries whatever root estimate
the base infers.
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

# The root twist is perloco's only command, so it is the task query: for terrain,
# "where am I going" is the best available "where should I look".
DEFAULT_QUERY_GROUPS = ("q_task_cmd", "q_proprio", CLS_GROUP)


@dataclass(frozen=True)
class ObsCtx(_ScanCtx, CamSpec):
    """orcs's perloco context + which camera and encoder to read it through."""


# ---------------------------------------------------------------------------
# Extractor query channels — each group is ONE (B, d) vector -> ONE attn row
# ---------------------------------------------------------------------------

QUERY_GROUPS = {
    # task command: the root-twist command {v, w}, time-varying
    "q_task_cmd": lambda c: _grp(robot_motion_cmd_terms(c.p)),
    # proprio feedback: where am I now (posture, stance)
    "q_proprio": lambda c: _grp(proprio_terms()),
    # encoder global token — the one global-pool row (`cls_query_group`)
    CLS_GROUP: cls_query_group,
}


def adapter_stream_group(c: ObsCtx) -> ObservationGroupCfg:
    """The adapter's direct stream: the root-twist command, exact on hardware (not sensed)."""
    return _grp(robot_motion_cmd_terms(c.p))


def attach_ext_obs(
    cfg,
    *,
    query_channels: tuple[str, ...] = DEFAULT_QUERY_GROUPS,
    ctx: ObsCtx | None = None,
) -> None:
    """Wire the extractor: `kv_tokens` + one group per query row; the adapter stream.

      [base]      policy stream (untouched)                  -> tracks the motion
      [adapter]   root-twist command + z                     -> corrects the base
      [extractor] kv_tokens ⟨queried by⟩ query_channels -> z

    Command values reach control through the adapter stream; a query only steers where
    to look (the extractor pools, it never carries the query's value into z).
    """
    c = ctx or ObsCtx()
    cfg.observations[TOKEN_GROUP] = kv_tokens_group(c)
    for name in query_channels:
        cfg.observations[name] = QUERY_GROUPS[name](c)
    cfg.observations["augmentation"] = adapter_stream_group(c)
