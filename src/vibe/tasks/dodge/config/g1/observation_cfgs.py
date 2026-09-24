"""observation_cfgs.py — dodge's VISION groups.

Read against `orcs.tasks.dodge.observation_cfgs`: the two differ in the adapter's stream.

    [base]      policy + tokenizer streams   — mocke (frozen SONIC contract)
    [adapter]   orcs: ball_{pos,vel}_b + root state + root-twist command
                vibe: z ALONE — the stream is deleted
    [critic]    privileged full state        — orcs's, verbatim (ball included)

The reference is a held stand, so its root-twist command is zero every frame: a constant
input informs nothing and would only add deploy ports. Root state needs odometry and a state
estimator. What is left is z, so every departure from the stand is what the camera saw.

Two query rows, not three: with a constant command, a `q_task_cmd` row would be a fixed
pooling pattern, i.e. a second mean-pool row beside `q_cls`.
"""

from __future__ import annotations

from dataclasses import dataclass

from orcs.core.obs import grp as _grp
from orcs.core.obs import proprio_terms
from orcs.tasks.dodge.observation_cfgs import ObsCtx as _BallKinCtx
from orcs.tasks.dodge.observation_cfgs import robot_motion_cmd_terms

from vibe.core.observation_cfgs import (
    CLS_GROUP,
    TOKEN_GROUP,
    CamSpec,
    cls_query_group,
    kv_tokens_group,
)

__all__ = [
    "ObsCtx", "DEFAULT_QUERY_GROUPS", "QUERY_GROUPS",
    "attach_ext_obs",
]

DEFAULT_QUERY_GROUPS = ("q_proprio", CLS_GROUP)


@dataclass(frozen=True)
class ObsCtx(_BallKinCtx, CamSpec):
    """orcs's dodge context + which camera and encoder to read it through."""


# ---------------------------------------------------------------------------
# Extractor query channels — each group is ONE (B, d) vector -> ONE attn row
# ---------------------------------------------------------------------------

QUERY_GROUPS = {
    # proprio feedback: where am I now (posture, limbs, which way is down)
    "q_proprio": lambda c: _grp(proprio_terms()),
    # the root-twist command: a constant here (held stand), so off by default
    "q_motion_cmd": lambda c: _grp(robot_motion_cmd_terms(c.p)),
    # encoder global token — the one global-pool row (`cls_query_group`)
    CLS_GROUP: cls_query_group,
}


def attach_ext_obs(
    cfg,
    *,
    query_channels: tuple[str, ...] = DEFAULT_QUERY_GROUPS,
    ctx: ObsCtx | None = None,
) -> None:
    """Wire the extractor: `kv_tokens` + one group per query row; delete `augmentation`.

      [base]      policy stream (untouched)                  -> holds the stand
      [adapter]   z alone                                    -> authors the evasion
      [extractor] kv_tokens ⟨queried by⟩ query_channels -> z
    """
    c = ctx or ObsCtx()
    cfg.observations[TOKEN_GROUP] = kv_tokens_group(c)
    for name in query_channels:
        cfg.observations[name] = QUERY_GROUPS[name](c)
    cfg.observations.pop("augmentation", None)
