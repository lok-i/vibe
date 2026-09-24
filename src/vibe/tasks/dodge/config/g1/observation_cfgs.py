"""observation_cfgs.py — dodge's VISION groups.

Read this against `orcs.tasks.dodge.observation_cfgs`: the two differ in the
ball half of ONE group, and that is the whole experiment.

    [base]      policy + tokenizer streams   — mocke (frozen SONIC contract)
    [adapter]   AUGMENTATION                 — orcs: ball_{pos,vel}_b + root
                                                     state + sys1 twist cmd
                                               vibe: DELETED — the adapter
                                                     reads z and nothing else
    [critic]    privileged full state        — orcs's, VERBATIM (ball included)

**The adapter stream is empty on purpose, and that is this task's whole point**
(2026-08-06). Dodge is the minimal statement of task-optimal behaviour
adaptation: a reference that does not perform the task, and NO sys1 above it —
so there is no command to feed forward. Everything else follows mechanically:

  `ball_state_terms`  the swap — ball kinematics become pixels.
  root state          odometry + estimated base velocity, neither on hardware.
  root-twist cmd      `v_cmd == w_cmd == 0` for every frame of every episode.
                      A constant input is absorbed into the first layer's bias:
                      it informs nothing, and keeping it would put two ports on
                      the exported graph that a deployer has to supply and that
                      cannot matter.

What is left is z. Every departure from a nominal stand is then attributable to
what the camera saw, with no second channel to argue about.

**Only TWO query rows, and that is a result, not a shortcut.** repose and uolm
spend a row on `q_task_cmd` because they have a task command that varies — a
goal colour, a goal pose. Dodge's reference is a held stand and its command
stream is constant, so a `q_task_cmd`/`q_motion_cmd` row would carry the same
vector every step: attention with a constant query is a fixed pooling pattern,
i.e. a mean-pool row, and a diffuse row IS a mean-pool row. `q_cls` already
occupies that slot as the one sanctioned global-pool path into z, so a second
one only duplicates `proj` params. What is left is the honest pair: where am I
(`q_proprio`) and what is globally in view (`q_cls`).
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
    # sys1's root-twist command. PARKED, and unlike repose/uolm not merely
    # unproven — it is a CONSTANT here (held-stand reference), so as a query it
    # is a mean-pool row by construction. It becomes a real channel the day a
    # sys1 above this task emits a non-trivial twist.
    "q_motion_cmd": lambda c: _grp(robot_motion_cmd_terms(c.p)),
    # encoder global token — the one sanctioned global-pool row (see core).
    CLS_GROUP: cls_query_group,
}


# ---------------------------------------------------------------------------
# The named groups
# ---------------------------------------------------------------------------

def attach_ext_obs(
    cfg,
    *,
    query_channels: tuple[str, ...] = DEFAULT_QUERY_GROUPS,
    ctx: ObsCtx | None = None,
) -> None:
    """Rewire the obs into the sys0 extractor hierarchy (docs/infra/agents.md §3).

      [base]      policy stream (untouched)          -> holds the nominal stand
      [adapter]   z, ALONE                           -> authors the evasion
      [extractor] KV_TOKENS ⟨queried by⟩ q_proprio, q_cls -> z

    Note what the middle row means HERE: on every other vibe task the adapter
    CORRECTS a reference that already performs the task, conditioned on a
    command from sys1. On dodge the reference stands still and there IS no sys1
    — so z is the only input to the adapter, and every departure from a stand is
    attributable to what the camera saw. That is the cleanest statement of
    task-optimal behaviour adaptation this repo can make, and it is why the
    `augmentation` group is deleted rather than filled.
    """
    c = ctx or ObsCtx()
    cfg.observations[TOKEN_GROUP] = kv_tokens_group(c)
    for name in query_channels:
        cfg.observations[name] = QUERY_GROUPS[name](c)
    cfg.observations.pop("augmentation", None)
