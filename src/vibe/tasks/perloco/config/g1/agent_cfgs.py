"""PPO runner config — the perloco VISION agent.

One row: AdaptSonic + the cross-attention extractor, plain PPO. The actor and
the runner spine are `vibe.core.rl` (shared with repose — a frozen SONIC base is
not a task's property); the privileged and from-scratch rows are orcs's.

  adapt_sonic_ext_agent_cfg()  frozen SONIC + LoRA + extractor, no predictor

No aux objectives here. `-Sfd`/`-Lfd` need a prediction TARGET, and a target set
is task-specific by construction — when perloco earns one, it declares it here
the way repose does.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from vibe.core.rl import adapt_sonic_agent_cfg, attach_extractor
from vibe.tasks.perloco.config.g1 import observation_cfgs


def adapt_sonic_ext_agent_cfg(
    experiment_name: str,
    *,
    rank: int = 16,
    alpha: float = 1.0,
    query_groups: tuple[str, ...] = observation_cfgs.DEFAULT_QUERY_GROUPS,
    extractor: str = "cross_attention",
) -> RslRlOnPolicyRunnerCfg:
    """Frozen SONIC base + LoRA adapter, conditioned on z instead of a scan.

    Extractor-only (`-Ext`): the extractor is trained by PPO gradients alone, so
    this row answers "does a task gradient reach a vision encoder at all" before
    any auxiliary objective is spent on the question. Requires the env cfg built
    with `aux=True` (its token + query groups feed the extractor — the coupling
    is group NAMES only, shared via `observation_cfgs`).
    """
    cfg = adapt_sonic_agent_cfg(experiment_name, rank=rank, alpha=alpha)
    attach_extractor(cfg, query_groups, extractor=extractor)
    return cfg
