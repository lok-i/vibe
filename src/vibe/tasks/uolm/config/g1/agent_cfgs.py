"""PPO runner config — the uolm VISION agent.

One row: AdaptSonic + the cross-attention extractor, plain PPO. The actor and
the runner spine are `vibe.core.rl` (shared with repose and perloco — a frozen
SONIC base is not a task's property); the privileged and from-scratch rows are
orcs's.

  adapt_sonic_ext_agent_cfg()  frozen SONIC + LoRA + extractor, no predictor

No aux objectives here. `-Sfd`/`-Lfd` need a prediction TARGET, and a target set
is task-specific by construction — when uolm earns one, it declares it here the
way repose does.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from vibe.core.rl import adapt_sonic_agent_cfg, attach_extractor
from vibe.tasks.uolm.config.g1 import observation_cfgs

# Sized off the privileged twin (`Orcs-Uolm-AdaptSonic`), not chosen here: `Adapter.scale
# = alpha / rank`, so alpha tracks rank or raising rank SHRINKS the delta, and
# 28 is the CEILING — the decoder's output layer is (512 -> 29), and at
# rank >= min(in, out) the adapter silently takes a full-rank branch at
# alpha (not alpha/rank) gain. Widening the conditioning port with z does not
# move that ceiling; the 29-dim output is what binds.
_ADAPTER_RANK = 28
# Exploration band, same provenance: a multiplier on the ckpt's converged
# per-dim std (frozen std, so this is the only sanctioned way to buy
# exploration off a frozen base). uolm is the row where the adapter must pull
# the base OFF its tracking manifold to move an object.
_STD_SCALE = 1.3


def adapt_sonic_ext_agent_cfg(
    experiment_name: str,
    *,
    rank: int = _ADAPTER_RANK,
    alpha: float = _ADAPTER_RANK,
    std_scale: float = _STD_SCALE,
    query_groups: tuple[str, ...] = observation_cfgs.DEFAULT_QUERY_GROUPS,
    extractor: str = "cross_attention",
) -> RslRlOnPolicyRunnerCfg:
    """Frozen SONIC base + LoRA adapter, conditioned on z instead of object state.

    Extractor-only (`-Ext`): the extractor is trained by PPO gradients alone, so
    this row answers "does a task gradient reach a vision encoder at all" before
    any auxiliary objective is spent on the question. LoRA sizing and the std
    band are its privileged twin's, so a delta against `Orcs-Uolm-AdaptSonic` is
    the exteroception and nothing else. The env's token + query groups feed the
    extractor by NAME (`observation_cfgs`).
    """
    cfg = adapt_sonic_agent_cfg(experiment_name, rank=rank, alpha=alpha,
                                std_scale=std_scale)
    attach_extractor(cfg, query_groups, extractor=extractor)
    return cfg
