"""PPO runner config — the dodge VISION agent.

One row: AdaptSonic + the cross-attention extractor, plain PPO. The actor and
the runner spine are `vibe.core.rl` (shared with repose, perloco and uolm — a
frozen SONIC base is not a task's property); the privileged row is orcs's.

  adapt_sonic_ext_agent_cfg()  frozen SONIC + LoRA + extractor, no predictor

No aux objectives here. `-Sfd`/`-Lfd` need a prediction TARGET, and a target set
is task-specific by construction — when dodge earns one, it declares it here the
way repose does.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from vibe.core.rl import adapt_sonic_agent_cfg, attach_extractor
from vibe.tasks.dodge.config.g1 import observation_cfgs


def adapt_sonic_ext_agent_cfg(
    experiment_name: str,
    *,
    query_groups: tuple[str, ...] = observation_cfgs.DEFAULT_QUERY_GROUPS,
    extractor: str = "cross_attention",
    **adapter_kw,
) -> RslRlOnPolicyRunnerCfg:
    """Frozen SONIC base + LoRA adapter, conditioned on z instead of ball state.

    Extractor-only (`-Ext`): the extractor is trained by PPO gradients alone, so
    this row answers "does a task gradient reach a vision encoder at all" before
    any auxiliary objective is spent on the question. Requires the env cfg built
    with `aux=True` (its token + query groups feed the extractor — the coupling
    is group NAMES only, shared via `observation_cfgs`).

    TWO query rows here against repose's and uolm's three; the reason is in
    `observation_cfgs` and it is a property of the task, not a saving.

    **LoRA sizing and the std band are left at the core defaults on purpose** —
    rank 16, alpha 1.0 (scale 1/16), std 1.0 — because that is what the
    privileged twin trains at, and a delta against `Orcs-Dodge-AdaptSonic` is
    only the exteroception if every other knob matches. `**adapter_kw` forwards
    to `adapt_sonic_agent_cfg`, so retuning is possible; retuning ONE row is the
    mistake. If this row ever needs more authority than the twin, the ceiling
    that binds is the twin's rank 20, not this row's 28 (`orcs.tasks.dodge`).
    """
    cfg = adapt_sonic_agent_cfg(experiment_name, **adapter_kw)
    # `stream_groups=()` — the adapter reads z and NOTHING else. Dodge has no
    # sys1, so there is no command stream to name; the env drops `augmentation`
    # for the same reason (`observation_cfgs.attach_ext_obs`).
    attach_extractor(cfg, query_groups, extractor=extractor, stream_groups=())
    return cfg
