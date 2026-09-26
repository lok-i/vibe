"""THE vision agent spine — task-blind, peer of `orcs.core.rl`.

orcs's zoo builds the privileged agents; this builds the ones that read a
camera. Two tiers, same shape as orcs's:

    runner(name)                 runner/algo/critic defaults + the three knobs
                                 rsl_rl reads but mjlab never declares
    adapt_sonic_agent_cfg()      frozen SONIC base + LoRA on the decoder
    attach_extractor()           swap that actor to its Extractor variant

The PPO hyperparameters themselves come from `orcs.core.rl` — one definition of
"what a PPO run here means" across the privileged and the visual tasks.

A task picks an agent, names its experiment, and passes ITS query rows. Nothing
here names a cube, a terrain, or an auxiliary objective — the aux variants are
the owning task's, since a target set is task-specific by construction.
"""

from __future__ import annotations

import dataclasses

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from orcs.core.rl import (
    CRITIC_HIDDEN,
    NUM_STEPS_PER_ENV,
    SONIC_CKPT,
    sonic_adapter_actor,
)

from vibe.core import observation_cfgs as oc

__all__ = [
    "VibeRunnerCfg", "runner", "MAX_ITERATIONS", "SAVE_INTERVAL",
    "EXTRACTOR_CFGS", "adapt_sonic_agent_cfg", "attach_extractor",
]

MAX_ITERATIONS = 60_000
SAVE_INTERVAL = 2000


@dataclasses.dataclass
class VibeRunnerCfg(RslRlOnPolicyRunnerCfg):
    """mjlab's runner cfg + the three knobs rsl_rl reads but mjlab never declares.

    `alg.from_cfg` reads them off `asdict(cfg)`, so a key only reaches it if a
    field carries it.
    """

    wandb_project: str = "vibe"
    """W&B project, one for every vibe task (mjlab's default is "mjlab")."""

    torch_compile_mode: str | None = None
    """torch.compile for actor+critic. OFF — measured ~1% at 4096 envs with bf16 on.
    Use "default" if you enable it; the autotune modes crash in Inductor codegen on
    torch 2.12, CUDA-graph modes are rejected for multi-model algos."""

    check_for_nan: bool = True
    """Per-step NaN guard on obs/rewards/dones. ON — costs ~450 host syncs/iteration,
    measured at 1.9%. Declared here so that cost is a decision, not an accident
    (rsl_rl always read the key; mjlab never declared it, so it ran on invisibly)."""

    amp_dtype: str | None = None
    """Body-autocast dtype for actor+critic — "bfloat16" | "float16" | None.

    bf16 is ~1.3x on learning at reward parity ON AN RTX 5090; a 3090 or L40S
    drifts (~1e-1, reward stalls), so it is opt-in (docs/tasks.md#train). The head
    stays fp32 and the SONIC encoder stays out of autocast entirely (FSQ's rounding
    grid); both invariants
    live in `rsl_rl.modules.AmpMixin`, and breaking either is what collapsed the
    first attempt. Gate on `Diagnostics/logp_drift_mb0` (~1e-3 healthy, ~1e-1 = the
    seam leaks). Under `play` use `VIBE_AMP=bfloat16` — play discards agent-cfg
    overrides."""

    drop_query_rows: str = ""
    """Extractor query rows to REMOVE — the attention ablation, one flag.

        --agent.drop-query-rows q_proprio          (comma or space separated)

    Empty (the default) changes nothing, so every existing task and checkpoint is
    untouched. The env still BUILDS every query group; only the extractor stops
    reading one — so two arms differ in the ACTOR and in nothing else, and the
    unused group costs ~60 floats/env/step against 4032 for the tokens.

    z stays 128-d in every arm (`proj: m*attn_dim -> latent_dim`), so the adapter's
    budget is constant and the ablation measures ROUTING, not capacity.

    A plain string rather than a tuple: mjlab's TYRO_FLAGS carry
    `UsePythonSyntaxForLiteralCollections`, so a collection flag reads `"['a','b']"`,
    which a shell or job script globs. `--agent.actor.extractor-cfg.<group>.query-groups`
    sets the KEEP set outright.
    """

    def __post_init__(self) -> None:
        """Prune the dropped rows. Runs at build time (a no-op — `actor` is still the
        base dataclass then) and again after a tyro override, which is where it bites."""
        drop = {r for r in self.drop_query_rows.replace(",", " ").split() if r}
        if not drop or not isinstance(self.actor, dict):
            return
        for ext in (self.actor.get("extractor_cfg") or {}).values():
            rows = ext.get("query_groups")
            if rows is None:  # an extractor with no query rows at all (cnn)
                continue
            assert drop <= set(rows), (
                f"--agent.drop-query-rows {sorted(drop - set(rows))}: not a query row "
                f"of this task (has {list(rows)})")
            ext["query_groups"] = [g for g in rows if g not in drop]


def runner(
    experiment_name: str, obs_groups: dict | None = None
) -> RslRlOnPolicyRunnerCfg:
    """Common runner cfg: adaptive-KL PPO (byte-identical to mjlab's stock G1
    tracking algo) + MLP critic. Factories override only what genuinely differs
    (the actor; obs routing for goal-command/vision)."""
    return VibeRunnerCfg(
        experiment_name=experiment_name,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=MAX_ITERATIONS,
        save_interval=SAVE_INTERVAL,
        obs_groups=obs_groups or {"actor": ("policy",), "critic": ("critic",)},
        critic=RslRlModelCfg(
            hidden_dims=CRITIC_HIDDEN,
            obs_normalization=True,
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            clip_param=0.2,
            entropy_coef=0.005,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


# ---------------------------------------------------------------------------
# The extractor — camera to z, one entry per architecture
# ---------------------------------------------------------------------------

# `(obs group it reads, cfg)` per architecture. Roles are EXPLICIT
# (token_terms/query_groups), never shape-inferred; group names are shared with the
# env side via observation_cfgs. latent 128 ~ the plain adapter stream and layer_norm
# gives z scale parity, so BOTH rows hand the adapter the same budget — the vision
# stack is then the only difference between them:
#
#   cross_attention  frozen task-agnostic encoder -> task-specific attention pool
#   cnn              task-specific trainable encoder over raw pixels (the baseline)
#
# num_heads is reserved (the extractor asserts 1).
EXTRACTOR_CFGS = {
    "cross_attention": lambda query_groups: (oc.TOKEN_GROUP, {
        "class_name": "rsl_rl.modules.CrossAttentionExtractor",
        "token_terms": list(oc.TOKEN_TERMS),
        "query_groups": list(query_groups),
        "latent_dim": 128, "attn_dim": 64, "num_heads": 1,
        "num_learned_queries": 0, "layer_norm": True,
    }),
    "cnn": lambda query_groups: (oc.CAMERA_GROUP, {
        "class_name": "vibe.core.cnn_encoder.CnnEncoder",
        "latent_dim": 128, "layer_norm": True,
        # mjlab's yam vision stack, unchanged.
        "output_channels": [16, 32], "kernel_size": [5, 3], "stride": [2, 2],
        "padding": "zeros", "activation": "elu", "max_pool": False,
        "spatial_softmax_temperature": 1.0,
    }),
}


def extractor_cfg(query_groups, extractor: str = "cross_attention") -> dict:
    """`{obs group: <extractor cfg>}` with this task's query rows bound in."""
    group, cfg = EXTRACTOR_CFGS[extractor](query_groups)
    return {group: cfg}


# ---------------------------------------------------------------------------
# AdaptSonic — frozen SONIC base + LoRA
# ---------------------------------------------------------------------------

def adapt_sonic_agent_cfg(
    experiment_name: str,
    *,
    rank: int = 16,
    alpha: float = 1.0,
    base_checkpoint: str = SONIC_CKPT,
    std_scale: float | dict[str, float] = 1.0,
) -> RslRlOnPolicyRunnerCfg:
    """Frozen SONIC base + LoRA adapter on the decoder (augmentation stream).

    The ACTOR is orcs's `sonic_adapter_actor` verbatim — one definition of what
    "adapt SONIC" means across the privileged and the visual rows, so a task can
    match its privileged twin's LoRA sizing (`rank`/`alpha`) and exploration band
    (`std_scale`) by passing the same numbers. vibe adds its runner (amp/nan/
    compile knobs) and the placement flags the vision path needs.
    `attach_extractor` turns this into the Extractor variant.
    """
    cfg = runner(experiment_name)
    cfg.actor = {  # type: ignore[assignment]
        **sonic_adapter_actor(rank=rank, alpha=alpha,
                              base_checkpoint=base_checkpoint,
                              std_scale=std_scale),
        # Explicit: adapter placement decides what a PPO update may SKIP, not just
        # what trains (rsl_rl SonicWithAdapterModel).
        # adapt_encoder off => the frozen encoder is a pure function => cache_tokens
        # is legal, and is worth -15% of the actor's update, bit-exact.
        "adapt_encoder": False,
        "adapt_decoder": True,
        "cache_tokens": True,
    }
    return cfg


def attach_extractor(
    cfg: RslRlOnPolicyRunnerCfg,
    query_groups,
    *,
    extractor: str = "cross_attention",
    stream_groups: tuple[str, ...] = ("augmentation",),
) -> None:
    """Swap the SONIC adapter actor to its Extractor variant.

    The adapter stream becomes `stream_groups` + z. `query_groups` are the task's
    attention rows — one (B, d) group per row, empty for architectures that have
    none. No env-side coupling beyond the group NAMES, shared via
    `observation_cfgs`.

    `stream_groups=()` is the z-ONLY adapter: dodge's, whose command stream is a
    constant, so its env drops the `augmentation` group entirely.
    """
    ext_cfg = extractor_cfg(query_groups, extractor)
    cfg.actor["class_name"] = "rsl_rl.models.ExtractorSonicAdapterModel"  # type: ignore[index]
    cfg.actor["adapter_obs_group"] = [*stream_groups, *ext_cfg]  # type: ignore[index]
    cfg.actor["extractor_cfg"] = ext_cfg  # type: ignore[index]
