"""PPO runner configs — the VISION agents.

vibe adapts a frozen SONIC WBC with exteroception; the privileged/from-scratch
architectures live in `orcs.core.rl` — THE agent zoo — and are not re-declared
here.

  adapt_sonic_agent_cfg()  frozen SONIC + LoRA, ObjKin / ImgFeat[+extractor,PPOAux]
  adapt_sonic_cnn_agent_cfg()  the same base, ImgRgb through a trainable CNN

The runner/algo/critic spine (`_runner`) comes from `vibe.core.rl`; each factory
sets only its actor.
"""

from __future__ import annotations

import dataclasses

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# The PPO spine moved from `orcs.tasks.uolm.rl_cfg` to `orcs.core.rl` when
# perloco arrived and a second task needed it — and shed its underscores on the
# way, because a shared spine is public API, not a task's private detail.
from orcs.core.rl import NUM_STEPS_PER_ENV

# The runner spine + the extractor swap are `vibe.core.rl` — shared with every
# vibe task that reads a camera. What stays here is repose's: the CNN row and
# the aux objectives (a target set is task-specific).
from vibe.core.rl import adapt_sonic_agent_cfg as _adapt_sonic
from vibe.core.rl import attach_extractor
from vibe.tasks.repose.config.g1 import observation_cfgs

# ---------------------------------------------------------------------------
# Aux (PPOAux): feature encoder + auxiliary objectives over frozen Theia feats
# ---------------------------------------------------------------------------

# lr 1e-4 = OpenTrack's world_model_learning_rate; joint mode already gives the aux a
# gradient at every PPO minibatch (20 steps/update) — parent recipe: more steps, lower lr.
_AUX_COMMON = {
    "predictor_hidden_dims": [256, 256],
    "learning_rate": 1.0e-4,
    "num_mini_batches": 4,
    # the aux objective binds to the extractor registered under the token group
    "extractor_group": observation_cfgs.TOKEN_GROUP,
}
# ONE K for both FD variants (= OpenTrack unroll_length); each recurses in ITS state space:
# Lfd in latent z, Sfd in the supervised target space (K-step windows, true-s seed per
# window). The retired Reg probe is now an Sfd config: unroll_steps=1,
# start_with_current_step=True, autoregress=False — dynamics-free decodability.
# Conditioning/target groups ride the rsl_rl defaults (prediction_{conditioning,target});
# conditioning experiments = edit that group's terms in _helpers, not here.
_AUX_UNROLL = 10

# Aux batch budget, in encoder-gradient ROWS PER ENV — the unit that survives the two
# variants sampling in different units (Sfd: env columns x all windows; Lfd: flat (t, env)
# window starts). This value IS Sfd's implied budget, so Lfd inherits it and Sfd keeps its
# own formula untouched (byte-identical to the 0p7p3 baseline run). Per-ENV, so an
# --env.scene.num-envs change rescales both identically (rl_cfg cannot see num_envs).
#   Sfd: (T // K) windows x K steps / num_mini_batches = (24 // 10) * 10 / 4 = 5.0
#   Lfd: 5.0 * num_envs / K sampled starts  ->  same encoder rows per aux call
_AUX_ROWS_PER_ENV = (NUM_STEPS_PER_ENV // _AUX_UNROLL) * _AUX_UNROLL / _AUX_COMMON["num_mini_batches"]

_AUX_VARIANTS = {
    # SSL: K-step latent FD vs an EMA target (multimodal_rl ForwardDynamics mechanics;
    # autoregressive=True adds the open-loop chain on top of the parent-exact TF sum).
    # condition_group rides the rsl_rl default (prediction_conditioning) — same robot-state
    # r the Sfd predictor reads, into the transition only. aux_weight stays 1.0 for both:
    # each loss is already in normalized units (Sfd: EmpiricalNormalization'd target,
    # Lfd: LayerNorm'd z), so no per-variant scaling is tuned. Audit: enc_grad_frac.
    "lfd": {"class_name": "rsl_rl.extensions.LatentFdAux",
            "ema_tau": 0.99, "unroll_steps": _AUX_UNROLL, "autoregressive": True,
            "mini_batch_rows_per_env": _AUX_ROWS_PER_ENV,
            **_AUX_COMMON, "predictor_hidden_dims": [128]},
    # SL: supervised FD — autoregression over K-step windows in the target space
    # (object state + upface color + task-reward rates; robot dims would dilute the
    # extractor gradient ~5x — the conditioning supplies the robot side)
    "sfd": {"class_name": "rsl_rl.extensions.StateFdAux",
            "unroll_steps": _AUX_UNROLL, "autoregress": True,
            "start_with_current_step": False, **_AUX_COMMON},
}


def _algo_with_aux(algo: RslRlPpoAlgorithmCfg, aux: str) -> dict:
    """Swap the algorithm to PPOAux (joint mode) with the chosen auxiliary objective.

    Algorithm-only: PPO hyperparams pass through untouched (asdict), runner
    fields (max_iterations etc.) are never involved. Joint mode: loss_ppo +
    aux_weight * aux_loss per PPO minibatch, one Adam per param; extractor and
    aux heads sit in fixed-LR groups (extractor_lr defaults to the aux lr).
    """
    algo_dict = dataclasses.asdict(algo)
    algo_dict["class_name"] = "rsl_rl.algorithms.PPOAux"
    algo_dict["aux_cfg"] = dict(_AUX_VARIANTS[aux])  # copy — the variant dict is shared
    algo_dict["aux_mode"] = "joint"
    algo_dict["aux_weight"] = 1.0
    return algo_dict


def _attach_aux_actor(
    cfg: RslRlOnPolicyRunnerCfg, aux: str, extractor: str = "cross_attention"
) -> None:
    """Swap the SONIC adapter actor to its Extractor variant, + PPOAux if asked.

    aux ∈ {"lfd","sfd"} attach the objective; "ext" keeps plain PPO (the
    extractor-only baseline). `extractor` picks the architecture from
    `vibe.core.rl.EXTRACTOR_CFGS`; the env's token/query groups feed it by NAME.
    """
    attach_extractor(cfg, observation_cfgs.DEFAULT_QUERY_GROUPS, extractor=extractor)
    if aux != "ext":
        cfg.algorithm = _algo_with_aux(cfg.algorithm, aux)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# AdaptSonic — frozen SONIC base + LoRA
# ---------------------------------------------------------------------------

def adapt_sonic_agent_cfg(
    experiment_name: str,
    *,
    rank: int = 16,
    alpha: float = 1.0,
    aux: str | None = None,
    extractor: str = "cross_attention",
) -> RslRlOnPolicyRunnerCfg:
    """Frozen SONIC base + LoRA adapter on the decoder (augmentation stream).

    The actor itself is `vibe.core.rl` — every vibe task adapts the same base.
    aux: None (plain PPO, flat feats in the stream) | "ext" | "sfd" | "lfd";
    anything but None needs the env cfg built with aux=True.
    """
    cfg = _adapt_sonic(experiment_name, rank=rank, alpha=alpha)
    if aux is not None:
        _attach_aux_actor(cfg, aux, extractor)
    return cfg


def adapt_sonic_cnn_agent_cfg(
    experiment_name: str, *, rank: int = 16, alpha: float = 1.0
) -> RslRlOnPolicyRunnerCfg:
    """The ImgRgb baseline: same frozen SONIC + LoRA, pixels through a CNN.

    The only difference from `adapt_sonic_agent_cfg(aux="ext")` is what feeds z —
    a trainable task-specific CNN instead of a frozen encoder + attention pool.
    """
    cfg = _adapt_sonic(experiment_name, rank=rank, alpha=alpha)
    attach_extractor(cfg, (), extractor="cnn")
    return cfg
