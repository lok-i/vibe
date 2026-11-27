"""Frozen-policy replay across rollout/update batch sizes; no learning or W&B.

Uses real training observations, a released actor and its fixed normalizers.
Example: .venv/bin/python scripts/probe_policy_precision.py --checkpoint PATH
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import mjlab  # noqa: F401 -- finish task registration before importing ORCS
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from vibe.tasks.repose.release_training import TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/precision-probe.json"))
    args = parser.parse_args()
    configure_torch_backends()
    torch.manual_seed(42)
    cfg = replace(load_env_cfg(TASK), release_critic=True, render_backend="warp")
    cfg.scene.num_envs = 4096
    cfg.seed = 42
    cfg.commands["motion"].init_phase_anneal_iterations = 50000
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cuda:0"))
    try:
        obs = env.get_observations()
        agent = asdict(load_rl_cfg(TASK))
        groups = resolve_obs_groups(obs, agent["obs_groups"], ["actor", "critic"])
        actor_cls = resolve_callable(agent["actor"].pop("class_name"))
        actor = actor_cls(obs, groups, "actor", env.num_actions, **agent["actor"]).cuda()
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
        del payload
        actor.train()
        with torch.inference_mode():
            for _ in range(24):
                obs, _, _, _ = env.step(actor(obs))
        # Clone outside inference mode so update-style autograd is valid.
        obs = obs.clone().to("cuda:0")
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        noise = torch.randn(4096, env.num_actions, device="cuda")
        results = []
        for name, dtype, reduction, fp32 in (
            ("bf16_default", torch.bfloat16, True, "high"),
            ("bf16_conservative", torch.bfloat16, False, "high"),
            ("fp32_ieee", None, False, "highest"),
        ):
            torch.set_float32_matmul_precision(fp32)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduction
            actor.set_amp_dtype(dtype)
            with torch.inference_mode():
                actor(obs, stochastic_output=True)
                mean = actor.output_mean.clone()
                std = actor.output_std.clone()
                cached = actor.cache_obs(obs)
            mean, std, cached = mean.clone(), std.clone(), cached.clone()
            actions = mean + std * noise
            old_logp = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
            row = {"mode": name, "batches": {}}
            for batch in (4096, 24576):
                repeats = batch // 4096
                replay = TensorDict({
                    k: v.repeat((repeats,) + (1,) * (v.ndim - 1))
                    for k, v in cached.items()
                }, batch_size=[batch], device="cuda:0")
                # Grad enabled, as in PPO, but never optimize or update normalizers.
                prediction = actor(replay)[:4096]
                predicted_std = std
                logp = torch.distributions.Normal(prediction, predicted_std).log_prob(actions).sum(-1)
                delta = logp - old_logp
                row["batches"][str(batch)] = {
                    "mean_action_abs_diff": (prediction - mean).abs().mean().item(),
                    "mean_logp_abs_diff": delta.abs().mean().item(),
                    "kl": (0.5 * ((prediction - mean) / std).square().sum(-1)).mean().item(),
                    "ratio_clip_fraction": ((delta.exp() - 1).abs() > 0.2).float().mean().item(),
                }
                del prediction, logp, delta, replay
            if any(not torch.equal(v, before[k]) for k, v in actor.state_dict().items()):
                raise RuntimeError("Actor parameters/normalizer buffers changed during replay")
            results.append(row)
            print(json.dumps(row), flush=True)
        report = {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "seed": 42,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "observations": {k: list(v.shape) for k, v in obs.items()},
            "inputs": "Fixed real training observations, released actor, frozen normalizers; no optimizer",
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
