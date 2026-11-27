"""Small local simulator/policy checks, with an optional single PPO update.

Run with the repository .venv. Results and optional RGB frames go under logs/.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import mjlab  # noqa: F401 — task discovery must precede imports of ORCS/ViBe
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from vibe.core.rendering import VisualDrCfg
from vibe.release import ensure_released_model

TASKS = (
    "Vibe-Repose-BigCubeFloor-ImgFeat-Ext",
    "Vibe-Repose-SmallCubeTable-ImgFeat-Ext",
    "Vibe-Uolm-ImgFeat-Ext",
    "Vibe-Dodge-ConeFast-ImgFeat-Ext",
    "Vibe-PerLoco-Grail-ImgFeat-Ext",
    "Vibe-PerLoco-OmRe-ImgFeat-Ext",
)


def finite(value) -> None:
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "Non-finite observation, reward or action"
    elif hasattr(value, "values"):
        for child in value.values():
            finite(child)


def run(task: str, backend: str, play: bool, args) -> dict:
    cfg = load_env_cfg(task, play=play)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.deterministic = args.deterministic
    plane = cfg.scene.terrain is not None and cfg.scene.terrain.terrain_type == "plane"
    options = VisualDrCfg(lighting=True, floor_material=plane, floor_textures=plane)
    cfg = replace(cfg, render_backend=backend,
                  visual_dr=options if args.visual_dr and not play else VisualDrCfg())
    os.environ["VIBE_RENDER_BACKEND"] = backend
    ManagerBasedRlEnv.seed(args.seed)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        context = env.scene.sensor_context
        assert context is not None
        assert context.render_context.backend == backend, "Renderer silently fell back"
        assert env.sim.step_graph is not None, "Physics CUDA graph was lost"
        assert bool(env.sim.wp_model.opt.deterministic) == args.deterministic
        assert (env.sim.sense_graph is not None) == (backend == "warp")
        policy = runner = None
        wrapped = None
        if args.agent != "zero":
            agent_cfg = load_rl_cfg(task)
            if args.learn:
                agent_cfg.num_steps_per_env = 8
                agent_cfg.algorithm.num_learning_epochs = 1
                agent_cfg.algorithm.num_mini_batches = 1
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
            runner = runner_cls(wrapped, asdict(agent_cfg), device="cuda:0")
            if args.agent == "release":
                checkpoint = ensure_released_model(task)
                runner.load(str(checkpoint), load_cfg={"actor": True},
                            strict=True, map_location="cuda:0")
            policy = runner.get_inference_policy(device="cuda:0")
        obs, _ = (wrapped or env).reset()
        if args.viser:
            import viser
            from mjlab.viewer.viser.scene import MjlabViserScene

            server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
            try:
                viewer_scene = MjlabViserScene(
                    server, env.sim.mj_model, args.num_envs,
                    sim_model=env.sim.model, expanded_fields=env.sim.expanded_fields,
                )
                viewer_scene.update(env.sim.data)
            finally:
                server.stop()
        if args.save_images:
            from PIL import Image

            frame = env.scene["head_cam"].data.rgb[0].cpu().numpy()
            name = f"{task}-{backend}-{'play' if play else 'train'}-{args.agent}.png"
            Image.fromarray(frame).save(args.output.parent / name)
        resets = 0
        for step in range(args.steps):
            with torch.no_grad():
                actions = (policy(obs) if policy is not None else torch.zeros(
                    (args.num_envs, env.action_manager.total_action_dim), device="cuda:0"))
                finite(actions)
                if wrapped is None:
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    resets += int((terminated | truncated).sum())
                else:
                    obs, reward, done, _ = wrapped.step(actions)
                    resets += int(done.sum())
                finite(obs)
                finite(reward)
            if step == args.steps // 2:
                env.reset(env_ids=torch.tensor([0, args.num_envs - 1], device="cuda:0").unique())
                obs = wrapped.get_observations() if wrapped is not None else env.get_observations()
        rgb = env.scene["head_cam"].data.rgb
        assert bool(env.sim.wp_model.opt.deterministic) == args.deterministic
        assert rgb.dtype == torch.uint8 and rgb.shape[0] == args.num_envs
        assert int(rgb.max()) > int(rgb.min()), "Camera produced a constant frame"
        variants = {name: assignment.cpu().tolist()
                    for name, assignment in env.sim.world_to_variant.items()}
        if "-Uolm-" in task and args.num_envs >= 6:
            assert len(set(variants["object"])) == 6, "Not all Uolm object variants exercised"
        if args.learn:
            model = runner.alg.get_policy()
            before = {name: p.detach().clone() for name, p in model.named_parameters()
                      if p.requires_grad}
            runner.learn(num_learning_iterations=1)  # log_dir=None: no writer/checkpoints
            assert runner.logger.writer is None
            changed = False
            for name, param in model.named_parameters():
                finite(param)
                if name in before and not torch.equal(before[name], param):
                    changed = True
            assert changed, "PPO smoke did not update any trainable actor parameters"
        return {"status": "ok", "rgb_shape": list(rgb.shape),
                "deterministic": cfg.sim.deterministic,
                "visual_dr": asdict(cfg.visual_dr),
                "variants": variants,
                "viser_update": args.viser,
                "learning_iterations": int(args.learn),
                "actions": env.action_manager.total_action_dim, "automatic_resets": resets,
                "physics_graph": env.sim.step_graph is not None,
                "sensing_graph": env.sim.sense_graph is not None,
                "raycast_sensors": len(context.raycast_sensors)}
    finally:
        env.close()
        wp.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--backend", choices=("warp", "optix", "both"), default="warp")
    parser.add_argument("--mode", choices=("train", "play", "both"), default="both")
    parser.add_argument("--agent", choices=("zero", "initial", "release"), default="zero")
    parser.add_argument("--learn", action="store_true",
                        help="One PPO iteration: 8 steps, 1 epoch/minibatch, no logging or saves.")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visual-dr", action="store_true",
                        help="Train: lighting everywhere, material/textures on plane floors only.")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--viser", action="store_true",
                        help="Build/update a local Viser scene once, without opening a browser.")
    parser.add_argument("--output", type=Path, default=Path("logs/smoke/simulator.json"))
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 8 or not 2 <= args.steps <= 512:
        parser.error("Smoke checks require 1–8 environments and 2–512 steps.")
    if args.learn and (args.agent != "initial" or args.mode != "train"):
        parser.error("--learn requires --agent initial --mode train.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(Path("logs/warp-cache").resolve())
    rows = []
    for task in args.tasks:
        for backend in (("warp", "optix") if args.backend == "both" else (args.backend,)):
            if backend == "optix" and "-PerLoco-" in task and args.backend == "both":
                rows.append({"task": task, "backend": backend, "status": "unsupported",
                             "reason": "PerLoco supports native Warp only"})
                continue
            for play in ((False, True) if args.mode == "both" else (args.mode == "play",)):
                row = {"task": task, "backend": backend, "play": play, "agent": args.agent}
                try:
                    row.update(run(task, backend, play, args))
                except Exception as exc:
                    traceback.print_exc()
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
                print("SMOKE_RESULT", json.dumps(row), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    raise SystemExit(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    main()
