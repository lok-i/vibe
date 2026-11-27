"""Paired-process rollout/PPO qualification and separate, uninstrumented timing.

Examples (run from the checkout with .venv/bin/python):
  scripts/repeat_check.py --task Vibe-Repose-BigCubeFloor-ImgFeat-Ext --num-envs 4096
  scripts/repeat_check.py --task Vibe-Dodge-ConeFast-ImgFeat-Ext --num-envs 2048 --mode ppo
  scripts/repeat_check.py --mode benchmark --num-envs 4096 --deterministic False

The default rollout uses the released actor's mean with training noise/events.
No raw observations or policy weights are included in the comparison records.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path


def parse_bool(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("Use True or False")
    return value.lower() == "true"


def worker(args):
    import mjlab  # noqa: F401 -- task discovery must precede ORCS imports
    import torch
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.sim.sim import Simulation
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    from vibe.core.reproducibility import BufferMonitor, digest
    from vibe.release import ensure_released_model

    directory = args.output / f"run-{args.worker}"
    directory.mkdir(parents=True, exist_ok=False)
    wp.config.kernel_cache_dir = str(Path("logs/warp-cache").resolve())
    configure_torch_backends()
    if args.graphs == "off":
        Simulation._can_capture_graphs = lambda self: False
    cfg = load_env_cfg(args.task, play=False)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.deterministic = args.deterministic
    cfg.sim.mujoco.jacobian = args.jacobian
    ManagerBasedRlEnv.seed(args.seed)

    # Device-wide used memory includes Warp allocations outside Torch's allocator.
    # This is a sampled high-water mark, not an exact allocator peak.
    memory = []
    stop = threading.Event()

    def sample_memory():
        while not stop.is_set():
            free, total = torch.cuda.mem_get_info()
            memory.append(total - free)
            stop.wait(0.1)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        assert bool(env.sim.wp_model.opt.deterministic) == args.deterministic
        assert env.sim.use_cuda_graph == (args.graphs == "on")
        assert env.scene.sensor_context.render_context.backend == "warp"
        agent_cfg = load_rl_cfg(args.task)
        agent_cfg.seed = args.seed
        agent_cfg.logger = "tensorboard"
        agent_cfg.upload_model = False
        if args.rollout_length:
            agent_cfg.num_steps_per_env = args.rollout_length
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        learning = args.mode != "rollout"
        runner = runner_cls(wrapped, asdict(agent_cfg),
                            log_dir=str(directory) if learning else None, device="cuda:0")
        dump_yaml(directory / "params/env.yaml", asdict(cfg))
        dump_yaml(directory / "params/agent.yaml", asdict(agent_cfg))
        summary = {"task": args.task, "num_envs": args.num_envs,
                   "seed": args.seed, "deterministic": args.deterministic,
                   "graphs": args.graphs, "gpu": torch.cuda.get_device_name(),
                   "mode": args.mode, "steps": args.steps, "iterations": args.iterations,
                   "jacobian": args.jacobian, "is_sparse": env.sim.wp_model.is_sparse}
        summary["versions"] = {name: version(name) for name in
                               ("mjlab", "mujoco-core", "mujoco-warp", "warp-lang",
                                "rsl-rl-lib", "torch", "torchvision")}
        with (directory / "records.jsonl").open("w") as output:
            def record(row):
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()

            if not learning:
                checkpoint = ensure_released_model(args.task)
                counter = env.common_step_counter
                runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                            map_location="cuda:0")
                # Actor-only evaluation must not advance the training curriculum
                # to the released checkpoint's saved environment counter.
                env.common_step_counter = counter
                policy = runner.get_inference_policy(device="cuda:0")
                monitor = BufferMonitor(env.sim)
                env.sim.add_data_listener(monitor.sample)
                obs, _ = env.reset()
                automatic_resets = 0
                thrower = (env.event_manager.get_term_cfg("throw_ball").func
                           if "throw_ball" in cfg.events else None)
                flight = thrower._flight.clone() if thrower is not None else None
                flight_starts = 0
                with torch.inference_mode():
                    for step in range(args.steps):
                        action = policy(wrapped._to_tensordict(obs))
                        if agent_cfg.clip_actions is not None:
                            action = action.clamp(-agent_cfg.clip_actions, agent_cfg.clip_actions)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        tensors = {"action": action, "reward": reward,
                                   "terminated": terminated, "truncated": truncated,
                                   **{name: getattr(env.sim.data, name)[:] for name in
                                      ("qpos", "qvel", "qacc_warmstart", "sensordata")},
                                   **{f"obs/{name}": value for name, value in obs.items()}}
                        record({"step": step, **{k: digest(v) for k, v in tensors.items()}})
                        automatic_resets += int((terminated | truncated).sum())
                        if thrower is not None:
                            flight_starts += int((thrower._flight > flight).sum())
                            flight.copy_(thrower._flight)
                        if step == args.steps // 2:
                            obs, _ = env.reset(env_ids=torch.tensor([0], device=env.device))
                        if step % 100 == 0:
                            print(f"repeat step {step}/{args.steps}", flush=True)
                summary.update(buffers=monitor.report(), automatic_resets=automatic_resets,
                               explicit_partial_resets=1, observed_flight_starts=flight_starts)
                if thrower is not None and args.steps >= 1000:
                    assert flight_starts > 0, "Dodge repeat did not exercise throws"
            else:
                scalar_rows = {}
                timings = []
                original_init = runner.logger.init_logging_writer
                original_log = runner.logger.log

                def init_writer():
                    original_init()
                    original_add = runner.logger.writer.add_scalar

                    def add_scalar(tag, value, global_step=None, *pos, **kw):
                        if not tag.startswith("Perf/") and not tag.endswith("/time"):
                            scalar_rows[tag] = digest(float(value))
                        return original_add(tag, value, global_step, *pos, **kw)

                    runner.logger.writer.add_scalar = add_scalar

                def log(**kw):
                    original_log(**kw)
                    timings.append({"collection": kw["collect_time"], "learning": kw["learn_time"]})
                    if args.mode == "ppo":
                        record({"iteration": kw["it"],
                                **{f"scalar/{k}": v for k, v in scalar_rows.items()},
                                **{f"state/{k}": digest(v) for k, v in runner.alg.save().items()},
                                "rng/torch": digest(torch.get_rng_state()),
                                "rng/cuda": digest(torch.cuda.get_rng_state())})
                    scalar_rows.clear()

                runner.logger.init_logging_writer = init_writer
                runner.logger.log = log
                # No hashing or capacity sampling in the timing mode.
                runner.learn(num_learning_iterations=args.iterations)
                warmed = timings[min(5, len(timings) - 1):]
                summary["timings"] = timings
                summary["mean_collection_s"] = statistics.mean(t["collection"] for t in warmed)
                summary["mean_learning_s"] = statistics.mean(t["learning"] for t in warmed)
                # Separate physics microbenchmark after PPO, same scene/capacities.
                if args.mode == "benchmark":
                    for _ in range(20):
                        env.sim.step()
                    wp.synchronize()
                    start = time.perf_counter()
                    for _ in range(200):
                        env.sim.step()
                    wp.synchronize()
                    summary["physics_step_s"] = (time.perf_counter() - start) / 200
        summary["sampled_peak_device_bytes"] = max(memory)
        summary["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        stop.set()
        sampler.join(timeout=2)
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Vibe-Repose-BigCubeFloor-ImgFeat-Ext")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rollout-length", type=int, default=0,
                        help="Small local PPO smoke only; 0 preserves the task default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", "--env.sim.deterministic", type=parse_bool, default=True)
    parser.add_argument("--graphs", choices=("on", "off"), default="on")
    parser.add_argument("--jacobian", choices=("auto", "dense", "sparse"), default="auto",
                        help="Explicit solver-layout diagnostic; task default is auto.")
    parser.add_argument("--mode", choices=("rollout", "ppo", "benchmark"), default="rollout")
    parser.add_argument("--output", type=Path, default=Path("logs/repeat-check"))
    parser.add_argument("--worker", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 2 or not 1 <= args.iterations <= 100:
        parser.error("Positive environment count, >=2 steps, and 1–100 PPO iterations required")
    if args.worker is not None:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    count = 1 if args.mode == "benchmark" else 2
    for repeat in range(count):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:],
                   "--worker", str(repeat)]
        with (args.output / f"worker-{repeat}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONHASHSEED": str(args.seed)})
    if count == 2:
        import mjlab  # noqa: F401 -- preserve import order for helper package

        from vibe.core.reproducibility import first_difference

        records = [args.output / f"run-{i}/records.jsonl" for i in range(2)]
        with records[0].open() as a, records[1].open() as b:
            difference = first_difference((json.loads(s) for s in a), (json.loads(s) for s in b))
        result = {"equal": difference is None, "first_difference": difference,
                  "deterministic": args.deterministic, "mode": args.mode}
        (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if args.deterministic and difference is not None:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
