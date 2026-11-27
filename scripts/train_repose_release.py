"""Train the released Repose recipe on the upgraded backend; see docs/tasks.md."""

import argparse
import os

import mjlab  # noqa: F401 -- task registration must finish first
from mjlab.scripts.train import launch_training

from vibe.tasks.repose.release_training import TASK, training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("warp", "optix"), default="warp")
    parser.add_argument("--run-name", default="mjlab-upgrade")
    parser.add_argument("--iterations", type=int, default=60000,
                        help="Lower only for a bounded smoke test.")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Lower only for a bounded smoke test.")
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
    args = parser.parse_args()
    if not 1 <= args.iterations <= 60000 or args.num_envs < 1:
        parser.error("Use 1–60000 iterations and a positive environment count")
    os.environ["VIBE_RENDER_BACKEND"] = args.backend
    cfg = training_config(args.backend)
    cfg.agent.run_name = args.run_name
    cfg.agent.max_iterations = args.iterations
    cfg.agent.logger = args.logger
    cfg.env.scene.num_envs = args.num_envs
    print(f"Released Repose recipe: backend={args.backend}, seed=42, "
          f"envs={args.num_envs}, iterations={args.iterations}, bf16, "
          "deterministic=False, release critic, fresh adapter training", flush=True)
    launch_training(TASK, cfg)


if __name__ == "__main__":
    main()
