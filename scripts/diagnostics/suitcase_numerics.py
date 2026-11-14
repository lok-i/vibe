"""Check, collect or replay the bounded suitcase numerical experiment.

Run with the repository .venv. Only `train` starts a simulation; `check` is CPU
only. This entry point does not install packages, fetch code or start tmux.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import mjlab  # noqa: F401 — register before importing orcs/vibe
from mjlab.scripts import train

from vibe.core.numerics import (
    CaptureOptions,
    DiagnosticStop,
    replay_capture,
    write_json,
)
from vibe.tasks.uolm.numerics import check_checkpoint, make_capture_runner

ROOT = Path(__file__).resolve().parents[2]
TASK = "Vibe-Uolm-Suitcase-ImgFeat-Ext"
BRANCH = "lok-i/uolm-suitcase-vision"
DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/vibe_uolm_suitcase/"
    "2026-11-08_01-15-11_suitcase_s42_baseline_vhagar/model_6000.pt"
)


def experiment_config(checkpoint, updates=1000):
    if not 1 <= updates <= 1000:
        raise ValueError("The experiment is bounded to 1–1000 additional updates")
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != "model_6000.pt":
        raise ValueError("This experiment starts from model_6000.pt")
    cfg = train.TrainConfig.from_task(TASK)
    cfg.env.scene.num_envs = 4096
    cfg.env.scene.env_spacing = 2.5
    cfg.env.scene.terrain.env_spacing = 2.5
    motion = cfg.env.commands["motion"]
    motion.sampling_mode = "start"
    motion.init_phase_anneal_iterations = 20000
    motion.start_from_zero = False
    cfg.env.events["virtual_object_force"].params["natural_frequency"] = 0.0
    cfg.agent.seed = 42
    cfg.agent.amp_dtype = None
    cfg.agent.torch_compile_mode = None
    cfg.agent.num_steps_per_env = 24
    cfg.agent.max_iterations = updates
    cfg.agent.resume = True
    cfg.agent.load_run = checkpoint.parent.name
    cfg.agent.load_checkpoint = checkpoint.name
    cfg.agent.run_name = "suitcase_numerics_s42_from6000_tf32"
    cfg.agent.save_interval = 100
    cfg.agent.upload_model = False
    cfg.agent.logger = "wandb"
    cfg.agent.wandb_project = "vibe"
    cfg.agent.algorithm.class_name = "vibe.core.numerics:DiagnosticPPO"
    return cfg


def preflight(checkpoint, check_runtime=True):
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        raise ValueError("NVIDIA_TF32_OVERRIDE=0 conflicts with the TF32-enabled control")
    report = check_checkpoint(checkpoint)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if branch != BRANCH:
        raise ValueError(f"Use {BRANCH}; current branch is {branch}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise ValueError(f"Commit/review the development checkout before running:\n{dirty}")
    for script, args in (("sync_deps.sh", []), ("sync_data.sh", ["hoiretarget"])):
        subprocess.run(["bash", f"scripts/setup/{script}", *args, "--check"], cwd=ROOT, check=True)
    expected = json.loads(Path(__file__).with_name("suitcase_runtime.json").read_text())
    actual = {name: version(name) for name in expected}
    drift = {name: {"expected": wanted, "actual": actual[name]}
             for name, wanted in expected.items() if actual[name] != wanted}
    if check_runtime and drift:
        raise ValueError(f"Runtime differs from the failed control; do not install automatically: {drift}")
    with Path(checkpoint).open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report.update({"runtime": actual, "runtime_drift": drift, "branch": branch,
                   "vibe_commit": subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "checkpoint_sha256": checkpoint_hash})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "train"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, default=ROOT / DEFAULT_CHECKPOINT)
        command.add_argument("--updates", type=int, default=1000)
        command.add_argument("--minutes", type=float, default=90)
        if name == "check":
            command.add_argument("--config-only", action="store_true")
    replay = sub.add_parser("replay")
    replay.add_argument("capture", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--precision", choices=("tf32", "ieee"), required=True)
    replay.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.command == "replay":
        result = replay_capture(args.capture, args.output, args.precision, args.device)
        print(result["outcome"])
        return
    if not 0 < args.minutes <= 90:
        parser.error("--minutes must be in (0, 90]")
    options = CaptureOptions(max_seconds=args.minutes * 60)
    cfg = experiment_config(args.checkpoint, args.updates)
    if args.command == "check" and args.config_only:
        # No environment, checkpoint, network or GPU initialization.
        print(json.dumps({"task": TASK, "agent": asdict(cfg.agent),
                          "num_envs": cfg.env.scene.num_envs,
                          "scene_spacing": cfg.env.scene.env_spacing,
                          "terrain_spacing": cfg.env.scene.terrain.env_spacing,
                          "natural_frequency": cfg.env.events["virtual_object_force"].params["natural_frequency"],
                          "anneal": cfg.env.commands["motion"].init_phase_anneal_iterations,
                          "capture": asdict(options)}, indent=2, default=str))
        return
    # The standard runner resolves the checkpoint under log_root/experiment.
    expected = ROOT / cfg.log_root / cfg.agent.experiment_name / cfg.agent.load_run / cfg.agent.load_checkpoint
    if args.checkpoint.resolve() != expected.resolve():
        parser.error(f"Checkpoint must be under {ROOT / cfg.log_root / cfg.agent.experiment_name}")
    report = preflight(args.checkpoint)
    if args.command == "check":
        print(json.dumps(report, indent=2))
        return
    os.chdir(ROOT)
    for key in ("WANDB_RUN_ID", "WANDB_RESUME"):
        os.environ.pop(key, None)
    os.environ.update(WANDB_USERNAME="lkrajan", WANDB_ENTITY="lkrajan", WANDB_PROJECT="vibe",
                      WANDB_RUN_GROUP="suitcase-numerics", WANDB_MODE="online")
    runner_cls = make_capture_runner(options, report)
    original = train.load_runner_cls
    train.load_runner_cls = lambda task: runner_cls if task == TASK else original(task)

    def stop(signum, frame):
        raise DiagnosticStop(f"Received signal {signum}")

    previous = signal.signal(signal.SIGTERM, stop)
    exit_code = 0
    stop_reason = "Update budget completed"
    try:
        train.launch_training(TASK, copy.deepcopy(cfg))
    except DiagnosticStop as exc:
        print(f"[numerics] stopped: {exc}", file=sys.stderr)
        exit_code = 2
        stop_reason = str(exc)
    except BaseException as exc:
        exit_code = 1
        stop_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        train.load_runner_cls = original
        signal.signal(signal.SIGTERM, previous)
        import wandb

        if wandb.run is not None:
            wandb.run.summary["numerics_exit_code"] = exit_code
            wandb.run.summary["numerics_stop_reason"] = stop_reason
            wandb.finish(exit_code=exit_code)
        write_json(ROOT / "logs/uolm_suitcase/numerics_last_exit.json", {
            "exit_code": exit_code, "reason": stop_reason,
        })
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
