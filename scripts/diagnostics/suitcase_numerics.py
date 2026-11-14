"""Check, collect or replay the bounded suitcase numerical experiment.

Run with the repository .venv. Only `train` starts a simulation; `check` is CPU
only. This entry point does not install packages, fetch code or start tmux.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
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
from vibe.tasks.uolm.numerics import (
    check_checkpoint,
    completed_capture,
    make_capture_runner,
)

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


def preflight(checkpoint, check_runtime=True, expected_iteration=6000, expected_counter=None):
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        raise ValueError("NVIDIA_TF32_OVERRIDE=0 conflicts with the TF32-enabled control")
    report = check_checkpoint(checkpoint, expected_iteration, expected_counter)
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


def extension_config(checkpoint, report, target_updates, run_name):
    """Finish stage one to 20k actual updates, then run the 1k phase-zero tail."""
    if target_updates not in (20000, 21000):
        raise ValueError("Continuation targets are 20000 and 21000 completed updates")
    completed = report["common_step_counter"] // 24
    if not 0 < completed < target_updates:
        raise ValueError("Checkpoint is already at/beyond the requested target")
    phase_zero = target_updates == 21000
    if phase_zero and completed != 20000:
        raise ValueError("The phase-zero tail must start after exactly 20000 updates")
    cfg = experiment_config(DEFAULT_CHECKPOINT, 1)
    checkpoint = Path(checkpoint).resolve()
    cfg.agent.load_run, cfg.agent.load_checkpoint = checkpoint.parent.name, checkpoint.name
    cfg.agent.max_iterations = target_updates - completed
    cfg.agent.run_name = run_name
    cfg.agent.save_interval = 1500
    cfg.env.commands["motion"].start_from_zero = phase_zero
    return cfg


def execute_training(cfg, options, report, *, align_completed_updates=False):
    os.chdir(ROOT)
    for key in ("WANDB_RUN_ID", "WANDB_RESUME"):
        os.environ.pop(key, None)
    os.environ.update(WANDB_USERNAME="lkrajan", WANDB_ENTITY="lkrajan", WANDB_PROJECT="vibe",
                      WANDB_RUN_GROUP="suitcase-numerics", WANDB_MODE="online")
    runner_cls = make_capture_runner(
        options, report, expected_iteration=report["iteration"],
        expected_counter=report["common_step_counter"], align_completed_updates=align_completed_updates,
    )
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
        exit_code, stop_reason = 2, str(exc)
    except BaseException as exc:
        exit_code, stop_reason = 1, f"{type(exc).__name__}: {exc}"
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


def require_idle_gpu():
    rows = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name", "--format=csv,noheader",
    ], text=True).splitlines()
    allowed = {"/usr/bin/ptyxis", "/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell"}
    busy = [(pid.strip(), name.strip()) for pid, name in csv.reader(rows)
            if pid.strip() != str(os.getpid()) and name.strip() not in allowed]
    if busy:
        raise RuntimeError(f"GPU has another process; continuation not started: {busy}")


def queued_continuation(source_run, source_pid, source_exitcode, output):
    """Wait for the exact source process, then gate both continuation stages."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental duplicate continuation supervisors.
    with (output / "queue.lock").open("x") as stream:
        stream.write(str(os.getpid()) + "\n")
    state = {"source_run": str(Path(source_run).resolve()), "source_pid": source_pid,
             "target_completed_updates": 21000, "stages": []}
    started = time.monotonic()
    process_stat = Path(f"/proc/{source_pid}/stat")

    def process_marker():
        try:
            return process_stat.read_text().rsplit(") ", 1)[1].split()[19]
        except FileNotFoundError:
            return None

    marker = process_marker()
    state.update(status="waiting_for_diagnostic", source_process_start=marker)
    write_json(output / "queue.json", state)
    print(f"[queue] Waiting for {source_run} to complete cleanly through 6999", flush=True)
    try:
        while marker is not None and process_marker() == marker:
            if time.monotonic() - started > 2 * 3600:
                raise TimeoutError("Source process exceeded the continuation wait budget")
            time.sleep(10)
        # The parent shell writes exitcode after the Python child has exited.
        source_status = ""
        for _ in range(30):
            if Path(source_exitcode).exists():
                source_status = Path(source_exitcode).read_text().strip()
                if source_status:
                    break
            time.sleep(1)
        if source_status != "0":
            raise RuntimeError("Diagnostic source exited nonzero; continuation refused")
        report = completed_capture(source_run, 6000, 1000, 144024 + 1000 * 24)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        for target, hours, phase in ((20000, 24, "stage1"), (21000, 2, "phase_zero")):
            require_idle_gpu()
            run_name = f"suitcase_continue_{stamp}_{phase}"
            checkpoint = report["checkpoint"]
            state.update(status=f"running_{phase}", checkpoint=checkpoint)
            write_json(output / "queue.json", state)
            print(f"[queue] Starting ViBe {phase} from {checkpoint} to {target} updates", flush=True)
            command = [
                "timeout", "--signal=TERM", "--kill-after=60s", f"{hours}h", sys.executable,
                str(Path(__file__).resolve()), "extend", "--checkpoint", checkpoint,
                "--expected-iteration", str(report["iteration"]),
                "--expected-counter", str(report["common_step_counter"]),
                "--target-updates", str(target), "--run-name", run_name,
            ]
            with (output / f"{phase}.log").open("x") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            runs = list((ROOT / "logs/rsl_rl/vibe_uolm_suitcase").glob(f"*_{run_name}"))
            if len(runs) != 1:
                raise RuntimeError(f"Cannot identify the continuation run: {runs}")
            completed = report["common_step_counter"] // 24
            first_label = completed if target == 20000 else report["iteration"]
            report = completed_capture(runs[0], first_label, target - completed, target * 24)
            state["stages"].append({"target": target, "run_dir": str(runs[0]), "checkpoint": report})
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="stopped", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "queue.json", state)


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
    extend = sub.add_parser("extend")
    extend.add_argument("--checkpoint", type=Path, required=True)
    extend.add_argument("--expected-iteration", type=int, required=True)
    extend.add_argument("--expected-counter", type=int, required=True)
    extend.add_argument("--target-updates", type=int, choices=(20000, 21000), required=True)
    extend.add_argument("--run-name", required=True)
    after = sub.add_parser("after")
    after.add_argument("--source-run", type=Path, required=True)
    after.add_argument("--source-pid", type=int, required=True)
    after.add_argument("--source-exitcode", type=Path, required=True)
    after.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "replay":
        result = replay_capture(args.capture, args.output, args.precision, args.device)
        print(result["outcome"])
        return
    if args.command == "after":
        queued_continuation(args.source_run, args.source_pid, args.source_exitcode, args.output)
        return
    if args.command == "extend":
        report = preflight(args.checkpoint, expected_iteration=args.expected_iteration,
                           expected_counter=args.expected_counter)
        cfg = extension_config(args.checkpoint, report, args.target_updates, args.run_name)
        expected = ROOT / cfg.log_root / cfg.agent.experiment_name / cfg.agent.load_run / cfg.agent.load_checkpoint
        if args.checkpoint.resolve() != expected.resolve():
            parser.error("Continuation checkpoint must be under this repository's experiment directory")
        execute_training(cfg, CaptureOptions(max_seconds=(24 if args.target_updates == 20000 else 2) * 3600),
                         report, align_completed_updates=args.target_updates == 20000)
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
    execute_training(cfg, options, report)


if __name__ == "__main__":
    main()
