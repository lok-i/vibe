"""Suitcase continuation, curriculum restoration and diagnostic runner lifecycle."""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner
from tensordict import TensorDict

from vibe.core.numerics import (
    CaptureOptions,
    DiagnosticPPO,
    FlightRecorder,
    finite,
    sync_adaptive_lr,
    tensor_leaves,
    write_json,
)


def check_checkpoint(path, expected_iteration=6000, expected_counter=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["iter"] != expected_iteration:
        raise ValueError(f"Expected iteration {expected_iteration}, got {checkpoint['iter']}")
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        if not checkpoint.get(key):
            raise ValueError(f"Missing checkpoint state: {key}")
    finite(checkpoint, "checkpoint")
    counter = checkpoint["infos"]["env_state"]["common_step_counter"]
    expected_counter = (expected_iteration + 1) * 24 if expected_counter is None else expected_counter
    if counter != expected_counter or counter <= 0 or counter % 24:
        raise ValueError(f"Unexpected curriculum counter: {counter}")
    rates = {group["lr"] for group in checkpoint["optimizer_state_dict"]["param_groups"]}
    if len(rates) != 1 or not all(0 < rate < float("inf") for rate in rates):
        raise ValueError(f"Invalid saved learning rates: {rates}")
    return {"checkpoint": str(Path(path).resolve()), "iteration": checkpoint["iter"],
            "common_step_counter": counter, "learning_rate": rates.pop(),
            "tensors_checked": sum(1 for _ in tensor_leaves(checkpoint))}


def disabled_vof_log(extras):
    """Clarify this experiment's logger only; leave ORCS's wrench event intact."""
    log = dict(extras.get("log", {}))
    log.pop("VirtualObjectForce/alpha", None)
    log["VirtualObjectForce/enabled"] = 0.0
    return {**extras, "log": log}


def completed_capture(run_dir, first_iteration, num_updates, final_counter):
    """Gate continuation on a complete, finite window and its exact final checkpoint."""
    run_dir = Path(run_dir)
    stop = json.loads((run_dir / "numerics/stop.json").read_text())
    if (stop["reason"] != "Update budget completed" or not stop["update_ready"]
            or stop["iteration"] != first_iteration + num_updates):
        raise ValueError(f"Source run did not complete cleanly: {stop['reason']}")
    count = 0
    with (run_dir / "numerics/updates.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            record = json.loads(line)
            if record["iteration"] != first_iteration + count - 1:
                raise ValueError("Missing or repeated updates in the source diagnostic window")
            if not record["losses"] or not all(
                isinstance(x, (int, float)) and math.isfinite(x) for x in record["losses"].values()
            ):
                raise ValueError("Nonfinite source update; continuation refused")
    if count != num_updates:
        raise ValueError(f"Expected {num_updates} completed updates, got {count}")
    if not (run_dir / "numerics/capture.pt").is_file():
        raise ValueError("Final diagnostic capture is missing")
    checkpoint = run_dir / f"model_{first_iteration + num_updates - 1}.pt"
    return check_checkpoint(checkpoint, first_iteration + num_updates - 1, final_counter)


def make_capture_runner(options=None, metadata=None, *, expected_iteration=6000,
                        expected_counter=None, align_completed_updates=False):
    options = options or CaptureOptions()
    class CaptureRunner(MjlabOnPolicyRunner):
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            train_cfg["numerics"] = {"capture": asdict(options), "control": "lkrajan/vibe/9t1lpbr5"}
            recipe = copy.deepcopy(train_cfg)
            super().__init__(env, train_cfg, log_dir, device)
            if type(self.alg) is not DiagnosticPPO:
                raise TypeError("Capture runner requires DiagnosticPPO")
            output = Path(log_dir) / "numerics"
            output.mkdir(parents=True, exist_ok=True)
            # One full rollout in CPU RAM; disk is used only on stop/end. The
            # multiplier reserves room for model/optimizer copies and serialization.
            rollout_bytes = sum(t.numel() * t.element_size()
                                for _, t in tensor_leaves(self.alg.storage.observations))
            if shutil.disk_usage(output).free < 3 * rollout_bytes + 1024**3:
                raise RuntimeError("Insufficient disk space for the diagnostic capture")
            meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(meminfo["MemAvailable"].split()[0]) * 1024
            if available < 3 * rollout_bytes + 2 * 1024**3:
                raise RuntimeError("Insufficient host RAM for the diagnostic capture")
            self.recorder = FlightRecorder(
                self.alg, recipe, TensorDict(env.unwrapped.obs_buf, batch_size=[env.num_envs]), output, options,
                {**(metadata or {}), "rollout_observation_bytes": rollout_bytes},
            )
            self.alg.recorder = self.recorder
            step = self.env.step

            def observed_step(actions):
                result = step(actions)
                # The parent runner's NaN check precedes process_env_step. Keep
                # its offending observation available if that guard fires first.
                self.recorder.current_obs = result[0]
                return result

            self.env.step = observed_step
            self.vof_disabled = env.unwrapped.cfg.events["virtual_object_force"].params["natural_frequency"] == 0
            process = self.logger.process_env_step

            def log_step(rewards, dones, extras, intrinsic_rewards=None):
                return process(rewards, dones, disabled_vof_log(extras) if self.vof_disabled else extras,
                               intrinsic_rewards)

            self.logger.process_env_step = log_step

        def load(self, path, load_cfg=None, strict=True, map_location=None):
            if load_cfg is not None or not strict:
                raise ValueError("The diagnostic continuation requires a full strict checkpoint load")
            report = check_checkpoint(path, expected_iteration, expected_counter)
            infos = super().load(path, load_cfg, strict, map_location or "cpu")
            sync_adaptive_lr(self.alg)
            env = self.env.unwrapped
            if env.common_step_counter != report["common_step_counter"]:
                raise RuntimeError("Curriculum counter was not restored")
            # The counter event normally refreshes on the next env step. Refresh
            # it BEFORE resetting episodes, so initial RSI uses the restored phase.
            event = env.cfg.events["policy_update_counter"].params
            env.policy_update_count = (event.get("init_policy_update_count", 0)
                                       + env.common_step_counter // event["num_steps_per_env"])
            counter = env.common_step_counter
            self.env.reset()
            if env.common_step_counter != counter:
                raise RuntimeError("Episode reset changed the curriculum counter")
            if align_completed_updates:
                # Diagnostic resume repeats label 6000. Continue the remaining
                # stage using the actual completed-rollout count, not that label.
                self.current_learning_iteration = counter // event["num_steps_per_env"]
            self.recorder.iteration = self.current_learning_iteration
            self.recorder.metadata["resume"] = report
            write_json(self.recorder.output / "resume.json", {
                **report, "policy_update_count": env.policy_update_count,
                "next_iteration_label": self.current_learning_iteration,
                "ppo_learning_rate": self.alg.learning_rate,
                "episode_state_restored": False, "rng_state_restored": False,
            })
            return infos

        def learn(self, *args, **kwargs):
            reason = "Update budget completed"
            failed = False
            try:
                return super().learn(*args, **kwargs)
            except BaseException as exc:
                failed = True
                reason = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                try:
                    self.recorder.dump(reason)
                finally:
                    self.recorder.close()
                    if failed:
                        self.env.close()

    return CaptureRunner
