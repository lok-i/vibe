"""Opt-in PPO flight recorder and simulator-free update replay.

This is an investigation tool, not a replacement optimizer. The pinned rsl_rl
PPO calls two diagnostic hooks around its existing clipping/Adam step. We read
the caller's loss tensors at the first hook, after backward and BEFORE clipping
or Adam. No training expression is copied or changed. Unsupported algorithms,
recurrent models, compilation and distributed training are rejected.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import random
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

STORAGE_FIELDS = (
    "observations", "actions", "rewards", "dones", "values", "returns",
    "advantages", "actions_log_prob", "distribution_params", "step",
)


class DiagnosticStop(RuntimeError):
    """An intentional stop; its capture contains the reason and execution stage."""


@dataclass(frozen=True)
class CaptureOptions:
    value_limit: float = 100.0
    spike_factor: float = 50.0
    history_size: int = 100
    warmup_batches: int = 20
    max_seconds: float = 5400.0

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (
            self.value_limit, self.spike_factor, self.max_seconds,
        )) or not 1 <= self.warmup_batches <= self.history_size:
            raise ValueError("Capture limits must be positive, with a valid history window")


def cpu_copy(value):
    """Independent snapshots, including nested visual TensorDict groups."""
    if isinstance(value, TensorDict):
        return value.detach().to("cpu").clone()
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def tensor_leaves(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (dict, TensorDict)):
        for key, child in value.items():
            yield from tensor_leaves(child, f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for key, child in enumerate(value):
            yield from tensor_leaves(child, f"{prefix}/{key}")


def finite(value, label):
    for name, tensor in tensor_leaves(value, label):
        if not torch.isfinite(tensor).all():
            raise DiagnosticStop(f"Nonfinite tensor: {name}")


def tensor_stats(tensor):
    x = tensor.detach().reshape(-1)
    if x.numel() == 0:
        return {"shape": list(tensor.shape), "empty": True}
    good = torch.isfinite(x)
    # Report an offending index even when reductions themselves are nonfinite.
    idx = int((~good).int().argmax()) if not good.all() else int(x.abs().argmax())
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "min": float(x.min()), "max": float(x.max()),
        "absmax": float(x.abs().max()), "nonfinite": int((~good).sum()),
        "extreme_index": list(np.unravel_index(idx, tensor.shape)),
    }


def stats(value):
    return {name: tensor_stats(tensor) for name, tensor in tensor_leaves(value)}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rng_state(device):
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    return cpu_copy(result)


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if torch.device(device).type != "cuda":
            raise ValueError("A CUDA capture requires CUDA replay")
        torch.cuda.set_rng_state(state["cuda"], device)


def source_hashes():
    import rsl_rl

    root = Path(rsl_rl.__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


def runtime_metadata(device):
    return {
        "packages": {name: version(name) for name in (
            "torch", "torchvision", "triton", "tensordict", "numpy", "mjlab",
            "mujoco", "mujoco-warp", "warp-lang",
        )},
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        "matmul_precision": torch.backends.cuda.matmul.fp32_precision,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_precision": torch.backends.cudnn.fp32_precision,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
        "environment": {name: os.environ.get(name) for name in (
            "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VIBE_AMP",
        )},
    }


def sync_adaptive_lr(alg):
    """PPO.load restores Adam, but not PPO's separate adaptive-rate variable."""
    rates = {group["lr"] for group in alg.optimizer.param_groups if not group.get("fixed_lr", False)}
    if len(rates) != 1 or not all(math.isfinite(x) and x > 0 for x in rates):
        raise ValueError(f"Expected one finite adaptive learning rate, got {rates}")
    alg.learning_rate = rates.pop()


class FlightRecorder:
    def __init__(self, alg, config, template, output, options=None, metadata=None):
        options = options or CaptureOptions()
        if (alg.is_multi_gpu or alg.rnd or alg.symmetry or alg.actor.is_recurrent
                or alg.critic.is_recurrent or config.get("torch_compile_mode")
                or config.get("amp_dtype") is not None):
            raise ValueError("Recorder supports single-device, feedforward, FP32 plain PPO only")
        if any(hasattr(m, "dualize_gradients") for m in (alg.actor, alg.critic)):
            raise ValueError("Recorder requires ordinary norm clipping")
        self.alg, self.options = alg, options
        self.config = copy.deepcopy(config)
        self.template = cpu_copy(template[:1])
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metadata = {**(metadata or {}), "runtime": runtime_metadata(alg.device),
                         "rsl_sources": source_hashes()}
        self.history = deque(maxlen=options.history_size)
        self.snapshot = None
        self.rollout_start = None
        self.records = []
        self.normalized = {}
        self.current_obs = None
        self.stage, self.iteration, self.batch_number = "initial", 0, 0
        self.started = time.monotonic()
        self.before_step = None
        self.saved = False
        self.handles = []
        for owner, model in (("actor", alg.actor), ("critic", alg.critic)):
            for name, module in model.named_modules():
                if isinstance(module, EmpiricalNormalization):
                    key = f"{owner}/{name}"
                    self.handles.append(module.register_forward_hook(self._normalizer_hook(key)))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def _normalizer_hook(self, key):
        def hook(module, inputs, output):
            self.normalized[key] = tensor_stats(output)
            finite(output, f"normalized/{key}")
        return hook

    def deadline(self):
        if time.monotonic() - self.started >= self.options.max_seconds:
            raise DiagnosticStop("Wall-clock budget reached")

    def begin_rollout(self, obs):
        self.deadline()
        self.snapshot = None  # release the previous large CPU rollout before replacing it
        self.saved = False
        self.records = []
        self.stage = "collection"
        self.rollout_start = {"model": cpu_copy(self.alg.save()),
                              "rng": rng_state(self.alg.device),
                              "learning_rate": self.alg.learning_rate}
        self.current_obs = obs

    def begin_update(self):
        self.deadline()
        self.stage = "update"
        self.batch_number = 0
        self.snapshot = self.capture(update_ready=True)
        finite(self.snapshot["model"], "pre_update_model")
        finite(self.snapshot["storage"], "rollout")

    def capture(self, update_ready):
        return {
            "format": 1, "update_ready": update_ready, "iteration": self.iteration,
            "config": self.config, "template": self.template, "metadata": self.metadata,
            "options": asdict(self.options), "history": list(self.history),
            "model": cpu_copy(self.alg.save()), "learning_rate": self.alg.learning_rate,
            "rng": rng_state(self.alg.device), "rollout_start": self.rollout_start,
            "storage": {key: cpu_copy(getattr(self.alg.storage, key)) for key in STORAGE_FIELDS},
            "num_envs": self.alg.storage.num_envs,
            "num_steps": self.alg.storage.num_transitions_per_env,
            "num_actions": self.alg.storage.actions.shape[-1],
            "permutation": None,
        }

    def dump(self, reason):
        if self.saved:
            return
        snapshot = self.snapshot if self.snapshot is not None else self.capture(update_ready=False)
        snapshot["stop"] = {"reason": str(reason), "stage": self.stage,
                            "iteration": self.iteration, "batch": self.batch_number}
        snapshot["records"] = self.records
        snapshot["last_normalized_outputs"] = self.normalized
        snapshot["observed_state"] = cpu_copy(self.alg.save())
        snapshot["observed_gradients"] = {
            owner: cpu_copy({name: p.grad for name, p in model.named_parameters() if p.grad is not None})
            for owner, model in (("actor", self.alg.actor), ("critic", self.alg.critic))
        }
        if not snapshot["update_ready"] and self.current_obs is not None:
            snapshot["offending_observations"] = cpu_copy(self.current_obs)
        temporary = self.output / "capture.pt.tmp"
        torch.save(snapshot, temporary)
        temporary.replace(self.output / "capture.pt")
        write_json(self.output / "stop.json", {
            **snapshot["stop"], "update_ready": snapshot["update_ready"],
            "records": self.records, "last_normalized_outputs": self.normalized,
            "gradient_stats": stats(snapshot["observed_gradients"]),
            "adam_stats": stats(snapshot["observed_state"]["optimizer_state_dict"]["state"]),
            "metadata": self.metadata,
        })
        self.saved = True
        print(f"[numerics] {reason}; capture: {self.output / 'capture.pt'}", flush=True)

    def before_optimizer(self, local):
        """Read the actual upstream loss, after backward and before any optimizer mutation."""
        required = {"loss", "value_loss", "surrogate_loss", "values", "batch", "generator", "log_ratio"}
        if not required <= local.keys():
            raise RuntimeError("Pinned PPO diagnostic hook contract changed")
        generator_locals = local["generator"].gi_frame.f_locals
        indices, batch_idx = generator_locals["indices"], generator_locals["batch_idx"]
        if self.snapshot["permutation"] is None:
            self.snapshot["permutation"] = cpu_copy(indices)
        self.batch_number += 1
        batch = local["batch"]
        row = int((local["values"].detach() - batch.returns).abs().reshape(-1).argmax())
        transition = int(batch_idx[row])
        value_loss = float(local["value_loss"].detach())
        record = {
            "kind": "minibatch", "batch": self.batch_number,
            "value_loss": value_loss, "loss": float(local["loss"].detach()),
            "surrogate_loss": float(local["surrogate_loss"].detach()),
            "learning_rate": self.alg.learning_rate,
            "values": tensor_stats(local["values"]), "targets": tensor_stats(batch.returns),
            "log_ratio": tensor_stats(local["log_ratio"]),
            "largest_residual_transition": {"step": transition // self.alg.storage.num_envs,
                                            "env": transition % self.alg.storage.num_envs},
            "normalized": dict(self.normalized),
        }
        self.records.append(record)
        finite({key: local[key] for key in ("loss", "value_loss", "surrogate_loss", "log_ratio")}, "loss")
        for name, model in (("actor", self.alg.actor), ("critic", self.alg.critic)):
            gradients = {key: p.grad for key, p in model.named_parameters() if p.grad is not None}
            finite(gradients, f"{name}/gradients")
            norms = torch.stack([torch.linalg.vector_norm(g) for g in gradients.values()])
            norm = torch.linalg.vector_norm(norms)
            record[f"{name}_grad_norm"] = float(norm)
            # Finite individual gradients can still overflow the norm used by clipping.
            finite(norm, f"{name}/gradient_norm")
        baseline = statistics.median(self.history) if len(self.history) >= self.options.warmup_batches else None
        record["value_baseline"] = baseline
        if value_loss > self.options.value_limit or (
            baseline is not None and value_loss > self.options.spike_factor * max(baseline, 1e-6)
        ):
            raise DiagnosticStop(f"Finite value-loss spike: {value_loss:g} (baseline {baseline})")
        self.history.append(value_loss)
        self.before_step = [(p, p.detach().clone()) for group in self.alg.optimizer.param_groups
                            for p in group["params"] if p.requires_grad]

    def after_optimizer(self):
        finite(self.alg.save(), "post_optimizer")
        self.records[-1]["parameter_delta_absmax"] = max(
            float((p.detach() - old).abs().max()) for p, old in self.before_step
        )
        self.before_step = None

    def completed_update(self, result):
        batches = [r for r in self.records if r["kind"] == "minibatch"]
        for key in ("value_loss", "actor_grad_norm", "critic_grad_norm", "parameter_delta_absmax"):
            result[1][f"Numerics/{key}_max"] = max(r[key] for r in batches)
        record = {"iteration": self.iteration, "losses": result[0],
                  "minibatches": batches}
        with (self.output / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        self.iteration += 1


class DiagnosticPPO(PPO):
    """Plain upstream PPO with an explicitly attached flight recorder."""

    recorder: FlightRecorder | None = None

    def act(self, obs):
        recorder = self.recorder
        if recorder:
            if self.storage.step == 0:
                recorder.begin_rollout(obs)
            recorder.deadline()
            recorder.current_obs = obs
            recorder.normalized = {}
            recorder.records.append({"kind": "observation", "step": self.storage.step,
                                     "stats": stats(obs)})
            finite(obs, "raw_observations")
        actions = super().act(obs)
        if recorder:
            finite((actions, self.transition.values, self.transition.actions_log_prob), "collection_output")
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self.recorder:
            self.recorder.current_obs = obs
            self.recorder.records.append({"kind": "reward", "step": self.storage.step,
                                          "stats": stats(rewards)})
            finite((obs, rewards, dones, extras.get("time_outs")), "env_step")
        super().process_env_step(obs, rewards, dones, extras)
        if self.recorder:
            for name, model in (("actor", self.actor), ("critic", self.critic)):
                finite(dict(model.named_buffers()), f"{name}/buffers_after_normalization")

    def compute_returns(self, obs):
        if self.recorder:
            self.recorder.stage = "returns"
            self.recorder.current_obs = obs
        super().compute_returns(obs)
        if self.recorder:
            finite((self.storage.returns, self.storage.advantages), "returns_and_advantages")

    def update(self):
        if not self.recorder:
            return super().update()
        self.recorder.begin_update()
        original = self.storage.mini_batch_generator

        def observed_batches(*args, **kwargs):
            generator = original(*args, **kwargs)
            for batch in generator:
                # These locals are also read by before_optimizer. Record the
                # order before forward, including failures in normalization.
                local = generator.gi_frame.f_locals
                indices, batch_idx = local["indices"], local["batch_idx"]
                if self.recorder.snapshot["permutation"] is None:
                    self.recorder.snapshot["permutation"] = cpu_copy(indices)
                yield batch
                del batch_idx

        self.storage.mini_batch_generator = observed_batches
        try:
            result = super().update()
            self.recorder.completed_update(result)
            return result
        finally:
            self.storage.mini_batch_generator = original

    def _extra_backward(self):
        if self.recorder:
            frame = inspect.currentframe()
            try:
                self.recorder.before_optimizer(frame.f_back.f_locals)
            finally:
                del frame

    def _post_optimizer_step(self):
        if self.recorder:
            self.recorder.after_optimizer()


def fixed_batches(storage, num_mini_batches, num_epochs=8):
    """Replay the captured permutation, consuming the original RNG draw as well.

    Batch layout follows the pinned RolloutStorage.mini_batch_generator. Saving
    the permutation makes sample order explicit across runtime versions.
    """
    size = storage.num_envs * storage.num_transitions_per_env // num_mini_batches
    torch.randperm(num_mini_batches * size, requires_grad=False, device=storage.device)
    indices = storage.replay_permutation.to(storage.device)
    fields = {"observations": "observations", "actions": "actions", "values": "values",
              "returns": "returns", "advantages": "advantages",
              "old_actions_log_prob": "actions_log_prob"}
    flat = {key: getattr(storage, attr).flatten(0, 1) for key, attr in fields.items()}
    params = tuple(p.flatten(0, 1) for p in storage.distribution_params)
    for _ in range(num_epochs):
        for i in range(num_mini_batches):
            batch_idx = indices[i * size:(i + 1) * size]
            yield RolloutStorage.Batch(
                **{key: value[batch_idx] for key, value in flat.items()},
                old_distribution_params=tuple(p[batch_idx] for p in params),
            )


def replay_capture(path, output, precision="tf32", device="cuda:0"):
    """Run exactly one captured PPO update. No environment, renderer or encoder."""
    if precision not in ("tf32", "ieee"):
        raise ValueError("Replay precision must be tf32 or ieee")
    if (Path(output) / "capture.pt").exists() or (Path(output) / "updates.jsonl").exists():
        raise FileExistsError("Use a fresh replay output directory; existing evidence is preserved")
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if snapshot["format"] != 1 or not snapshot["update_ready"] or snapshot["permutation"] is None:
        raise ValueError("Capture has no replayable PPO update; inspect collection/return diagnostics")
    if snapshot["metadata"]["rsl_sources"] != source_hashes():
        raise ValueError("rsl_rl source differs from capture; use the recorded dependency pin")
    runtime = snapshot["metadata"]["runtime"]
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "0" and precision == "tf32":
        raise ValueError("NVIDIA_TF32_OVERRIDE=0 prevents a TF32-enabled comparison")
    # ViBe's actual training patch uses this unified API (not the stock mjlab
    # backend setter). Preserve that path in the control arm. The captured
    # visual features already exclude encoder/cuDNN differences from the update.
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
    torch.backends.cudnn.fp32_precision = runtime["cudnn_precision"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = runtime["cudnn_deterministic"]
    torch.use_deterministic_algorithms(runtime["deterministic_algorithms"])
    torch.set_num_threads(runtime["num_threads"])
    config = copy.deepcopy(snapshot["config"])
    # Full model state is in the capture; the original base checkpoint path need
    # not exist on the replay host. This avoids loading a second external asset.
    if "base_checkpoint" in config["actor"]:
        config["actor"]["base_checkpoint"] = None
    # The regular mjlab runner removes these optional dataclass fields before
    # model construction. Replay bypasses that runner, so do the same cleanup.
    for key in ("actor", "critic"):
        for opt in ("cnn_cfg", "distribution_cfg"):
            if config[key].get(opt) is None:
                config[key].pop(opt, None)
        if config[key].get("rnn_type") is None:
            for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                config[key].pop(opt, None)
    template = snapshot["template"].to(device).expand(snapshot["num_envs"]).contiguous()
    env = SimpleNamespace(num_envs=snapshot["num_envs"], num_actions=snapshot["num_actions"])
    alg = PPO.construct_algorithm(template, env, copy.deepcopy(config), device)
    alg.load(snapshot["model"], None, True)
    alg.learning_rate = snapshot["learning_rate"]
    alg.train_mode()
    for key, value in snapshot["storage"].items():
        if isinstance(value, (torch.Tensor, TensorDict)):
            getattr(alg.storage, key).copy_(value.to(device))
        elif isinstance(value, tuple):
            setattr(alg.storage, key, tuple(v.to(device) for v in value))
        else:
            setattr(alg.storage, key, value)
    alg.storage.replay_permutation = snapshot["permutation"]
    alg.storage.mini_batch_generator = MethodType(fixed_batches, alg.storage)
    recorder = FlightRecorder(alg, config, template, output, CaptureOptions(**snapshot["options"]),
                              {"replay_of": str(Path(path).resolve()), "precision": precision})
    alg.recorder = recorder
    recorder.iteration = snapshot["iteration"]
    recorder.history.extend(snapshot["history"])
    restore_rng(snapshot["rng"], device)
    outcome = "completed"
    try:
        alg.update()
    except DiagnosticStop as exc:
        outcome = str(exc)
    except BaseException as exc:
        outcome = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        recorder.dump(outcome)
        recorder.close()
    return {"outcome": outcome, "records": recorder.records, "state": cpu_copy(alg.save()),
            "rng": rng_state(device)}
