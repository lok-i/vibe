"""Actual PPO updates on CPU: recorder transparency, tripwires, capture and replay."""

from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mjlab  # noqa: F401 — task registration order
import pytest
import torch
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from vibe.core.numerics import (
    CaptureOptions,
    DiagnosticStop,
    FlightRecorder,
    cpu_copy,
    replay_capture,
    restore_rng,
    rng_state,
    sync_adaptive_lr,
    tensor_leaves,
)
from vibe.tasks.uolm.numerics import (
    check_checkpoint,
    completed_capture,
    disabled_vof_log,
    make_capture_runner,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def recipe(diagnostic=True):
    return {
        "algorithm": {
            "class_name": "vibe.core.numerics:DiagnosticPPO" if diagnostic else "PPO",
            "num_learning_epochs": 2, "num_mini_batches": 2,
            "learning_rate": 3e-4, "schedule": "adaptive", "desired_kl": 0.01,
        },
        "actor": {"class_name": "MLPModel", "hidden_dims": [8], "obs_normalization": True,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.3}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [8], "obs_normalization": True},
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "num_steps_per_env": 4, "multi_gpu": None, "amp_dtype": None,
    }


def observation(n=8):
    return TensorDict({"policy": torch.randn(n, 3), "critic": torch.randn(n, 4)}, batch_size=[n])


def build(tmp_path, diagnostic=True, options=None):
    torch.manual_seed(123)
    obs = observation()
    cfg = recipe(diagnostic)
    alg = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=8, num_actions=2), copy.deepcopy(cfg), "cpu")
    alg.train_mode()
    if diagnostic:
        alg.recorder = FlightRecorder(alg, cfg, obs, tmp_path, options)
    return alg, obs


def collect(alg, obs):
    with torch.inference_mode():
        for step in range(alg.storage.num_transitions_per_env):
            actions = alg.act(obs)
            obs = observation(alg.storage.num_envs)
            rewards = 0.1 - 0.01 * actions.square().sum(-1)
            dones = torch.zeros(alg.storage.num_envs, dtype=torch.long)
            dones[step % alg.storage.num_envs] = 1
            alg.process_env_step(obs, rewards, dones, {"time_outs": dones.clone()})
        alg.compute_returns(obs)


def equal_trees(a, b):
    if isinstance(a, (torch.Tensor, TensorDict)):
        assert type(a) is type(b)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else (a == b).all()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_trees(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal_trees(x, y)
    else:
        assert a == b


def no_adam_step(before, after):
    # PPO's adaptive KL rate changes before backward. The tripwire must stop
    # parameter/moment/step-counter updates, not undo that existing LR decision.
    for key in ("actor_state_dict", "critic_state_dict"):
        equal_trees(before[key], after[key])
    equal_trees(before["optimizer_state_dict"]["state"], after["optimizer_state_dict"]["state"])


def test_recorder_preserves_collection_update_adam_and_rng_bitwise(tmp_path):
    reference, obs = build(tmp_path, diagnostic=False)
    collect(reference, obs)
    expected_rollout = cpu_copy(reference.storage.observations)
    expected = reference.update()
    state = cpu_copy(reference.save())
    random_after = torch.get_rng_state().clone()
    instrumented, obs = build(tmp_path / "instrumented")
    collect(instrumented, obs)
    equal_trees(expected_rollout, instrumented.storage.observations)
    actual = instrumented.update()
    equal_trees(expected[0], actual[0])
    for key, value in expected[1].items():
        equal_trees(value, actual[1][key])
    assert "Numerics/value_loss_max" in actual[1]
    equal_trees(state, instrumented.save())
    assert torch.equal(random_after, torch.get_rng_state())
    instrumented.recorder.close()


def test_capture_is_independent_and_replay_matches_full_update(tmp_path):
    alg, obs = build(tmp_path / "original")
    collect(alg, obs)
    alg.update()
    expected = cpu_copy(alg.save())
    random_after = torch.get_rng_state().clone()
    alg.recorder.dump("test finished")
    # A second live update must not mutate the saved CPU snapshot.
    before = cpu_copy(alg.recorder.snapshot["model"])
    with torch.no_grad():
        next(alg.actor.parameters()).add_(10)
    equal_trees(before, alg.recorder.snapshot["model"])
    alg.recorder.close()
    result = replay_capture(tmp_path / "original/capture.pt", tmp_path / "replay", device="cpu")
    assert result["outcome"] == "completed"
    equal_trees(expected, result["state"])
    assert torch.equal(random_after, result["rng"]["torch"])
    with pytest.raises(FileExistsError, match="fresh replay"):
        replay_capture(tmp_path / "original/capture.pt", tmp_path / "original", device="cpu")


def test_finite_loss_spike_stops_before_adam_and_replays(tmp_path):
    alg, obs = build(tmp_path / "original", options=CaptureOptions(value_limit=1))
    collect(alg, obs)
    alg.storage.returns.fill_(20)
    before = cpu_copy(alg.save())
    with pytest.raises(DiagnosticStop, match="Finite value-loss spike") as exc:
        alg.update()
    no_adam_step(before, alg.save())
    observed = cpu_copy(alg.save())
    alg.recorder.dump(exc.value)
    alg.recorder.close()
    replay = replay_capture(tmp_path / "original/capture.pt", tmp_path / "replay", device="cpu")
    assert "Finite value-loss spike" in replay["outcome"]
    equal_trees(observed, replay["state"])
    assert replay["records"][-1]["value_loss"] == alg.recorder.records[-1]["value_loss"]


@pytest.mark.parametrize("gradient", [float("inf"), 1e30])
def test_infinite_gradient_or_overflowing_norm_stops_before_adam(tmp_path, gradient):
    alg, obs = build(tmp_path)
    collect(alg, obs)
    before = cpu_copy(alg.save())
    handle = next(alg.critic.parameters()).register_hook(lambda grad: torch.full_like(grad, gradient))
    with pytest.raises(DiagnosticStop, match="gradient"):
        alg.update()
    no_adam_step(before, alg.save())
    handle.remove()
    alg.recorder.close()


def test_nonfinite_loss_and_normalizer_failures_are_captured(tmp_path):
    alg, obs = build(tmp_path / "loss")
    collect(alg, obs)
    # Finite inputs whose squared residual overflows during the actual loss.
    alg.storage.returns.fill_(1e30)
    with pytest.raises(DiagnosticStop, match="Nonfinite tensor: loss") as exc:
        alg.update()
    alg.recorder.dump(exc.value)
    assert alg.recorder.snapshot["permutation"] is not None
    alg.recorder.close()
    alg, obs = build(tmp_path / "normalizer")
    with torch.inference_mode():
        alg.act(obs)
        obs["critic"][:4].fill_(1e30)
        with pytest.raises(DiagnosticStop, match="buffers_after_normalization") as exc:
            alg.process_env_step(obs, torch.ones(8), torch.zeros(8), {})
    alg.recorder.dump(exc.value)
    payload = torch.load(tmp_path / "normalizer/capture.pt", weights_only=False)
    assert payload["update_ready"] is False
    assert "offending_observations" in payload
    with pytest.raises(ValueError, match="no replayable"):
        replay_capture(tmp_path / "normalizer/capture.pt", tmp_path / "no_replay", device="cpu")
    alg.recorder.close()


def test_raw_nonfinite_stops_before_actor_and_deadline_is_bounded(tmp_path):
    alg, obs = build(tmp_path)
    obs["policy"][0, 0] = float("nan")
    with pytest.raises(DiagnosticStop, match="raw_observations"):
        alg.act(obs)
    assert alg.transition.actions is None
    alg.recorder.started -= 5401
    with pytest.raises(DiagnosticStop, match="Wall-clock"):
        alg.act(observation())
    alg.recorder.close()


@pytest.mark.parametrize("align_completed", [False, True])
def test_load_restores_adam_adaptive_rate_and_curriculum_before_reset(tmp_path, align_completed):
    alg, obs = build(tmp_path / "source")
    collect(alg, obs)
    alg.update()  # create nonempty Adam state
    for group in alg.optimizer.param_groups:
        group["lr"] = 7.59375e-5
    checkpoint = {**cpu_copy(alg.save()), "iter": 6000,
                  "infos": {"env_state": {"common_step_counter": 144024}}}
    path = tmp_path / "model_6000.pt"
    torch.save(checkpoint, path)
    assert check_checkpoint(path)["common_step_counter"] == 144024
    alg.recorder.close()
    alg, _ = build(tmp_path / "resumed")  # a fresh process has fresh normalizer buffers
    runner_type = make_capture_runner(align_completed_updates=align_completed)
    runner = runner_type.__new__(runner_type)
    runner.alg = alg
    runner.alg.learning_rate = 1e-3
    env = SimpleNamespace(common_step_counter=0, policy_update_count=0,
                          cfg=SimpleNamespace(events={"policy_update_counter": SimpleNamespace(
                              params={"num_steps_per_env": 24})}))
    reset_counts = []
    runner.env = SimpleNamespace(unwrapped=env, reset=lambda: reset_counts.append(env.policy_update_count))
    runner.recorder = alg.recorder
    runner.load(str(path))
    assert reset_counts == [6001]
    assert runner.current_learning_iteration == (6001 if align_completed else 6000)
    assert runner.alg.learning_rate == 7.59375e-5
    equal_trees(checkpoint["optimizer_state_dict"], alg.optimizer.state_dict())
    alg.recorder.close()


def test_nonfinite_action_is_rejected_before_the_environment_can_step(tmp_path):
    alg, obs = build(tmp_path)
    handle = alg.actor.register_forward_hook(lambda module, inputs, output: output * float("inf"))
    with pytest.raises(DiagnosticStop, match="collection_output"):
        alg.act(obs)
    assert alg.storage.step == 0
    handle.remove()
    alg.recorder.close()


def test_post_adam_corruption_is_rejected_before_next_rollout(tmp_path):
    alg, obs = build(tmp_path)
    collect(alg, obs)

    def corrupt(optimizer, args, kwargs):
        with torch.no_grad():
            next(alg.critic.parameters()).fill_(float("nan"))

    handle = alg.optimizer.register_step_post_hook(corrupt)
    with pytest.raises(DiagnosticStop, match="post_optimizer") as exc:
        alg.update()
    alg.recorder.dump(exc.value)
    saved = torch.load(tmp_path / "capture.pt", weights_only=False)
    assert all(torch.isfinite(t).all() for _, t in tensor_leaves(saved["model"]))
    assert any(not torch.isfinite(t).all() for _, t in tensor_leaves(saved["observed_state"]))
    handle.remove()
    alg.recorder.close()


def test_disabled_vof_logging_does_not_mutate_mechanics_or_shared_extras():
    original = {"log": {"VirtualObjectForce/alpha": 0.5, "reward": 1}, "time_outs": torch.zeros(2)}
    result = disabled_vof_log(original)
    assert result["log"] == {"VirtualObjectForce/enabled": 0.0, "reward": 1}
    assert original["log"]["VirtualObjectForce/alpha"] == 0.5
    assert result["time_outs"] is original["time_outs"]


def test_experiment_preserves_task_and_ppo_settings():
    spec = importlib.util.spec_from_file_location("capture_cli", ROOT / "scripts/diagnostics/suitcase_numerics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.experiment_config(module.DEFAULT_CHECKPOINT)
    assert (cfg.env.scene.num_envs, cfg.env.scene.env_spacing, cfg.env.scene.terrain.env_spacing) == (4096, 2.5, 2.5)
    assert cfg.env.events["virtual_object_force"].params["natural_frequency"] == 0
    assert cfg.env.commands["motion"].init_phase_anneal_iterations == 20000
    assert cfg.env.commands["motion"].start_from_zero is False
    assert cfg.agent.max_iterations == 1000 and cfg.agent.num_steps_per_env == 24
    assert cfg.agent.wandb_project == "vibe"
    actual = asdict(cfg.agent.algorithm)
    baseline = asdict(load_rl_cfg(module.TASK).algorithm)
    actual.pop("class_name")
    baseline.pop("class_name")
    assert actual == baseline
    assert cfg.agent.actor == load_rl_cfg(module.TASK).actor
    assert cfg.agent.critic == load_rl_cfg(module.TASK).critic
    with pytest.raises(ValueError, match="bounded"):
        module.experiment_config(module.DEFAULT_CHECKPOINT, 21000)


def test_replay_real_sonic_visual_actor_with_cached_tokens(tmp_path):
    """Real frozen base + attention + LoRA; synthetic inputs, no renderer/GPU."""
    torch.manual_seed(42)
    n = 4
    obs = TensorDict({
        "policy": torch.randn(n, 930) * 0.01, "tokenizer": torch.randn(n, 640) * 0.01,
        "critic": torch.randn(n, 534) * 0.01, "augmentation": torch.randn(n, 18) * 0.01,
        "kv_tokens": TensorDict({"img_tokens": torch.randn(n, 21, 192)}, batch_size=[n]),
        "q_task_cmd": torch.randn(n, 9), "q_proprio": torch.randn(n, 64), "q_cls": torch.randn(n, 192),
    }, batch_size=[n])
    cfg = asdict(load_rl_cfg("Vibe-Uolm-Suitcase-ImgFeat-Ext"))
    cfg.update(num_steps_per_env=2, multi_gpu=None)
    cfg["algorithm"].update(class_name="vibe.core.numerics:DiagnosticPPO",
                            num_learning_epochs=1, num_mini_batches=2)
    # Use mjlab's normal preparation (None-valued model options are stripped).
    for key in ("actor", "critic"):
        cfg[key] = {k: v for k, v in cfg[key].items() if v is not None and not k.startswith("rnn_")}
    alg = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=n, num_actions=29), copy.deepcopy(cfg), "cpu")
    alg.train_mode()
    alg.recorder = FlightRecorder(alg, cfg, obs, tmp_path / "real", CaptureOptions(value_limit=1e10))
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.full((n,), .01), torch.zeros(n), {})
        alg.compute_returns(obs)
    assert "tokenizer" not in alg.storage.observations
    alg.update()
    expected = cpu_copy(alg.save())
    expected_rng = rng_state("cpu")
    alg.recorder.dump("real policy synthetic smoke complete")
    alg.recorder.close()
    result = replay_capture(tmp_path / "real/capture.pt", tmp_path / "real_replay", device="cpu")
    assert result["outcome"] == "completed"
    equal_trees(expected, result["state"])
    assert torch.equal(expected_rng["torch"], result["rng"]["torch"])
    # All frozen base tensors were carried in the replay, as well as trainable weights.
    assert any("encoder" in key for key in expected["actor_state_dict"])


def test_sync_lr_and_nested_snapshot_are_independent(tmp_path):
    alg, _ = build(tmp_path)
    alg.optimizer.param_groups[0]["lr"] = 7.59375e-5
    sync_adaptive_lr(alg)
    assert alg.learning_rate == 7.59375e-5
    nested = TensorDict({"tokens": TensorDict({"x": torch.ones(2, 3, 4)}, batch_size=[2])}, batch_size=[2])
    saved = cpu_copy(nested)
    nested["tokens"]["x"].zero_()
    assert saved["tokens"]["x"].eq(1).all()
    assert sum(t.numel() for _, t in tensor_leaves(saved)) == 24
    rng = rng_state("cpu")
    expected = torch.randperm(100)
    restore_rng(rng, "cpu")
    assert torch.equal(expected, torch.randperm(100))
    alg.recorder.close()


def test_runner_lifecycle_saves_capture_and_handles_parent_nan_guard(tmp_path):
    class CpuEnvironment:
        num_envs, num_actions, device = 8, 2, "cpu"

        def __init__(self, corrupt=False):
            self.unwrapped = self
            self.cfg = SimpleNamespace(events={"virtual_object_force": SimpleNamespace(
                params={"natural_frequency": 0})})
            self.obs_buf = observation()
            self.common_step_counter = 0
            self.closed = False
            self.corrupt = corrupt

        def get_observations(self):
            return self.obs_buf

        def step(self, actions):
            self.common_step_counter += 1
            self.obs_buf = observation()
            if self.corrupt:
                self.obs_buf["critic"][0, 0] = float("nan")
            return self.obs_buf, torch.ones(8) * .01, torch.zeros(8), {"log": {"VirtualObjectForce/alpha": .5}}

        def close(self):
            self.closed = True

    def runner(env, folder):
        cfg = recipe()
        cfg.update(logger="tensorboard", save_interval=100, upload_model=False,
                   check_for_nan=True, seed=42)
        return make_capture_runner()(env, cfg, str(folder), "cpu")

    healthy = runner(CpuEnvironment(), tmp_path / "healthy")
    healthy.learn(1, init_at_random_ep_len=False)
    assert (tmp_path / "healthy/numerics/capture.pt").exists()
    assert (tmp_path / "healthy/model_0.pt").exists()
    assert healthy.recorder.snapshot["permutation"] is not None
    unhealthy = runner(CpuEnvironment(corrupt=True), tmp_path / "unhealthy")
    with pytest.raises(ValueError, match="NaN"):
        unhealthy.learn(1, init_at_random_ep_len=False)
    assert unhealthy.env.closed
    snapshot = torch.load(tmp_path / "unhealthy/numerics/capture.pt", weights_only=False)
    assert snapshot["update_ready"] is False
    assert torch.isnan(snapshot["offending_observations"]["critic"][0, 0])
    unhealthy.logger.stop_logging_writer()


def capture_cli():
    spec = importlib.util.spec_from_file_location("capture_cli", ROOT / "scripts/diagnostics/suitcase_numerics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_window(tmp_path):
    alg, obs = build(tmp_path / "source")
    collect(alg, obs)
    alg.update()
    alg.recorder.close()
    root = tmp_path / "finished"
    (root / "numerics").mkdir(parents=True)
    (root / "numerics/stop.json").write_text(json.dumps({
        "reason": "Update budget completed", "update_ready": True, "iteration": 7000,
    }))
    (root / "numerics/updates.jsonl").write_text("".join(
        json.dumps({"iteration": i, "losses": {"value": .5, "surrogate": -.02, "entropy": 20}}) + "\n"
        for i in range(6000, 7000)
    ))
    (root / "numerics/capture.pt").write_bytes(b"capture already verified by recorder tests")
    checkpoint = {**cpu_copy(alg.save()), "iter": 6999,
                  "infos": {"env_state": {"common_step_counter": 168024}}}
    torch.save(checkpoint, root / "model_6999.pt")
    return root


def test_clean_gate_and_actual_update_counts_preserve_the_20k_plus_1k_schedule(tmp_path):
    root = completed_window(tmp_path)
    report = completed_capture(root, 6000, 1000, 168024)
    cli = capture_cli()
    cfg = cli.extension_config(report["checkpoint"], report, 20000, "stage1")
    assert cfg.agent.max_iterations == 12999  # count rollouts, not repeated resume labels
    assert cfg.env.commands["motion"].start_from_zero is False
    assert cfg.agent.save_interval == 1500
    assert cfg.agent.actor == load_rl_cfg(cli.TASK).actor
    tail = cli.extension_config(root / "model_19999.pt", {
        "iteration": 19999, "common_step_counter": 480000,
    }, 21000, "phase_zero")
    assert tail.agent.max_iterations == 1000
    assert tail.env.commands["motion"].start_from_zero is True
    assert tail.env.events["virtual_object_force"].params["natural_frequency"] == 0
    assert tail.agent.actor == cfg.agent.actor and tail.agent.critic == cfg.agent.critic
    assert tail.agent.algorithm == cfg.agent.algorithm
    with pytest.raises(ValueError, match="exactly 20000"):
        cli.extension_config(report["checkpoint"], report, 21000, "too_early")


@pytest.mark.parametrize("failure", ["trigger", "partial", "nonfinite_loss", "bad_checkpoint", "wrong_counter"])
def test_continuation_gate_refuses_failed_or_incomplete_windows(tmp_path, failure):
    root = completed_window(tmp_path)
    if failure == "trigger":
        (root / "numerics/stop.json").write_text(json.dumps({
            "reason": "DiagnosticStop: finite value-loss spike", "update_ready": True, "iteration": 7000,
        }))
    elif failure == "partial":
        path = root / "numerics/updates.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    elif failure == "nonfinite_loss":
        path = root / "numerics/updates.jsonl"
        path.write_text(path.read_text().replace('"value": 0.5', '"value": "nan"', 1))
    else:
        path = root / "model_6999.pt"
        checkpoint = torch.load(path, weights_only=False)
        if failure == "bad_checkpoint":
            next(iter(checkpoint["critic_state_dict"].values())).fill_(float("nan"))
        else:
            checkpoint["infos"]["env_state"]["common_step_counter"] -= 24
        torch.save(checkpoint, path)
    with pytest.raises((ValueError, DiagnosticStop)):
        completed_capture(root, 6000, 1000, 168024)


def test_queue_starts_both_stages_only_after_success_and_never_retries(tmp_path, monkeypatch):
    cli = capture_cli()
    source = completed_window(tmp_path)
    exitcode = tmp_path / "exitcode.txt"
    exitcode.write_text("0\n")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "require_idle_gpu", lambda: None)
    calls = []
    original_gate = cli.completed_capture

    def gate(path, first, count, counter):
        if Path(path) == source:
            return original_gate(path, first, count, counter)
        calls.append((first, count, counter))
        label = first + count - 1
        return {"checkpoint": str(Path(path) / f"model_{label}.pt"), "iteration": label,
                "common_step_counter": counter}

    def launch(command, **kwargs):
        run_name = command[command.index("--run-name") + 1]
        (tmp_path / "logs/rsl_rl/vibe_uolm_suitcase" / f"timestamp_{run_name}").mkdir(parents=True)

    monkeypatch.setattr(cli, "completed_capture", gate)
    monkeypatch.setattr(cli.subprocess, "run", launch)
    cli.queued_continuation(source, 999999999, exitcode, tmp_path / "queue")
    assert calls == [(7001, 12999, 480000), (19999, 1000, 504000)]
    assert json.loads((tmp_path / "queue/queue.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        cli.queued_continuation(source, 999999999, exitcode, tmp_path / "queue")
    exitcode.write_text("2\n")
    with pytest.raises(RuntimeError, match="nonzero"):
        cli.queued_continuation(source, 999999999, exitcode, tmp_path / "failed_queue")
    assert len(calls) == 2


def test_continuation_does_not_start_over_another_gpu_job(monkeypatch):
    cli = capture_cli()
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw:
                        f"{cli.os.getpid()}, /venv/bin/python\n123, /usr/bin/ptyxis\n")
    cli.require_idle_gpu()  # its own optional CUDA context and desktop apps are harmless
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *a, **kw:
                        "123, /usr/bin/ptyxis\n456, /another/project/.venv/bin/python\n")
    with pytest.raises(RuntimeError, match="another process"):
        cli.require_idle_gpu()
