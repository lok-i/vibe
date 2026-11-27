"""Configuration, comparison and overflow contracts; rollout checks are separate."""

from dataclasses import asdict
from types import SimpleNamespace

import mjlab
import mujoco_warp as mjwarp
import pytest
import torch
import tyro
from mjlab.scripts.play import PlayConfig
from mjlab.sim import SimulationCfg

from vibe.core.reproducibility import BufferMonitor, digest, first_difference


def test_determinism_cli_and_serialized_config():
    cfg = tyro.cli(PlayConfig, args=["--env.sim.deterministic", "True", "--env.seed", "42"],
                   config=mjlab.TYRO_FLAGS)
    assert cfg.env.sim.deterministic is True
    assert cfg.env.seed == 42
    assert SimulationCfg().deterministic is False
    sim = SimulationCfg(deterministic=True)
    assert asdict(sim)["deterministic"] is True
    opt = object.__new__(mjwarp.Option)
    sim.apply_wp_opt(opt)
    assert opt.deterministic == mjwarp.DeterminismType.ALL
    assert opt.warn_overflow == mjwarp.OverflowType.ALL
    SimulationCfg().apply_wp_opt(opt)
    assert opt.deterministic == mjwarp.DeterminismType.NONE


def test_bit_comparison_detects_perturbation_and_signed_zero():
    x = torch.tensor([0.0, 1.0])
    a = {"obs": digest(x), "step": 0}
    assert first_difference([a], [dict(a)]) is None
    x[1] = torch.nextafter(x[1], torch.tensor(float("inf")))
    difference = first_difference([a], [{**a, "obs": digest(x)}])
    assert difference["record"] == 0 and difference["field"] == "obs"
    assert digest(torch.tensor([0.0])) != digest(torch.tensor([-0.0]))
    assert digest(torch.ones(2, dtype=torch.bfloat16))
    assert first_difference([a], []) == {"record": 0, "field": "record_count"}


def test_buffer_monitor_checks_aggregate_contacts_and_each_world():
    sim = SimpleNamespace(
        data=SimpleNamespace(nacon=torch.tensor([10]), nefc=torch.tensor([3, 8]),
                             overflow=torch.zeros(2, dtype=torch.int32)),
        wp_data=SimpleNamespace(naconmax=20, njmax=16, naccdmax=12),
        wp_model=SimpleNamespace(is_sparse=False), cfg=SimpleNamespace(nconmax=10),
    )
    monitor = BufferMonitor(sim)
    monitor.sample()
    assert monitor.report()["contact_headroom"] == 2
    assert monitor.report()["constraint_headroom"] == 2
    sim.data.overflow[1] = int(mjwarp.OverflowType.LS_ITERATIONS)
    monitor.sample()
    assert monitor.report()["solver_limit_samples"] == 1
    sim.data.overflow[1] = 1
    with pytest.raises(RuntimeError, match="overflow"):
        monitor.sample()


def test_preserved_velocity_rewards_use_world_frame():
    from mjlab.tasks.tracking.mdp import rewards

    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=("pelvis",)),
        body_lin_vel_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
        robot_body_lin_vel_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
        body_ang_vel_w=torch.tensor([[[0.0, 1.0, 0.0]]]),
        robot_body_ang_vel_w=torch.tensor([[[0.0, 1.0, 0.0]]]),
        # No anchor-yaw/relative-velocity attributes: the old contract never reads them.
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    for fn in (rewards.motion_global_body_linear_velocity_error_exp,
               rewards.motion_global_body_angular_velocity_error_exp):
        torch.testing.assert_close(fn(env, "motion", 1.0), torch.ones(1))
