"""The standard CLI can select the released critic without changing task mechanics."""

from dataclasses import asdict, replace

import mjlab  # noqa: F401
import pytest
import tyro
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import load_env_cfg

from vibe.tasks.repose.release_training import CRITIC_TERMS, TASK


def test_release_critic_is_explicit_and_does_not_mutate_standard_task():
    normal = load_env_cfg(TASK)
    cfg = replace(normal, release_critic=True)
    assert tuple(cfg.observations["critic"].terms) == CRITIC_TERMS
    assert {"object_id", "robot_root_pos_env"} <= normal.observations["critic"].terms.keys()
    assert cfg.rewards == normal.rewards
    assert cfg.terminations == normal.terminations
    assert cfg.observations["policy"] == normal.observations["policy"]
    assert cfg.observations["kv_tokens"] == normal.observations["kv_tokens"]
    assert tuple(replace(cfg).observations["critic"].terms) == CRITIC_TERMS
    assert not load_env_cfg(TASK).release_critic


def test_release_critic_rejects_an_unreviewed_layout():
    cfg = load_env_cfg(TASK)
    cfg.observations["critic"].terms.pop("joint_pos")
    with pytest.raises(ValueError, match="critic changed"):
        replace(cfg, release_critic=True)


def test_standard_cli_selects_precision_and_release_critic():
    common = [
        "--env.release-critic", "True", "--env.render-backend", "warp",
        "--env.scene.num-envs", "4096", "--agent.seed", "42",
        "--env.commands.motion.init-phase-anneal-iterations", "50000",
        "--agent.max-iterations", "5000", "--agent.upload-model", "False",
    ]
    def parse(extra):
        return tyro.cli(TrainConfig, default=TrainConfig.from_task(TASK),
                        args=common + extra, config=mjlab.TYRO_FLAGS)

    bf16 = parse(["--agent.amp-dtype", "bfloat16",
                  "--agent.bf16-reduced-precision-reduction", "False"])
    off = parse(["--agent.amp-dtype", "None"])
    assert tuple(bf16.env.observations["critic"].terms) == CRITIC_TERMS
    assert tuple(off.env.observations["critic"].terms) == CRITIC_TERMS
    assert bf16.agent.bf16_reduced_precision_reduction is False
    assert off.agent.bf16_reduced_precision_reduction is None
    assert off.agent.amp_dtype is None
    assert bf16.env.scene.num_envs == off.env.scene.num_envs == 4096
    assert bf16.env.commands["motion"].init_phase_anneal_iterations == 50000
    assert bf16.agent.num_steps_per_env == 24
    assert bf16.agent.resume is False and bf16.agent.upload_model is False
    assert bf16.env.sim.deterministic is False
    assert bf16.env.sim.mujoco.jacobian == "auto"
    a, b = asdict(bf16.agent), asdict(off.agent)
    assert {k for k in a if a[k] != b[k]} == {
        "amp_dtype", "bf16_reduced_precision_reduction",
    }
