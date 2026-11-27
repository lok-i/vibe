"""Keep reproduction overrides explicit and separate from standard task defaults."""

from dataclasses import asdict
from functools import partial

import mjlab  # noqa: F401
from mjlab.tasks.registry import load_env_cfg

from vibe.tasks.repose.release_training import CRITIC_TERMS, TASK, training_config


def comparable(value):
    # Copies of partials compare by identity; compare their actual call contract.
    if isinstance(value, partial):
        return value.func, comparable(value.args), comparable(value.keywords)
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(comparable(v) for v in value)
    return value


def test_release_recipe_restores_critic_without_mutating_task():
    cfg = training_config()
    assert tuple(cfg.env.observations["critic"].terms) == CRITIC_TERMS
    normal = load_env_cfg(TASK)
    assert {"object_id", "robot_root_pos_env"} <= normal.observations["critic"].terms.keys()
    assert cfg.env.rewards == normal.rewards
    assert cfg.env.terminations == normal.terminations
    assert cfg.env.observations["policy"] == normal.observations["policy"]
    assert cfg.env.observations["kv_tokens"] == normal.observations["kv_tokens"]
    assert cfg.env.scene.num_envs == 4096
    assert cfg.env.commands["motion"].init_phase_anneal_iterations == 50000
    assert cfg.agent.num_steps_per_env == 24
    assert cfg.agent.max_iterations == 60000
    assert cfg.agent.seed == cfg.env.seed == 42
    assert cfg.agent.amp_dtype == "bfloat16"
    assert cfg.agent.resume is False and cfg.agent.upload_model is False
    assert cfg.env.sim.deterministic is False
    assert cfg.env.sim.mujoco.jacobian == "auto"


def test_renderer_pair_has_identical_learning_configuration(monkeypatch):
    monkeypatch.delenv("VIBE_RENDER_BACKEND", raising=False)
    warp, optix = training_config("warp"), training_config("optix")
    assert asdict(warp.agent) == asdict(optix.agent)
    assert optix.env.render_backend == "optix"
    assert next(s for s in optix.env.scene.sensors if s.name == "head_cam").backend == "optix"
    optix.env.render_backend = "warp"
    next(s for s in optix.env.scene.sensors if s.name == "head_cam").backend = "warp"
    assert comparable(asdict(warp.env)) == comparable(asdict(optix.env))
