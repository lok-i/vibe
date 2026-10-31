"""ViBe play removes robot reset noise without changing the training domain."""

from copy import deepcopy
from functools import partial

import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg

import vibe  # noqa: F401 - register tasks before reading the registry
from vibe.tasks.dodge.config.g1.env_cfgs import (
    g1_dodge_cone_fast_env_cfg,
    g1_dodge_env_cfg,
)
from vibe.tasks.perloco.config.g1.env_cfgs import (
    g1_perloco_grail_env_cfg,
    g1_perloco_omre_env_cfg,
)
from vibe.tasks.repose.config.g1.env_cfgs import g1_repose_cube_env_cfg
from vibe.tasks.uolm.config.g1.env_cfgs import g1_uolm_env_cfg

VIBE_TASKS = [task for task in list_tasks() if task.startswith("Vibe-")]
TRAIN_VELOCITY_RANGE = {
    "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.15, 0.15),
    "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.5, 0.5),
}


def _reset_ranges(cfg):
    motion = cfg.commands["motion"]
    return motion.pose_range, motion.velocity_range, motion.joint_position_range


@pytest.mark.parametrize("task", VIBE_TASKS)
def test_play_clears_robot_reset_noise_and_train_keeps_its_domain(task):
    train = load_env_cfg(task)
    play = load_env_cfg(task, play=True)

    assert _reset_ranges(train) == ({}, TRAIN_VELOCITY_RANGE, (-0.05, 0.05))
    assert _reset_ranges(play) == ({}, {}, (0.0, 0.0))
    assert "perturb_robot" in train.events
    assert "perturb_robot" not in play.events


@pytest.mark.parametrize("factory", [
    pytest.param(g1_dodge_env_cfg, id="dodge"),
    pytest.param(g1_dodge_cone_fast_env_cfg, id="dodge-cone-fast"),
    pytest.param(g1_perloco_grail_env_cfg, id="perloco-grail"),
    pytest.param(g1_perloco_omre_env_cfg, id="perloco-omre"),
    pytest.param(g1_uolm_env_cfg, id="uolm"),
    pytest.param(partial(g1_repose_cube_env_cfg, extero="imgfeat", aux=True),
                 id="repose-big-cube"),
    pytest.param(partial(g1_repose_cube_env_cfg, scene="small_cube_table",
                         extero="imgfeat", aux=True), id="repose-small-cube"),
])
def test_play_factory_does_not_mutate_training_reset_ranges(factory):
    train = factory(play=False)
    expected = deepcopy(_reset_ranges(train))
    play = factory(play=True)

    assert _reset_ranges(train) == expected
    assert _reset_ranges(factory(play=False)) == expected
    assert play.commands["motion"].velocity_range is not train.commands["motion"].velocity_range
