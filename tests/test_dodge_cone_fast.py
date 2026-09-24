"""Contracts for the single-frame ConeFast task migration."""

from __future__ import annotations

from dataclasses import asdict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import vibe  # noqa: F401 - register tasks
from vibe.export.agent.case import get as get_export_case
from vibe.tasks.dodge.config.g1.env_cfgs import DODGE_CONE_FAST_THROW
from vibe.tasks.dodge.terrain import ROOM_LIGHT_NAMES, ROOM_SIZE

BASE = "Vibe-Dodge-ImgFeat-Ext"
CONE_FAST = "Vibe-Dodge-ConeFast-ImgFeat-Ext"


def _head_cam(cfg):
    return next(sensor for sensor in cfg.scene.sensors if sensor.name == "head_cam")


def test_cone_fast_matches_the_trained_environment_contract():
    cfg = load_env_cfg(CONE_FAST)

    assert cfg.events["throw_ball"].params == {
        "ball_name": "ball",
        **DODGE_CONE_FAST_THROW,
    }
    assert (_head_cam(cfg).width, _head_cam(cfg).height) == (112, 64)
    assert cfg.observations["kv_tokens"].terms["img_tokens"].params[
        "delay_steps"
    ] == 0
    assert cfg.scene.terrain.terrain_generator.size == ROOM_SIZE
    assert tuple(light.name for light in cfg.scene.terrain.lights[1:]) == (
        ROOM_LIGHT_NAMES
    )
    assert "rand_room_light_dir" in cfg.events


def test_cone_fast_reuses_the_current_branch_agent_bit_for_bit():
    assert asdict(load_rl_cfg(CONE_FAST)) == asdict(load_rl_cfg(BASE))


def test_legacy_dodge_task_keeps_its_existing_plane_and_camera():
    cfg = load_env_cfg(BASE)

    assert (_head_cam(cfg).width, _head_cam(cfg).height) == (112, 63)
    assert cfg.scene.terrain.terrain_type == "plane"
    assert "rand_room_light_dir" not in cfg.events


def test_cone_fast_declares_an_export_case():
    assert get_export_case(CONE_FAST) is not None
