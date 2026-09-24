"""Cube-specific task rewards — up-face variants of the goal-ori pair.

Repose task truth = "queried face on top" (2-DOF); the quat kernel
(object_goal_ori_reward) over-constrains by task-irrelevant yaw about
world-z. Kernel math in cube_faces.py.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from vibe.tasks.repose.mdp.cube_faces import (
    color_tilt_error,
    face_tilt_error,
)

__all__ = [
    "up_face_reward",
    "up_face_success_bonus",
    "up_color_reward",
    "up_color_success_bonus",
]


def _goal_face_idx(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Goal up-face index, precomputed on ReposeMotionCommand."""
    return env.command_manager.get_term(command_name).goal_up_face_idx


def _tilt(env: ManagerBasedRlEnv, object_cfg: SceneEntityCfg,
          command_name: str) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    return face_tilt_error(
        obj.data.root_link_quat_w, _goal_face_idx(env, command_name))


def up_face_reward(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """Gaussian kernel on face-tilt error: exp(-tilt^2 / std^2). Returns (B,)."""
    tilt = _tilt(env, object_cfg, command_name)
    return torch.exp(-tilt.square() / (std * std))


def up_face_success_bonus(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Binary bonus when face-tilt error < threshold. Returns (B,)."""
    tilt = _tilt(env, object_cfg, command_name)
    return (tilt < threshold).float()


def _color_tilt(env: ManagerBasedRlEnv, object_cfg: SceneEntityCfg,
                command_name: str) -> torch.Tensor:
    """Goal-COLOR tilt: min tilt over faces carrying the commanded color."""
    cmd = env.command_manager.get_term(command_name)
    obj = env.scene[object_cfg.name]
    return color_tilt_error(
        obj.data.root_link_quat_w, cmd.goal_color_idx,
        getattr(env, "_face_color_perm", None))


def up_color_reward(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """Gaussian kernel on goal-COLOR tilt: exp(-tilt^2 / std^2). Returns (B,).

    Color-channel twin of up_face_reward, defined WITHOUT the goal quat
    (color cmd + perm + live quat only) — the command stays the single source
    of task truth once goals are sampled off the demo manifold."""
    tilt = _color_tilt(env, object_cfg, command_name)
    return torch.exp(-tilt.square() / (std * std))


def up_color_success_bonus(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.3,
    max_ang_vel: float | None = None,
) -> torch.Tensor:
    """Binary bonus when goal-color tilt < threshold. Returns (B,).

    max_ang_vel (rad/s) gates the bonus on a settled object — no
    flicker-success while the cube tumbles through the right pose."""
    ok = _color_tilt(env, object_cfg, command_name) < threshold
    if max_ang_vel is not None:
        obj = env.scene[object_cfg.name]
        ok &= obj.data.root_link_ang_vel_w.norm(dim=-1) < max_ang_vel
    return ok.float()
