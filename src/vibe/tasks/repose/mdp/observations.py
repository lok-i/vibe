"""Repose-local observation terms — the cube's color/orientation obs.

`image_feature` moved to `vibe.core.mdp.observations` (2026-08-03) — it reads a
camera, it never knew about a cube. Re-exported rather than deleted: a saved
env cfg from an earlier run stores the term by its DEFINING module path, so
dropping the name here would break reloading one.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import subtract_frame_transforms
from orcs.tasks.uolm.mdp.observations import _quat_to_mat6d

from vibe.core.mdp.observations import image_feature

__all__ = [
    "base_ori_mat6d",
    "image_feature",
    "goal_ori_mat6d",
    "object_goal_color",
    "object_ori_error_mat6d",
    "object_upface_color",
]


def object_goal_color(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Goal up-face COLOR as one-hot(6) -> (B, 6).

    The color-goal command encoding (extero="imgfeat"): the actor's only task
    signal — the goal QUAT never reaches it (grounding the command in vision is
    the point). Phase 2: swap the identity table for a (6, D) offline
    text-embedding matrix — same term, richer code.
    """
    cmd = env.command_manager.get_term(command_name)
    return torch.eye(6, device=env.device)[cmd.goal_color_idx]


def object_upface_color(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """CURRENT up-face COLOR as one-hot(6) -> (B, 6).

    Aux prediction target (NOT a policy obs): the perm is image-only
    information, so decoding this from z supervises color extraction.
    """
    cmd = env.command_manager.get_term(command_name)
    return torch.eye(6, device=env.device)[cmd.current_color_idx]


def base_ori_mat6d(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Robot base orientation as mat6d -> (B, 6)."""
    return _quat_to_mat6d(env.scene["robot"].data.root_link_quat_w)


def goal_ori_mat6d(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """ReorientationCommand target orientation as mat6d -> (B, 6)."""
    target_quat = env.command_manager.get_command(command_name)
    return _quat_to_mat6d(target_quat)


def object_ori_error_mat6d(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Relative rotation from object to ReorientationCommand goal as mat6d -> (B, 6)."""
    obj = env.scene[object_cfg.name]
    target_quat = env.command_manager.get_command(command_name)
    _, err_quat = subtract_frame_transforms(
        obj.data.root_link_pos_w, obj.data.root_link_quat_w,
        obj.data.root_link_pos_w, target_quat,
    )
    return _quat_to_mat6d(err_quat)
