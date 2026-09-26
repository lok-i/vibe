"""Repose-local observation terms — the cube's color/orientation obs.

`image_feature` lives in `vibe.core.mdp.observations`; it is re-exported here
because a saved env cfg stores a term by its DEFINING module path.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from vibe.core.mdp.observations import image_feature

__all__ = ["image_feature", "object_goal_color", "object_upface_color"]


def object_goal_color(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Goal up-face COLOR as one-hot(6) -> (B, 6).

    The color-goal command encoding (extero="imgfeat"): the actor's only task
    signal — the goal QUAT never reaches it (grounding the command in vision is
    the point).
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
