"""Repose-local observation terms — the cube's color/orientation obs.

`image_feature` moved to `vibe.core.mdp.observations` (2026-08-03) — it reads a
camera, it never knew about a cube. Re-exported rather than deleted: a saved
env cfg from an earlier run stores the term by its DEFINING module path, so
dropping the name here would break reloading one.
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
