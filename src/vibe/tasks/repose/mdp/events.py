"""Repose's TASK-CHANNEL event: the per-env cube face->colour remap.

This is not a domain. `rand_face_colors` is what DEFINES the goal (the
commanded colour is read off the perm it writes), which is why it survives the
play build while every `rand_*` domain knob is stripped.

The camera / light / terrain-colour terms live in `vibe.core.mdp.events` and are
re-exported here: a saved env cfg records a term by its DEFINING module path.

Fields are declared via @requires_model_fields so the EventManager expands them
per-world BEFORE graph capture. The groundplane checker TEXTURE beats mat_rgba,
so the colour-task cfg flips the head cam's `use_textures=False` (env_cfgs).
"""

from __future__ import annotations

import functools

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat

from vibe.core.mdp.events import (  # noqa: F401 — re-export, see docstring
    GROUND_RGBAS as _GROUND_RGBAS,
)
from vibe.core.mdp.events import (  # noqa: F401 — re-export, see docstring
    rand_cam_extrinsics,
    rand_terrain_color,
)
from vibe.tasks.repose.mdp.cube_faces import face_colors, face_normals

__all__ = ["rand_face_colors", "rand_terrain_color", "rand_cam_extrinsics"]


@functools.lru_cache(maxsize=None)
def _face_perms(device: torch.device) -> torch.Tensor:
    """Face permutations induced by the 24 octahedral rotations -> (24, 6) long.

    perm[k, f] = face whose normal the k-th rotation sends face f's normal to.
    Rotations only (det +1): opposite pairs and neighbor cyclic order are
    preserved — the color arrangement is fixed up to a physical cube rotation.
    """
    from vibe.tasks.repose.mdp.commands import _cube_symmetry_quats

    n = face_normals(device)                              # (6, 3)
    rot = matrix_from_quat(_cube_symmetry_quats(device))  # (24, 3, 3)
    rotated = torch.einsum("kij,fj->kfi", rot, n)         # (24, 6, 3)
    return (rotated @ n.T).argmax(dim=-1)


@requires_model_fields("geom_rgba")
def rand_face_colors(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
    """Per-env face->color remap: one of the 24 octahedral colorings.

    Breaks the orientation<->color correspondence (same reference motion ends
    on a different color per env) while preserving relative color adjacency.
    Stashes ``env._face_color_perm`` (B, 6) long — perm[e, f] = color idx shown
    on geometric face f — read by ReposeMotionCommand.goal_color_idx.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    obj = env.scene["object"]
    names = list(obj.geom_names)
    gids = obj.indexing.geom_ids[
        [names.index(f"cube_face_{i}") for i in range(6)]]  # (6,)

    if not hasattr(env, "_face_color_perm"):
        env._face_color_perm = (
            torch.arange(6, device=env.device).expand(env.num_envs, 6).clone())

    perms = _face_perms(env.device)
    perm = perms[torch.randint(
        perms.shape[0], (len(env_ids),), device=env.device)]  # (n, 6)
    env._face_color_perm[env_ids] = perm
    env.sim.model.geom_rgba[env_ids[:, None], gids[None, :]] = (
        face_colors(env.device)[perm])
