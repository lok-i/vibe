"""Cube face kernel — up-face detection, tilt error, face colors.

Face order/axes/colors derive from :mod:`vibe.assets.repose`. The task
truth for repose is "queried face on top": a 2-DOF constraint on SO(3);
`up_face_idx` names the face a goal quat implies, `face_tilt_error` scores
only that constraint (yaw about world-z is task-irrelevant).
"""

from __future__ import annotations

import functools

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from vibe.assets.repose import FACE_COLOR_NAMES, FACE_COLORS, FACE_NORMALS

__all__ = [
    "face_normals",
    "face_colors",
    "face_color_names",
    "up_face_idx",
    "face_tilt_error",
    "all_face_tilts",
    "color_tilt_error",
]

@functools.lru_cache(maxsize=None)
def _face_table() -> tuple[tuple, tuple]:
    """(axes, rgbas) in canonical colored-face order."""
    return FACE_NORMALS, FACE_COLORS


@functools.lru_cache(maxsize=None)
def face_normals(device: torch.device) -> torch.Tensor:
    """Outward face normals in object frame -> (6, 3), canonical order."""
    axes, _ = _face_table()
    return torch.tensor(axes, device=device)


@functools.lru_cache(maxsize=None)
def face_colors(device: torch.device) -> torch.Tensor:
    """Face rgba -> (6, 4), canonical order."""
    _, rgbas = _face_table()
    return torch.tensor(rgbas, device=device)


@functools.lru_cache(maxsize=None)
def face_color_names() -> tuple[str, ...]:
    """Physical color name per face -> len-6 tuple."""
    return FACE_COLOR_NAMES


def up_face_idx(quat: torch.Tensor) -> torch.Tensor:
    """Face whose outward normal is most world-up under quat -> (B,) long.

    argmax_i (R n_i) . z == argmax_i (R^T z) . n_i, with R^T z = row 2 of R.
    """
    u = matrix_from_quat(quat)[:, 2, :]  # world-up in object frame (B, 3)
    n = face_normals(quat.device)
    return (u @ n.T).argmax(dim=-1)


def face_tilt_error(quat: torch.Tensor, face_idx: torch.Tensor) -> torch.Tensor:
    """Angle between face_idx's outward normal (rotated by quat) and world-up
    -> (B,) rad in [0, pi]."""
    u = matrix_from_quat(quat)[:, 2, :]
    n = face_normals(quat.device)[face_idx]
    cos = (u * n).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos)


def all_face_tilts(quat: torch.Tensor) -> torch.Tensor:
    """Tilt of every face's outward normal vs world-up -> (B, 6) rad."""
    u = matrix_from_quat(quat)[:, 2, :]
    cos = (u @ face_normals(quat.device).T).clamp(-1.0, 1.0)
    return torch.acos(cos)


def color_tilt_error(
    quat: torch.Tensor, color_idx: torch.Tensor, perm: torch.Tensor | None = None
) -> torch.Tensor:
    """Min tilt over the faces CARRYING color_idx -> (B,) rad.

    The color-channel task truth: reads only (color, perm, live quat) — never a
    goal quat. perm is the (B, 6) face->color remap (rand_face_colors), None =
    canonical identity (color idx == face idx). Min-over-matched survives
    non-bijective palettes; under today's bijection it equals
    face_tilt_error(quat, perm^-1[color])."""
    if perm is None:
        return face_tilt_error(quat, color_idx)
    tilts = all_face_tilts(quat)
    matched = torch.where(perm == color_idx.unsqueeze(1), tilts, torch.full_like(tilts, torch.pi))
    return matched.min(dim=1).values
