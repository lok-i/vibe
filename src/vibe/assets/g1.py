"""G1 render-cost variants for vibe's vision tasks.

The robot itself — flat-hand contact surgery, articulation, keyframe — is
**orcs's** (`orcs.assets.g1`), because contact geometry is a loco-manipulation
concern, not a vision one. This module composes on top of it and owns exactly
one thing: **which meshes the head camera has to render.** Physics is identical
across every variant here.

  G1_BASE_CFG      orcs's flat-hand G1, whole visual set          (play)
  G1_VIS_LEAN_CFG  same, out-of-frame visuals demoted             (train)
"""

from __future__ import annotations

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    G1_ARTICULATION,
    KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityCfg
from orcs.assets import flat_hand_spec
from orcs.assets import get_g1_flat_hand_cfg as G1_BASE_CFG

__all__ = [
    "G1_BASE_CFG",
    "G1_VIS_LEAN_CFG",
    "get_g1_flat_hand_lean_cfg",
]

# Bodies whose visual mesh the head camera can actually see (torso-mounted,
# fovy 42.5°, looking down-forward at the object): the lower arms + hands it
# manipulates with, and the lower legs + feet that enter frame when it stoops.
_CAM_VISIBLE_LINKS = (
    "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
    "knee_link", "ankle_pitch_link", "ankle_roll_link",
)
_VISUAL_GROUP = 2  # stock G1 visual meshes
_HIDDEN_GROUP = 4  # not in the head_cam's enabled_geom_groups=(0, 2)


def _hide_far_visuals(spec: mujoco.MjSpec) -> None:
    """Demote out-of-frame visual meshes to a group the head camera skips.

    mujoco_warp builds the render BVH from `enabled_geom_groups` only, so a
    demoted geom costs exactly nothing to render — same as deleting it, but the
    viewers still show a whole robot (native viewer: key `4`). Drops
    pelvis/hips/waist/torso/shoulders — 19 of 35 visual geoms, 206k of 307k
    faces. Measured on `-ImgFeat-`: render 1.65x, and over a 300-step x 256-env
    random-action rollout the head-cam image differs in 0.05% of pixels
    (torso+waist alone is bit-identical; the residual is knees at frame edge).
    """
    for body in spec.bodies:
        if any(body.name.endswith(link) for link in _CAM_VISIBLE_LINKS):
            continue
        for g in body.geoms:
            if g.group == _VISUAL_GROUP:
                g.group = _HIDDEN_GROUP


def _flat_hand_lean_spec() -> mujoco.MjSpec:
    spec = flat_hand_spec()  # orcs owns the contact surgery
    _hide_far_visuals(spec)  # vibe owns the render cost
    return spec


def _cfg(spec_fn) -> EntityCfg:
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=spec_fn,
        articulation=G1_ARTICULATION,
    )


def get_g1_flat_hand_lean_cfg() -> EntityCfg:
    """flat_hand + out-of-frame visuals hidden from the head camera (1.65x render)."""
    return _cfg(_flat_hand_lean_spec)


G1_VIS_LEAN_CFG = get_g1_flat_hand_lean_cfg
