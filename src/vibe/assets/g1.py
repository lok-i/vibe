"""G1 render-cost variants for vibe's vision tasks.

The robot itself — flat-hand contact surgery, articulation, keyframe — is
**orcs's** (`orcs.assets.g1`), because contact geometry is a loco-manipulation
concern, not a vision one. This module composes on top of it and owns exactly
one thing: **which meshes the head camera has to render.** Physics is identical
across every variant here.

  G1_BASE_CFG      orcs's flat-hand G1, whole visual set          (play)
  G1_VIS_LEAN_CFG  same, out-of-frame visuals demoted             (train)

`get_g1_mesh_hand_cfg` is the higher-fidelity rubber-hand collision variant —
dormant, kept because it is a real alternative if plate contact ever becomes the
limiting approximation.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    G1_ARTICULATION,
    G1_XML,
    KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityCfg
from orcs.assets import find_body, flat_hand_spec
from orcs.assets import get_g1_flat_hand_cfg as G1_BASE_CFG

__all__ = [
    "G1_BASE_CFG",
    "G1_VIS_LEAN_CFG",
    "get_g1_flat_hand_lean_cfg",
    "get_g1_mesh_hand_cfg",
]

_SIDES = ("left", "right")
_VISUAL_HAND_POS = {"left": (0.0415, 0.003, 0.0), "right": (0.0415, -0.003, 0.0)}

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


def _replace_hand_with_mesh(spec: mujoco.MjSpec) -> None:
    """Replace capsule hand colliders with rubber_hand mesh colliders."""
    for side in _SIDES:
        body = find_body(spec.worldbody, f"{side}_wrist_yaw_link")
        for g in body.geoms:
            if g.name == f"{side}_hand_collision":
                g.fromto[:] = np.nan  # unset fromto (required before mesh)
                g.type = mujoco.mjtGeom.mjGEOM_MESH
                g.meshname = f"{side}_rubber_hand"
                g.pos[:] = _VISUAL_HAND_POS[side]
                g.size[:] = 0
                break


def _mesh_hand_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    _replace_hand_with_mesh(spec)
    return spec


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


def get_g1_mesh_hand_cfg() -> EntityCfg:
    """G1 with rubber_hand mesh collision (highest contact fidelity; dormant)."""
    return _cfg(_mesh_hand_spec)


def get_g1_flat_hand_lean_cfg() -> EntityCfg:
    """flat_hand + out-of-frame visuals hidden from the head camera (1.65x render)."""
    return _cfg(_flat_hand_lean_spec)


G1_VIS_LEAN_CFG = get_g1_flat_hand_lean_cfg


# ---------------------------------------------------------------------------
# Interactive test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    import mujoco.viewer as viewer

    VARIANTS = {"mesh": _mesh_hand_spec, "flat": flat_hand_spec,
                "lean": _flat_hand_lean_spec, "stock": None}
    choice = sys.argv[1] if len(sys.argv) > 1 else "mesh"
    if choice not in VARIANTS:
        print(f"Usage: python -m vibe.assets.g1 [{'/'.join(VARIANTS)}]")
        sys.exit(1)

    if VARIANTS[choice] is not None:
        spec = VARIANTS[choice]()
    else:
        spec = mujoco.MjSpec.from_file(str(G1_XML))

    # Add a cube in front of the robot
    cube_body = spec.worldbody.add_body()
    cube_body.name = "cube"
    cube_body.pos[:] = [0.4, 0.0, 0.9]
    fj = cube_body.add_freejoint()
    fj.name = "cube_joint"
    cg = cube_body.add_geom()
    cg.type = mujoco.mjtGeom.mjGEOM_BOX
    cg.size[:] = [0.03, 0.03, 0.03]
    cg.mass = 0.1
    cg.rgba[:] = [0.2, 0.6, 1.0, 1.0]
    cg.condim = 4
    cg.friction[:] = [1.0, 0.005, 0.001]

    # Ground plane
    ground = spec.worldbody.add_geom()
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size[:] = [10, 10, 0.1]
    ground.rgba[:] = [0.8, 0.8, 0.8, 1.0]
    ground.conaffinity = 1
    ground.condim = 3

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Variant: {choice}")
    print("Native viewer controls:")
    print("  Ctrl+RightClick: apply force")
    print("  0-4: toggle geom groups (0=visual, 3=collision)")
    print("  ESC: quit")
    viewer.launch(model, data)
