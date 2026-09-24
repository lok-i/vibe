"""Primitive assets for Repose's colored-cube scenes."""

from __future__ import annotations

from functools import partial

import mujoco
from mjlab.entity import EntityCfg

FACE_COLORS = (
    (0.9, 0.2, 0.2, 1.0),  # +x  red
    (0.95, 0.45, 0.1, 1.0),  # -x  orange
    (0.2, 0.9, 0.2, 1.0),  # +y  green
    (0.9, 0.9, 0.2, 1.0),  # -y  yellow
    (0.2, 0.2, 0.9, 1.0),  # +z  blue
    (0.8, 0.2, 0.65, 1.0),  # -z  purplish pink
)
FACE_COLOR_NAMES = ("red", "orange", "green", "yellow", "blue", "pink")
FACE_NORMALS = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)

BIG_CUBE_HALF_EXTENT = 0.3048
BIG_CUBE_MASS = 1.5

# assets/custom_objects/box/box.obj is a 0.36 m cube.
SMALL_CUBE_HALF_EXTENT = 0.18
SMALL_CUBE_MASS = 0.3

TABLE_SIZE = (0.75, 0.75, 0.05)
TABLE_CENTER_HEIGHT = 1.0
TABLE_TOP_HEIGHT = TABLE_CENTER_HEIGHT + TABLE_SIZE[2] / 2.0

_FACE_SKIN = 0.002


def face_geoms(
    half_extent: float,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    """Colored shell poses and half-sizes in canonical face order."""
    h, s = float(half_extent), _FACE_SKIN
    return (
        ((h, 0.0, 0.0), (s, h, h)),
        ((-h, 0.0, 0.0), (s, h, h)),
        ((0.0, h, 0.0), (h, s, h)),
        ((0.0, -h, 0.0), (h, s, h)),
        ((0.0, 0.0, h), (h, h, s)),
        ((0.0, 0.0, -h), (h, h, s)),
    )


def colored_cube_spec(*, half_extent: float, mass: float) -> mujoco.MjSpec:
    """A fast primitive collider with six independently recolorable faces."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_core",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_extent,) * 3,
        mass=mass,
        rgba=(0.5, 0.5, 0.5, 0.0),
        group=3,
        contype=1,
        conaffinity=1,
    )
    for i, ((pos, size), rgba) in enumerate(
        zip(face_geoms(half_extent), FACE_COLORS, strict=True)
    ):
        body.add_geom(
            name=f"cube_face_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=rgba,
            contype=0,
            conaffinity=0,
            mass=0.0,
        )
    return spec


def colored_cube_entity_cfg(
    *, half_extent: float, mass: float, init_pos: tuple[float, float, float]
) -> EntityCfg:
    return EntityCfg(
        spec_fn=partial(colored_cube_spec, half_extent=half_extent, mass=mass),
        init_state=EntityCfg.InitialStateCfg(pos=init_pos),
    )


def table_spec(*, size: tuple[float, float, float] = TABLE_SIZE) -> mujoco.MjSpec:
    """A fixed primitive tabletop; mjlab gives fixed entities a mocap root."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table")
    body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=tuple(float(v) / 2.0 for v in size),
        contype=1,
        conaffinity=1,
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.55, 0.57, 0.62, 1.0),
    )
    return spec


def table_entity_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=table_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_CENTER_HEIGHT)),
    )
