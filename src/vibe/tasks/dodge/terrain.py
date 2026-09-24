"""Dodge's indoor room terrain and its lighting domain randomization."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)
from mjlab.terrains.utils import make_plane
from mjlab.utils import spec_config as spec_cfg

__all__ = [
    "ROOM_LIGHT_NAMES",
    "ROOM_SIZE",
    "WALL_HEIGHT",
    "RoomTerrainCfg",
    "apply_room_light_domain",
    "room_terrain_cfg",
]

ROOM_SIZE = (8.0, 8.0)
WALL_HEIGHT = 3.0
_THICKNESS = 0.05


@dataclass(kw_only=True)
class RoomTerrainCfg(SubTerrainCfg):
    """Floor, four walls, and ceiling built as separate non-contact slabs."""

    height: float = WALL_HEIGHT
    thickness: float = _THICKNESS

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng
        body = spec.body("terrain")
        sx, sy = self.size
        h, t = self.height, self.thickness
        geoms = list(make_plane(body, self.size, 0.0, center_zero=False))

        # Oversize orthogonal axes so the slabs overlap and leave no background
        # pixels at their seams. Separate inward-facing slabs are required by the
        # renderer; one enclosing box is culled when viewed from inside.
        ex, ey, ez = sx / 2 + t, sy / 2 + t, (h + t) / 2
        slabs = (
            ((t / 2, ey, ez), (sx + t / 2, sy / 2, ez)),
            ((t / 2, ey, ez), (-t / 2, sy / 2, ez)),
            ((ex, t / 2, ez), (sx / 2, sy + t / 2, ez)),
            ((ex, t / 2, ez), (sx / 2, -t / 2, ez)),
            ((ex, ey, t / 2), (sx / 2, sy / 2, h + t / 2)),
        )
        for size, pos in slabs:
            geoms.append(
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=size,
                    pos=pos,
                    contype=0,
                    conaffinity=0,
                )
            )

        return TerrainOutput(
            origin=np.array((sx / 2, sy / 2, 0.0)),
            geometries=[TerrainGeometry(geom=geom) for geom in geoms],
        )


ROOM_LIGHT_NAMES = ("room_0", "room_1", "room_2")


def _room_lights() -> tuple[spec_cfg.LightCfg, ...]:
    """Light vertical walls without changing the nominal floor exposure."""
    diagonal = 0.7071067811865476
    directions = (
        (diagonal, diagonal, 0.0),
        (-diagonal, -diagonal, 0.0),
        (0.0, 0.0, 1.0),
    )
    return tuple(
        spec_cfg.LightCfg(
            name=name,
            type="directional",
            pos=(0.0, 0.0, WALL_HEIGHT / 2),
            dir=direction,
            castshadow=False,
        )
        for name, direction in zip(ROOM_LIGHT_NAMES, directions, strict=True)
    )


def apply_room_light_domain(
    cfg: ManagerBasedRlEnvCfg, *, play: bool = False, jitter: float = 0.3
) -> None:
    """Randomize room-light directions per training environment."""
    if play:
        return
    cfg.events["rand_room_light_dir"] = EventTermCfg(
        func=dr.light_dir,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "terrain", light_names=ROOM_LIGHT_NAMES
            ),
            "operation": "add",
            "ranges": {axis: (-jitter, jitter) for axis in range(3)},
        },
    )


_SUN = spec_cfg.LightCfg(
    name="sun", pos=(0.0, 0.0, 1.5), type="directional"
)


def room_terrain_cfg(size: tuple[float, float] = ROOM_SIZE) -> TerrainEntityCfg:
    """Build one 8 m room shared by every independent simulation world."""
    return TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=size,
            num_rows=1,
            num_cols=1,
            border_width=0.0,
            color_scheme="none",
            sub_terrains={"room": RoomTerrainCfg(size=size)},
        ),
        lights=(_SUN,) + _room_lights(),
    )
