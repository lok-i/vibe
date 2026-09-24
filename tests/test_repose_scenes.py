"""CPU-only contracts for Repose's physical-scene axis."""

from __future__ import annotations

import mujoco
import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg

import vibe  # noqa: F401 — register tasks
from vibe.assets.repose import (
    BIG_CUBE_HALF_EXTENT,
    FACE_COLOR_NAMES,
    FACE_COLORS,
    SMALL_CUBE_HALF_EXTENT,
    SMALL_CUBE_MASS,
    TABLE_CENTER_HEIGHT,
    TABLE_SIZE,
    colored_cube_spec,
    table_spec,
)

BIG_ROWS = {
    "Vibe-Repose-BigCubeFloor-ObjKin",
    "Vibe-Repose-BigCubeFloor-ImgFeat",
    "Vibe-Repose-BigCubeFloor-ImgRgb",
    "Vibe-Repose-BigCubeFloor-ImgFeat-Ext",
    "Vibe-Repose-BigCubeFloor-ImgFeat-Sfd",
    "Vibe-Repose-BigCubeFloor-ImgFeat-Lfd",
}
SMALL_ROW = "Vibe-Repose-SmallCubeTable-ImgFeat-Ext"


def test_cube_palette_matches_physical_opposite_pairs():
    assert FACE_COLOR_NAMES == (
        "red",
        "orange",
        "green",
        "yellow",
        "blue",
        "pink",
    )
    assert FACE_COLORS[0] != FACE_COLORS[1]
    assert FACE_COLORS[2] != FACE_COLORS[3]
    assert FACE_COLORS[4] != FACE_COLORS[5]


def test_repose_task_roster_names_the_physical_scene():
    registered = {t for t in list_tasks() if t.startswith("Vibe-Repose-")}
    assert registered == BIG_ROWS | {SMALL_ROW}
    assert all("AdaptSonic" not in task for task in registered)


def test_small_cube_and_table_are_primitive_boxes():
    assert SMALL_CUBE_HALF_EXTENT == pytest.approx(0.18)
    assert SMALL_CUBE_MASS == pytest.approx(0.3)

    cube = colored_cube_spec(
        half_extent=SMALL_CUBE_HALF_EXTENT, mass=SMALL_CUBE_MASS
    ).compile()
    core = cube.geom("cube_core")
    assert core.type == mujoco.mjtGeom.mjGEOM_BOX
    assert tuple(core.size) == pytest.approx((SMALL_CUBE_HALF_EXTENT,) * 3)
    assert cube.body("object").mass == pytest.approx(SMALL_CUBE_MASS)

    table = table_spec().compile()
    top = table.geom("table_top")
    assert top.type == mujoco.mjtGeom.mjGEOM_BOX
    assert tuple(top.size) == pytest.approx(tuple(v / 2.0 for v in TABLE_SIZE))


def test_repose_scene_cfgs_select_dataset_geometry_and_support():
    big = load_env_cfg("Vibe-Repose-BigCubeFloor-ObjKin")
    small = load_env_cfg(SMALL_ROW)

    assert tuple(big.scene.entities) == ("object", "robot")
    assert big.commands["motion"].cube_half_extent == BIG_CUBE_HALF_EXTENT
    assert big.commands["motion"].table_entity_name is None

    assert tuple(small.scene.entities) == ("object", "table", "robot")
    assert small.scene.env_spacing == 5.0
    assert small.commands["motion"].cube_half_extent == SMALL_CUBE_HALF_EXTENT
    assert small.commands["motion"].table_entity_name == "table"
    assert small.commands["motion"].table_center_height == TABLE_CENTER_HEIGHT
    assert small.commands["motion"].dataset_dir.endswith("/custom/box_manip")
    assert small.episode_length_s == 11.0


@pytest.mark.parametrize("task", sorted(BIG_ROWS | {SMALL_ROW}))
def test_film_recenter_is_play_only(task: str) -> None:
    """Every clip opens at the env origin under play — and NEVER under train.

    The transform is an exact symmetry of the task, but it moves the reference,
    so a leak into train would be a silent dataset change (`_recenter_clips`).
    """
    assert load_env_cfg(task, play=True).commands["motion"].film_recenter
    assert not load_env_cfg(task, play=False).commands["motion"].film_recenter
