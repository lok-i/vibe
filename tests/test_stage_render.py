"""The film STAGE — one ground colour across every task, under play only.

Two claims, and the collage is only coherent if both hold: every registered
vibe task pins the SAME floor at play time (and none of them at train time),
and that colour stays separable from everything standing on it.
"""

from __future__ import annotations

import mjlab  # noqa: F401 — must precede orcs (CLAUDE.md: the import-order trap)
import numpy as np
import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg
from orcs.assets.ball import ball_entity_cfg

import vibe  # noqa: F401 — registers the tasks
from vibe.assets.repose import FACE_COLORS
from vibe.core.env_cfgs import STAGE_RGBA
from vibe.core.mdp.events import _MIN_PAIR_DIST, GROUND_RGBAS, set_terrain_color

VIBE_TASKS = sorted(t for t in list_tasks() if t.startswith("Vibe-"))

_DEAD_ZONE = (0.50, 0.72)
"""Nominal luminance band where a floor renders ON TOP of the robot's silver.

Measured, not derived (`STAGE_RGBA`): a flat floor rides above its nominal under
the sun's ~1.2 gain while the G1's curved silver renders BELOW its own, so the
two cross somewhere in here. The band is the UNION over the framings measured,
because the robot's rendered silver moves with pose — at a far, small-robot pose
nominal 0.68 measured a dL of exactly 0.00; at the framing a clip is cut at, the
same tone measured 0.26. A radius around silver's 0.7 nominal sees neither.
"""

_TRAIN_PLANE = {
    # The train floor each plane task shipped BEFORE the stage existed, recorded
    # so a future edit to `apply_stage_render` cannot quietly become a TRAINING
    # change. repose keeps mjlab's checker (its head cam ignores textures, which
    # is exactly why the viewer still drew one); uolm and dodge were already
    # flattened to the nominal grey by `flat_floor`.
    t: [("groundplane", (1.0, 1.0, 1.0, 1.0))] if "-Repose-" in t
    else [(None, (0.6, 0.6, 0.6, 1.0))]
    for t in VIBE_TASKS
}


def _rgb_dist(a, b) -> float:
    """0-255 Euclidean, the unit `_MIN_PAIR_DIST` is written in."""
    return float(np.linalg.norm((np.asarray(a[:3]) - np.asarray(b[:3])) * 255.0))


@pytest.mark.parametrize("task", VIBE_TASKS)
def test_play_pins_the_stage(task: str) -> None:
    """Play draws STAGE_RGBA, and nothing else recolours the ground under it."""
    cfg = load_env_cfg(task, play=True)
    term = cfg.events.get("set_terrain_color")
    assert term is not None, f"{task} films on its own floor"
    assert term.func is set_terrain_color
    assert term.params["rgba"] == STAGE_RGBA
    assert "rand_terrain_color" not in cfg.events, (
        "a random floor beside the stage is a random floor — the stage REPLACES "
        "it (repose keeps that event alive under play on purpose)")


@pytest.mark.parametrize("task", VIBE_TASKS)
def test_train_has_no_stage(task: str) -> None:
    """The stage is a filming knob. Training keeps its randomized floor."""
    assert "set_terrain_color" not in load_env_cfg(task, play=False).events


@pytest.mark.parametrize("task", VIBE_TASKS)
def test_the_stage_never_reaches_the_TRAIN_material(task: str) -> None:
    """The other half of it: the play build strips the plane's TEXTURE (mjlab's
    checker BEATS both rgba fields, so a per-world write alone paints a floor
    the viewer never shows) — and the train build must keep whatever it had."""
    train, play = (load_env_cfg(task, play=p).scene.terrain for p in (False, True))
    if getattr(train, "terrain_type", None) != "plane":
        pytest.skip("generated terrain — no material to strip")
    assert all(m.texture is None for m in play.materials), "checker survives play"
    assert all(m.rgba == STAGE_RGBA for m in play.materials)
    got = [(m.texture, tuple(m.rgba)) for m in train.materials]
    assert got == _TRAIN_PLANE[task], (
        "the stage leaked into the TRAIN material — it is a play-only knob")


def test_stage_sits_inside_the_palette_span() -> None:
    """Not an ENTRY any more (0.76 is between 0.94 and 0.50) — but the eval
    floor must stay an interpolation of what the encoder trained on, never an
    extrapolation past it. Bracketed on every channel."""
    lo = np.min([c[:3] for c in GROUND_RGBAS], axis=0)
    hi = np.max([c[:3] for c in GROUND_RGBAS], axis=0)
    assert np.all(lo <= np.array(STAGE_RGBA[:3])) and np.all(np.array(STAGE_RGBA[:3]) <= hi)


def test_stage_clears_the_bar_against_TASK_objects() -> None:
    """The thing the policy must SEE cannot alias the floor it sits on.

    The palette's own bar, applied where it means something: a cube face or a
    ball drawing the floor's colour deletes the exteroception. (perloco's curbs
    are not here — they keep their own hue, `ground_rgba`.)
    """
    actors = {f"cube_{i}": c for i, c in enumerate(FACE_COLORS)}
    actors["dodge_ball"] = ball_entity_cfg.__defaults__[2]
    worst = min((_rgb_dist(STAGE_RGBA, c), n) for n, c in actors.items())
    assert worst[0] >= _MIN_PAIR_DIST, (
        f"stage {STAGE_RGBA} is {worst[0]:.0f} from {worst[1]} — under the "
        f"{_MIN_PAIR_DIST:.0f} bar a task object has to clear")


def test_the_stage_is_not_in_the_robot_dead_zone() -> None:
    """The SCENE half of the bar, and it is deliberately NOT a nominal distance.

    Nominal rgb-dist is the wrong instrument for a curved, specular robot on a
    flat-lit floor: the G1's silver renders BELOW its 0.7 nominal (0.59 at the
    framing a clip is cut at),
    while the floor rides ABOVE its own under a ~1.2 sun gain. So the thing to
    keep away from is the RENDERED dead zone (`_DEAD_ZONE` nominal
    luminance, where grey 0.50 measured a hopeless dL of 0.04), not a radius
    around silver — `STAGE_RGBA` sits 27 from silver and reads fine.

    Rendering needs a GPU, so the numbers live in `STAGE_RGBA`'s docstring and
    the doc; what CI can hold is the band they imply.
    """
    lum = float(np.dot(STAGE_RGBA[:3], (0.2126, 0.7152, 0.0722)))
    assert not (_DEAD_ZONE[0] <= lum <= _DEAD_ZONE[1]), (
        f"stage luminance {lum:.2f} is inside the rendered dead zone "
        f"{_DEAD_ZONE} — the robot washes into the floor (measured dL 0.04)")
