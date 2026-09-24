"""The RENDER domain — per-env-constant camera / light / scene-colour variation.

Task-agnostic by construction: a camera mount error and a sun angle are
properties of the robot and the room, not of a cube, a curb or a ball. What
stays task-local is anything that DEFINES a task (repose's face-colour perm is
the goal channel, not a domain — `vibe.tasks.repose.mdp.events`).

Recipe + sharp bits: docs/perception/render_domain.md — fields are declared via
@requires_model_fields so the EventManager expands them per-world BEFORE graph
capture, and colour must land on `mat_rgba` as well as `geom_rgba` (material
beats geom at render; a plane's groundplane carries one). A checker TEXTURE
beats both, so a scene whose floor is recoloured must lose its texture first
(`vibe.core.env_cfgs.flat_floor`).
"""

from __future__ import annotations

import functools

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

__all__ = ["GROUND_RGBAS", "rand_terrain_color", "set_terrain_color",
           "rand_cam_extrinsics"]

# Muted ground palette (collect_frames parity) — every entry is >100 rgb-dist
# (0-255 Euclidean) from all 6 repose cube face colors so a color-threshold cube
# mask can never alias the floor. Min margin across the set is ~104
# (tan/off-white); validate new tones the same way before adding.
GROUND_RGBAS = (
    (0.50, 0.50, 0.50, 1.0),  # gray
    (0.35, 0.27, 0.20, 1.0),  # brown
    (0.18, 0.31, 0.31, 1.0),  # dark slate
    (0.82, 0.71, 0.55, 1.0),  # tan
    (0.94, 0.94, 0.94, 1.0),  # off-white
    (0.25, 0.25, 0.30, 1.0),  # charcoal
    (0.02, 0.02, 0.02, 1.0),  # black
    (0.10, 0.30, 0.15, 1.0),  # forest
    (0.33, 0.34, 0.12, 1.0),  # olive drab
    (0.24, 0.30, 0.18, 1.0),  # moss
    (0.40, 0.12, 0.12, 1.0),  # maroon
    (0.30, 0.15, 0.12, 1.0),  # dark red-brown
    (0.10, 0.12, 0.32, 1.0),  # navy
    (0.30, 0.40, 0.50, 1.0),  # steel blue
    (0.28, 0.14, 0.28, 1.0),  # aubergine
    (0.20, 0.36, 0.38, 1.0),  # teal-gray
)

_MIN_PAIR_DIST = 100.0
"""Minimum 0-255 RGB distance between the ground colour and a raised feature's.

Same margin the palette already guarantees against repose's cube. A curb that
draws the same colour as the floor it sits on is not a harder domain, it is an
unlit scene: at 112x63 with shadows off, an unshaded coplanar edge carries
almost no signal, so the sample would delete the exteroception rather than
randomize it. 100 leaves 100 of 256 ordered pairs and >=3 partners for every
entry — variety is not what is scarce here.
"""


@functools.lru_cache(maxsize=None)
def _valid_pairs(rgbas: tuple, min_dist: float, device: torch.device) -> torch.Tensor:
    """(P, 2) long — every (base, feature) colour pair at least `min_dist` apart."""
    pal = torch.tensor(rgbas, device=device, dtype=torch.float32)[:, :3] * 255.0
    d = torch.cdist(pal, pal)
    pairs = torch.nonzero(d >= min_dist)
    assert pairs.numel(), (
        f"no colour pair in the palette is {min_dist} apart — widen the palette "
        "or lower min_dist, do not silently recolour a curb into its floor")
    return pairs


@requires_model_fields("geom_rgba", "mat_rgba")
def rand_terrain_color(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    rgbas: tuple = GROUND_RGBAS,
    ground_rgba: tuple[float, float, float, float] | None = None,
    min_dist: float = _MIN_PAIR_DIST,
) -> None:
    """Uniform per-env terrain colour from a fixed palette.

    Written to BOTH the terrain `geom_rgba` and its materials' `mat_rgba`
    (material beats geom at render; a groundplane carries one).

    `ground_rgba` splits the terrain in two: every geom whose NOMINAL colour is
    that one is ground and draws colour A, everything else is raised geometry
    and draws colour B, with A and B at least `min_dist` apart. `None` is the
    single-colour case (a flat plane has nothing to separate from) and is
    bit-identical to the pre-split term.

    The split keys on the nominal COLOUR and not on a geom name because mjlab
    renames every terrain geom to `terrain_<i>` when it merges the tiles, so a
    role-carrying name does not survive the build. The colour does, and it is
    the same channel the terrain generator already uses to express role
    (`color_scheme="height"` writes `TerrainGeometry.color`) — which is why
    `orcs.tasks.perloco.terrain.FLOOR_RGBA` is public.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ter = env.scene.terrain
    assert ter is not None, "rand_terrain_color needs a scene terrain"

    pal = torch.tensor(rgbas, device=env.device, dtype=torch.float32)
    pairs = _valid_pairs(tuple(rgbas), min_dist, env.device)
    pick = pairs[torch.randint(len(pairs), (len(env_ids),), device=env.device)]
    base, feature = pal[pick[:, 0]], pal[pick[:, 1]]

    gids = ter.indexing.geom_ids
    is_feature = _feature_mask(env, gids, ground_rgba)

    env.sim.model.geom_rgba[env_ids[:, None], gids[None, :]] = torch.where(
        is_feature[None, :, None], feature[:, None, :], base[:, None, :])
    mids = ter.indexing.mat_ids
    if mids.numel():
        env.sim.model.mat_rgba[env_ids[:, None], mids[None, :]] = base[:, None, :]


def _feature_mask(
    env: ManagerBasedRlEnv,
    gids: torch.Tensor,
    ground_rgba: tuple[float, float, float, float] | None,
) -> torch.Tensor:
    """(G,) bool — which terrain geoms are RAISED, keyed on their nominal colour.

    All-False when `ground_rgba` is None (a flat plane is ground, entirely).
    """
    if ground_rgba is None:
        return torch.zeros(len(gids), dtype=torch.bool, device=env.device)
    nominal = _nominal_geom_rgba(env)[gids]  # (G, 4), pre-DR
    ref = torch.tensor(ground_rgba, device=env.device, dtype=nominal.dtype)
    is_feature = (nominal - ref).abs().amax(-1) > 1e-4
    assert bool(is_feature.any()) and bool((~is_feature).any()), (
        f"nominal terrain colours do not split on {ground_rgba} "
        f"({int(is_feature.sum())} of {len(gids)} geoms differ) — the ground "
        "colour changed upstream and the curb would render in the floor's "
        "colour, which deletes the exteroception rather than randomizing it")
    return is_feature


@requires_model_fields("geom_rgba", "mat_rgba")
def set_terrain_color(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    rgba: tuple[float, float, float, float],
    ground_rgba: tuple[float, float, float, float] | None = None,
) -> None:
    """ONE fixed ground colour, every env — the deterministic twin of the above.

    The stage a clip is filmed on, not a domain: `vibe.core.env_cfgs.STAGE_RGBA`
    under `play` only, so four task families collage as one shoot instead of
    four different rooms (`docs/perception/render_domain.md`).

    RAISED geometry keeps its nominal colour — a perloco curb or a uolm prop is
    the thing being looked at, and only the flat ground under it has to match
    across tasks. `ground_rgba=None` has nothing raised to spare and paints the
    whole terrain (a plane; dodge's room, walls included).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ter = env.scene.terrain
    assert ter is not None, "set_terrain_color needs a scene terrain"

    gids = ter.indexing.geom_ids
    ground = gids[~_feature_mask(env, gids, ground_rgba)]
    col = torch.tensor(rgba, device=env.device, dtype=torch.float32)
    env.sim.model.geom_rgba[env_ids[:, None], ground[None, :]] = col
    # material beats geom at render, and a groundplane carries one — same write
    # path (and the same all-materials reach) as `rand_terrain_color`'s base.
    mids = ter.indexing.mat_ids
    if mids.numel():
        env.sim.model.mat_rgba[env_ids[:, None], mids[None, :]] = col


def _nominal_geom_rgba(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Pre-DR `geom_rgba` as (G, 4) — env 0's, since the nominal is the build's."""
    d = env.sim.get_default_field("geom_rgba")
    return d[0] if "geom_rgba" in env.sim.per_world_default_fields else d


def _nominal(env, field: str, cid: int, env_ids: torch.Tensor) -> torch.Tensor:
    """Pre-DR default of one camera's `field`, as (len(env_ids), k).

    Mirrors mjlab's `dr._core._select_default_values`: per-world fields keep each
    world's own baseline, everything else is shared across envs.
    """
    d = env.sim.get_default_field(field)
    if field in env.sim.per_world_default_fields:
        return d[env_ids, cid]
    return d[cid].unsqueeze(0).expand(len(env_ids), -1)


@requires_model_fields("cam_pos", "cam_quat")
def rand_cam_extrinsics(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    sensor_name: str,
    pos_range: tuple[float, float] = (-0.02, 0.02),
    rot_range: tuple[float, float] = (-0.035, 0.035),
) -> None:
    """Per-env camera MOUNTING error: position + RPY, off the nominal pose.

    mjlab ships `dr.cam_{pos,quat}`, but they resolve the camera through an
    Entity's own spec, and the head cam is a scene SENSOR welded onto the MERGED
    spec (`CameraSensorCfg.parent_body`) — it belongs to no entity, so
    `SceneEntityCfg(..., camera_names=...)` cannot see it. Hence this local term.
    Same math as mjlab's (perturbation PRE-multiplied onto the DEFAULT quat, read
    through `sim.get_default_field`, so repeated calls never accumulate) and the
    same per-world write path as `rand_terrain_color`.

    Defaults are TIGHTER than fcrl's ±4 cm / ±3°: at 112x63 the 69° horizontal
    FOV spans 7 patch columns (~9.9°/patch), so ±2° is a fifth of a patch — a
    plausible bolt-up error rather than a re-aimed camera.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    cid = env.sim.mj_model.camera(sensor_name).id
    n = len(env_ids)

    def _u(rng: tuple[float, float]) -> torch.Tensor:
        return torch.empty(n, 3, device=env.device).uniform_(*rng)

    rpy = _u(rot_range)
    q = quat_mul(
        quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2]),
        _nominal(env, "cam_quat", cid, env_ids).contiguous(),
    )
    env.sim.model.cam_pos[env_ids, cid] = (
        _nominal(env, "cam_pos", cid, env_ids) + _u(pos_range))
    env.sim.model.cam_quat[env_ids, cid] = q
