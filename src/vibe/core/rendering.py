"""Batch renderer selection and opt-in native visual domain randomization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, RayCastSensorCfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg

RenderBackend = Literal["warp", "optix"]


@dataclass
class VisualDrCfg:
    """Additional startup variation; existing camera/direction/palette DR stays on.

    Material/texture variation targets a plane floor only, preserving generated
    terrain's contrast and manipulated-object appearance. All options are off
    by default and are rejected in a play configuration.
    """

    lighting: bool = False
    floor_material: bool = False
    floor_textures: bool = False


def configure_backend(cfg: ManagerBasedRlEnvCfg, backend: str) -> None:
    """Resolve a requested backend before constructing any GPU resources."""
    if backend not in ("warp", "optix"):
        raise ValueError(f"Unknown render backend {backend!r}; choose 'warp' or 'optix'.")
    sensors = cfg.scene.sensors or ()
    if backend == "optix" and any(isinstance(s, RayCastSensorCfg) for s in sensors):
        raise ValueError(
            "PerLoco and other raycast-sensor tasks support only the native Warp "
            "renderer. Use --env.render-backend warp or VIBE_RENDER_BACKEND=warp."
        )
    cfg.scene.sensors = tuple(
        replace(sensor, backend=backend) if isinstance(sensor, CameraSensorCfg) else sensor
        for sensor in sensors
    )


def apply_visual_dr(cfg: ManagerBasedRlEnvCfg, options: VisualDrCfg, *, play: bool) -> None:
    """Attach native terms with selected assets; never randomize task labels."""
    if not any((options.lighting, options.floor_material, options.floor_textures)):
        return
    if play:
        raise ValueError("Additional visual DR is training-only; leave visual_dr off in play.")
    terrain = cfg.scene.terrain
    if terrain is None:
        raise ValueError("Visual DR requires a terrain entity.")
    if (options.floor_material or options.floor_textures) and terrain.terrain_type != "plane":
        raise ValueError(
            "Floor material/texture DR requires plane terrain. Generated terrain "
            "and rooms retain their existing materials and contrast."
        )
    if options.lighting:
        for name, func in (("diffuse", dr.light_diffuse), ("ambient", dr.light_ambient)):
            cfg.events[f"visual_dr_light_{name}"] = EventTermCfg(
                func=func, mode="startup",
                params={"asset_cfg": SceneEntityCfg("terrain", light_names=".*"),
                        "ranges": (0.8, 1.2), "operation": "scale"},
            )
    if options.floor_material or options.floor_textures:
        # Repose's training plane initially has the upstream checker material.
        # Give the floor its own explicit material; leave every object untouched.
        material = MaterialCfg(
            name="groundplane", rgba=(0.6, 0.6, 0.6, 1.0), reflectance=0.0,
            geom_names_expr=("terrain$",),
        )
        cfg.scene.terrain = terrain = replace(terrain, materials=(material,))
    if options.floor_material:
        for name, func, bounds in (
            ("specular", dr.mat_specular, (0.0, 0.15)),
            ("shininess", dr.mat_shininess, (0.1, 0.5)),
        ):
            cfg.events[f"visual_dr_floor_{name}"] = EventTermCfg(
                func=func, mode="startup",
                params={"asset_cfg": SceneEntityCfg("terrain", material_names="groundplane"),
                        "ranges": bounds, "operation": "abs"},
            )
    if options.floor_textures:
        # Low-contrast, low-frequency grayscale modulates the existing palette.
        # No image downloads and no textures attached to the semantic cube faces.
        textures = tuple(
            TextureCfg(name=name, type="2d", builtin=pattern, width=64, height=64,
                       rgb1=(1.0, 1.0, 1.0), rgb2=(0.85, 0.85, 0.85))
            for name, pattern in (("vibe_floor_flat", "flat"), ("vibe_floor_checker", "checker"))
        )
        material = replace(terrain.materials[0], texture=textures[0].name,
                           texuniform=True, texrepeat=(1.0, 1.0))
        cfg.scene.terrain = replace(terrain, textures=textures, materials=(material,))
        cfg.scene.sensors = tuple(
            replace(sensor, use_textures=True) if isinstance(sensor, CameraSensorCfg) else sensor
            for sensor in cfg.scene.sensors or ()
        )
        cfg.events["visual_dr_floor_texture"] = EventTermCfg(
            func=dr.mat_texid, mode="startup",
            params={"asset_cfg": SceneEntityCfg(
                "terrain", material_names="groundplane", texture_names="vibe_floor_.*")},
        )
