"""The head camera — task-agnostic, because the robot is.

One camera spec shared by every vibe task, so a resolution or FOV change
re-bases every task's attention metrics together instead of one at a time.
Task-specific sensors (contact graphs, ray casts) stay in the task.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import CameraSensorCfg

__all__ = ["HEAD_CAM_NAME", "head_cam_cfg", "attach_head_cam",
           "PLANNER_CAM_NAME", "planner_cam_cfg", "attach_planner_cam"]

HEAD_CAM_NAME = "head_cam"
PLANNER_CAM_NAME = "planner_cam"


def head_cam_cfg(**overrides) -> CameraSensorCfg:
    """Torso-mounted RGB head camera, looking 45 deg down-forward.

    D435 RGB stream, 16:9 at 112x63. MuJoCo fovy = VERTICAL FOV, so the
    horizontal one follows the aspect: 2*atan(tan(42.5deg/2)*112/63) ~ 69 deg H.
    (fovy=58 would be the DEPTH module's FOV — wrong stream for an rgb sensor.)
    Token grid follows the frame: the backbone patchifies at stride 16 with no
    resize, so P = floor(112/16) x floor(63/16) = 7 x 3 = 21 — the ZAttention
    normalizer. NOT logged (constant per run, implied by these numbers), so
    recompute it by hand on any change here: it re-bases every attention number
    and re-grids the offline ZGrounding patch mapping, which is why the offline
    eval reads P off the attention tensor and cross-checks it against this cfg
    (docs/perception/metrics.md §1, §5.2).

    The 45 deg down pitch sees ~0.46 m to ~2.4 m ahead on flat ground — the
    manipulation workspace for repose, the stepping window for perloco.
    """
    return CameraSensorCfg(**{
        "name": HEAD_CAM_NAME,
        "parent_body": "robot/torso_link",
        "pos": (0.05, 0.0, 0.4318),
        "quat": (0.6533, 0.2706, -0.2706, -0.6533),
        "fovy": 42.5,
        "height": 63,
        "width": 112,
        "data_types": ("rgb",),
        "enabled_geom_groups": (0, 2),
        "use_shadows": False,
        "use_textures": True,
        **overrides,
    })


def attach_head_cam(cfg: ManagerBasedRlEnvCfg, **overrides) -> None:
    """Append the head camera to a scene's sensors."""
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (head_cam_cfg(**overrides),)


# ---------------------------------------------------------------------------
# the planner's stream — the SAME lens, read at sensor resolution, with depth
# ---------------------------------------------------------------------------

def planner_cam_cfg(head: CameraSensorCfg, *, width: int = 640, height: int = 360,
                 **overrides) -> CameraSensorCfg:
    """the planner's view: head cam mount + FOV, full color res, rgb + depth.

    640x360 is EXACTLY 5.714x the head cam's 112x63 — same mount, same fovy,
    same 16:9 aspect, so the policy's tensor is a pure downscale of this frame and no
    FOV argument can differ between the two. On hardware that is the D435i color
    stream with `rs.align(color)` depth; the policy takes the letterbox of it.

    The three render flags are COPIED from the head cam, never restated:
    mujoco_warp requires every camera in a scene to share them, and the color
    relabel domain flips `use_textures` on the head cam at cfg-build time.
    """
    return CameraSensorCfg(**{
        "name": PLANNER_CAM_NAME,
        "parent_body": head.parent_body,
        "pos": head.pos,
        "quat": head.quat,
        "fovy": head.fovy,
        "height": height,
        "width": width,
        "data_types": ("rgb", "depth"),
        "use_textures": head.use_textures,
        "use_shadows": head.use_shadows,
        "enabled_geom_groups": head.enabled_geom_groups,
        **overrides,
    })


def attach_planner_cam(cfg: ManagerBasedRlEnvCfg, **overrides) -> CameraSensorCfg:
    """Append the planner's camera beside the head cam it derives from -> the new cfg.

    Play/demo path ONLY — never in a training cfg. It costs a second render and
    the policy cannot see it, so the task registration (and every saved run cfg) stays
    untouched.
    """
    sensors = cfg.scene.sensors or ()
    head = next((s for s in sensors if getattr(s, "name", None) == HEAD_CAM_NAME), None)
    if head is None:
        raise ValueError(f"no {HEAD_CAM_NAME} to derive {PLANNER_CAM_NAME} from")
    if any(getattr(s, "name", None) == PLANNER_CAM_NAME for s in sensors):
        raise ValueError(f"{PLANNER_CAM_NAME} already attached")
    cam = planner_cam_cfg(head, **overrides)
    cfg.scene.sensors = sensors + (cam,)
    return cam
