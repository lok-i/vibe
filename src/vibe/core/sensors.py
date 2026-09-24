"""The head camera — task-agnostic, because the robot is.

One camera spec shared by every vibe task, so a resolution or FOV change
re-bases every task's attention metrics together instead of one at a time.
Task-specific sensors (contact graphs, ray casts) stay in the task.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import CameraSensorCfg

__all__ = ["HEAD_CAM_NAME", "head_cam_cfg", "attach_head_cam"]

HEAD_CAM_NAME = "head_cam"


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
