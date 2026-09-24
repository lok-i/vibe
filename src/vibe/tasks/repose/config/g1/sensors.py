"""Repose sensor configs — what the env can measure.

The contact sensors are orcs's (shared with every object-manip task); the head
camera is vibe's, because vision is what vibe is for. Re-exported together so
"which sensors does this env have" has one answer in one file.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from orcs.tasks.uolm.sensors import (
    CONTACT_GRAPH_BODY_NAMES,
    CONTACT_GRAPH_SENSOR_NAME,
    GROUND_CONTACT_SENSOR_NAME,
    HAND_BODY_NAMES,
    STRICT_KILL_BODIES,
    ground_contact_sensor,
    object_contact_graph_sensor,
)

from vibe.core.sensors import HEAD_CAM_NAME, attach_head_cam
from vibe.tasks.repose import mdp

__all__ = [
    "CONTACT_GRAPH_BODY_NAMES",
    "CONTACT_GRAPH_SENSOR_NAME",
    "GROUND_CONTACT_SENSOR_NAME",
    "HAND_BODY_NAMES",
    "HEAD_CAM_NAME",
    "object_contact_graph_sensor",
    "repose_ground_contact_sensor",
    "add_camera_sensor",
]


def repose_ground_contact_sensor():
    """The fall kill-switch: everything but the feet.

    STRICTER than the loco-manip default orcs ships. Reposing a cube is not
    supposed to involve kneeling or bracing, so a knee or forearm reaching the
    ground IS a failure here — inheriting uolm's permissive set would quietly
    stop terminating on falls.
    """
    return ground_contact_sensor(STRICT_KILL_BODIES, exclude=(".*foot.*",))


def add_camera_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
    """Attach the head camera + the object-in-FOV duty-cycle metric.

    The camera itself is `vibe.core.sensors` (every vibe task shares one head
    cam, so a resolution change re-bases them together); the metric stays here
    — it needs an `object` entity, which only this task family has.
    """
    attach_head_cam(cfg)
    # Episode_Metrics/object_in_fov — vision duty cycle (rises with active perception).
    cfg.metrics = dict(cfg.metrics or {})
    cfg.metrics["object_in_fov"] = MetricsTermCfg(
        func=mdp.object_in_fov,
        params={"sensor_name": HEAD_CAM_NAME, "object_cfg": SceneEntityCfg("object")},
    )
