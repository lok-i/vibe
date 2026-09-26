"""Dodge sensors — the head camera, re-aimed.

The only thing dodge changes about vibe's shared camera is where it points, and
that one override is why this file exists rather than a constant buried in the
factory.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from vibe.core import sensors as core_sensors

__all__ = ["DODGE_CAM_QUAT", "attach_dodge_cam"]

DODGE_CAM_QUAT = (0.491198, 0.508650, -0.508650, -0.491198)
"""Head camera pitched 2 deg UP, against the shared default's 45 DOWN.

A ball arrives through the air, so the shared ground-aimed camera points at the floor while
the threat flies over it. Chosen from a sweep of in-flight ball positions under the frozen base:
this pitch holds the ball in frame far more often than the first guess (10 deg down), where
misses were mostly the ball exiting ABOVE. A wider lens was the wrong lever: it costs angular
resolution, and the ball is already ~7 px against a 16 px patch. Same 112x63, same fovy.

The optimum belongs to THIS throw distribution — change the launch geometry, re-run the
sweep. At 2 deg up the camera sees no ground inside ~4 m: fine for a standing dodge, not for
a walking task.
"""


def attach_dodge_cam(cfg: ManagerBasedRlEnvCfg, **overrides) -> None:
    """The shared head camera, aimed at the air instead of the floor."""
    core_sensors.attach_head_cam(cfg, quat=DODGE_CAM_QUAT, **overrides)
