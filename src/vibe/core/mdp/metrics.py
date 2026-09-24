"""Camera/perception metrics (``Episode_Metrics/*``) — task-agnostic.

Anything that needs only a camera sensor + an entity to project into it lives
here, so every vibe task that carries an object reads the SAME duty cycle.
Object *tracking* metrics stay with the command that owns them.
"""

from __future__ import annotations

import math

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

__all__ = ["ObjectCamProjection", "object_in_fov"]


class ObjectCamProjection:
    """Pinhole projection of the object into a camera sensor's image plane.

    THE camera geometry, once: object root pos → camera frame (parent body pose ∘ the
    sensor's fixed mount offset) → normalized image coords. `object_in_fov` is the
    in-frustum indicator built on it; an offline representation eval reads `uv` to
    ask *which image patch the
    object falls in*, so attention can be scored against ground truth. One implementation
    keeps the training metric and the offline read from drifting apart.

    `uv` is (B, 2) in [0, 1]², origin TOP-LEFT (image convention: v grows downward,
    camera-frame +y is up). Values outside [0, 1] mean outside the frustum; they stay
    unclamped so a caller can see how far out. Pixel coords are `uv * (width, height)`.
    """

    def __init__(self, env, sensor_name: str, object_name: str = "object", robot_name: str = "robot") -> None:
        c = env.scene.sensors[sensor_name].cfg
        self._obj, self._robot = object_name, robot_name
        self._ti = env.scene[robot_name].find_bodies(c.parent_body.split("/")[-1])[0][0]
        self._pos_off = torch.tensor(c.pos, dtype=torch.float, device=env.device)
        self._quat_off = torch.tensor(c.quat, dtype=torch.float, device=env.device)  # wxyz
        self.height, self.width = int(c.height), int(c.width)
        self._half_v = math.radians(c.fovy) / 2.0
        self._half_h = math.atan((c.width / c.height) * math.tan(self._half_v))
        self._tan_v = math.tan(self._half_v)
        self._tan_h = math.tan(self._half_h)

    def __call__(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (uv (B,2) in [0,1]² top-left origin, in_front (B,) bool)."""
        robot = env.scene[self._robot]
        p_t = robot.data.body_link_pos_w[:, self._ti]   # (B, 3) camera parent body
        q_t = robot.data.body_link_quat_w[:, self._ti]  # (B, 4)
        p_cam = p_t + quat_apply(q_t, self._pos_off.expand_as(p_t))
        q_cam = quat_mul(q_t, self._quat_off.expand(q_t.shape))
        d = quat_apply_inverse(q_cam, env.scene[self._obj].data.root_link_pos_w - p_cam)
        fwd = -d[:, 2]  # MuJoCo camera looks down local -z
        in_front = fwd > 1e-4
        fwd = fwd.clamp(min=1e-4)
        # Tangent-space NDC in [-1, 1] across the frustum. |ndc| < 1 is the same test as
        # |atan2(.)| < half_angle (tan is monotone on [0, pi/2) and fwd > 0), just kept in
        # tangent space so the same numbers give the pixel position for free.
        u_ndc = (d[:, 0] / fwd) / self._tan_h
        v_ndc = (d[:, 1] / fwd) / self._tan_v
        uv = torch.stack([(u_ndc + 1.0) * 0.5, (1.0 - v_ndc) * 0.5], dim=-1)
        return uv, in_front

    def in_fov(self, env) -> torch.Tensor:
        """(B,) float ∈ {0,1}: object inside the frustum."""
        uv, in_front = self(env)
        inside = (uv > 0.0).all(dim=-1) & (uv < 1.0).all(dim=-1)
        return (in_front & inside).float()


class object_in_fov(ManagerTermBase):
    """Object-in-head_cam-frustum indicator → the **vision duty cycle**.

    Geometric, no render (`ObjectCamProjection`): object root pos → camera frame (torso
    body pose ∘ the sensor's fixed mount offset), tested against the MuJoCo ``-z``
    frustum (``fovy`` + aspect-derived ``fovx``). Returns (B,) ∈ {0,1}; MetricsManager
    ``reduce="mean"`` reports the per-episode fraction (``Episode_Metrics/object_in_fov``),
    expected to RISE as active perception emerges (low+flat ⇒ the lever is camera aim/FOV,
    not the extractor — docs/perception/encoders.md).
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self._proj = ObjectCamProjection(
            env,
            sensor_name=cfg.params["sensor_name"],
            object_name=cfg.params.get("object_cfg", SceneEntityCfg("object")).name,
            robot_name=cfg.params.get("robot_name", "robot"),
        )

    def __call__(self, env, **_kw) -> torch.Tensor:
        return self._proj.in_fov(env)
