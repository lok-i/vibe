"""Dodge sensors — the head camera, re-aimed.

The only thing dodge changes about vibe's shared camera is where it points, and
that one override is why this file exists rather than a constant buried in the
factory.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from vibe.core import sensors as core_sensors

__all__ = ["DODGE_CAM_PITCH_DOWN_DEG", "DODGE_CAM_QUAT", "attach_dodge_cam"]

DODGE_CAM_PITCH_DOWN_DEG = -2.0  # negative = UP
DODGE_CAM_QUAT = (0.491198, 0.508650, -0.508650, -0.491198)
"""Head camera pitched 2 deg UP, against the shared default's 45 DOWN.

`vibe.core.sensors.head_cam_cfg` aims 45 deg down because repose needs a
manipulation workspace and perloco a stepping window — both on the GROUND. A
ball arrives through the air, so that aim points at the floor while the threat
flies over it. Same D435 module, same 112x63, same 42.5 fovy — only the mount
angle moves.

**Measured, not chosen.** 21.8k in-flight ball samples under the frozen base
(64 envs), logging the ball's angular offset from the camera axis, then
thresholding offline for every candidate frustum. At the first guess of 10 deg
down the duty cycle was 55.7%, and the losses were almost entirely VERTICAL:

    exits above (el > +21)   76.5% of misses      elevation p10/med/p90
    exits below (el < -21)   21.9%                  -18 / +13 / +28 deg
    exits left/right         5.3%                 azimuth   -23 /  -2 / +20

i.e. the ball's median elevation sat +13 deg ABOVE an axis deliberately aimed
down, and azimuth never came close to the +/-34 deg horizontal edge. Re-centring
on that distribution is the whole fix:

    extra UP-pitch     69H (this)   86H     120H     157H     ball px @ 3 m
    +0  (10 down)        55.7%     79.1%   95.2%    99.7%     7.4 / 6.0 / 4.3 / 3.3
    +8                   80.0%     89.2%   93.8%    98.8%
    +12 (2 UP, THIS)     85.3%     88.5%   93.0%    98.0%

A wider lens is the wrong lever here even though it looks like the obvious one:
it buys coverage with ANGULAR RESOLUTION, and resolution on a small object is
exactly what is scarce — the ball is already 7.4 px against a 16 px patch, and a
157 deg fisheye leaves 3.3 px, under a quarter of a patch on a side. Pitch is
free. Note also that the two are substitutes: past ~120H, pitching up makes
coverage slightly WORSE (95.2 -> 92.0), since a wide frustum already covers the
high ball and starts clipping the low one.

Derived, not tuned: the quaternion is the camera frame with `look` = (cos p, 0,
-sin p), `right` = -y, `up` = right x look. Constructing it at p = 45 reproduces
`head_cam_cfg`'s literal exactly, which is the check that this uses the same
convention.

**The +12 optimum is a property of THIS throw distribution** (2-3 m, +/-25 deg
cone, ~0.6 s flight, the two threat heights). Change the launch geometry and it
moves — re-run the sweep, do not re-guess.

Two consequences worth knowing. **Resolution is untouched, so P stays 7x3 = 21**
and every ZAttention number remains on the shared normalizer; what does NOT
carry is attention CONTENT, so compare dodge to dodge. And at 2 deg up the
camera sees no ground inside ~4 m — irrelevant for a standing dodge, but this
mount does not transfer to a walking task without another sweep.
"""


def attach_dodge_cam(cfg: ManagerBasedRlEnvCfg, **overrides) -> None:
    """The shared head camera, aimed at the air instead of the floor."""
    core_sensors.attach_head_cam(cfg, quat=DODGE_CAM_QUAT, **overrides)
