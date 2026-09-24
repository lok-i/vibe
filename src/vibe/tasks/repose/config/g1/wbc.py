"""Frozen-SONIC assembly — robot/action wiring + the base's obs streams.

Everything comes from `mocke.sonic.profile`: that is the frozen base's I/O
CONTRACT, coupled bit-for-bit to the ported checkpoint, so nothing here may
reshape it. `robot_cfg` picks the VISUAL set only (`vibe.assets.g1`).
"""

from __future__ import annotations

from typing import Callable

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mocke.sonic import profile as sonic_profile

from vibe.assets.g1 import G1_VIS_LEAN_CFG

__all__ = ["wire_robot", "policy_obs_group", "extra_obs_groups"]


def wire_robot(
    cfg: ManagerBasedRlEnvCfg,
    robot_cfg: Callable[[], EntityCfg] = G1_VIS_LEAN_CFG,
) -> None:
    """Robot entity + action term. SONIC regroups hip_pitch and emits MJ-order,
    so its action term must be built FROM the robot it wired."""
    robot = sonic_profile.robot_cfg(base=robot_cfg())
    cfg.scene.entities["robot"] = robot
    cfg.actions["joint_pos"] = sonic_profile.action_cfg(robot)


def policy_obs_group() -> ObservationGroupCfg:
    """The frozen base's input stream."""
    return ObservationGroupCfg(
        terms=sonic_profile.policy_obs_terms(),
        concatenate_terms=True,
        enable_corruption=True,
        nan_policy="sanitize",
        nan_check_per_term=True,
    )


def extra_obs_groups() -> dict[str, ObservationGroupCfg]:
    """Base streams beyond `policy` — SONIC's tokenizer window."""
    return sonic_profile.extra_obs_groups("motion")
