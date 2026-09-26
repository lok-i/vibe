"""Robot-agnostic base env config for the repose (cube reorientation) task.

Provides:
  - Colored cube scene (large/floor or small/table)
  - The task reward layer + base terminations
  - Scene, sim, viewer defaults

Robot, action, command, obs and the tracking reward layer are wired per-robot in
`config/g1/env_cfgs.py`. Nothing here names a command term: this layer describes
the CUBE and the task, not how the task is driven.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.mdp.terminations import illegal_contact
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from vibe.assets.repose import (
    BIG_CUBE_HALF_EXTENT,
    BIG_CUBE_MASS,
    SMALL_CUBE_HALF_EXTENT,
    SMALL_CUBE_MASS,
    colored_cube_entity_cfg,
    table_entity_cfg,
)
from vibe.tasks.repose import mdp

# ---------------------------------------------------------------------------
# Base env config factory (robot-agnostic)
# ---------------------------------------------------------------------------

def make_repose_cube_env_cfg(*, small_cube_table: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the base repose task config (cube + task rewards + scene).

    Robot, action scale, command and obs are set per-robot in the G1 layer.
    """

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # override per-robot
            use_default_offset=True,
        ),
    }

    events = {
        "reset_default": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
    }

    rewards = {
        # task layer = up-face only: repose truth is a 2-DOF constraint, and a
        # quat kernel over-constrains by task-irrelevant yaw.
        "object_goal": RewardTermCfg(
            func=mdp.up_face_reward,
            weight=1.0,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "command_name": "motion",
                "std": 0.5,
            },
        ),
        "success_bonus": RewardTermCfg(
            func=mdp.up_face_success_bonus,
            weight=10.0,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "command_name": "motion",
                "threshold": 0.3,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "illegal_contact": TerminationTermCfg(func=illegal_contact),
    }

    if small_cube_table:
        cube = colored_cube_entity_cfg(
            half_extent=SMALL_CUBE_HALF_EXTENT,
            mass=SMALL_CUBE_MASS,
            init_pos=(0.6, 0.0, SMALL_CUBE_HALF_EXTENT),
        )
        entities = {"object": cube, "table": table_entity_cfg()}
        env_spacing = 5.0
    else:
        cube = colored_cube_entity_cfg(
            half_extent=BIG_CUBE_HALF_EXTENT,
            mass=BIG_CUBE_MASS,
            init_pos=(0.6, 0.0, 0.21),
        )
        entities = {"object": cube}
        env_spacing = 2.0

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities=entities,
            num_envs=1,
            env_spacing=env_spacing,
        ),
        observations={},   # 3-stream layout, set by the G1 layer
        actions=actions,
        commands={},       # motion command, set by the G1 layer
        events=events,
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=4.0,
            elevation=-20.0,
            azimuth=135.0,
        ),
        sim=SimulationCfg(
            nconmax=100,
            njmax=300,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0,
    )
