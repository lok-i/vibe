"""Explicit critic compatibility for the released Repose recipe (011pgzbh).

Selected through the normal train CLI with --env.release-critic True.
Training hyperparameters remain ordinary CLI options; this only selects the
original ordered 534-input critic, leaving the actor and task mechanics intact.
"""

from dataclasses import dataclass, replace

from vibe.core.env_cfgs import VibeEnvCfg

TASK = "Vibe-Repose-BigCubeFloor-ImgFeat-Ext"
CRITIC_TERMS = (
    "command", "motion_anchor_pos_b", "motion_anchor_ori_b",
    "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel",
    "base_lin_vel", "actions", "object_pos_w", "object_ori_mat6d_w",
    "object_lin_vel_w", "object_ang_vel_w", "object_goal_ori",
    "object_ref_pos_b", "object_ref_ori_b", "bodywise_contact_cmd",
    "robot_root_lin_vel_cmd", "robot_root_ang_vel_cmd", "reward_vec",
    "object_goal_color",
)


@dataclass
class ReposeEnvCfg(VibeEnvCfg):
    release_critic: bool = False
    """Use the released colour-task critic layout; default keeps today's layout."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.release_critic:
            return
        terms = self.observations["critic"].terms
        expected = set(CRITIC_TERMS) | {"object_id", "robot_root_pos_env"}
        # Reconstructing a config (tyro/dataclasses.replace) must be idempotent.
        if set(terms) not in (expected, set(CRITIC_TERMS)):
            raise ValueError("Repose critic changed; review the release critic layout")
        self.observations = dict(self.observations)
        self.observations["critic"] = replace(
            self.observations["critic"],
            terms={name: terms[name] for name in CRITIC_TERMS},
        )
