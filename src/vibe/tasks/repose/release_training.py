"""Training recipe for the released Repose run, with the current simulator.

The source run is vbp/repose/011pgzbh. This is a new training run from the frozen
SONIC base, not a resume from its released adapter. Standard task factories are
unchanged; this explicit recipe restores its original 534-input critic.
"""

from dataclasses import replace

from mjlab.scripts.train import TrainConfig

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


def training_config(backend: str = "warp") -> TrainConfig:
    """Build a fresh config; renderer is the only learning-config variation."""
    if backend not in ("warp", "optix"):
        raise ValueError(f"Unsupported renderer: {backend}")
    cfg = TrainConfig.from_task(TASK)
    env = replace(cfg.env, render_backend=backend)
    terms = env.observations["critic"].terms
    expected = set(CRITIC_TERMS) | {"object_id", "robot_root_pos_env"}
    if set(terms) != expected:
        raise ValueError("Repose critic changed; review the release training recipe")
    env.observations["critic"].terms = {name: terms[name] for name in CRITIC_TERMS}
    env.scene.num_envs = 4096
    env.seed = 42
    env.sim.deterministic = False
    env.sim.mujoco.jacobian = "auto"
    env.commands["motion"].init_phase_anneal_iterations = 50000

    agent = cfg.agent
    agent.seed = 42
    agent.num_steps_per_env = 24
    agent.max_iterations = 60000
    agent.save_interval = 2000
    agent.amp_dtype = "bfloat16"
    agent.experiment_name = "0p7p15/0p7p12/ExtendSimToReal"
    agent.wandb_project = "repose"
    agent.logger = "wandb"
    agent.resume = False
    # Keep metric/config logging online, and checkpoints on the training host.
    agent.upload_model = False
    return replace(cfg, env=env)
