"""G1 repose env config — single entry point, two axes.

  g1_repose_cube_env_cfg(extero, aux, ...)

  extero  : "objkin" object kinematic state; goal = quat, reward = up_face
          | "imgrgb" raw head-camera RGB (the trainable-CNN baseline)
          | "imgfeat" frozen encoder features from the same camera
  aux     : rewire obs for the extractor/PPOAux pipeline (imgfeat only)

Both VISION values share one task channel: the goal reaches the actor as an
up-face COLOR one-hot (never the quat), a per-env face->color remap + terrain
palette break the orientation<->color correspondence, and the task rewards swap
to the up_color pair. So imgrgb and imgfeat differ in ONE thing — which obs
group the camera lands in — which is what makes the two comparable.

The command is always the motion command and init is always RSI; the frozen base
is always SONIC. vibe is visual behavior adaptation: a frozen base, adapted.

Module layout: `wbc.py` (frozen-base assembly) · `sensors.py` (camera + contact)
· `observation_cfgs.py` (every obs group) · `agent_cfgs.py` (the actors).
"""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.mdp import rewards as tracking_rewards
from orcs.assets import OBJECT_BODY_NAME
from orcs.core.obs import apply_obs_noise
from orcs.tasks.uolm.robustness import apply_robustness

from vibe.assets.g1 import G1_BASE_CFG, G1_VIS_LEAN_CFG
from vibe.assets.repose import BIG_CUBE_HALF_EXTENT, SMALL_CUBE_HALF_EXTENT
from vibe.core import paths
from vibe.core.env_cfgs import (
    QUERY_NOISY_GROUPS,
    apply_render_domain,
    apply_stage_render,
    with_vision_knobs,
)
from vibe.tasks.repose import mdp
from vibe.tasks.repose.config.g1 import observation_cfgs as oc
from vibe.tasks.repose.config.g1 import sensors, wbc
from vibe.tasks.repose.mdp.commands import ReposeMotionCommandCfg
from vibe.tasks.repose.repose_cube_env_cfg import make_repose_cube_env_cfg

ReposeScene = Literal["big_cube_floor", "small_cube_table"]

_G1_REPOSE_DATASETS: dict[ReposeScene, str | list[str]] = {
    "big_cube_floor": [str(p) for p in paths.G1_REPOSE_BIG_CUBE_FLOOR_DATASET],
    "small_cube_table": str(paths.G1_REPOSE_SMALL_CUBE_TABLE_DATASET),
}

# Post-motion padding: the reference freezes at the clip's last frame and the
# policy must hold steady-state until `exceeded_motion_by_eps` truncates. Single
# source for the termination AND the episode length (episode_length =
# max_clip_len + ε, so the episode — not the motion — owns resets).
_MOTION_PAD_EPS_SEC = 2.0


def _first_clip_file(dataset_dir: str | list[str]) -> str:
    """First motion.npz in the dataset — MotionCommand's init needs one file."""
    for d in mdp.motion_dirs(dataset_dir):
        for s in sorted(p for p in d.iterdir() if p.is_dir()):
            mf = s / "motion.npz"
            if mf.exists():
                return str(mf)
    raise FileNotFoundError(f"No motion.npz under {dataset_dir}")


_MAX_CLIP_LEN_CACHE: dict[str, int] = {}


def _max_clip_len_steps(dataset_dir: str | list[str]) -> int:
    """Longest clip (frames) in the dataset — cached, scanned once per root."""
    key = str(dataset_dir)
    if key not in _MAX_CLIP_LEN_CACHE:
        import numpy as np

        max_len = 0
        for d in mdp.motion_dirs(dataset_dir):
            for s in sorted(p for p in d.iterdir() if p.is_dir()):
                mf = s / "motion.npz"
                if mf.exists():
                    max_len = max(max_len, int(np.load(mf)["joint_pos"].shape[0]))
        if max_len == 0:
            raise FileNotFoundError(f"No motion.npz under {dataset_dir}")
        _MAX_CLIP_LEN_CACHE[key] = max_len
    return _MAX_CLIP_LEN_CACHE[key]


def _load_exclude_motions(path: str) -> tuple[str, ...]:
    """Exclusion list, a .json list[str] of "<motion>" or "<motion>/<sampleX>".
    Relative paths resolve against the repo root."""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.is_absolute():
        p = paths.VIBE_ROOT / p
    entries = json.loads(p.read_text())
    assert all(isinstance(e, str) for e in entries), f"non-string entry in {p}"
    return tuple(dict.fromkeys(entries))  # dedupe, order-preserving


# ---------------------------------------------------------------------------
# Sim2real obs noise + render domain — shared, not redefined: the noise table is
# `orcs.core.obs.OBS_NOISE`, the render domain `vibe.core.env_cfgs`.
# ---------------------------------------------------------------------------

_NOISY_GROUPS = ("policy",) + QUERY_NOISY_GROUPS
"""Deployed streams ONLY. Deliberately excluded:

    augmentation      the motion COMMAND, not sensed — exact on hardware
    critic            privileged, never deploys
    prediction_*      train-time predictor; noising its conditioning buys an
                      error floor and no transfer (the target is sim truth)
    kv_tokens/q_cls   image path — a separate axis, not this one
"""


def apply_color_relabel(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Color-relabel render domain: per-env face->color perm + terrain palette.

    The env-side half of the vision rows' colour channel. Startup mode = one perm per
    env for the whole run, so each env has a consistent commanded colour
    (`ReposeMotionCommand.goal_color_idx`).
    """
    cfg.events["rand_face_colors"] = EventTermCfg(
        func=mdp.rand_face_colors, mode="startup")
    cfg.events["rand_terrain_color"] = EventTermCfg(
        func=mdp.rand_terrain_color, mode="startup")
    # flat mat_rgba only wins with textures off (texture > mat_rgba)
    for s in cfg.scene.sensors or ():
        if getattr(s, "name", None) == sensors.HEAD_CAM_NAME:
            s.use_textures = False
    return cfg


# ---------------------------------------------------------------------------
# Obs layout
# ---------------------------------------------------------------------------

def _aug_obs(cfg: ManagerBasedRlEnvCfg, extero: str) -> None:
    """3-stream obs: policy (frozen WBC input) / augmentation (task) / critic.

    policy + the tokenizer stream come from the WBC assembler and are mocke's
    contract; `augmentation` and `critic` are assembled from the signal library.
    Repose's goal is orientation-only, so the goal-pos term of the exhaustive
    object-manip schema is dropped everywhere (keeps legacy ckpt obs dims).
    """
    ctx = oc.ObsCtx(obj=SceneEntityCfg("object"), p={"command_name": "motion"})
    cfg.observations = {
        "policy": wbc.policy_obs_group(),
        **wbc.extra_obs_groups(),
        "augmentation": (
            oc.objkin_augmentation_group(ctx) if extero == "objkin"
            # imgrgb reads the same camera through the `camera` group instead,
            # so the two vision rows share this stream term for term.
            else oc.vision_augmentation_group(ctx, feat=extero == "imgfeat")),
        "critic": oc.critic_group(ctx),
    }
    if extero == "imgrgb":
        cfg.observations[oc.CAMERA_GROUP] = oc.camera_group(ctx)
    for grp in cfg.observations.values():
        grp.terms.pop("object_goal_pos", None)


# ---------------------------------------------------------------------------
# THE factory
# ---------------------------------------------------------------------------

def g1_repose_cube_env_cfg(
    *,
    scene: ReposeScene = "big_cube_floor",
    extero: Literal["objkin", "imgrgb", "imgfeat"] = "objkin",
    aux: bool = False,
    play: bool = False,
    dataset_dir: str | list[str] | None = None,
    exclude_motions_file: str | None = None,
    num_steps_per_env: int = 24,
) -> ManagerBasedRlEnvCfg:
    """THE G1 repose env config factory.

    Args:
        scene:   Physical object/support scene and its default motion dataset.
        extero:  "objkin" = object kinematic state (goal quat obs, up_face
                 rewards); "imgrgb" / "imgfeat" = the head camera, raw or through
                 the frozen encoder — both on the COLOR task channel (module
                 docstring).
        aux:     Rewire obs for the extractor/PPOAux pipeline: a `kv_tokens` dict
                 group + one query group per active channel + the
                 `prediction_{target,conditioning}` side-inputs
                 (docs/architecture.md).
        play:    Play-mode overrides (no noise, no anneal, relaxed terminations).
        dataset_dir: Multi-clip dataset root(s) (default: the repose cube clips).
        exclude_motions_file: .json clip exclusion list (`_load_exclude_motions`),
                 e.g. the unlearnable-clip list the tasks register with.
        num_steps_per_env: PPO rollout length (for PolicyUpdateCounter).
    """
    if aux:
        assert extero == "imgfeat", "aux obs require extero='imgfeat'"
    if scene not in _G1_REPOSE_DATASETS:
        raise ValueError(f"unknown Repose scene: {scene!r}")
    vision = extero in ("imgfeat", "imgrgb")
    has_table = scene == "small_cube_table"

    # ── base shell + frozen-WBC robot/action wiring ──
    cfg = make_repose_cube_env_cfg(small_cube_table=has_table)
    wbc.wire_robot(cfg, robot_cfg=G1_BASE_CFG if play else G1_VIS_LEAN_CFG)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        sensors.repose_ground_contact_sensor(),
    )
    cfg.terminations["illegal_contact"].params["sensor_name"] = (
        sensors.GROUND_CONTACT_SENSOR_NAME)

    ds = dataset_dir or _G1_REPOSE_DATASETS[scene]
    _p = {"command_name": "motion"}

    # ── the motion command (the base cfg ships none) ──
    cfg.commands["motion"] = ReposeMotionCommandCfg(
        motion_file=_first_clip_file(ds),
        dataset_dir=ds,
        future_steps=5,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        contact_graph_body_names=sensors.CONTACT_GRAPH_BODY_NAMES,
        contact_graph_sensor_name=sensors.CONTACT_GRAPH_SENSOR_NAME,
        cube_half_extent=(
            SMALL_CUBE_HALF_EXTENT if has_table else BIG_CUBE_HALF_EXTENT
        ),
        table_entity_name="table" if has_table else None,
        exclude_motions=(_load_exclude_motions(exclude_motions_file)
                         if exclude_motions_file else None),
    )
    # per-body object-filtered contact sensor (one multi-primary sensor;
    # contact-graph nodes, single source: sensors.CONTACT_GRAPH_BODY_NAMES)
    cfg.scene.sensors = cfg.scene.sensors + (sensors.object_contact_graph_sensor(),)

    # episode = longest clip + ε hold padding, so resets come from
    # terminations/time_out, never from running out of motion.
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = _max_clip_len_steps(ds) * step_dt + _MOTION_PAD_EPS_SEC
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "illegal_contact": cfg.terminations["illegal_contact"],
        "bad_anchor_pos": TerminationTermCfg(
            func=mdp.bad_anchor_pos, params={**_p, "threshold": 0.3}),
        "bad_anchor_ori": TerminationTermCfg(
            func=mdp.bad_anchor_ori, params={**_p, "threshold": 0.8}),
        "bad_object_pos": TerminationTermCfg(
            func=mdp.bad_object_pos, params={**_p, "threshold": 0.3}),
        "bad_object_ori": TerminationTermCfg(
            func=mdp.bad_object_ori, params={**_p, "threshold": 0.8}),
        "exceeded_motion": TerminationTermCfg(
            func=mdp.exceeded_motion_by_eps, time_out=True,
            params={**_p, "epsilon_steps": int(_MOTION_PAD_EPS_SEC / step_dt)}),
    }

    cfg.events["policy_update_counter"] = EventTermCfg(
        func=mdp.PolicyUpdateCounter,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={"num_steps_per_env": num_steps_per_env},
    )
    # No VOF (orcs's decaying PD wrench toward the demo): the cube is light enough
    # that the curriculum never paid for itself.

    # ── robot-motion rewards: track the demo trajectories ──
    # The task layer comes from the base dict (object_goal + success_bonus +
    # regularizers). Re-point it at the motion command, stack dense tracking
    # terms on top.
    cfg.rewards["object_goal"].params["command_name"] = "motion"
    cfg.rewards["success_bonus"].params["command_name"] = "motion"
    cfg.rewards.update({
        "root_pos": RewardTermCfg(
            func=tracking_rewards.motion_global_anchor_position_error_exp,
            weight=0.5, params={**_p, "std": 0.3}),
        "root_ori": RewardTermCfg(
            func=tracking_rewards.motion_global_anchor_orientation_error_exp,
            weight=0.5, params={**_p, "std": 0.4}),
        "body_pos": RewardTermCfg(
            func=tracking_rewards.motion_relative_body_position_error_exp,
            weight=1.0, params={**_p, "std": 0.3}),
        "body_ori": RewardTermCfg(
            func=tracking_rewards.motion_relative_body_orientation_error_exp,
            weight=1.0, params={**_p, "std": 0.4}),
        "body_lin_vel": RewardTermCfg(
            func=tracking_rewards.motion_global_body_linear_velocity_error_exp,
            weight=1.0, params={**_p, "std": 1.0}),
        "body_ang_vel": RewardTermCfg(
            func=tracking_rewards.motion_global_body_angular_velocity_error_exp,
            weight=1.0, params={**_p, "std": 3.14}),
        "object_pos": RewardTermCfg(
            func=mdp.object_pos_tracking_reward,
            weight=2.0, params={**_p, "std": 0.3}),
        "object_ori": RewardTermCfg(
            func=mdp.object_ori_tracking_reward,
            weight=1.0, params={**_p, "std": 0.4}),
        # contact consistency: per-body binary match vs the demo, mean over
        # graph nodes (partial credit)
        "contact_consistency": RewardTermCfg(
            func=mdp.object_contact_consistency,
            weight=1.0, params={**_p,
                                "sensor_name": sensors.CONTACT_GRAPH_SENSOR_NAME,
                                "contact_force_threshold": 0.1}),
    })

    # ── obs layout ──
    _aug_obs(cfg, extero)
    if vision:
        sensors.add_camera_sensor(cfg)

    # ── aux obs rewiring (PPOAux pipeline) ──
    if aux:
        oc.attach_aux_obs(cfg)

    # ── vision ⇒ colour task channel: break orientation<->colour, the goal
    #    reaches the actor as COLOUR only ──
    if vision:
        apply_color_relabel(cfg)
        # task rewards -> the color-channel pair (no goal quat; keys stay stable
        # — reward_vec / task_reward_vec reference them by name). Success gated
        # on a settled object (no flicker-success mid-tumble).
        _r, _b = cfg.rewards["object_goal"], cfg.rewards["success_bonus"]
        cfg.rewards["object_goal"] = RewardTermCfg(
            func=mdp.up_color_reward, weight=_r.weight, params=dict(_r.params))
        cfg.rewards["success_bonus"] = RewardTermCfg(
            func=mdp.up_color_success_bonus, weight=_b.weight,
            params={**_b.params, "max_ang_vel": 0.5})
        if aux:
            # actor: color one-hot replaces the goal quat in the task query
            cfg.observations["q_task_cmd"] = oc._grp(
                {"object_goal_color": oc._T(mdp.object_goal_color, _p)})
            # aux target += current up-face color (one-hot; the perm is
            # image-only -> supervises color extraction) + task-layer reward
            # rates (MuZero-style; task-only subset — WBC tracking terms = f(ref)
            # are unidentifiable from (z, cond) and would just add an error floor)
            tgt = cfg.observations["prediction_target"].terms
            tgt["upface_color"] = ObservationTermCfg(
                func=mdp.object_upface_color, params=_p)
            tgt["task_reward_vec"] = ObservationTermCfg(
                func=mdp.unweighted_reward_vector,
                params={"enabled": True,
                        "terms": ["object_goal", "success_bonus"]})
            # HANDS ONLY, not the full contact graph: a knee<->object or
            # foot<->ground force is not in the head cam's FOV, so it fails the
            # ego-observable rule the rest of this target obeys — its residual
            # would be gradient noise on the shared encoder. The demo schedule
            # agrees: 99% of robot<->object contact frames are hand contact.
            tgt["bodywise_saturated_force"] = ObservationTermCfg(
                func=mdp.bodywise_saturated_force,
                params={**_p, "sensor_name": sensors.CONTACT_GRAPH_SENSOR_NAME,
                        "body_names": sensors.HAND_BODY_NAMES})
        else:
            # vanilla adapter (no extractor): color one-hot replaces the goal
            # quat in the augmentation stream
            aug = cfg.observations["augmentation"].terms
            aug.pop("object_goal_ori")
            aug["object_goal_color"] = ObservationTermCfg(
                func=mdp.object_goal_color, params=_p)
        # critic: privileged — sees BOTH the quat (already there) and the color
        cfg.observations["critic"].terms["object_goal_color"] = (
            ObservationTermCfg(func=mdp.object_goal_color, params=_p))

    # ── robustness domain: state (isr + pushes) + param (physical DR) ──
    apply_robustness(cfg, object_name=OBJECT_BODY_NAME,
                     sensor_name=sensors.CONTACT_GRAPH_SENSOR_NAME,
                     hand_body_names=sensors.HAND_BODY_NAMES)
    if vision:
        # `terrain_color=False`: repose already has one, and it belongs to
        # `apply_color_relabel` — the task CHANNEL, which is why it survives the
        # play build while the camera/light knobs do not.
        apply_render_domain(cfg, play=play, terrain_color=False)
    apply_obs_noise(cfg, _NOISY_GROUPS)

    # INVARIANT: play overrides are LAST. They SUBTRACT from the assembled
    # training domain, so anything wired below this line leaks into play/eval.
    if play:
        _play_overrides(cfg)
        # AFTER them: `_play_overrides` keeps `rand_terrain_color` alive (the
        # colour relabel is repose's task channel), and the stage replaces it.
        apply_stage_render(cfg, play=play)

    # LAST, and after the play overrides on purpose: this adds a FIELD, never a
    # term or an event, so it cannot change what the play invariant sees.
    return with_vision_knobs(cfg)


def _play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Play-mode: no corruption, no anneal/VOF, no tracking kills, no DR.

    Runs LAST in the factory — see the invariant note there. Every domain knob
    added above must be subtracted here, or play/eval silently trains-domain.
    """
    for name in _NOISY_GROUPS:
        grp = cfg.observations.get(name)
        if grp is not None:
            grp.enable_corruption = False

    for event in (
                "policy_update_counter",
                "virtual_object_force",
                "perturb_robot",
                "perturb_object",
                "rand_object_inertia",
                "rand_object_friction",
                "rand_robot_friction",
                "rand_robot_com",
                "rand_encoder_bias",
                # rand_cam_extrinsics / rand_light_dir are not listed: the
                # shared `apply_render_domain` is a no-op under play, so there
                # is nothing to subtract (vibe.core.env_cfgs).
                ):
        cfg.events.pop(event, None)

    for k in ("bad_anchor_pos", "bad_anchor_ori", "bad_object_pos", "bad_object_ori"):
        cfg.terminations.pop(k, None)

    mc = cfg.commands["motion"]
    mc.start_from_zero = True
    mc.film_recenter = True  # every clip opens at the origin, facing +x — for the camera
    mc.pose_range = {}
    mc.velocity_range = {}
    mc.joint_position_range = (0.0, 0.0)
    mc.object_init_pose_range = {}
    mc.object_in_contact_velocity_range = {}

    jp = cfg.observations["policy"].terms.get("joint_pos")
    if jp is not None:
        jp.params = {**(jp.params or {}), "biased": False}
