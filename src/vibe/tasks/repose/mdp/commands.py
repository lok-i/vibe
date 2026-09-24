"""ReposeMotionCommand — orcs's ObjectMotionCommand, specialized to the colored cube.

Success is orientation-only (the up-face COLOR, read through the per-env face->color
perm), and the goal is drawn as a floating colored badge.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_from_matrix, quat_mul
from orcs.tasks.uolm.mdp.commands import ObjectMotionCommand, ObjectMotionCommandCfg

from vibe.assets.repose import BIG_CUBE_HALF_EXTENT, FACE_COLORS, TABLE_CENTER_HEIGHT
from vibe.tasks.repose.mdp.cube_faces import (
    color_tilt_error,
    face_color_names,
    face_colors,
    up_face_idx,
)

__all__ = ["ReposeMotionCommand", "ReposeMotionCommandCfg"]


def _cube_symmetry_quats(device: torch.device) -> torch.Tensor:
    """The 24 axis-aligned cube orientations (octahedral rotation group) -> (24, 4)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = torch.zeros(3, 3)
            for row, (col, s) in enumerate(zip(perm, signs, strict=True)):
                m[row, col] = s
            if torch.det(m) > 0:
                mats.append(m)
    return quat_from_matrix(torch.stack(mats).to(device))


_VIZ_HALF = 0.15
_VIZ_SKIN = 0.004
# Inherited metrics repose does not read. REPOSE-LOCAL: the parents keep
# publishing them (other task families may want them), we just never log them.
#   robot motion-tracker error -> the tracking REWARDS already price the
#     trade-off; repose is loco-MANIPULATION, the tracker error is not the task.
#   sampling_* -> `sampling_mode="start"`, the adaptive-sampling path never runs,
#     so these are constants (uniform).
#   init_phase_max -> duplicate of the `PhaseAnnealing/init_phase_max` log key.
# Filtered at the PUBLISH point (`reset`, which is what turns self.metrics into
# `Metrics/motion/*`), never popped from self.metrics: the parents keep computing
# and writing these — `_update_command` writes `metrics["init_phase_max"][:]` in
# place — so removing the entries would break them. Un-publish, don't un-write.
_DROP_METRICS = (
    "error_anchor_pos", "error_anchor_rot",
    "error_anchor_lin_vel", "error_anchor_ang_vel",
    "error_body_pos", "error_body_rot",
    "error_body_lin_vel", "error_body_ang_vel",
    "error_joint_pos", "error_joint_vel",
    "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin",
    "init_phase_max",
)


class ReposeMotionCommand(ObjectMotionCommand):
    """Cube repose: orientation-only success, floating cube goal marker.

    Tracks the goal's dominating up-face (cube_faces.up_face_idx on the
    clip-end quat): the color-on-top the task actually asks for, read through
    the per-env face->color perm (`error_color_tilt`). A separate face-tilt
    readout would be the SAME number — the perm is a bijection, so
    color_tilt_error(q, perm[f]) == face_tilt_error(q, f) identically."""

    cfg: "ReposeMotionCommandCfg"

    def __init__(self, cfg: "ReposeMotionCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.film_recenter:
            self._recenter_clips()
        self._goal_up_face_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        for name in ("error_color_tilt", "at_goal_color", "at_goal_color_ever"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _recenter_clips(self) -> None:
        """Rigid z-transform per clip: open at the env origin, facing +x.

        An exact symmetry of this task, which is why a filming aid may touch the
        reference at all — flat plane, z gravity, robot-vs-reference rewards, and
        the table is placed FROM the clip goal (`_reset_task`) so it comes along.
        The whole clip moves by ONE transform; shifting frame 0 alone would tear
        the trajectory. What it does change is the SUN's azimuth relative to the
        motion, hence the shadow — which is why it is play-only.
        """
        m = self.motion
        pos = [m.body_pos_w, m.obj_pos]
        quat = [m.body_quat_w, m.obj_quat]
        vec = [m.body_lin_vel_w, m.body_ang_vel_w, m.obj_lin_vel, m.obj_ang_vel]
        # World-frame SMPL channels ride along; `smpl_joints` does NOT — it is the
        # encoder's own frame, and repose reads none of the three either way.
        for arr, dst in ((getattr(m, "smpl_joints_viz", None), pos),
                         (getattr(m, "smpl_root_quat", None), quat)):
            if arr is not None:
                dst.append(arr)

        for i in range(m.n_clips):
            s = slice(int(m.clip_offsets[i]), int(m.clip_ends[i]))
            p0, q0 = m.body_pos_w[s][0, 0], m.body_quat_w[s][0, 0]
            w, x, y, z = q0
            yaw = -torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            c, sn = torch.cos(yaw), torch.sin(yaw)
            o, unit = torch.zeros_like(c), torch.ones_like(c)
            # R.T, for row-vector `v @ rot_t`. z untouched: the floor is honest.
            rot_t = torch.stack([torch.stack([c, sn, o]),
                                 torch.stack([-sn, c, o]),
                                 torch.stack([o, o, unit])])
            shift = p0 * torch.tensor([1.0, 1.0, 0.0], device=self.device)
            qz = torch.stack([torch.cos(yaw / 2), o, o, torch.sin(yaw / 2)])

            for arr in pos:
                arr[s] = (arr[s] - shift) @ rot_t
            for arr in vec:
                arr[s] = arr[s] @ rot_t
            for arr in quat:
                flat = arr[s].reshape(-1, 4)
                arr[s] = quat_mul(qz.expand_as(flat), flat).reshape(arr[s].shape)

        print(f"[repose] film_recenter: {m.n_clips} clips zeroed to xy (0, 0), yaw 0")

    def reset(self, env_ids):
        """Harvest metrics, minus the inherited ones repose does not read."""
        extras = super().reset(env_ids)
        for name in _DROP_METRICS:
            extras.pop(name, None)
        return extras

    def _reset_task(
        self,
        env_ids: torch.Tensor,
        clip_ids: torch.Tensor,
        t: torch.Tensor,
        origins: torch.Tensor,
    ) -> None:
        super()._reset_task(env_ids, clip_ids, t, origins)
        if self.cfg.table_entity_name is None:
            return

        # Each clip ends at a different world-frame XY. Keep one small table
        # per world directly under that world's sampled goal.
        pose = self._object_goal_pos[env_ids].new_zeros((len(env_ids), 7))
        pose[:, :2] = self._object_goal_pos[env_ids, :2] + origins[:, :2]
        pose[:, 2] = self.cfg.table_center_height + origins[:, 2]
        pose[:, 3] = 1.0
        self._env.scene[self.cfg.table_entity_name].write_mocap_pose_to_sim(
            pose, env_ids=env_ids
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self._goal_up_face_idx[env_ids] = up_face_idx(
            self._object_goal_quat[env_ids])

    @property
    def goal_up_face_idx(self) -> torch.Tensor:
        """Goal up-face index -> (B,) long, canonical face order."""
        return self._goal_up_face_idx

    @property
    def goal_color_idx(self) -> torch.Tensor:
        """Goal COLOR idx under the per-env face->color remap -> (B,) long.

        rand_face_colors stashes env._face_color_perm; absent (no recolor
        event) the canonical identity mapping applies (color idx == face idx).
        """
        perm = getattr(self._env, "_face_color_perm", None)
        if perm is None:
            return self._goal_up_face_idx
        return perm.gather(1, self._goal_up_face_idx.unsqueeze(1)).squeeze(1)

    @property
    def current_color_idx(self) -> torch.Tensor:
        """CURRENT up-face COLOR idx (live object ori, perm-remapped) -> (B,) long.

        The aux color target: perm is per-env and unreadable from state/proprio,
        so predicting this forces the extractor through the pixels.
        """
        face = up_face_idx(self.object.data.root_link_quat_w)
        perm = getattr(self._env, "_face_color_perm", None)
        if perm is None:
            return face
        return perm.gather(1, face.unsqueeze(1)).squeeze(1)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        # repose task = reorientation: success ignores position
        self.metrics["at_goal"] = (
            self.metrics["error_object_ori_goal"] < self.cfg.success_ori_threshold
        ).float()
        # THE task truth: min tilt over faces carrying the goal color. Reads
        # (color, perm, live quat) — never a goal quat, so it survives a future
        # non-bijective palette; under today's bijection it equals the face-tilt.
        ctilt = color_tilt_error(
            self.object.data.root_link_quat_w, self.goal_color_idx,
            getattr(self._env, "_face_color_perm", None))
        at_goal_color = (ctilt < self.cfg.success_ori_threshold).float()
        self.metrics["error_color_tilt"] = ctilt
        self.metrics["at_goal_color"] = at_goal_color
        # ANYTIME success. CommandTerm.reset logs mean(metric[env_ids]) AT the
        # reset step, so `at_goal_color` is a TERMINAL-step rate; this running
        # max is its any-frame counterpart (the offline eval's headline number,
        # and the axis on which train and eval ranked runs differently).
        self.metrics["at_goal_color_ever"] = torch.maximum(
            self.metrics["at_goal_color_ever"], at_goal_color)
        self._refresh_goal_color_gui()

    # --- viser GUI: stock motion scrubber + live goal-color readout ---

    def create_gui(self, name, server, get_env_idx,
                   on_change=None, request_action=None) -> None:
        """Motion scrubber (inherited) + goal-color chip for the selected env."""
        super().create_gui(name, server, get_env_idx, on_change, request_action)
        self._gui_color_html = server.gui.add_html("")
        self._gui_get_env_idx = get_env_idx
        self._gui_color_state: tuple[int, int] | None = None
        self._refresh_goal_color_gui()

    def _refresh_goal_color_gui(self) -> None:
        """Cheap per-step refresh (env switch or goal resample changes it)."""
        if getattr(self, "_gui_color_html", None) is None:
            return
        e = int(self._gui_get_env_idx())
        ci = int(self.goal_color_idx[e])
        if self._gui_color_state == (e, ci):
            return
        self._gui_color_state = (e, ci)
        r, g, b = (int(255 * v) for v in face_colors(self.device)[ci, :3])
        self._gui_color_html.content = (
            '<div style="display:flex;align-items:center;gap:8px;padding:2px 10px;">'
            f'<span style="width:14px;height:14px;border-radius:3px;'
            f'background:rgb({r},{g},{b});border:1px solid #8888;"></span>'
            f'<span>goal: <b>{face_color_names()[ci]}</b> face up (env {e})</span>'
            '</div>')

    def _debug_vis_goal(self, visualizer, batch: int) -> None:
        # 2D badge: goal is a color-on-top target, so cube orientation is
        # irrelevant -> vertical square (the up-face color) facing along x so
        # it stays legible in the recording, ringed by a success disc (green
        # once the goal color is up, grey otherwise).
        origin = self._env.scene.env_origins[batch].cpu().numpy()
        goal_center = origin + np.array([0.0, 0.0, self.cfg.viz_height])
        eye = np.eye(3, dtype=np.float64)

        # success disc: thin, co-centered with the square (no x offset -> no
        # side that shows disc-over-color when filmed from behind). gate on
        # at_goal_COLOR (matches the badge = up-face color); at_goal is
        # full-orientation and stays off when the right color is up but spun.
        at_goal = self.metrics["at_goal_color"][batch] > 0.5
        disc_color = (0.15, 0.9, 0.25, 0.3) if at_goal else (0.0, 0.0, 0.0, 0.0)
        visualizer.add_cylinder(
            start=goal_center - np.array([_VIZ_SKIN, 0.0, 0.0]),
            end=goal_center + np.array([_VIZ_SKIN, 0.0, 0.0]),
            radius=_VIZ_HALF * 1.7,
            color=disc_color, label=f"goal_success_{batch}",
        )

        # target square: side-face dims (_VIZ_HALF), thicker in x than the disc
        # so the colored panel protrudes past it on BOTH faces (front & back)
        up_color = FACE_COLORS[int(self.goal_color_idx[batch])]
        visualizer.add_box(
            center=goal_center,
            size=np.array([3.0 * _VIZ_SKIN, _VIZ_HALF, _VIZ_HALF], dtype=np.float64),
            mat=eye, color=up_color,
            label=f"goal_{face_color_names()[int(self.goal_color_idx[batch])]}_up_{batch}",
        )


@dataclass(kw_only=True)
class ReposeMotionCommandCfg(ObjectMotionCommandCfg):
    """ObjectMotionCommandCfg + repose viz/success specialization."""

    success_ori_threshold: float = 0.25
    cube_half_extent: float = BIG_CUBE_HALF_EXTENT
    viz_height: float = 1.5
    """Height above env origin to place the goal-cube marker."""
    table_entity_name: str | None = None
    table_center_height: float = TABLE_CENTER_HEIGHT
    film_recenter: bool = False
    """Rigid-transform every clip to open at the env origin facing +x. PLAY ONLY.

    A fixed camera cannot hold a shot when 30 of the 78 BigCubeFloor clips open
    3.6 m out and up to 88 deg off-axis — and it is the same 30 on both counts,
    so zeroing xy alone would leave them framed sideways.
    """

    def build(self, env) -> ReposeMotionCommand:
        return ReposeMotionCommand(self, env)
