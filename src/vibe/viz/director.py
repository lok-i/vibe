"""Free-camera director: viser knobs bound both ways to the client view.

``position = lookat + d·[-cosE·cosA, -cosE·sinA, sinE]`` is mjviser's own mapping
(``mjviser.scene.create_scene_gui``), and it inverts exactly — so a pose set by
mouse, refined by knob, and filmed by MuJoCo is one pose in three places.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import mujoco
import numpy as np
import viser

from vibe.viz.session import pose_dir

_TRACK_FREE = "free"


def _read_pose(path: Path) -> "ShotPose":
    """Tolerant of pose files written by another version — unknown keys drop,
    missing ones take the default, so an old shot never hard-fails a session."""
    data = json.loads(path.read_text())
    names = {f.name for f in fields(ShotPose)}
    return ShotPose(**{k: v for k, v in data.items() if k in names})


@dataclass
class ShotPose:
    """A filmable camera pose. ``lookat`` is world when free, body-relative when tracking."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.8)
    distance: float = 4.0
    azimuth: float = 90.0
    elevation: float = -15.0
    fovy: float = 45.0
    track: str = _TRACK_FREE
    damping: float = 0.0
    context_envs: int = 0

    def offset(self) -> np.ndarray:
        """Camera position minus lookat, from (distance, azimuth, elevation)."""
        a, e = np.deg2rad(self.azimuth), np.deg2rad(self.elevation)
        return self.distance * np.array(
            [-np.cos(e) * np.cos(a), -np.cos(e) * np.sin(a), np.sin(e)]
        )

    @staticmethod
    def from_offset(lookat: np.ndarray, offset: np.ndarray) -> tuple[float, float, float]:
        """(distance, azimuth, elevation) from a camera offset — inverse of ``offset``."""
        del lookat
        d = float(np.linalg.norm(offset))
        if d < 1e-6:
            return 1e-6, 90.0, 0.0
        elev = float(np.rad2deg(np.arcsin(np.clip(offset[2] / d, -1.0, 1.0))))
        azim = float(np.rad2deg(np.arctan2(-offset[1], -offset[0])))
        return d, azim, elev


def mjv_camera(model: mujoco.MjModel, p: "ShotPose", lookat: np.ndarray) -> mujoco.MjvCamera:
    """A filmable MuJoCo camera from a ShotPose.

    The knobs are in VISER's convention; MuJoCo's elevation is its negation. Verified
    against ``mjv_updateScene``: for the same (lookat, d, azimuth) the two camera
    offsets agree in x and y and differ in z by a sign. Feeding the knob straight
    through puts the camera under the floor, looking up.
    """
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    cam.fixedcamid, cam.trackbodyid = -1, -1
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = p.distance, p.azimuth, -p.elevation
    return cam


@dataclass
class _Handles:
    """GUI handles, kept only so ``pose()`` can read them back."""

    items: dict = field(default_factory=dict)


class CameraDirector:
    """Camera knobs + view sync. Owns the pose the recorder films with."""

    def __init__(self, server: viser.ViserServer, env, viser_scene, mj_model: mujoco.MjModel) -> None:
        self._server = server
        self._env = env
        self._scene = viser_scene
        self._model = mj_model
        self._h = _Handles()
        p = ShotPose()

        with server.gui.add_folder("Director"):
            g = server.gui
            self._h.items = {
                "lookat": g.add_vector3("lookat", initial_value=p.lookat, step=0.05),
                "distance": g.add_slider("distance", min=0.5, max=40.0, step=0.05,
                                         initial_value=p.distance),
                "azimuth": g.add_slider("azimuth", min=-180.0, max=180.0, step=0.5,
                                        initial_value=p.azimuth),
                "elevation": g.add_slider("elevation", min=-89.0, max=89.0, step=0.5,
                                          initial_value=p.elevation),
                "fovy": g.add_slider("fovy", min=15.0, max=110.0, step=1.0,
                                     initial_value=p.fovy),
                "track": g.add_dropdown("track", options=self._track_options(),
                                        initial_value=p.track,
                                        hint="Entity the SHOT follows — and the viewfinder "
                                             "with it, so preview IS the take. Drive tracking "
                                             "from HERE, not the Scene panel's `Track camera`."),
                "damping": g.add_slider("damping", min=0.0, max=0.98, step=0.02,
                                        initial_value=p.damping,
                                        hint="EMA on the tracked lookat — MuJoCo's tracking "
                                             "camera is rigid and a jumping robot shakes."),
                "context_envs": g.add_slider("context envs", min=0, max=8, step=1,
                                             initial_value=p.context_envs,
                                             hint="Neighbouring envs drawn into the SAME frame — "
                                                  "this is how several objects/motions get into "
                                                  "one clip. 4 = the 5-env overview."),
            }
            sync = g.add_button("Sync from view", icon=viser.Icon.CAMERA_DOWN)
            push = g.add_button("Push to view", icon=viser.Icon.CAMERA_UP)
            # Task-scoped by default: one angle serves every checkpoint of a task,
            # so the pose must NOT follow the clips into a per-checkpoint folder.
            self._pose_file = g.add_text(
                "pose file", initial_value=str(pose_dir() / "pose.json"),
                hint="Relative paths are fine. Rename per shot (sagittal.json, overview.json).")
            save = g.add_button("Save pose", icon=viser.Icon.DEVICE_FLOPPY)
            load = g.add_button("Load pose", icon=viser.Icon.FOLDER_OPEN)
            self._status = g.add_markdown("")

        sync.on_click(lambda _: self.sync_from_view())
        push.on_click(lambda _: self.push_to_view())
        save.on_click(lambda _: self._save())
        load.on_click(lambda _: self._load())
        for name in ("fovy", "distance", "azimuth", "elevation", "lookat"):
            self._h.items[name].on_update(lambda _: self.push_to_view())
        self._h.items["track"].on_update(lambda _: self._bind_viewfinder())
        self._report("knobs live · mouse the view, then `Sync from view`")
        self._autoload()
        self._bind_viewfinder()

    # Pose I/O.

    def pose(self) -> ShotPose:
        v = {k: h.value for k, h in self._h.items.items()}
        return ShotPose(
            lookat=tuple(float(x) for x in v["lookat"]),  # type: ignore[arg-type]
            distance=float(v["distance"]), azimuth=float(v["azimuth"]),
            elevation=float(v["elevation"]), fovy=float(v["fovy"]),
            track=str(v["track"]), damping=float(v["damping"]),
            context_envs=int(v["context_envs"]),
        )

    def set_pose(self, p: ShotPose) -> None:
        for k, val in asdict(p).items():
            h = self._h.items.get(k)
            if h is None:
                continue
            h.value = tuple(val) if k == "lookat" else val
        self._bind_viewfinder()

    # View sync. `_scene_offset` is -tracked_pos when viser tracking is on
    # (mjviser.scene), so viser coords are world + offset in both directions.

    def sync_from_view(self) -> None:
        clients = list(self._server.get_clients().values())
        if not clients:
            self._report("**no client connected** — open the viewer first")
            return
        cam = clients[0].camera
        off = self._scene_offset()
        lookat = np.asarray(cam.look_at, dtype=float) - off
        d, az, el = ShotPose.from_offset(lookat, np.asarray(cam.position, dtype=float) - off - lookat)
        h = self._h.items
        h["lookat"].value = tuple(float(x) for x in self._to_track_frame(lookat))
        h["distance"].value, h["azimuth"].value, h["elevation"].value = d, az, el
        h["fovy"].value = float(np.rad2deg(cam.fov))
        self._report(f"synced · d={d:.2f} az={az:.1f} el={el:.1f}")

    def push_to_view(self) -> None:
        p = self.pose()
        lookat = self._to_world(np.asarray(p.lookat, dtype=float)) + self._scene_offset()
        for client in self._server.get_clients().values():
            client.camera.position = lookat + p.offset()
            client.camera.look_at = lookat
            client.camera.fov = float(np.deg2rad(p.fovy))

    # World lookat for a pose, resolved against the tracked body this frame.

    def world_lookat(self, p: ShotPose, env_idx: int) -> np.ndarray:
        base = np.asarray(p.lookat, dtype=float)
        if p.track == _TRACK_FREE:
            return base
        return base + self._track_pos(p.track, env_idx)

    def mjv_camera(self, p: ShotPose, lookat: np.ndarray) -> mujoco.MjvCamera:
        return mjv_camera(self._model, p, lookat)

    # Internals.

    def _track_options(self) -> tuple[str, ...]:
        """Scene entity roots — tracking a whole entity, never a raw body id."""
        return (_TRACK_FREE, *self._env.unwrapped.scene.entities.keys())

    def _track_pos(self, name: str, env_idx: int) -> np.ndarray:
        entity = self._env.unwrapped.scene.entities.get(name)
        if entity is None or not hasattr(entity.data, "root_link_pos_w"):
            return np.zeros(3)  # e.g. terrain — a fixed entity has no root link
        return entity.data.root_link_pos_w[env_idx].detach().cpu().numpy().astype(float)

    def _bind_viewfinder(self) -> None:
        """Point mjviser's scene tracking at the entity we FILM, so what you frame is
        what gets rendered. mjviser picks the first non-fixed body — the OBJECT in
        every vibe scene — and never offers a way to change it.

        It offsets the SCENE by ``-tracked_pos`` rather than moving the camera, which
        is the recorder's geometry exactly: ``push_to_view`` lands on a constant
        ``lookat`` (``(base + track_pos) - track_pos``) while the world slides under a
        camera held still. Both sides track POSITION only, so nothing is left over.
        """
        ent = self._env.unwrapped.scene.entities.get(str(self._h.items["track"].value))
        # Same guard as `_track_pos`: a fixed entity (terrain) has no root link.
        bid = (ent.indexing.root_body_id
               if ent is not None and hasattr(ent.data, "root_link_pos_w") else None)

        def _mutate() -> None:
            self._scene._tracked_body_id = bid
            self._scene.camera_tracking_enabled = bid is not None

        apply = getattr(self._scene, "_apply_visualization_change", None)
        _mutate() if apply is None else apply(_mutate)
        self.push_to_view()

    def _env_idx(self) -> int:
        return int(getattr(self._scene, "env_idx", 0))

    def _to_track_frame(self, world_lookat: np.ndarray) -> np.ndarray:
        """World lookat → the stored frame (relative when an entity is tracked)."""
        p = self.pose()
        if p.track == _TRACK_FREE:
            return world_lookat
        return world_lookat - self._track_pos(p.track, self._env_idx())

    def _to_world(self, lookat: np.ndarray) -> np.ndarray:
        p = self.pose()
        if p.track == _TRACK_FREE:
            return lookat
        return lookat + self._track_pos(p.track, self._env_idx())

    def _scene_offset(self) -> np.ndarray:
        return np.asarray(getattr(self._scene, "_scene_offset", np.zeros(3)), dtype=float)

    def _save(self) -> None:
        path = Path(self._pose_file.value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.pose()), indent=2))
        self._report(f"saved `{path}`")

    def _load(self) -> None:
        path = Path(self._pose_file.value).expanduser()
        if not path.exists():
            self._report(f"**missing** `{path}`")
            return
        self.set_pose(_read_pose(path))
        self._report(f"loaded `{path}`")

    def _autoload(self) -> None:
        """A task opens on its own saved view. The knobs take it immediately; the
        3D view can't (no client yet at setup), so late joiners get pushed on connect."""
        path = Path(self._pose_file.value).expanduser()
        if not path.exists():
            return
        try:
            self.set_pose(_read_pose(path))
        except Exception as exc:  # noqa: BLE001 — a stale pose must not cost the viewer
            self._report(f"**ignored** `{path}` — {type(exc).__name__}: {exc}")
            return
        self._report(f"loaded this task's `{path.name}` · `Save pose` overwrites it")

        @self._server.on_client_connect
        def _(_client) -> None:
            self.push_to_view()

    def _report(self, msg: str) -> None:
        self._status.content = msg

    def cleanup(self) -> None:
        for h in (*self._h.items.values(), self._pose_file, self._status):
            h.remove()
