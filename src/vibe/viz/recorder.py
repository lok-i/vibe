"""Take recorder: MuJoCo-rendered composite clips, one mp4 per episode.

Films with ``mjlab.viewer.OffscreenRenderer`` (native quality, any resolution),
not the viser canvas — viser is the viewfinder. Frames are captured in the STEP
path, so ``fps = 1/step_dt`` is exact however slowly the render runs.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from dataclasses import asdict
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
import viser
from mjlab.viewer.offscreen_renderer import OffscreenRenderer

from vibe.viz.attention import upsample
from vibe.viz.director import CameraDirector, ShotPose
from vibe.viz.session import play_context, resolve_out_root

_RES = {"1920x1080": (1920, 1080), "1280x720": (1280, 720), "2560x1440": (2560, 1440)}
_MARGIN, _GAP, _BORDER = 12, 14, 2
_EPISODE = re.compile(r"episode_(\d+)\.mp4")


def _bordered(img: np.ndarray) -> np.ndarray:
    return np.pad(img, ((_BORDER,) * 2, (_BORDER,) * 2, (0, 0)), constant_values=235)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Caption strip under an inset. No-ops if PIL is unavailable."""
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        return img
    h = max(14, img.shape[0] // 8)
    strip = Image.new("RGB", (img.shape[1], h), (18, 18, 18))
    draw = ImageDraw.Draw(strip)
    draw.text((4, max(0, (h - 11) // 2)), text, fill=(235, 235, 235))
    return np.concatenate([img, np.asarray(strip, dtype=np.uint8)], axis=0)


class TakeRecorder:
    """Record button + composite pipeline. One mp4 per episode of the selected env."""

    def __init__(self, server: viser.ViserServer, env, director: CameraDirector,
                 viser_scene, fpv_fn, attn_fn, reset_fn=None, meta_fn=None,
                 paused_fn=None) -> None:
        self._server, self._env, self._director = server, env, director
        self._scene = viser_scene
        self._fpv_fn, self._attn_fn = fpv_fn, attn_fn
        self._reset_fn, self._meta_fn, self._paused_fn = reset_fn, meta_fn, paused_fn
        self._fps = int(round(1.0 / env.unwrapped.step_dt))

        self._renderer: OffscreenRenderer | None = None
        self._rcfg = None
        self._pose: ShotPose | None = None
        self._lookat_ema: np.ndarray | None = None
        self._writers: dict[str, media.VideoWriter] = {}
        self._recording = self._armed = False
        self._episode = 0
        self._clips = 0  # this take's, not the folder's
        self._frames = 0
        self._active: tuple[str, ...] = ()
        self._out: Path | None = None
        self._pending_build: tuple[int, int] | None = None
        self._pose_res = (1920, 1080)  # snapshotted at arm: the writers and the
        self._stale = False            # renderer must agree even if the knob moves
        self._lock = threading.Lock()

        out, why = resolve_out_root()
        with server.gui.add_folder("Record"):
            g = server.gui
            self._take = g.add_text("take", initial_value="",
                                    hint="Optional subfolder — one per angle you try.")
            self._res = g.add_dropdown("resolution", options=tuple(_RES), initial_value="1920x1080")
            self._inset = g.add_slider("inset %", min=0, max=45, step=1, initial_value=19)
            self._corner = g.add_dropdown(
                "inset corner", options=("top-left", "top-right", "bottom-left", "bottom-right"),
                initial_value="top-left", hint="Keep the insets off whatever the robot does.")
            self._panes = g.add_checkbox("export panes", initial_value=False,
                                         hint="Also write fpv/attn as separate synced mp4s.")
            self._debug_vis = g.add_checkbox("debug viz in film", initial_value=True)
            self._reset_on_rec = g.add_checkbox(
                "reset on record", initial_value=True,
                hint="Start the take from the top. Off = wait for the running episode to end.")
            self._rec_btn = g.add_button("● Record", color="red", icon=viser.Icon.PLAYER_RECORD)
            self._stop_btn = g.add_button("■ Stop", icon=viser.Icon.PLAYER_STOP, visible=False)
            self._status = g.add_markdown("")
        self._rec_btn.on_click(lambda _: self.arm())
        self._stop_btn.on_click(lambda _: self.stop())
        self._report(f"idle · out `{out}` ({why})")

    # Control.

    def arm(self) -> None:
        """Snapshot the shot. Runs on the viser CALLBACK thread, so it touches no GL:
        the renderer and the writers are built in ``on_step`` (the loop thread) —
        a MuJoCo GL context is bound to the thread that created it."""
        self._pose = self._director.pose()
        self._lookat_ema = None
        w, h = _RES[self._res.value]
        self._pending_build = self._pose_res = (w, h)
        self._out = self._session_dir()
        self._out.mkdir(parents=True, exist_ok=True)
        self._episode = self._next_index()
        self._clips = 0
        # Freeze the pane set for the whole take: a panel that appears mid-clip
        # would otherwise grow an inset halfway through the film.
        idx = self._env_idx()
        self._active = tuple(n for n, f in (("fpv", self._fpv_fn), ("attn", self._attn_fn))
                             if f(idx) is not None)
        self._armed, self._recording = True, False
        self._rec_btn.visible, self._stop_btn.visible = False, True
        task, ckpt, agent = play_context()
        meta = {"pose": asdict(self._pose), "res": [w, h], "fps": self._fps,
                "env_idx": self._env_idx(), "task": task, "checkpoint": ckpt, "agent": agent,
                "panes": list(self._active)}
        if self._meta_fn is not None:
            meta.update(self._meta_fn())
        (self._out / "take.json").write_text(json.dumps(meta, indent=2))
        if self._at_episode_start():
            self._report(f"**armed** · `{self._out}`{self._roll_hint()}")
        elif self._reset_on_rec.value and self._reset_fn is not None:
            self._reset_fn()  # queued action, drained on the main loop thread
            self._report(f"**armed** · reset requested, take starts clean{self._roll_hint()}")
        else:
            self._report(f"**armed** · waiting for this episode to end{self._roll_hint()}")

    def stop(self) -> None:
        """Callback-thread safe: the writers are lock-guarded (so the clip finalizes
        now), the renderer is only FLAGGED (its GL context is not ours to close)."""
        with self._lock:
            self._recording = False
            self._close_writers()
        self._armed = False
        self._pending_build = None
        self._stale = True
        self._rec_btn.visible, self._stop_btn.visible = True, False
        self._report(f"stopped · {self._clips} clip(s) in `{self._out}`")

    def _roll_hint(self) -> str:
        return " · **press ▶ to roll**" if (self._paused_fn and self._paused_fn()) else ""

    # Step hook — called once per env step, never from the render loop. Everything
    # here (GL, ffmpeg pipes) stays on this one thread.

    def on_step(self) -> None:
        if self._stale:
            self._release()
        if not self._armed:
            return
        try:
            if self._pending_build is not None:
                self._build_renderer(*self._pending_build)
                self._pending_build = None
            if not self._recording:
                if not self._at_episode_start():
                    return
                self._begin_episode()
            self._write_frame()
            if bool(self._env.unwrapped.reset_buf[self._env_idx()].item()):
                self._end_episode()
        except Exception as exc:  # noqa: BLE001 — a bad take must not kill the rollout
            import traceback

            traceback.print_exc()
            self.stop()
            hint = (" · no display: relaunch with `MUJOCO_GL=egl`"
                    if isinstance(exc, mujoco.FatalError) else "")
            self._report(f"**recording failed** — {type(exc).__name__}: {exc}{hint}")

    # Frame production.

    def _write_frame(self) -> None:
        assert self._renderer is not None and self._pose is not None
        p, idx = self._pose, self._env_idx()
        lookat = self._director.world_lookat(p, idx)
        if p.damping > 0:
            self._lookat_ema = (lookat if self._lookat_ema is None
                                else p.damping * self._lookat_ema + (1 - p.damping) * lookat)
            lookat = self._lookat_ema
        self._rcfg.env_idx = idx  # type: ignore[union-attr]
        cb = None
        if self._debug_vis.value and hasattr(self._env.unwrapped, "update_visualizers"):
            cb = self._env.unwrapped.update_visualizers
        self._renderer.update(self._env.unwrapped.sim.data, debug_vis_callback=cb,
                              camera=self._director.mjv_camera(p, lookat))
        frame = self._renderer.render()

        fns = {"fpv": self._fpv_fn, "attn": self._attn_fn}
        panes = {n: fns[n](idx) for n in self._active}
        self._composite(frame, panes)
        with self._lock:  # Stop may fire from the GUI thread between any two writes
            if not self._recording:
                return
            self._writers["main"].add_image(frame)
            for name, img in panes.items():
                if img is not None and (w := self._writers.get(name)) is not None:
                    w.add_image(img)
        self._frames += 1
        if self._frames % 25 == 0:
            self._report(f"**● REC** episode {self._episode:02d} · {self._frames} frames")

    def _composite(self, frame: np.ndarray, panes: dict[str, np.ndarray | None]) -> None:
        """Paste insets into the chosen corner, in place. Nearest upscale — the FPV
        is 112x63 and bilinear would flatter it into something the policy never saw."""
        pct = self._inset.value
        imgs = [(n, im) for n, im in panes.items() if im is not None]
        if pct <= 0 or not imgs:
            return
        target = frame.shape[0] * pct / 100.0
        tiles = [_label(_bordered(upsample(im, max(1, int(round(target / im.shape[0]))))), n)
                 for n, im in imgs]
        strip_w = sum(t.shape[1] for t in tiles) + _GAP * (len(tiles) - 1)
        strip_h = max(t.shape[0] for t in tiles)
        corner = self._corner.value
        fh, fw = frame.shape[:2]
        y = _MARGIN if corner.startswith("top") else fh - _MARGIN - strip_h
        x = _MARGIN if corner.endswith("left") else fw - _MARGIN - strip_w
        for tile in tiles:
            h, w = tile.shape[:2]
            if y < 0 or x < 0 or y + h > fh or x + w > fw:
                continue
            frame[y:y + h, x:x + w] = tile
            x += w + _GAP

    # Episode bookkeeping.

    def _begin_episode(self) -> None:
        assert self._out is not None
        w, h = self._pose_res
        self._writers = {"main": media.VideoWriter(
            self._out / f"episode_{self._episode:02d}.mp4", shape=(h, w), fps=self._fps)}
        self._writers["main"].__enter__()
        if self._panes.value:
            fns = {"fpv": self._fpv_fn, "attn": self._attn_fn}
            for name in self._active:
                if (img := fns[name](self._env_idx())) is None:
                    continue
                wr = media.VideoWriter(
                    self._out / f"episode_{self._episode:02d}.{name}.mp4",
                    shape=img.shape[:2], fps=self._fps)
                wr.__enter__()
                self._writers[name] = wr
        self._recording, self._frames = True, 0
        self._clips += 1
        self._report(f"**● REC** episode {self._episode:02d}")

    def _end_episode(self) -> None:
        with self._lock:
            self._close_writers()
        self._report(f"saved `episode_{self._episode:02d}.mp4` ({self._frames} frames)")
        self._episode += 1
        self._recording = False

    def _close_writers(self) -> None:
        for w in self._writers.values():
            w.close()
        self._writers = {}

    def _at_episode_start(self) -> bool:
        """True within the first two steps of the selected env's episode."""
        return int(self._env.unwrapped.episode_length_buf[self._env_idx()].item()) <= 2

    def _next_index(self) -> int:
        assert self._out is not None
        used = [int(m[1]) for p in self._out.glob("episode_*.mp4")
                if (m := _EPISODE.fullmatch(p.name))]
        return max(used) + 1 if used else 0

    def _session_dir(self) -> Path:
        root, _ = resolve_out_root()
        return root / self._take.value if self._take.value else root

    # Renderer.

    def _build_renderer(self, width: int, height: int) -> None:
        """Fresh renderer per take — it bakes resolution and shadow extent at init."""
        if self._renderer is not None:
            self._renderer.close()
        sim = self._env.unwrapped.sim
        assert self._pose is not None
        cfg = copy.copy(self._env.unwrapped.cfg.viewer)
        cfg.width, cfg.height = width, height
        cfg.distance, cfg.fovy = self._pose.distance, self._pose.fovy
        cfg.origin_type = cfg.OriginType.WORLD
        cfg.max_extra_envs = self._pose.context_envs
        cfg.env_idx = self._env_idx()
        self._rcfg = cfg
        # deepcopy: OffscreenRenderer mutates the model it is handed (offwidth,
        # extent, shadows) and the viser scene is using the live one.
        self._renderer = OffscreenRenderer(
            model=copy.deepcopy(sim.mj_model), cfg=cfg,
            scene=self._env.unwrapped.scene, sim_model=sim.model,
            expanded_fields=sim.expanded_fields,  # geom_dataid → per-world mesh variants
        )
        self._renderer.initialize()

    def _release(self) -> None:
        """Close the renderer from the thread that created its GL context."""
        self._stale = False
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _env_idx(self) -> int:
        return int(getattr(self._scene, "env_idx", 0))

    def _report(self, msg: str) -> None:
        self._status.content = msg

    def cleanup(self) -> None:
        self.stop()
        self._release()
        for h in (self._take, self._res, self._inset, self._corner, self._panes, self._debug_vis,
                  self._reset_on_rec, self._rec_btn, self._stop_btn, self._status):
            h.remove()
