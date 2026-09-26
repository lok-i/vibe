"""Cross-attention mask overlay for the viser play viewer.

The CrossAttentionExtractor caches its last forward's attention
(``cross_attention.LATEST_ATTENTION``, agent side); the head_cam sensor holds the FPV rgb
(env side). This panel reads both each camera frame and paints the (Q, P) attention over the
image: a stride-16 backbone on the head cam gives an (H//16, W//16) grid, one row per query
group, labelled by the extractor's `query_labels`.
"""

from __future__ import annotations

import contextlib
import os
from collections import deque

import numpy as np
import viser
import viser.uplot

from vibe.viz.attention import colorbar as _colorbar
from vibe.viz.attention import colorize as _colorize
from vibe.viz.attention import entropy as _entropy
from vibe.viz.attention import overlay as _overlay
from vibe.viz.attention import patch_grid as _patch_grid
from vibe.viz.attention import select_map as _select_map
from vibe.viz.attention import upsample as _upsample
from vibe.viz.director import CameraDirector
from vibe.viz.recorder import TakeRecorder

_HISTORY = 200  # flatness-trace length (frames)
# Distinct series colors (matplotlib tab-style), goal first.
_SERIES_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


class AttnCameraPanel:
    """Viser GUI panel: attention overlay on the FPV + a per-query montage strip.

    Self-contained — needs only the viser server, the head_cam sensor, and the
    live extractor (for its query layout). Flatness (normalized attention entropy,
    1 = uniform/meaningless → 0 = concentrated) streams to a single live plot so
    the queries can be read against each other on one shared scale.
    """

    def __init__(
        self, server: viser.ViserServer, sensor, extractor, display_scale: int = 3,
        order: float | None = None,
    ) -> None:
        self._server = server
        self._sensor = sensor
        self._tap = extractor
        self._scale = display_scale
        h, w = sensor.cfg.height, sensor.cfg.width
        self._hw = (h, w)
        q = extractor.last_attn.shape[1]
        # row labels come straight from the extractor's query layout (query groups + learned)
        self._labels = list(getattr(extractor, "query_labels", None) or [f"q{i}" for i in range(q)])
        self._last_env_idx = -1
        self._hist: dict[str, deque[float]] = {lbl: deque(maxlen=_HISTORY) for lbl in self._labels}
        self._x = np.arange(-_HISTORY + 1, 1, dtype=np.float64)

        blank = np.zeros((h * display_scale, w * display_scale, 3), dtype=np.uint8)
        with server.gui.add_folder("Attention", order=order):
            self._query = server.gui.add_dropdown(
                "Query", options=(*self._labels, "mean", "max"), initial_value=self._labels[0]
            )
            self._alpha = server.gui.add_slider(
                "Overlay α", min=0.0, max=1.0, step=0.05, initial_value=0.55
            )
            self._overlay_h = server.gui.add_image(image=blank, label="attn_overlay", format="jpeg")
            # Colorbar legend (turbo): low attention → high attention.
            server.gui.add_image(
                image=_upsample(_colorbar(w), display_scale), label="scale", format="png"
            )
            server.gui.add_markdown("`low` — blue · green · yellow — `high`")
            self._montage_caption = server.gui.add_markdown("montage `[…]`")
            self._montage_h = server.gui.add_image(image=blank, label="attn_per_query", format="jpeg")
            self._plot = server.gui.add_uplot(
                data=(self._x, *([np.zeros(_HISTORY)] * len(self._labels))),
                series=(
                    viser.uplot.Series(label="step"),
                    *(
                        viser.uplot.Series(label=lbl, stroke=_SERIES_COLORS[i % len(_SERIES_COLORS)], width=2)
                        for i, lbl in enumerate(self._labels)
                    ),
                ),
                scales={
                    "x": viser.uplot.Scale(time=False, auto=False, range=(-_HISTORY, 0)),
                    "y": viser.uplot.Scale(auto=False, range=(0.0, 1.0)),
                },
                legend=viser.uplot.Legend(show=True),
                title="flatness  (1 = uniform → 0 = focused)",
                aspect=2.0,
            )

    @property
    def query(self) -> str:
        return str(self._query.value)

    def overlay_frame(self, env_idx: int) -> np.ndarray | None:
        """The selected query's overlay at NATIVE camera resolution, for the recorder."""
        attn_t = getattr(self._tap, "last_attn", None)
        rgb = self._sensor.data.rgb
        if attn_t is None or rgb is None or env_idx >= attn_t.shape[0]:
            return None
        attn = attn_t[env_idx].float().cpu().numpy()
        return _overlay(rgb[env_idx].cpu().numpy(), self._select(attn), self._alpha.value)

    def _select(self, attn: np.ndarray) -> np.ndarray:
        """(Q, P) → the chosen (P,) map."""
        return _select_map(attn, self._labels, self._query.value)

    def update(self, extractor, env_idx: int) -> None:
        self._tap = extractor  # kept so `overlay_frame` can paint between GUI ticks
        attn_t = getattr(extractor, "last_attn", None)
        rgb = self._sensor.data.rgb
        if attn_t is None or rgb is None or env_idx >= attn_t.shape[0]:
            return
        if env_idx != self._last_env_idx:  # env switch → drop stale traces
            for h in self._hist.values():
                h.clear()
            self._last_env_idx = env_idx
        attn = attn_t[env_idx].float().cpu().numpy()  # (Q, P)
        rgb_np = rgb[env_idx].cpu().numpy()  # (H, W, 3) uint8
        q, p = attn.shape
        hp, wp = _patch_grid(p, *self._hw)

        # Overlay: selected query α-blended on the FPV.
        blend = _overlay(rgb_np, self._select(attn), self._alpha.value)
        self._overlay_h.image = _upsample(blend, self._scale)

        # Montage: every raw query heatmap in forward order, small gap between.
        gap = np.zeros((self._hw[0], 2, 3), dtype=np.uint8)
        maps = [_colorize(attn[i].reshape(hp, wp), self._hw) for i in range(q)]
        strip = maps[0]
        for m in maps[1:]:
            strip = np.concatenate([strip, gap, m], axis=1)
        self._montage_h.image = _upsample(strip, self._scale)
        self._montage_caption.content = f"montage `{self._labels}` — grid {hp}×{wp}, P={p}"

        # Flatness trace: normalized entropy per query onto the shared-scale plot.
        ent = _entropy(attn)
        for lbl, e in zip(self._labels, ent, strict=False):
            self._hist[lbl].append(float(e))
        n = len(next(iter(self._hist.values())))
        if n:
            ys = [np.fromiter(self._hist[lbl], np.float64, n) for lbl in self._labels]
            self._plot.data = (self._x[-n:], *ys)

    def cleanup(self) -> None:
        for h in (self._query, self._alpha, self._overlay_h, self._montage_caption, self._montage_h, self._plot):
            h.remove()


# ---------------------------------------------------------------------------
# Viewer subclass: builds the panel iff head_cam + a live cross-attn tap exist.
# ---------------------------------------------------------------------------

from mjlab.viewer.viser.viewer import ViserPlayViewer  # noqa: E402


def _find_head_cam(env):
    from mjlab.sensor.camera_sensor import CameraSensor

    for s in env.unwrapped.scene.sensors.values():
        if isinstance(s, CameraSensor) and s.camera_name == "head_cam":
            return s
    return None


class AttnViserPlayViewer(ViserPlayViewer):
    """``ViserPlayViewer`` + attention overlay + camera director + take recorder.

    The attention panel is built lazily on the first policy forward that populates
    the tap: this gates on a live ``CrossAttentionExtractor`` (mlp / non-vision runs
    never build it) and sidesteps setup-vs-first-forward ordering. The director and
    recorder are unconditional — a clip of the robot alone is still a clip.
    """

    def setup(self) -> None:
        super().setup()
        self._attn_panel: AttnCameraPanel | None = None
        self._attn_sensor = _find_head_cam(self.env)
        self._director = CameraDirector(
            self._server, self.env, self._scene, self.env.unwrapped.sim.mj_model
        )
        self._recorder = TakeRecorder(
            self._server, self.env, self._director, self._scene,
            fpv_fn=self._fpv_frame, attn_fn=self._attn_frame, reset_fn=self.request_reset,
            meta_fn=lambda: {"attn_query": self._attn_panel.query if self._attn_panel else None},
            paused_fn=lambda: self._is_paused,
        )
        if os.environ.get("VIBE_PAUSED") or os.environ.get("VIBE_REC"):  # VIBE_REC: old name
            self.pause()  # set the shot before a single step is taken

    # Frame taps for the recorder: whatever the panels are showing, as numpy.

    def _fpv_frame(self, env_idx: int) -> np.ndarray | None:
        if self._attn_sensor is None or (rgb := self._attn_sensor.data.rgb) is None:
            return None
        return rgb[env_idx].cpu().numpy()

    def _attn_frame(self, env_idx: int) -> np.ndarray | None:
        if self._attn_panel is None:
            return None
        return self._attn_panel.overlay_frame(env_idx)

    def _execute_step(self) -> bool:
        """Capture in the STEP path — ``tick()`` renders at its own rate and would
        drop or duplicate steps, time-warping the clip."""
        ok = super()._execute_step()
        if ok:
            self._recorder.on_step()
        return ok

    def _update_camera_feeds(self, sim, has_pending_updates: bool) -> None:
        super()._update_camera_feeds(sim, has_pending_updates)
        if self._attn_sensor is None or not self._should_update_cameras(
            self._is_paused, has_pending_updates
        ):
            return
        from rsl_rl.modules import cross_attention

        tap = cross_attention.LATEST_ATTENTION
        if tap is None or tap.last_attn is None:
            return
        if self._attn_panel is None:
            tab, order = self._above_info()
            with tab:
                self._attn_panel = AttnCameraPanel(self._server, self._attn_sensor, tap, order=order)
        self._attn_panel.update(tap, self._scene.env_idx)

    def _above_info(self):
        """(Controls tab, an order just above its Info folder), so Attention is the tab's
        first box. mjlab keeps no handle on either: both are reached from its Info
        html through viser's container registry. Falls back to the root, last."""
        try:
            gui = self._server.gui
            info = gui._container_handle_from_uuid[self._status_html._impl.parent_container_id]
            tab = gui._container_handle_from_uuid[info._impl.parent_container_id]
            return tab, info.order - 0.5
        except (AttributeError, KeyError):
            return contextlib.nullcontext(), None

    def close(self) -> None:
        self._recorder.cleanup()
        super().close()
