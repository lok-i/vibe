"""Every patch vibe applies to mjlab, at `import vibe`. Idempotent; no mjlab fork.

| patch | why | inert when |
|---|---|---|
| `MotionCommandCfg` sentinel in train/play | both scripts treat any `MotionCommandCfg` as single-file tracking (train passes `registry_name`, play demands `--motion-file`); the multi-clip cfgs load their own dataset | a genuine single-file tracking task |
| `_patch_play_init_agent` | `--agent auto`, `initial`, `release` on play | `--agent zero`, `random`, `trained` |
| `_patch_play_attn_viewer` | viser play gets the attention overlay, camera director and recorder | no head cam / no extractor (panel not built) |
| `_patch_viser_dark_mode` | viser starts dark | — |
| `_muffle_mesh_support_warning` | drops host-side GJK "mesh_support" spam on variant scenes (physics runs on mujoco_warp) | every other MuJoCo warning prints |
| `_patch_put_data_nccdmax` | caps mujoco_warp's CCD scratch per world (`VIBE_NCCDMAX`, default 64) | the caller passes `nccdmax` itself |
| `_patch_inductor_tf32` | one tf32 API, so Inductor does not crash on the mix | without `--agent.torch-compile-mode` (besides the precision call) |
| `_patch_amp_env_override` | `VIBE_AMP=bfloat16` sets the agent amp dtype, play included | `VIBE_AMP` unset |
| `_patch_variant_scene_indexing` | a camera on a variant scene left every entity at body id -1 | the scene spec is not stale |

The script patches are ONE global each and orcs applies its own at `import orcs`; vibe patches
last, so it must cover the union (`orcs.MULTI_CLIP_CFGS`, the parent's `--agent` choices).
"""

from __future__ import annotations

import importlib
from typing import Literal, get_args, get_type_hints

import orcs
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

from vibe.tasks.repose.mdp.commands import ReposeMotionCommandCfg

# Scripts that gate on isinstance(cmd, MotionCommandCfg).
_PATCHED_SCRIPTS = ("mjlab.scripts.train", "mjlab.scripts.play")

# Every multi-clip cfg reachable in this process. THE UNION MATTERS: orcs's own
# `apply()` already ran at `import orcs` with only ITS cfgs, and the sentinel is
# ONE global on the mjlab scripts — so whoever patches last decides for all of
# them. vibe imports orcs, therefore vibe patches last, therefore vibe must
# exempt orcs's cfgs too or `play Orcs-*` breaks inside a vibe env.
#
# Taken from `orcs.MULTI_CLIP_CFGS`, never respelled: this list was already one
# task stale (perloco's TerrainMotionCommandCfg was missing) and nothing said
# so. (ReposeMotionCommandCfg subclasses ObjectMotionCommandCfg, so it is
# already covered — named anyway, because relying on that is one refactor from
# silent.)
_MULTI_CLIP_CFGS = (*orcs.MULTI_CLIP_CFGS, ReposeMotionCommandCfg)


class _SingleFileMotionMeta(type):
    """``isinstance`` is True only for *single-file* tracking cfgs."""

    def __instancecheck__(cls, obj: object) -> bool:
        return isinstance(obj, MotionCommandCfg) and not isinstance(
            obj, _MULTI_CLIP_CFGS
        )


class _SingleFileMotionCfg(metaclass=_SingleFileMotionMeta):
    """Drop-in for the scripts' ``MotionCommandCfg`` isinstance target."""


_APPLIED = False


def apply() -> None:
    """Patch mjlab in place. Idempotent — orcs patches three of the same targets
    at ITS import, and wrapping an already-wrapped function stacks silently."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    for name in _PATCHED_SCRIPTS:
        mod = importlib.import_module(name)
        mod.MotionCommandCfg = _SingleFileMotionCfg  # type: ignore[attr-defined]
    _patch_play_init_agent()
    _patch_play_attn_viewer()
    _patch_viser_dark_mode()
    _muffle_mesh_support_warning()
    _patch_put_data_nccdmax()
    _patch_inductor_tf32()
    _patch_amp_env_override()
    _patch_variant_scene_indexing()


def _patch_variant_scene_indexing() -> None:
    """A camera on a VARIANT scene silently aliases every entity to body -1.

    mjSpec assigns element ids at COMPILE and invalidates them on the next
    structural edit. `Scene.__init__` adds sensors AFTER attaching entities, so
    a camera leaves the scene spec dirty (`b.id == -1` for every body). The
    non-variant path recompiles that same spec inside `Simulation`, which
    restores them — but `build_variant_model` compiles a `spec.copy()`, so the
    scene's own spec stays dirty and `Entity._compute_indexing` reads -1 into
    `body_ids`. `data.xpos[:, -1]` is the LAST body, so EVERY entity reports the
    same pose: on `Vibe-Uolm-*` the object's pose became the robot's, in the
    obs, the rewards and the terminations at once, with no error anywhere.

    Fix: compile the scene spec before indexing is computed, and only when it is
    actually stale (one extra MjModel build per env construction, never on the
    healthy path). Measured: object body id -1 -> 2, the same id the camera-free
    scene gets. mjlab's to fix properly (compile the original, not a copy);
    until then this is the fix, and it is inert for every non-variant task.
    """
    import functools

    from mjlab.scene import Scene

    if getattr(Scene.initialize, "_vibe_variant_ids", False):
        return
    _orig = Scene.initialize

    @functools.wraps(_orig)
    def initialize(self, *args, **kwargs):
        stale = any(b.id < 0 for e in self.entities.values()
                    for b in e.spec.bodies[1:])
        if stale:
            self.spec.compile()
        return _orig(self, *args, **kwargs)

    initialize._vibe_variant_ids = True  # type: ignore[attr-defined]
    Scene.initialize = initialize


def _patch_amp_env_override() -> None:
    """``VIBE_AMP=bfloat16`` sets the agent's amp dtype on every entry point.

    ``play`` discards every agent-cfg override (``scripts/play.py``:
    ``del remaining_args, agent_cfg``), so ``--agent.amp-dtype`` reaches ``train`` but
    never ``play`` — which is exactly where you want to watch what reduced precision
    does to a frozen WBC. One env var covers play, train and the bench::

        VIBE_AMP=bfloat16 play <task> --viewer native --agent initial

    Rebound on the importing modules too: the scripts do ``from ... import
    load_rl_cfg``, so patching only the registry would miss them.
    """
    import os

    dtype = os.environ.get("VIBE_AMP") or None
    if dtype is None:
        return

    import functools

    import mjlab.tasks.registry as _registry

    _orig = _registry.load_rl_cfg

    @functools.wraps(_orig)
    def load_rl_cfg(*args, **kwargs):
        cfg = _orig(*args, **kwargs)
        if not hasattr(cfg, "amp_dtype"):
            raise TypeError(
                f"VIBE_AMP={dtype} but {type(cfg).__name__} has no `amp_dtype` field "
                "(only VibeRunnerCfg declares it)."
            )
        cfg.amp_dtype = dtype
        print(f"[vibe] VIBE_AMP: agent amp_dtype = {dtype}")
        return cfg

    _registry.load_rl_cfg = load_rl_cfg
    for name in ("mjlab.scripts.play", "mjlab.scripts.train"):
        try:
            importlib.import_module(name).load_rl_cfg = load_rl_cfg  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            print(f"[vibe] VIBE_AMP not applied to {name}: {exc}")


def _patch_inductor_tf32() -> None:
    """Keep the two torch tf32 APIs consistent — Inductor crashes on the mix.

    mjlab's `configure_torch_backends` sets only the NEW api
    (`torch.backends.cuda.matmul.fp32_precision`), while Inductor passes still read
    the LEGACY `allow_tf32` getter; torch 2.12 tracks which was used and raises
    "checking the matmul precision without a specific backend name" during codegen
    of the first compiled forward. Two halves, because disabling `shape_padding`
    alone was not enough: also swap in the UNIFIED
    `torch.set_float32_matmul_precision`, which sets both.

    The unified precision call is live behavior on every run; the rest is inert
    without `--agent.torch-compile-mode`. Note it does NOT clear the autotune-path
    crash — only `mode="default"` does.
    """
    import torch

    try:
        import torch._inductor.config as ic

        ic.shape_padding = False
    except Exception as exc:  # noqa: BLE001
        print(f"[vibe] inductor shape_padding not disabled: {exc}")

    def _configure(allow_tf32: bool = True, deterministic: bool = False) -> None:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    import mjlab.utils.torch as _mut

    _mut.configure_torch_backends = _configure
    for name in ("mjlab.scripts.train", "mjlab.scripts.play"):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "configure_torch_backends"):
                mod.configure_torch_backends = _configure
        except Exception as exc:  # noqa: BLE001
            print(f"[vibe] tf32 patch skipped for {name}: {exc}")


def _patch_viser_dark_mode() -> None:
    """Viser viewer starts in dark mode (spares the per-run Dev Settings tick).

    Wraps ``ViserPlayViewer.__init__`` (covers subclasses, e.g. the attention
    viewer) to call viser's ``configure_theme(dark_mode=True)`` right after the
    server exists; mjlab never configures a theme, so nothing is overridden.
    """
    import functools

    try:
        from mjlab.viewer.viser.viewer import ViserPlayViewer
    except Exception as exc:  # noqa: BLE001
        print(f"[vibe] viser dark-mode patch unavailable ({exc})")
        return
    if getattr(ViserPlayViewer, "_vibe_dark_mode", False):
        return
    ViserPlayViewer._vibe_dark_mode = True
    _orig_init = ViserPlayViewer.__init__

    @functools.wraps(_orig_init)
    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            self._server.gui.configure_theme(dark_mode=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[vibe] dark mode not applied: {exc}")

    ViserPlayViewer.__init__ = __init__


def _patch_play_attn_viewer() -> None:
    """Swap play's ``ViserPlayViewer`` for the cross-attention overlay subclass.

    Inert everywhere it doesn't apply: the panel builds only when a head_cam
    sensor AND a live ``CrossAttentionExtractor`` tap both exist (see
    ``vibe.viz.AttnViserPlayViewer.setup``). A failed viz import (viser /
    matplotlib) never breaks play — we simply keep the stock viewer.
    """
    play = importlib.import_module("mjlab.scripts.play")
    if getattr(play, "_vibe_attn_viewer", False):
        return
    try:
        from vibe.viz import AttnViserPlayViewer
    except Exception as exc:  # noqa: BLE001
        print(f"[vibe] attention overlay unavailable ({exc}); using stock viewer.")
        return
    play._vibe_attn_viewer = True
    play.ViserPlayViewer = AttnViserPlayViewer


def _patch_put_data_nccdmax() -> None:
    """Bound mujoco_warp's CCD workspace — the VRAM whale for mesh scenes.

    mjwarp sizes its GJK/EPA + multiccd scratch arrays by ``naccdmax`` rows
    (~6.8 KB/row at ccd_iterations=50), and ``naccdmax`` silently defaults to
    ``nconmax * nworld`` because mjlab's ``Simulation._finish_init`` never
    passes ``nccdmax``. Only mesh-pair candidate contacts consume CCD rows,
    so for the omni-object scenes (nconmax=150+, 12k worlds) the default
    allocates tens of GB of workspace that box/capsule contacts never touch.

    Wrap ``put_data`` to cap CCD rows per world at ``VIBE_NCCDMAX`` (default
    64, clamped to nconmax). Undershoot is loud, not silent: mjwarp printf's
    "CCD overflow - please increase naccdmax" to stderr and drops the
    contact — watch run logs and raise the env var if it appears.
    """
    import functools
    import os

    import mujoco_warp as mjwarp

    if getattr(mjwarp.put_data, "_vibe_nccdmax", False):
        return
    _orig = mjwarp.put_data

    @functools.wraps(_orig)
    def put_data(*args, **kwargs):
        nconmax = kwargs.get("nconmax")
        if (nconmax is not None
                and kwargs.get("nccdmax") is None
                and kwargs.get("naccdmax") is None):
            nccdmax = int(os.environ.get("VIBE_NCCDMAX", "64"))
            kwargs["nccdmax"] = min(nccdmax, nconmax)
        return _orig(*args, **kwargs)

    put_data._vibe_nccdmax = True  # type: ignore[attr-defined]
    mjwarp.put_data = put_data


def _muffle_mesh_support_warning() -> None:
    """Silence libmujoco's "mesh_support could not find support vertex" spam.

    Emitted by the HOST model's GJK for the omni-object variant scenes — padded
    mesh slots give degenerate support queries. Physics runs on mujoco_warp, so
    the host warning is cosmetic. Every OTHER MuJoCo warning still prints.
    """
    import mujoco

    def _warn(msg: str) -> None:
        if "mesh_support could not find support vertex" in msg:
            return
        print(f"WARNING: {msg}")

    mujoco.set_mju_user_warning(_warn)


def _patch_play_init_agent() -> None:
    """Add ``--agent {auto,initial,release}`` to mjlab's play script (no core edits, no fork).

    "initial" = instantiate the task's ACTUAL agent (actor cfg, base_checkpoint
    and all) but load NO training checkpoint — the freshly constructed policy
    is rolled out. For an adapter agent this is the frozen base bit-exact
    (zero-init LoRA). Unlike ``--agent zero`` (zero ACTIONS), this exercises the
    real model stack: obs plumbing, base-ckpt load, normalizers, action heads.

    Mechanics: main() resolves ``PlayConfig`` and ``run_play`` as module
    globals at call time, so swapping both on the module is enough. tyro
    picks up the widened Literal; video/ckpt-hotswap are trained-only and
    stay untouched.

    This patch STACKS on orcs's (orcs patches at its import, vibe last), so the
    choices are the parent's UNION vibe's, never respelled: a hand-written Literal
    silently dropped orcs's ``release`` and broke ``play Orcs-* --agent release``
    in a vibe env. "release" routes by manifest — a task in vibe's ``release.json``
    resolves here, any other falls through to orcs's handler.
    """
    import dataclasses
    from dataclasses import asdict, field, make_dataclass

    import torch

    play = importlib.import_module("mjlab.scripts.play")
    if getattr(play, "_vibe_init_agent", False):
        return
    play._vibe_init_agent = True
    _orig_run_play = play.run_play
    _OrigPlayConfig = play.PlayConfig

    # The three ways a checkpoint can be named. "auto" reads these, so a source
    # added upstream must be added here or it resolves to `initial` and silently
    # rolls out an untrained policy.
    _CKPT_SOURCES = ("wandb_run_path", "registry_name", "checkpoint_file")

    # Default `auto`, against mjlab's "trained": bare `play <task>` looks at the
    # scene with the real model stack (initial), while a named checkpoint with no
    # `--agent` plays those weights (trained). The resolution is always printed.
    # Choices = the parent's UNION vibe's, as a real Literal (make_dataclass:
    # a class-body annotation is a string here, and cannot name a local).
    _agents = get_args(get_type_hints(_OrigPlayConfig)["agent"])
    _agents = tuple(dict.fromkeys(("auto", *_agents, "initial", "release")))
    PlayConfig = make_dataclass(
        "PlayConfig", [("agent", Literal[_agents], field(default="auto"))],
        bases=(_OrigPlayConfig,), frozen=True)

    def run_play(task_id: str, cfg):
        from vibe import release

        requested = cfg.agent
        named = [s for s in _CKPT_SOURCES if getattr(cfg, s, None)]
        flags = ", ".join("--" + s.replace("_", "-") for s in named)
        if cfg.agent == "initial" and named:
            raise ValueError(f"--agent initial loads no checkpoint; drop {flags}")
        if cfg.agent == "release" and task_id in release.released_model_ids():
            if named:
                raise ValueError(f"--agent release cannot be combined with {flags}")
            ckpt = release.ensure_released_model(task_id)
            cfg = dataclasses.replace(cfg, agent="trained", checkpoint_file=str(ckpt))
        elif cfg.agent == "auto":
            resolved = "trained" if named else "initial"
            print(f"[INFO]: agent=auto -> {resolved}"
                  + (f" ({', '.join(named)} given)" if named
                     else " (no checkpoint named)"))
            cfg = dataclasses.replace(cfg, agent=resolved)
        # Stash for the take recorder (`vibe.viz.session`): play never exposes the
        # resolved checkpoint, and a clip is filed under the agent that was ASKED for.
        play._vibe_play_ctx = (task_id, cfg, "release" if requested == "release" else cfg.agent)
        if cfg.agent != "initial":
            return _orig_run_play(task_id, cfg)

        play.configure_torch_backends()
        device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        env_cfg = play.load_env_cfg(task_id, play=True)
        agent_cfg = play.load_rl_cfg(task_id)
        if cfg.no_terminations:
            env_cfg.terminations = {}
            print("[INFO]: Terminations disabled")
        if cfg.num_envs is not None:
            env_cfg.scene.num_envs = cfg.num_envs

        env = play.ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
        env = play.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = play.load_runner_cls(task_id) or play.MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        policy = runner.get_inference_policy(device=device)
        base = (agent_cfg.actor.get("base_checkpoint")
                if isinstance(agent_cfg.actor, dict) else None)
        print("[INFO]: agent=initial — freshly constructed policy, NO training "
              f"checkpoint{f' (frozen base: {base})' if base else ''}")

        import os as _os
        if cfg.viewer == "auto":
            has_display = bool(_os.environ.get("DISPLAY")
                               or _os.environ.get("WAYLAND_DISPLAY"))
            resolved_viewer = "native" if has_display else "viser"
        else:
            resolved_viewer = cfg.viewer
        if resolved_viewer == "native":
            play.NativeMujocoViewer(env, policy).run()
        elif resolved_viewer == "viser":
            play.ViserPlayViewer(env, policy, checkpoint_manager=None).run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
        env.close()

    play.PlayConfig = PlayConfig
    play.run_play = run_play
