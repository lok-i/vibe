"""What play is currently running, and where its artifacts belong.

Shared by the director (pose files, task-scoped) and the recorder (clips,
checkpoint-scoped) — the two differ on purpose: one angle serves every
checkpoint of a task, one clip belongs to the weights that produced it.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path


def play_context() -> tuple[str | None, str | None, str | None]:
    """(task, checkpoint_file, agent) stashed by the play patch; all None off-play.

    `agent` is the one asked for, resolved: `auto` -> initial|trained, `release` kept."""
    try:
        play = importlib.import_module("mjlab.scripts.play")
    except Exception:  # noqa: BLE001
        return None, None, None
    ctx = getattr(play, "_vibe_play_ctx", None)
    if ctx is None:
        return None, None, None
    task, cfg, agent = ctx
    return task, getattr(cfg, "checkpoint_file", None), agent


def resolve_out_root() -> tuple[Path, str]:
    """(folder, label) for this session's clips — the checkpoint's own dir when there is one.

    A released checkpoint is the exception: its dir is the download cache, so its clips
    go to `videos/<task>/release/` instead."""
    if (env_dir := os.environ.get("VIBE_REC_DIR")):
        return Path(env_dir), "VIBE_REC_DIR"
    task, ckpt, agent = play_context()
    if ckpt is not None and agent != "release":
        return Path(ckpt).parent / "videos" / Path(ckpt).stem, "checkpoint dir"
    return Path("videos") / (task or "unknown") / (agent or "agent"), "repo videos/"


def pose_dir() -> Path:
    """``videos/<task>/`` — one level ABOVE the per-agent clip folders.

    Task-scoped, never checkpoint-scoped: the same angle across `initial` and
    every trained checkpoint is exactly what makes two clips comparable.
    """
    task, _, _ = play_context()
    return Path(os.environ.get("VIBE_POSE_DIR", "videos")) / (task or "unknown")
