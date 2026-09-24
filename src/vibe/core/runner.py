"""Vibe runner shim.

mjlab's train script flags any env with a "motion" command as a tracking task and passes
``registry_name`` to the runner; the default MjlabOnPolicyRunner doesn't accept it. This
shim absorbs the kwarg (vibe motion files are always local, never a WandB artifact).
"""

from __future__ import annotations

from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv


class VibeOnPolicyRunner(MjlabOnPolicyRunner):
    """MjlabOnPolicyRunner that tolerates the tracking-task ``registry_name`` kwarg."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
