"""Vibe runner shim.

mjlab's train script flags any env with a "motion" command as a tracking task and passes
``registry_name`` to the runner; the default MjlabOnPolicyRunner doesn't accept it. This
shim absorbs the kwarg (vibe motion files are always local, never a WandB artifact).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
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


def configure_precision(train_cfg: dict, log_dir: str | None) -> dict:
    """Apply an explicit ViBe precision option before constructing the models."""
    requested = train_cfg.get("bf16_reduced_precision_reduction")
    if requested is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = requested
    precision = {
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "amp_dtype": train_cfg.get("amp_dtype"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    # The logger stores this alongside the requested agent configuration.
    train_cfg["torch_precision"] = precision
    print(f"[ViBe] Effective Torch precision: {json.dumps(precision)}", flush=True)
    if log_dir is not None:
        path = Path(log_dir) / "params" / "precision.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(precision, indent=2) + "\n")
    return precision
