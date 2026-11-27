"""Precision overrides must apply before model construction and be recorded."""

import json

import mjlab  # noqa: F401
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

from vibe.core.runner import MjlabOnPolicyRunner, VibeOnPolicyRunner


@pytest.mark.parametrize("initial,requested", [(True, None), (False, None),
                                                 (True, False), (False, True)])
@pytest.mark.parametrize("runner_cls", [MjlabOnPolicyRunner, VibeOnPolicyRunner])
def test_runner_precision_before_model_construction(
    monkeypatch, tmp_path, initial, requested, runner_cls,
):
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    expected = initial if requested is None else requested
    seen = []
    def construct(self, env, cfg, log_dir, device):
        seen.append(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    monkeypatch.setattr(OnPolicyRunner, "__init__", construct)
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = initial
        cfg = {"bf16_reduced_precision_reduction": requested, "amp_dtype": "bfloat16",
               "logger": "tensorboard"}
        runner_cls(None, cfg, str(tmp_path))
        assert seen == [expected]
        recorded = json.loads((tmp_path / "params/precision.json").read_text())
        assert recorded == cfg["torch_precision"]
        assert recorded["bf16_reduced_precision_reduction"] == expected
        assert recorded["amp_dtype"] == "bfloat16"
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
