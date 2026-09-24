"""Register the G1 dodgeball VISION tasks with mjlab.

Naming: `Vibe-<Task>-<Extero>[-<Suffix>]` (docs/infra/naming.md). Dodge carries
no `<RobotMotionRew>` slot — orcs's dodge is motion tracking against a held
stand, and a one-valued axis is not an axis.

  Vibe-Dodge-ImgFeat-Ext
  Vibe-Dodge-ConeFast-ImgFeat-Ext

The privileged twin is `Orcs-Dodge-AdaptSonic`, and the pair is the experiment:
same ball, same throws, same nominal-stand reference, same rewards, same
terminations, same frozen base, same LoRA sizing and std band, same critic (ball
state included) — the ADAPTER's exteroception is the only difference.

Registration needs the nominal-stand reference (`orcs-make-nominal`). Absent, it
is SKIPPED, never raised — orcs's rule, and vibe inherits it:

    python -c "import vibe; print(vibe.tasks.dodge.config.g1.SKIP_REASON)"
"""

from functools import partial

from orcs.core.registry import register_all

from vibe.export.agent.case import register as register_export_case
from vibe.tasks.dodge.config.g1.agent_cfgs import adapt_sonic_ext_agent_cfg
from vibe.tasks.dodge.config.g1.env_cfgs import (
    g1_dodge_cone_fast_env_cfg,
    g1_dodge_env_cfg,
)
from vibe.tasks.dodge.config.g1.export_case import POLICY_EXPORT_TESTS

# No `runner_cls`: the nominal-stand reference rides a plain
# `MultiClipMotionCommandCfg`, which is `orcs.MULTI_CLIP_CFGS` itself, so the
# compat sentinel already stops mjlab's train script from treating this as a
# single-file tracking task — there is no `registry_name` kwarg to absorb.

_TASKS = (
    ("Vibe-Dodge-ImgFeat-Ext",
     g1_dodge_env_cfg,
     partial(adapt_sonic_ext_agent_cfg, "vibe_dodge")),
    ("Vibe-Dodge-ConeFast-ImgFeat-Ext",
     g1_dodge_cone_fast_env_cfg,
     partial(adapt_sonic_ext_agent_cfg, "vibe_dodge")),
)

SKIP_REASON: dict[str, str] = register_all(_TASKS)
"""{task_id: why it could not register}. Empty when every task registered."""

# The ONNX export test case (export_case.py holds the episode + why it is that
# one). Declared unconditionally — it is data, and reads no dataset.
for _case in POLICY_EXPORT_TESTS:
    register_export_case(_case)
