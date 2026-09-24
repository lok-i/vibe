"""Register the G1 perceptive-locomotion VISION tasks with mjlab.

Naming: `Vibe-<Task>-<Source>-<Extero>[-<Suffix>]` (docs/infra/naming.md).
PerLoco carries no `<RobotMotionRew>` slot — it is motion tracking, and a
one-valued axis is not an axis.

  Vibe-PerLoco-OmRe-ImgFeat-Ext
  Vibe-PerLoco-Grail-ImgFeat-Ext

The privileged twin is `Orcs-PerLoco-<Source>-AdaptSonic`, and each pair is the
experiment: same terrain, same rewards, same RSI, same frozen base, same
critic (height scan included) — the ADAPTER's exteroception is the only
difference. Read a delta between them as the cost of seeing terrain through a
camera, because nothing else is free to move.

Two sources, one obs layout: the vision swap does not depend on which terrain,
so `observation_cfgs` and `agent_cfgs` are shared and only the orcs factory
differs (docs/infra/naming.md — `<Source>` is in the id because provenance changes
code, and here it changes orcs's code, not vibe's).

Registration needs staged terrain data (`scripts/setup/sync_data.sh omre grail`). Absent, it
is SKIPPED, never raised — orcs's rule, and vibe inherits it so an incomplete
checkout still gets the repose tasks:

    python -c "import vibe; print(vibe.tasks.perloco.config.g1.SKIP_REASON)"
"""

from functools import partial

from orcs.core.registry import register_all

from vibe.export.agent.case import register as register_export_case
from vibe.tasks.perloco.config.g1.agent_cfgs import adapt_sonic_ext_agent_cfg
from vibe.tasks.perloco.config.g1.env_cfgs import (
    g1_perloco_grail_env_cfg,
    g1_perloco_omre_env_cfg,
)
from vibe.tasks.perloco.config.g1.export_case import policy_export_test

# No `runner_cls`: `TerrainMotionCommandCfg` rides `orcs.MULTI_CLIP_CFGS`, so the
# compat sentinel already stops mjlab's train script from treating this as a
# single-file tracking task — there is no `registry_name` kwarg to absorb.

_TASKS = (
    ("Vibe-PerLoco-OmRe-ImgFeat-Ext",
     g1_perloco_omre_env_cfg,
     partial(adapt_sonic_ext_agent_cfg, "vibe_perloco_omre")),
    ("Vibe-PerLoco-Grail-ImgFeat-Ext",
     g1_perloco_grail_env_cfg,
     partial(adapt_sonic_ext_agent_cfg, "vibe_perloco_grail")),
)

SKIP_REASON: dict[str, str] = register_all(_TASKS)
"""{task_id: why it could not register}. Empty when every task registered."""

# The ONNX export test cases (export_case.py holds the episode + why it is that
# one). Declared unconditionally — they are data, and read no dataset.
for _task_id, *_ in _TASKS:
    register_export_case(policy_export_test(_task_id))
