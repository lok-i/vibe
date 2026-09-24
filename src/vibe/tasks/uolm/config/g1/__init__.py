"""Register the G1 uni-object loco-manipulation VISION task with mjlab.

Naming: `Vibe-<Task>-<Extero>[-<Suffix>]` (docs/infra/naming.md). Uolm carries
no `<RobotMotionRew>` slot — orcs's uolm is motion tracking, and a one-valued
axis is not an axis.

  Vibe-Uolm-ImgFeat-Ext

The privileged twin is `Orcs-Uolm-AdaptSonic`, and the pair is the experiment:
same roster, same demo clips, same VOF, same rewards, same robustness domain,
same frozen base, same LoRA sizing and std band, same critic (object state and
object_id included) — the ADAPTER's exteroception is the only difference.

Registration needs the retargeted-motion dataset and the object assets. Absent,
it is SKIPPED, never raised — orcs's rule, and vibe inherits it so an incomplete
checkout still gets the other tasks:

    python -c "import vibe; print(vibe.tasks.uolm.config.g1.SKIP_REASON)"
"""

from functools import partial

from orcs.core.registry import register_all

from vibe.export.agent.case import register as register_export_case
from vibe.tasks.uolm.config.g1.agent_cfgs import adapt_sonic_ext_agent_cfg
from vibe.tasks.uolm.config.g1.env_cfgs import g1_uolm_env_cfg
from vibe.tasks.uolm.config.g1.export_case import POLICY_EXPORT_TEST

# No `runner_cls`: `ObjectMotionCommandCfg` rides `orcs.MULTI_CLIP_CFGS`, so the
# compat sentinel already stops mjlab's train script from treating this as a
# single-file tracking task — there is no `registry_name` kwarg to absorb.

_TASKS = (
    ("Vibe-Uolm-ImgFeat-Ext",
     g1_uolm_env_cfg,
     partial(adapt_sonic_ext_agent_cfg, "vibe_uolm")),
)

SKIP_REASON: dict[str, str] = register_all(_TASKS)
"""{task_id: why it could not register}. Empty when every task registered."""

# The ONNX export test case (export_case.py holds the episode + why it is that
# one). Declared unconditionally — it is data, and reads no dataset.
register_export_case(POLICY_EXPORT_TEST)
