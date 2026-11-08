"""Uolm's export episode: ONE clip per roster object, six in all.

Identity here is the object, not the clip: `sim.world_to_variant` is fixed at
sim init, so with two worlds the export runs variant 0 and variant 1 (suitcase,
trashcan) whatever the clip pin says. Two consequences, both measured:

  1. the pin is per OBJECT — a keep list that starves one roster object hands
     `multinomial` an all-zero allowance row for the world holding it;
  2. the roster itself must stay WHOLE. `object_names=("suitcase",)` — or any
     subset — plus the head camera dies in `sim.sense()` with a CUDA illegal
     memory access (unfilled variant slots at `dataid -1`); the same subset
     without a camera is fine, and the full roster with a camera is fine. So the
     export case pins clips and never touches the roster.

Every pinned clip carries a `contact_matrix.npz`, including plasticbox since the
convex-decomposition contact schedules were regenerated.
"""

from __future__ import annotations

from vibe.export.agent.case import ExportCase

CLIPS = (
    "omomo/sub1_suitcase_010/sample1",    # suitcase
    "omomo/sub7_trashcan_007/sample1",    # trashcan
    "omomo/sub1_largetable_053/sample1",  # largetable
    "omomo/sub8_plasticbox_000/sample1",  # plasticbox
    "custom/tire_roll/sample1",           # tire
    "custom/woodchair2_flip/sample2",     # woodchair2
)
"""One clip per `ordered_object_names` entry, in roster order."""

POLICY_EXPORT_TEST = ExportCase(
    task_id="Vibe-Uolm-ImgFeat-Ext",
    clips=CLIPS,
)
