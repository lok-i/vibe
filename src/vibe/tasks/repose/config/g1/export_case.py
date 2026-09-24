"""One pinned export episode per Repose physical scene.

Identity here is the clip alone — one object, one scene — so the pin is a single
entry and the two worlds are genuinely the same episode.

The clip must carry a `contact_matrix.npz` (`ContactSchedule` builds its legend
from the loaded set and refuses an empty one) and must be off the unlearnable
kill list the tasks register with (`apply` asserts the second, the contract test
the first).
"""

from __future__ import annotations

from vibe.export.agent.case import ExportCase

SCENE_CLIPS = {
    "big_cube_floor": ("custom/cube_frontflip/sample1",),
    "small_cube_table": ("custom/box_manip/sample14",),
}


def policy_export_test(task_id: str, scene: str) -> ExportCase:
    """The same episode for every agent row of one physical scene.

    Moving the episode with the agent would vary two things at once and make the
    artifact check unreadable.
    """
    return ExportCase(task_id=task_id, clips=SCENE_CLIPS[scene])
