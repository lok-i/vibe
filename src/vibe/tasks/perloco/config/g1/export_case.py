"""PerLoco's export episode: nothing to pin.

Identity here is the TILE, and the staged dataset already holds exactly one clip
per tile (9/9 on the omni grid) — so under play's `start_from_zero` the two
worlds' episodes are fixed the moment the terrain assigns their rows. A keep
list would have to name every tile to say the same thing, and starving one is an
env-build failure ("tile X is in the grid with no clips").

The tile a world stands on is the curriculum's to move, so `seed` is what pins
this run, not a clip list.
"""

from __future__ import annotations

from vibe.export.agent.case import ExportCase


def policy_export_test(task_id: str) -> ExportCase:
    """Same shape for both sources — the vision swap does not depend on which
    terrain, and neither does the artifact check."""
    return ExportCase(task_id=task_id)
