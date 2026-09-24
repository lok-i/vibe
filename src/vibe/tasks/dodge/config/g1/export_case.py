"""Dodge's export episode: nothing to pin but the seed.

The reference is a nominal stand — one clip, and every frame of it is the same
pose — so there is no clip choice to make. What varies between two worlds is the
THROW, drawn by an event, which means `seed` (taken before the env is built) is
the whole of this task's episode identity.
"""

from __future__ import annotations

from vibe.export.agent.case import ExportCase

POLICY_EXPORT_TESTS = (
    ExportCase(task_id="Vibe-Dodge-ImgFeat-Ext"),
    ExportCase(task_id="Vibe-Dodge-ConeFast-ImgFeat-Ext"),
)
