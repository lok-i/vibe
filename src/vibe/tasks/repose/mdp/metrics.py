"""Moved to `vibe.core.mdp.metrics` — the projection needs a camera and an
entity, never a cube, and uolm's vision row reads the same duty cycle.

Re-exported here because a saved env cfg records a term by its DEFINING module
path: deleting this name would break reloading an earlier repose run's cfg.
"""

from vibe.core.mdp.metrics import ObjectCamProjection, object_in_fov

__all__ = ["ObjectCamProjection", "object_in_fov"]
