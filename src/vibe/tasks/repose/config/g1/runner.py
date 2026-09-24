"""Re-export of the shared runner shim.

Moved to `vibe.core.runner` (2026-08-03) — absorbing mjlab's tracking-task
kwarg is not a repose concern. Kept here because a saved run cfg records the
runner class by its module path.
"""

from vibe.core.runner import VibeOnPolicyRunner

__all__ = ["VibeOnPolicyRunner"]
