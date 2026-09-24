"""The task families. One subpackage each, all four the same shape.

A namespace, not a registrar — `vibe/__init__.py` imports these to register
them, and importing `vibe.tasks` alone deliberately registers nothing.

Moved here 2026-08-25, from `vibe.<family>`. The old paths still import
(`vibe.core._legacy_paths`) because a saved env cfg records an mdp term by its
DEFINING module path.
"""
