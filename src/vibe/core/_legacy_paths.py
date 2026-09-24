"""`vibe.<family>` still imports after the fold into `vibe.tasks.<family>`.

A saved env cfg records an mdp term by its DEFINING module path, so deleting a
module path deletes the ability to reload every run that used it. A finder
rather than N shim files: it is lazy (a legacy path costs nothing until someone
asks for it), it covers submodules that do not exist yet, and there is one place
to delete when the last pre-fold checkpoint is retired.

The aliased module IS the new one — same object in `sys.modules` under both
names — so `isinstance` and `func is` comparisons hold across the two spellings.
"""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

__all__ = ["MOVED", "install"]

MOVED = ("repose", "perloco", "uolm", "dodge")
"""Families that used to live at `vibe.<name>`."""


def _target(name: str) -> str | None:
    """`vibe.repose.mdp.events` -> `vibe.tasks.repose.mdp.events`, else None."""
    head = name.split(".")
    if len(head) < 2 or head[0] != "vibe" or head[1] not in MOVED:
        return None
    return "vibe.tasks." + ".".join(head[1:])


class _MovedTasksFinder(MetaPathFinder, Loader):
    def find_spec(self, name, path=None, target=None) -> ModuleSpec | None:
        del path, target
        return None if _target(name) is None else ModuleSpec(name, self)

    def create_module(self, spec: ModuleSpec):
        # Returning the real module makes both names one object.
        return importlib.import_module(_target(spec.name))

    def exec_module(self, module) -> None:
        """Already executed by the import above."""


def install() -> None:
    """Idempotent, and FIRST in `sys.meta_path` — last does not work.

    A legacy package resolves its own submodules through its `__path__`, which
    is the NEW directory — so `PathFinder` finds `.../tasks/dodge/config/g1` on
    it and loads a SECOND copy under the old name. Two copies of a task module
    is two registrations, and mjlab raises on the second. Nothing can be
    shadowed by going first: `_target` matches only names that no longer exist.
    """
    if not any(isinstance(f, _MovedTasksFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MovedTasksFinder())
