"""The per-task export test case — the EPISODE one export run is checked on.

The exporter knows nothing about clips, objects or terrain. A task declares its
own case beside `register_mjlab_task`, and what a case pins is that task's
EPISODE IDENTITY — whose arity is the task's business, not the exporter's:

| task | identity | clips declared |
|---|---|---|
| repose | clip | 1 — both worlds track it |
| uolm | object variant | 6 — one per roster object (env->object is fixed at sim init) |
| perloco | tile | 0 — the staged dataset is already one clip per tile |
| dodge | throw | 0 — one nominal stand; `seed` pins the throw |

`clips` is a KEEP list applied as its COMPLEMENT through `exclude_motions` (the
public exclusion grammar, `orcs.core.data.scan.matches_exclude`). Never through
`dataset_dir`: rewriting the root to a motion folder is valid only for a flat
scan, and every bucketed scan (`scan_grouped`) takes a str at a fixed depth —
that rewrite is what broke every task but repose.

A keep list must cover EVERY bucket. Starve one — a roster object with no
surviving clip, a grid tile with none — and it fails at env build, not here:
`_clip_allowance` hands `multinomial` an all-zero row. The pytest contract test
(`tests/test_export_cases.py`) is what holds that invariant.

**A case pins the episode, NOT the trajectory.** Two runs of the same case start
from the same state and diverge anyway: mujoco-warp is not bitwise reproducible
across processes (~1e-7/step, measured) and FSQ turns that into a token flip
within ~3 steps. So the VERDICT repeats (gate ~5e-6 against a 1e-5 tolerance,
survival 128/128) while the digits do not — read the pass/fail, never the
last decimal.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from orcs.core.data.scan import matches_exclude, scan_flat

__all__ = ["ExportCase", "register", "get", "cases", "apply", "clip_id"]


@dataclass(frozen=True)
class ExportCase:
    """One export check: which episodes it runs, from which seed, what must pass."""

    task_id: str
    clips: tuple[str, ...] = ()
    """`<dataset>/<motion>/<sampleN>` keep list; empty = this dataset is already
    deterministic under play's `start_from_zero`."""
    seed: int = 0
    """Drawn BEFORE the env is built, so startup events, the variant assignment
    and the first reset's clip draw are all pinned by it."""
    steps: int = 128
    min_steps: int = 128
    """Closed-loop steps the exported graph must survive (0 = report only)."""
    command_name: str = "motion"


_CASES: dict[str, ExportCase] = {}


def register(case: ExportCase) -> ExportCase:
    """Declare a task's case at import time; returns it so a module can bind it."""
    _CASES[case.task_id] = case
    return case


def get(task_id: str) -> ExportCase | None:
    return _CASES.get(task_id)


def cases() -> dict[str, ExportCase]:
    """Every declared case — the contract test's input."""
    return dict(_CASES)


def clip_id(motion_file: str) -> str:
    """`.../<dataset>/<motion>/<sampleN>/motion.npz` -> `<dataset>/<motion>/<sampleN>`."""
    return "/".join(motion_file.split("/")[-4:-1])


def apply(env_cfg: ManagerBasedRlEnvCfg, case: ExportCase) -> tuple[str, ...]:
    """Pin the case's clips on a play cfg. Returns what survives.

    The task's own `exclude_motions` is REPLACED, not extended — the complement
    already excludes everything outside the keep list, and a pinned clip may not
    sit on that list (asserted), so the surviving set is identical either way.
    """
    if not case.clips:
        return ()
    cmd = env_cfg.commands[case.command_name]
    every = {clip_id(f) for f in scan_flat(cmd.dataset_dir)}
    missing = sorted(set(case.clips) - every)
    if missing:
        raise FileNotFoundError(
            f"{case.task_id}: pinned clip(s) absent from {cmd.dataset_dir}: {missing}")
    killed = [c for c in case.clips
              if matches_exclude(set(cmd.exclude_motions or ()), *c.split("/"))]
    if killed:
        raise ValueError(
            f"{case.task_id}: pinned clip(s) sit on the task's OWN exclusion list: "
            f"{killed} — the task killed those for a reason; pin another.")
    cmd.exclude_motions = tuple(sorted(every - set(case.clips)))
    return tuple(sorted(case.clips))
