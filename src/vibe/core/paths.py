"""Filesystem roots — the single source of path truth.

No caller does ``__file__`` depth math, so moving a module never silently
orphans a dataset. Resolution order for every root: env override -> marker
walk-up.

  VIBE_ROOT       repo root      (default: walk up for _MARKERS)
  VIBE_DATA_ROOT  datasets       (default: <repo>/data)
  VIBE_DEPS_ROOT  synced deps    (default: <repo>/dependencies)

`bind_orcs` propagates these roots to orcs. It is the EXPLICIT channel, not the
load-bearing one: entry-point import order across `mjlab.tasks` is not
guaranteed, so orcs may resolve its own paths before this ever runs. orcs's own
rule ("when vendored, the host owns the data") is what carries — this makes the
same answer explicit, and is what a VIBE_DATA_ROOT override rides.
`setdefault`, so an explicit ORCS_* in the environment still wins.

Setup scripts need the same roots from a shell. They read them from HERE rather
than re-deriving, so there is one definition:

    eval "$(python -m vibe.core.paths --env)"
"""

from __future__ import annotations

import os
from pathlib import Path

from assets.paths import PACKAGE_ROOT as ASSETS_SOURCE

_MARKERS = ("pyproject.toml", "deps.lock")
"""Files that mark the repo root — both must be present."""


def _env(var: str) -> Path | None:
    val = os.environ.get(var)
    return Path(val).expanduser().resolve() if val else None


def _walk_up() -> Path:
    here = Path(__file__).resolve()
    for d in here.parents:
        if all((d / m).exists() for m in _MARKERS):
            return d
    raise FileNotFoundError(
        f"vibe repo root not found above {here} — no ancestor holds all of "
        f"{_MARKERS}. Set VIBE_ROOT to override."
    )


VIBE_ROOT: Path = _env("VIBE_ROOT") or _walk_up()
DATA_ROOT: Path = _env("VIBE_DATA_ROOT") or VIBE_ROOT / "data"
DEPS_ROOT: Path = _env("VIBE_DEPS_ROOT") or VIBE_ROOT / "dependencies"

LOGS_ROOT: Path = VIBE_ROOT / "logs"

_G1_CUSTOM = DATA_ROOT / "retargeted_motions/data/unitree_g1/custom"

G1_REPOSE_BIG_CUBE_FLOOR_DATASET: list[Path] = [
    _G1_CUSTOM / "cube_frontflip",
    _G1_CUSTOM / "cube_sideflip",
]
G1_REPOSE_SMALL_CUBE_TABLE_DATASET = _G1_CUSTOM / "box_manip"
"""Repose datasets. Each root may be a motion folder (holds
``sampleX/motion.npz``) or an ancestor of motion folders; ``mdp.motion_dirs``
finds either layout depth-invariantly.

The big cube moves on the floor. The small cube's clips finish on a table whose
per-environment XY is taken from the sampled clip's final object pose.
"""


ORCS_ENV: dict[str, str] = {
    "ORCS_DATA_ROOT": str(DATA_ROOT),
    "ORCS_DEPS_ROOT": str(DEPS_ROOT),
    "ORCS_ASSETS_SOURCE": str(ASSETS_SOURCE),
}
"""What orcs is told, in one place — read by `bind_orcs` (runtime) and by
`--env` (setup scripts). `ORCS_DEPS_ROOT` is here because staging clones the
GRAIL reference repo into it; it landed in this repo's `dependencies/` by the
same walk-up that found `data/`, which is not a thing to leave implicit."""


def bind_orcs() -> None:
    """Propagate this repo's roots to orcs, for when they are not the defaults.

    Only has an effect before `import orcs` (orcs resolves at import). Harmless
    to skip: orcs disqualifies its own vendored checkout and resolves against
    the host repo, which is this one.
    """
    for var, val in ORCS_ENV.items():
        os.environ.setdefault(var, val)


def lfs_include() -> list[str]:
    """Globs, relative to the `retargeted_motions` checkout, of every blob a vibe task reads.

    Derived, never listed: repose from the `G1_REPOSE_*` roots above, uolm from ORCS's
    own roster resolver (`get_motion_files_for_objects`, at its object + exclude lists).
    That resolver reads only `metadata.json`, which is not LFS, so it runs on a
    pointer-only checkout — which is what lets `sync_data.sh` fetch the ~0.1 GB the
    rosters name instead of the whole 3.8 GB dataset. `*.npz` only: the per-clip mp4s
    and csvs are ~90% of the bytes and no runtime reads them.

    Run it as a FILE (`python src/vibe/core/paths.py --lfs-include`), never `-m`: on a
    pointer tree `import vibe` raises (repose registration loads a clip), and `-m` imports
    the package first. As a file only mjlab's entry-point scan imports vibe, and that scan
    catches — so the resolve survives the very state it exists to get out of.
    """
    bind_orcs()  # before mjlab: its scan is what imports orcs
    import mjlab  # noqa: F401 — mjlab before orcs (CLAUDE.md, the import-order trap)
    from orcs.tasks.uolm.env_cfg import (
        _DEFAULT_OBJECT_NAMES,
        _EXCLUDE_MOTIONS,
        _G1_DATASETS_ROOT,
    )
    from orcs.tasks.uolm.mdp.demo_loader import get_motion_files_for_objects

    root = DATA_ROOT / "retargeted_motions"
    _, uolm = get_motion_files_for_objects(
        list(_DEFAULT_OBJECT_NAMES), _G1_DATASETS_ROOT, list(_EXCLUDE_MOTIONS))
    repose = [*G1_REPOSE_BIG_CUBE_FLOOR_DATASET, G1_REPOSE_SMALL_CUBE_TABLE_DATASET]
    return ([f"{d.relative_to(root)}/**/*.npz" for d in repose]
            + sorted({f"{Path(f).parent.relative_to(root)}/*.npz" for f in uolm}))


def _main() -> None:
    """`python -m vibe.core.paths --env` -> a shell-evalable export block.
    `--lfs-include` -> `include <glob>` lines (prefixed: importing vibe chatters on stdout)."""
    import sys

    if "--lfs-include" in sys.argv[1:]:
        for g in lfs_include():
            print(f"include {g}")
        return
    if "--env" not in sys.argv[1:]:
        raise SystemExit("usage: python -m vibe.core.paths --env | --lfs-include")
    for var, val in ORCS_ENV.items():
        print(f"export {var}={val!r}")


if __name__ == "__main__":
    _main()


def default_motion_file() -> str:
    """First big-cube clip (deterministic ordering)."""
    for d in G1_REPOSE_BIG_CUBE_FLOOR_DATASET:
        clips = sorted(Path(d).rglob("motion.npz"))  # depth-agnostic
        if clips:
            return str(clips[0])
    return ""
