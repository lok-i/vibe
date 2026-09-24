"""vibe — visual behavior adaptation for exteroceptive whole-body control.

Importing this package registers every task family under `vibe.tasks` with
mjlab and patches the mjlab scripts. `core.paths.bind_orcs()` runs first to
propagate a non-default data root to orcs — orcs finds this repo on its own
otherwise, so this only matters under a VIBE_DATA_ROOT override.
"""

from vibe.core import _legacy_paths as _legacy
from vibe.core import paths as _paths

_paths.bind_orcs()
_legacy.install()  # `vibe.<family>` -> `vibe.tasks.<family>`, for saved cfgs

from vibe.core._mjlab_compat import apply as _apply_mjlab_compat  # noqa: E402
from vibe.tasks import dodge as _dodge  # noqa: E402, F401 — task registration
from vibe.tasks import perloco as _perloco  # noqa: E402, F401 — task registration
from vibe.tasks import repose as _repose  # noqa: E402, F401 — task registration
from vibe.tasks import uolm as _uolm  # noqa: E402, F401 — task registration

_apply_mjlab_compat()
