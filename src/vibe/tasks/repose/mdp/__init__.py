"""Repose mdp — orcs's uolm term library + the cube-specific terms.

Every object-manipulation term (commands, tracking rewards, contact schedule,
VOF, RSI, terminations) lives in `orcs.tasks.uolm.mdp` — repose is the
single-object, vision-carrying special case and adds only what is genuinely
its own:

  commands.py     ReposeMotionCommand — orientation-only success, color goal
  cube_faces.py   the 6-face / 24-symmetry geometry
  observations.py color obs + the frozen vision encoder (`image_feature`)
  rewards.py      up_face / up_color task kernels
  events.py       the color-relabel render domain
  metrics.py      camera projection + the object-in-FOV duty cycle
"""

from orcs.tasks.uolm.mdp import *  # noqa: F403

from .commands import *  # noqa: F403
from .events import *  # noqa: F403
from .metrics import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
