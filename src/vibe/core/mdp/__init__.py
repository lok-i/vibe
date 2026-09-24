"""vibe's task-agnostic mdp terms — the frozen vision encoder + camera metrics.

A task's `mdp` package star-imports this alongside orcs's and mocke's, so
`mdp.image_feature` resolves the same way in every vibe task.
"""

from .events import *  # noqa: F403
from .metrics import *  # noqa: F403
from .observations import *  # noqa: F403
