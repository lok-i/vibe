"""vibe's public model release — orcs's verified downloader over vibe's own manifest.

The manifest is ``vibe/release.json`` (schema ``vibe.release.v1``, repo
``lkrajan/vibe``); checkpoints cache under ``~/.cache/vibe/releases/<revision>``,
moved by ``VIBE_RELEASE_ROOT``. The download/verify logic is orcs's, parameterized
by package (``orcs.release``), never copied. ``play <task> --agent release`` reaches
here through ``core._mjlab_compat``.

    vibe-download-released-models [TASK ...] [--list] [--force]
"""

from __future__ import annotations

from functools import partial

from orcs import release as _release
from orcs.cli import download_released_models as _cli

PACKAGE = "vibe"

release_manifest = partial(_release.release_manifest, PACKAGE)
released_model_ids = partial(_release.released_model_ids, PACKAGE)
ensure_released_model = partial(_release.ensure_released_model, package=PACKAGE)
main = partial(_cli.main, PACKAGE)

if __name__ == "__main__":
    main()
