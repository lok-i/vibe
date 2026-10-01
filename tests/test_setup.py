"""Static setup contracts: no network, installs, or environment mutation."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "scripts/setup"
ORCS_SHA = "f1ba79d57218f4be1ef2f5b7886a04e4f3d662da"


def test_setup_scripts_parse() -> None:
    for name in (
        "_common.sh",
        "sync_deps.sh",
        "sync_data.sh",
        "download_released_models.sh",
    ):
        subprocess.run(["bash", "-n", str(SETUP / name)], check=True)


def test_orcs_pin_is_the_lean_setup_build() -> None:
    lock = json.loads((REPO / "deps.lock").read_text())
    assert lock["orcs"]["sha"] == ORCS_SHA
    assert lock["orcs"]["pip_install"] is True
    assert lock["orcs"]["pip_no_deps"] is True


def test_vibe_installs_orcs_grail_staging_dependency() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert "joblib" in project["dependencies"]


def test_setup_requires_vibes_active_local_venv() -> None:
    common = (SETUP / "_common.sh").read_text()
    deps = (SETUP / "sync_deps.sh").read_text()

    assert 'local venv="$REPO_ROOT/.venv"' in common
    assert "uv venv --prompt vibe" in common
    assert "source .venv/bin/activate" in common
    assert "PIP_CMD=(uv pip)" in common
    assert "DEPS_PIP_CMD" not in common
    assert '${VIRTUAL_ENV:-$REPO_ROOT/.venv}' not in common
    assert "use_venv create" not in deps


def test_sync_without_activation_stops_before_installing() -> None:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        ["bash", str(SETUP / "sync_deps.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Vibe" in result.stdout
    assert "source .venv/bin/activate" in result.stdout


def test_vibe_does_not_run_orcs_top_level_sync_scripts() -> None:
    data = (SETUP / "sync_data.sh").read_text()
    deps = (SETUP / "sync_deps.sh").read_text()

    assert "orcs/scripts/setup/sync_deps.sh" not in deps + data
    assert "orcs/scripts/setup/sync_data.sh" not in deps + data
    assert "orcs/scripts/setup/sync_dependencies.sh" not in deps + data
