"""Static setup contracts: no network, installs, or environment mutation."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "scripts/setup"


def test_setup_scripts_parse() -> None:
    for name in (
        "_common.sh",
        "sync_deps.sh",
        "sync_data.sh",
        "download_released_models.sh",
    ):
        subprocess.run(["bash", "-n", str(SETUP / name)], check=True)


def test_simulator_sources_and_manifests_agree() -> None:
    lock = json.loads((REPO / "deps.lock").read_text())
    for name in ("orcs", "mjlab", "mujoco_warp", "rsl_rl"):
        assert len(lock[name]["sha"]) == 40
        assert lock[name]["pip_install"] is True
        assert not lock[name].get("pip_no_deps"), "Dependencies must co-resolve"
    assert lock["mjlab"]["patches"]
    manifests = [tomllib.loads((root / "pyproject.toml").read_text())["project"]
                 for root in (REPO, REPO / "dependencies/orcs")]
    for prefix in ("mjlab==", "mujoco-core[rendering]==", "mujoco-warp==", "warp-lang=="):
        requirements = [{r for r in m["dependencies"] if r.startswith(prefix)}
                        for m in manifests]
        assert requirements[0] and requirements[0] == requirements[1]
    constraints = set((REPO / "requirements.lock").read_text().splitlines())
    for requirement in manifests[0]["dependencies"]:
        if "==" in requirement and not requirement.startswith(("mjlab==", "mujoco-warp==")):
            assert requirement.replace("[rendering]", "") in constraints


def test_vibe_installs_orcs_grail_staging_dependency() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert "joblib" in project["dependencies"]


def test_sync_rejects_an_unrelated_active_environment(tmp_path) -> None:
    other = tmp_path / "other-env"
    other.mkdir()
    sentinel = other / "pyvenv.cfg"
    sentinel.write_text("unrelated environment\n")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(other)
    result = subprocess.run(
        ["bash", str(SETUP / "sync_deps.sh")],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "another environment is active" in result.stdout
    assert "[ FETCH" not in result.stdout and "[ENV] installer" not in result.stdout
    assert sentinel.read_text() == "unrelated environment\n"


def test_vibe_does_not_run_orcs_top_level_sync_scripts() -> None:
    data = (SETUP / "sync_data.sh").read_text()
    deps = (SETUP / "sync_deps.sh").read_text()

    assert "orcs/scripts/setup/sync_deps.sh" not in deps + data
    assert "orcs/scripts/setup/sync_data.sh" not in deps + data
    assert "orcs/scripts/setup/sync_dependencies.sh" not in deps + data
