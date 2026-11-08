"""Static setup contracts: no network, installs, or environment mutation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "scripts/setup"
ORCS_SHA = "b347b6f0c1e488df619011dd23a869bd7c6f95e2"


def test_setup_scripts_parse() -> None:
    for name in (
        "_common.sh",
        "sync_deps.sh",
        "sync_data.sh",
        "download_released_models.sh",
    ):
        subprocess.run(["bash", "-n", str(SETUP / name)], check=True)


def test_orcs_pin_includes_native_suitcase_and_repertoire_benchmark() -> None:
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


@pytest.mark.parametrize("mode", ["hoiretarget", "all"])
@pytest.mark.parametrize("verify_status", [0, 1])
def test_hoi_check_uses_consumer_roots_and_propagates_verification_failure(
    tmp_path, mode, verify_status,
) -> None:
    """Exercise shell control flow without installing, downloading or staging."""
    setup = tmp_path / "scripts/setup"
    setup.mkdir(parents=True)
    for name in ("sync_data.sh", "_common.sh"):
        shutil.copyfile(SETUP / name, setup / name)
    (tmp_path / "deps.lock").write_text("{}")
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1].endswith('/src/vibe/core/paths.py') and sys.argv[2:] == ['--env']:\n"
        "    print('export ORCS_DATA_ROOT=' + repr(os.environ['TEST_DATA_ROOT']))\n"
        "elif sys.argv[1:] == ['-m', 'orcs.cli.stage_hoi_motions', '--check']:\n"
        "    assert os.environ['ORCS_DATA_ROOT'] == os.environ['TEST_DATA_ROOT']\n"
        "    pathlib.Path(os.environ['TEST_CHECK_LOG']).write_text('verified')\n"
        "    sys.exit(int(os.environ['TEST_VERIFY_STATUS']))\n"
        "else:\n"
        "    raise AssertionError(sys.argv)\n"
    )
    python.chmod(0o755)
    log = tmp_path / "check.log"
    env = {**os.environ, "TEST_DATA_ROOT": str(tmp_path / "native data"),
           "TEST_CHECK_LOG": str(log), "TEST_VERIFY_STATUS": str(verify_status)}
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        ["bash", str(setup / "sync_data.sh"), mode, "--check"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == verify_status, result.stdout + result.stderr
    assert log.read_text() == "verified"


@pytest.mark.parametrize("mode", ["inhouse", "omre", "grail"])
def test_existing_data_modes_keep_the_git_pin_check(tmp_path, mode):
    setup = tmp_path / "scripts/setup"
    setup.mkdir(parents=True)
    for name in ("sync_data.sh", "_common.sh"):
        shutil.copyfile(SETUP / name, setup / name)
    (tmp_path / "deps.lock").write_text(json.dumps({
        "motions": {"url": "unused", "sha": "0" * 40, "path": "data/missing"},
    }))
    result = subprocess.run(
        ["bash", str(setup / "sync_data.sh"), mode, "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "MISSING" in result.stdout
