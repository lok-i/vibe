"""Contract test for the per-task ONNX export cases — no GPU, no checkpoint.

It proves exactly what would otherwise explode inside `export-agent`, minutes in
and after a wandb download: a task with no declared episode, a pinned clip that
left the dataset, a pin that lands on the task's own kill list, a contact
schedule with no legend, an identity bucket starved of clips.

BEHAVIOR is not here — the numerics are the CLI's job, `export-agent <task-id>
--check`, which needs a checkpoint and a GPU. Same split as orcs: pytest tests
contracts, a real run tests behavior.

    pytest tests/test_export_cases.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg
from orcs.core.data.scan import scan_flat

import vibe  # noqa: F401 — task + case registration
from vibe.export.agent import case as export_case

VIBE_TASKS = sorted(t for t in list_tasks() if t.startswith("Vibe-"))
"""Registered vibe tasks. A task whose dataset is unstaged never registers
(`SKIP_REASON`), so an incomplete checkout tests less rather than failing."""


def test_tasks_are_registered():
    assert VIBE_TASKS, "no Vibe-* task registered — check the entry point / datasets"


def test_task_ids_omit_singleton_agent_axis():
    assert all("AdaptSonic" not in task_id for task_id in VIBE_TASKS)


@pytest.mark.parametrize("task_id", VIBE_TASKS)
def test_task_declares_an_export_case(task_id):
    assert export_case.get(task_id) is not None, (
        f"{task_id} has no export case. Declare one in its config/g1/export_case.py and "
        "register it beside register_mjlab_task — an export with no pinned episode is not "
        "reproducible.")


@pytest.mark.parametrize("task_id", VIBE_TASKS)
def test_export_case_pins_a_live_episode(task_id):
    """The case applies, and what survives is exactly the keep list."""
    case = export_case.get(task_id)
    env_cfg = load_env_cfg(task_id, play=True)
    kept = export_case.apply(env_cfg, case)  # raises on missing / self-excluded clips
    assert set(kept) == set(case.clips)
    if not kept:
        return
    cmd = env_cfg.commands[case.command_name]
    surviving = scan_flat(cmd.dataset_dir, cmd.exclude_motions)
    assert {export_case.clip_id(f) for f in surviving} == set(kept)


@pytest.mark.parametrize("task_id", VIBE_TASKS)
def test_export_case_feeds_every_consumer(task_id):
    """No starved bucket, and a contact legend where one is loaded.

    Both branch on a COMMAND CFG FIELD, never a task name — a new task with the
    same field gets the same check for free.
    """
    case = export_case.get(task_id)
    env_cfg = load_env_cfg(task_id, play=True)
    if not export_case.apply(env_cfg, case):
        return
    cmd = env_cfg.commands[case.command_name]
    surviving = scan_flat(cmd.dataset_dir, cmd.exclude_motions)

    if getattr(cmd, "contact_graph_body_names", None):
        # ContactSchedule builds its legend from the loaded SET and refuses an
        # empty one — one clip carrying a matrix is enough, not all of them.
        assert any((Path(f).parent / "contact_matrix.npz").exists() for f in surviving), (
            f"{task_id}: no pinned clip carries a contact_matrix.npz — ContactSchedule "
            "cannot establish its legend and the env will not build.")

    if roster := getattr(cmd, "ordered_object_names", None):
        # env->object is fixed at sim init; an object with no surviving clip hands
        # `multinomial` an all-zero allowance row for the world holding it.
        from orcs.tasks.uolm.mdp.demo_loader import load_motion_files_from_datasets

        by_object = load_motion_files_from_datasets(
            cmd.dataset_dir, exclude_motions=list(cmd.exclude_motions or ()))
        starved = [o for o in roster if not by_object.get(o)]
        assert not starved, (
            f"{task_id}: roster object(s) {starved} keep no clip — pin one per object.")
