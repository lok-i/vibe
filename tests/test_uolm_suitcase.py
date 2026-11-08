"""Suitcase roster, actor boundary, renderer assignment and benchmark contracts."""

from dataclasses import asdict

import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from orcs.core.motion_benchmark.contract import load_spec
from orcs.tasks.uolm.env_cfg import uolm_env_cfg
from orcs.tasks.uolm.roster import load_roster

import vibe  # noqa: F401 — task discovery
from vibe.export.agent.case import ExportCase, apply, get
from vibe.tasks.uolm.config.g1 import SKIP_REASON
from vibe.tasks.uolm.config.g1.env_cfgs import g1_uolm_env_cfg, g1_uolm_suitcase_env_cfg

TASK = "Vibe-Uolm-Suitcase-ImgFeat-Ext"
pytestmark = pytest.mark.skipif(TASK not in list_tasks(), reason=SKIP_REASON.get(TASK, ""))


def test_native_roster_is_isolated_from_the_legacy_uolm_task():
    cfg = load_env_cfg(TASK)
    motion = cfg.commands["motion"]
    roster = load_roster("suitcase")
    assert len(roster.motions) == 63
    assert motion.include_motions == roster.motions
    assert motion.ordered_object_names == ("suitcase",)
    assert motion.sampling_mode == "start"
    assert motion.exclude_motions == ()
    legacy = load_env_cfg("Vibe-Uolm-ImgFeat-Ext").commands["motion"]
    assert legacy.include_motions is None
    assert len(legacy.ordered_object_names) == 6
    assert legacy.dataset_dir != motion.dataset_dir


def test_full_render_model_keeps_only_suitcase_active():
    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play)
        entity = cfg.scene.entities["object"]
        assert len(entity.variants) == 6
        assert next(iter(entity.variants)) == "suitcase"
        assert entity.assignment(7) == [0] * 7
        assert cfg.commands["motion"].ordered_object_names == ("suitcase",)


def test_shared_mechanics_and_privileged_critic_match_orcs():
    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play)
        twin = uolm_env_cfg(roster="suitcase", play=play)
        assert cfg.commands == twin.commands
        assert cfg.rewards == twin.rewards
        assert cfg.terminations == twin.terminations
        for group in ("policy", "tokenizer", "critic"):
            assert cfg.observations[group] == twin.observations[group]
        assert tuple(cfg.observations["augmentation"].terms) == (
            "bodywise_contact_cmd", "robot_root_lin_vel_cmd", "robot_root_ang_vel_cmd")
        for name, event in twin.events.items():
            assert cfg.events[name] == event


def test_suitcase_reuses_the_existing_visual_agent():
    agent = asdict(load_rl_cfg(TASK))
    legacy = asdict(load_rl_cfg("Vibe-Uolm-ImgFeat-Ext"))
    assert agent.pop("experiment_name") == "vibe_uolm_suitcase"
    legacy.pop("experiment_name")
    assert agent == legacy


@pytest.mark.parametrize("nominal", [False, True])
def test_benchmark_removes_assistance_and_cleans_the_requested_domain(nominal):
    cfg = load_env_cfg(TASK)
    original = load_env_cfg(TASK)
    spec = load_spec(TASK)
    spec.prepare(cfg, nominal)
    assert "virtual_object_force" not in cfg.events
    assert "policy_update_counter" not in cfg.events
    assert cfg.terminations == original.terminations
    assert cfg.commands["motion"].include_motions == original.commands["motion"].include_motions
    assert cfg.observations["q_proprio"].enable_corruption is not nominal
    assert ("rand_cam_extrinsics" in cfg.events) is not nominal
    assert ("rand_light_dir" in cfg.events) is not nominal
    assert ("rand_terrain_color" in cfg.events) is not nominal
    assert ("set_terrain_color" in cfg.events) is nominal
    assert spec.completion_term == "exceeded_motion"
    spec.configure_batch(cfg, [{"context": 0}] * 5)
    assert cfg.scene.entities["object"].assignment(5) == [0] * 5
    assert original.observations["q_proprio"].enable_corruption
    assert "virtual_object_force" in original.events


def test_export_pins_a_required_include_without_mutating_training():
    from orcs.core.data.retargeted import get_motion_files_for_objects

    cfg = load_env_cfg(TASK, play=True)
    case = get(TASK)
    apply(cfg, case)
    cmd = cfg.commands["motion"]
    _, files = get_motion_files_for_objects(
        list(cmd.ordered_object_names), cmd.dataset_dir,
        list(cmd.exclude_motions), cmd.include_motions,
    )
    assert len(files) == 1
    assert cmd.include_motions == case.clips
    assert len(load_env_cfg(TASK).commands["motion"].include_motions) == 63


def test_export_cannot_add_a_clip_outside_the_tasks_include_roster():
    cfg = load_env_cfg(TASK, play=True)
    cmd = cfg.commands["motion"]
    outside = cmd.include_motions[0]
    cmd.include_motions = cmd.include_motions[1:]
    before = cmd.include_motions, cmd.exclude_motions
    with pytest.raises(ValueError, match="outside the task's include roster"):
        apply(cfg, ExportCase(TASK, clips=(outside,)))
    assert (cmd.include_motions, cmd.exclude_motions) == before


def test_missing_suitcase_export_does_not_require_changing_legacy_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="Native roster is not staged"):
        g1_uolm_suitcase_env_cfg(dataset_dir=str(tmp_path))
    assert len(g1_uolm_env_cfg().commands["motion"].ordered_object_names) == 6
