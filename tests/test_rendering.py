"""CPU regression checks for backend guards, native DR and pose compatibility."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from mjlab.entity.data import EntityData
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

from vibe.core.mdp.events import rand_cam_extrinsics
from vibe.core.rendering import VisualDrCfg, configure_backend

REPOSE = "Vibe-Repose-BigCubeFloor-ImgFeat-Ext"


def test_perloco_rejects_optix_before_gpu_creation():
    cfg = load_env_cfg("Vibe-PerLoco-Grail-ImgFeat-Ext")
    assert cfg.render_backend == "warp"
    with pytest.raises(ValueError, match="PerLoco.*only the native Warp"):
        configure_backend(cfg, "optix")
    assert len(cfg.scene.sensors) >= 2  # camera and privileged scanner remain


@pytest.mark.parametrize("task", ["Vibe-PerLoco-Grail-ImgFeat-Ext", "Vibe-Dodge-ConeFast-ImgFeat-Ext"])
def test_floor_dr_rejects_generated_terrain_and_rooms(task):
    cfg = load_env_cfg(task)
    with pytest.raises(ValueError, match="requires plane terrain"):
        replace(cfg, visual_dr=VisualDrCfg(floor_textures=True))
    lighting = replace(cfg, visual_dr=VisualDrCfg(lighting=True))
    assert "visual_dr_light_diffuse" in lighting.events
    assert cfg.scene.terrain.materials == lighting.scene.terrain.materials


def test_renderer_selection_is_recorded_and_does_not_mutate_other_configs():
    original = load_env_cfg(REPOSE)
    changed = replace(original, render_backend="optix")
    assert changed.render_backend == "optix"
    assert next(s for s in changed.scene.sensors if s.name == "head_cam").backend == "optix"
    assert next(s for s in original.scene.sensors if s.name == "head_cam").backend == "warp"
    with pytest.raises(ValueError, match="Unknown render backend"):
        configure_backend(original, "missing")


def test_extra_visual_dr_preserves_cube_faces_and_play_domain():
    cfg = replace(load_env_cfg(REPOSE), visual_dr=VisualDrCfg(True, True, True))
    cfg.scene.num_envs = 2
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    for i in range(6):
        assert model.geom(f"object/cube_face_{i}").matid == -1
    assert model.ntex >= 2
    assert cfg.events["rand_face_colors"].func is load_env_cfg(REPOSE).events["rand_face_colors"].func
    assert cfg.events["visual_dr_floor_texture"].params["asset_cfg"].name == "terrain"
    assert cfg.events["visual_dr_light_diffuse"].mode == "startup"
    play = load_env_cfg(REPOSE, play=True)
    assert play.play and not any(name.startswith("visual_dr_") for name in play.events)
    with pytest.raises(ValueError, match="training-only"):
        replace(play, visual_dr=VisualDrCfg(lighting=True))


def test_native_camera_dr_targets_subset_without_accumulating():
    cfg = load_env_cfg(REPOSE)
    cfg.scene.num_envs = 3
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    robot = scene["robot"]
    robot.indexing = robot._compute_indexing(model, "cpu")
    defaults = {k: torch.from_numpy(getattr(model, k).copy()).float()
                for k in ("cam_pos", "cam_quat")}
    live = SimpleNamespace(**{k: value[None].repeat(3, 1, 1) for k, value in defaults.items()})
    env = SimpleNamespace(scene=scene, num_envs=3, device="cpu", sim=SimpleNamespace(
        model=live, per_world_default_fields=set(), get_default_field=defaults.__getitem__))
    ids = torch.tensor([0, 2])
    rand_cam_extrinsics(env, ids, "head_cam", (0.02, 0.02), (0.035, 0.035))
    first = {k: getattr(live, k).clone() for k in defaults}
    rand_cam_extrinsics(env, ids, "head_cam", (0.02, 0.02), (0.035, 0.035))
    cid = model.camera("head_cam").id
    torch.testing.assert_close(live.cam_pos[ids, cid] - defaults["cam_pos"][cid],
                               torch.full((2, 3), 0.02))
    torch.testing.assert_close(live.cam_quat[ids, cid].norm(dim=-1), torch.ones(2))
    for field in defaults:
        value = getattr(live, field)
        torch.testing.assert_close(value, first[field])
        torch.testing.assert_close(value[1], defaults[field])
        other = [i for i in range(model.ncam) if i != cid]
        torch.testing.assert_close(value[:, other], defaults[field][other].expand(3, -1, -1))


@pytest.mark.parametrize("kind", ["geom", "site"])
def test_pose_api_reads_scalar_first_quaternions(kind):
    half = 2.0 ** -0.5
    positions = torch.tensor([[[1.0, 2.0, 3.0]]])
    # 90 degrees about Z, in MuJoCo's wxyz convention; no matrix field exists.
    rotations = torch.tensor([[[half, 0.0, 0.0, half]]])
    entity = SimpleNamespace(
        indexing=SimpleNamespace(**{f"{kind}_ids": torch.tensor([0])}),
        data=SimpleNamespace(**{f"{kind}_xpos": positions, f"{kind}_xquat": rotations}),
    )
    pose = getattr(EntityData, f"{kind}_pose_w").fget(entity)
    torch.testing.assert_close(pose, torch.tensor([[[1.0, 2.0, 3.0, half, 0.0, 0.0, half]]]))
