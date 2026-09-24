"""CPU-only contracts for the two per-RUN vision knobs.

`--env.img-encoder` and `--agent.drop-query-rows` vary a run, not a task id, so
neither has a task to guard it. These are the guards:

  1. both flags are INERT by default — every existing task and checkpoint is
     byte-identical without them;
  2. `img_encoder` rebinds EVERY `image_feature` term and nothing else (the
     failure it exists to prevent is rebinding `kv_tokens` and not `q_cls`,
     which silently loads two backbones and queries the wrong global token);
  3. `drop_query_rows` removes exactly the named rows and refuses a name the
     task does not have.

No model is loaded and no env is built — these are cfg contracts.
"""

from __future__ import annotations

import dataclasses

import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import vibe  # noqa: F401 — register tasks
from vibe.core.env_cfgs import VibeEnvCfg, with_vision_knobs
from vibe.core.mdp.observations import image_feature
from vibe.core.observation_cfgs import IMG_ENCODER

TASK = "Vibe-Repose-BigCubeFloor-ImgFeat-Ext"
OTHER_ENCODER = "facebook/dinov3-vits16plus-pretrain-lvd1689m"


def _encoder_terms(cfg) -> dict[str, str]:
    """{"<group>.<term>": model_name} for every frozen-encoder term in a cfg."""
    return {f"{g}.{n}": t.params["model_name"]
            for g, grp in cfg.observations.items()
            for n, t in grp.terms.items() if t.func is image_feature}


def _rebuild(cfg):
    """What tyro does: reconstruct the dataclass from its fields, re-running
    `__post_init__`. Overriding a field then rebuilding IS the CLI path."""
    return type(cfg)(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)})


# --- img_encoder ------------------------------------------------------------

def test_vision_task_is_a_vibe_env_cfg_with_the_knob_off():
    cfg = load_env_cfg(TASK)
    assert isinstance(cfg, VibeEnvCfg)
    assert cfg.img_encoder is None, "the knob must default to OFF"
    assert set(_encoder_terms(cfg).values()) == {IMG_ENCODER}


def test_img_encoder_rebinds_every_encoder_term_together():
    cfg = load_env_cfg(TASK)
    before = _encoder_terms(cfg)
    # the whole point: MORE THAN ONE term reads the backbone
    assert len(before) >= 2, f"expected kv_tokens + q_cls, got {list(before)}"

    cfg.img_encoder = OTHER_ENCODER
    cfg = _rebuild(cfg)
    after = _encoder_terms(cfg)

    assert set(after) == set(before), "no encoder term may appear or vanish"
    assert set(after.values()) == {OTHER_ENCODER}, f"terms left behind: {after}"


def test_img_encoder_touches_nothing_but_model_name():
    cfg = load_env_cfg(TASK)
    keep = {f"{g}.{n}": {k: v for k, v in t.params.items() if k != "model_name"}
            for g, grp in cfg.observations.items()
            for n, t in grp.terms.items() if t.func is image_feature}
    groups_before = sorted(cfg.observations)

    cfg.img_encoder = OTHER_ENCODER
    cfg = _rebuild(cfg)

    assert sorted(cfg.observations) == groups_before
    for g, grp in cfg.observations.items():
        for n, t in grp.terms.items():
            if t.func is image_feature:
                rest = {k: v for k, v in t.params.items() if k != "model_name"}
                assert rest == keep[f"{g}.{n}"], f"{g}.{n} params drifted"


def test_img_encoder_is_idempotent():
    cfg = load_env_cfg(TASK)
    cfg.img_encoder = OTHER_ENCODER
    once = _encoder_terms(_rebuild(cfg))
    assert _encoder_terms(_rebuild(_rebuild(cfg))) == once


def test_with_vision_knobs_is_idempotent():
    cfg = load_env_cfg(TASK)
    assert with_vision_knobs(cfg) is cfg


# --- drop_query_rows --------------------------------------------------------

def _rows(rl_cfg) -> list[str]:
    (ext,) = rl_cfg.actor["extractor_cfg"].values()
    return ext["query_groups"]


def test_query_rows_default_to_the_full_set():
    rl = load_rl_cfg(TASK)
    assert rl.drop_query_rows == "", "the ablation knob must default to OFF"
    assert _rows(rl) == ["q_task_cmd", "q_proprio", "q_cls"]


@pytest.mark.parametrize("drop", ["q_task_cmd", "q_cls", "q_proprio"])
def test_drop_query_rows_removes_exactly_one_row(drop):
    rl = load_rl_cfg(TASK)
    full = list(_rows(rl))
    rl.drop_query_rows = drop
    rl = _rebuild(rl)
    assert _rows(rl) == [g for g in full if g != drop]
    # the env is untouched: the group is still BUILT, only unread
    assert drop in load_env_cfg(TASK).observations


def test_drop_query_rows_accepts_comma_or_space():
    for spec in ("q_cls,q_proprio", "q_cls q_proprio", "q_cls, q_proprio"):
        rl = load_rl_cfg(TASK)
        rl.drop_query_rows = spec
        assert _rows(_rebuild(rl)) == ["q_task_cmd"], spec


def test_drop_query_rows_rejects_a_row_the_task_does_not_have():
    rl = load_rl_cfg(TASK)
    rl.drop_query_rows = "q_motion_cmd"  # defined, but PARKED for this task
    with pytest.raises(AssertionError, match="q_motion_cmd"):
        _rebuild(rl)


def test_dropping_a_row_leaves_the_latent_budget_alone():
    """z stays 128-d, so an ablation measures routing and not capacity."""
    rl = load_rl_cfg(TASK)
    (before,) = rl.actor["extractor_cfg"].values()
    rl.drop_query_rows = "q_proprio"
    (after,) = _rebuild(rl).actor["extractor_cfg"].values()
    assert after["latent_dim"] == before["latent_dim"] == 128
    assert after["attn_dim"] == before["attn_dim"]


def test_cnn_row_has_no_query_rows_to_drop():
    """The ImgRgb extractor declares none; the knob must skip it, not crash."""
    rl = load_rl_cfg("Vibe-Repose-BigCubeFloor-ImgRgb")
    (ext,) = rl.actor["extractor_cfg"].values()
    assert "query_groups" not in ext
    rl.drop_query_rows = "q_cls"
    _rebuild(rl)  # no raise
