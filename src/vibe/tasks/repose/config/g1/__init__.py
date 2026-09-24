"""Register G1 Repose tasks with mjlab.

Naming: ``Vibe-Repose-<Scene>-<Extero>[-<Suffix>]`` (docs/tasks.md). ``BigCubeFloor``
carries six rows; ``SmallCubeTable`` the ImgFeat-Ext row.

  ObjKin   object kinematic state — the privileged twin
  ImgRgb   raw head-camera RGB through a trainable CNN (the baseline)
  ImgFeat  frozen encoder features from the same camera

Both vision rows ride the COLOR task channel (env_cfgs): goal = up-face COLOR
one-hot (never the quat), per-env face->color remap + terrain palette DR, task
reward = up_color pair. Four agent families over ImgFeat (aux=True adds the
extractor layout):

  -ImgFeat      vanilla adapter (no extractor; color one-hot in the aug stream)
  -ImgFeat-Ext  extractor-only baseline (plain PPO, no predictor)
  -ImgFeat-Sfd  supervised FD (PPOAux; target = object state + upface color +
                task-reward rates + hand contact force, one ZPrediction/<term>
                wandb panel per target term, derived from the group)
  -ImgFeat-Lfd  latent FD vs EMA (PPOAux, SSL)

The privileged baselines for the other families live in orcs (`Orcs-*-AdaptSonic`).
"""

from mjlab.tasks.registry import register_mjlab_task

from vibe.export.agent.case import register as register_export_case
from vibe.tasks.repose.config.g1.agent_cfgs import (
    adapt_sonic_agent_cfg,
    adapt_sonic_cnn_agent_cfg,
)
from vibe.tasks.repose.config.g1.env_cfgs import g1_repose_cube_env_cfg
from vibe.tasks.repose.config.g1.export_case import policy_export_test
from vibe.tasks.repose.config.g1.runner import VibeOnPolicyRunner

# `experiment_name` defaults: `logs/rsl_rl/<name>/`, following the task id.
_BIG_EXP = "g1_repose_big_cube_floor"
_SMALL_EXP = "g1_repose_small_cube_table"

# Applied to all BigCubeFloor variants (relative → repo root in env_cfgs).
_EXCLUDE = "src/vibe/tasks/repose/config/g1/exclusions/unlearnable_3faw6xrn_f0eh005.json"

# Rows: (tag, rl_cfg, env_kw). PPOAux rows carry runner_cls + aux via env_kw.
_BIG_TASKS = [
    ("ObjKin", adapt_sonic_agent_cfg(_BIG_EXP), dict()),
    ("ImgFeat", adapt_sonic_agent_cfg(_BIG_EXP), dict(extero="imgfeat")),
    # The ImgRgb baseline: same base, same task, pixels through a trainable CNN
    # instead of a frozen encoder + attention pool. Raw RGB is ~5x the rollout
    # storage of the token path (84.7 vs 16.1 KB/env/step), so this row wants a
    # smaller --env.scene.num-envs than the ImgFeat ones.
    ("ImgRgb", adapt_sonic_cnn_agent_cfg(_BIG_EXP), dict(extero="imgrgb")),
]

# ImgFeat extractor family — {Ext,Sfd,Lfd} over the sonic adapter (the extractor
# rides the agent-cfg default, currently cross_attention). The Sfd target is
# WIDENED env-side: object state + current up-face color + task-layer reward
# rates + hand contact force. The `ZPrediction/<term>` wandb split is derived
# from the target obs group itself (one slice per term), so nothing about it is
# declared here.
for _aux, _tag in (("lfd", "Lfd"), ("sfd", "Sfd"), ("ext", "Ext")):
    _BIG_TASKS.append((
        f"ImgFeat-{_tag}",
        adapt_sonic_agent_cfg(_BIG_EXP, aux=_aux),
        dict(extero="imgfeat", aux=True, runner_cls=VibeOnPolicyRunner),
    ))


def _register(
    task_id: str,
    rl_cfg,
    *,
    scene: str,
    runner_cls=None,
    **env_kw,
) -> None:
    if scene == "big_cube_floor":
        env_kw["exclude_motions_file"] = _EXCLUDE
    env_kw["scene"] = scene
    register_mjlab_task(
        task_id=task_id,
        env_cfg=g1_repose_cube_env_cfg(**env_kw),
        play_env_cfg=g1_repose_cube_env_cfg(**env_kw, play=True),
        rl_cfg=rl_cfg,
        **({"runner_cls": runner_cls} if runner_cls else {}),
    )
    register_export_case(policy_export_test(task_id, scene))


for _tag, _rl_cfg, _raw_env_kw in _BIG_TASKS:
    _runner_cls = _raw_env_kw.get("runner_cls")
    _env_kw = {k: v for k, v in _raw_env_kw.items() if k != "runner_cls"}
    _register(
        f"Vibe-Repose-BigCubeFloor-{_tag}",
        _rl_cfg,
        scene="big_cube_floor",
        runner_cls=_runner_cls,
        **_env_kw,
    )

_register(
    "Vibe-Repose-SmallCubeTable-ImgFeat-Ext",
    adapt_sonic_agent_cfg(_SMALL_EXP, aux="ext"),
    scene="small_cube_table",
    extero="imgfeat",
    aux=True,
    runner_cls=VibeOnPolicyRunner,
)
