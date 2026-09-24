"""Deploy manifest: the ordered term table a C++ node needs to fill the ONNX inputs.

The exported graph takes one tensor per observation GROUP; this module records what goes
INSIDE each of them — term names, widths, offsets, history lengths — so the node builds every
buffer by NAME and asserts, instead of hand-counting offsets that silently shift the next time
a term is ablated. Written into the ONNX `metadata_props` and as a sibling `.manifest.json`.

The vision inputs are tagged with their encoder + slice (`tokens` vs `cls`): both come from ONE
frozen-backbone forward env-side, so the deploy-side vision package must publish the full
`(1 + P, C)` sequence, not just the patches.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction

_SCHEMA = "vibe.onnx.v1"
_VISION_FUNC = "image_feature"


def _action_metadata(env: ManagerBasedRlEnv) -> dict:
    """Joint order, PD gains, action scale and default pose for the deploy node.

    Same content as :func:`mjlab.rl.exporter_utils.get_base_metadata`, minus its
    ``active_terms["actor"]`` lookup — vibe's frozen-WBC tasks use a policy/augmentation
    obs layout with no "actor" group, and the per-input term tables cover that field anyway.
    Joint order is the checkpoint's, never assumed: SONIC is MuJoCo-ordered, textop IL-ordered.
    """
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, JointPositionAction)
    # Each spec actuator drives exactly one joint; index the gains in global joint order.
    joint_to_ctrl = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [joint_to_ctrl[j] for j in robot.joint_names if j in joint_to_ctrl]
    scale = action._scale
    return {
        "type": "joint_position",
        "joint_names": list(robot.joint_names),
        "scale": scale[0].cpu().tolist() if hasattr(scale, "cpu") else scale,
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "stiffness": env.sim.mj_model.actuator_gainprm[ctrl_ids, 0].tolist(),
        "damping": (-env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]).tolist(),
    }


def _term_signature(env: ManagerBasedRlEnv, group: str, term: str) -> tuple:
    """Identity of a term: its function, its params, and its width.

    Two groups match only if every term matches on all three — the equivalence proof behind
    :func:`duplicate_group_aliases`. Params are part of it because the same function under a
    different ``command_name`` is a different signal.
    """
    cfg = env.observation_manager.get_term_cfg(group, term)
    index = env.observation_manager.active_terms[group].index(term)
    dim = tuple(env.observation_manager.group_obs_term_dim[group][index])
    func = getattr(cfg.func, "__qualname__", type(cfg.func).__name__)
    return (term, func, repr(sorted((cfg.params or {}).items(), key=str)), dim)


def _vision_source(env: ManagerBasedRlEnv, group: str, term: str) -> dict | None:
    """Encoder / sensor / slice tag for a frozen-vision term (``None`` if not one)."""
    cfg = env.observation_manager.get_term_cfg(group, term)
    name = getattr(cfg.func, "__qualname__", type(cfg.func).__name__)
    if _VISION_FUNC not in name:
        return None
    params = cfg.params or {}
    return {
        "kind": "vision_encoder",
        "sensor": params.get("sensor_name"),
        "model_name": params.get("model_name"),
        "model_dtype": params.get("model_dtype", "float32"),
        # Both slices come from the SAME encoder forward: publish (1 + P, C) and split.
        "slice": params.get("output", "tokens"),
    }


def _term_table(env: ManagerBasedRlEnv, group: str, only: str | None = None) -> list[dict]:
    """Ordered term entries for one group (or the single term of a dict-group port)."""
    manager = env.observation_manager
    entries, offset = [], 0
    for index, term in enumerate(manager.active_terms[group]):
        shape = tuple(manager.group_obs_term_dim[group][index])
        width = 1
        for axis in shape:
            width *= axis
        if only is None or term == only:
            cfg = manager.get_term_cfg(group, term)
            entry = {
                "name": term,
                "shape": list(shape),
                "dim": width,
                "offset": offset,
                "history_length": getattr(cfg, "history_length", 0),
                "flatten_history_dim": getattr(cfg, "flatten_history_dim", True),
            }
            source = _vision_source(env, group, term)
            if source is not None:
                entry["source"] = source
            entries.append(entry)
        offset += width
    return entries


def duplicate_group_aliases(env: ManagerBasedRlEnv, layout: list[dict]) -> dict[str, str]:
    """Map content-identical input groups onto one canonical input.

    ``augmentation`` and ``q_motion_cmd`` are the same bundle by construction
    (``observation_cfgs.robot_motion_cmd_terms`` feeds both), so exporting them twice would
    make the node fill two identical buffers. Groups match only when every term matches on
    function, params and width; the first port in graph order wins.
    """
    canonical: dict[tuple, str] = {}
    aliases: dict[str, str] = {}
    for port in layout:
        if port["term"] is not None or len(port["groups"]) != 1:
            continue  # dict-group term ports are never deduped
        group = port["groups"][0]
        signature = tuple(
            _term_signature(env, group, t) for t in env.observation_manager.active_terms[group]
        )
        if signature in canonical:
            aliases[port["name"]] = canonical[signature]
        else:
            canonical[signature] = port["name"]
    return aliases


def git_sha(path: Path) -> str:
    """Short SHA of the repo containing ``path`` (``unknown`` outside a work tree)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_encoder_manifest(
    enc,
    *,
    height: int,
    width: int,
    num_patches: int,
    token_dim: int,
    has_cls: bool,
    compute_dtype: str,
    opset: int,
) -> dict:
    """vision.onnx.v1 — the manifest of a frozen-encoder export (`vibe.export.encoder`).

    Grid and dims are MEASURED from a real forward, never derived arithmetically;
    preprocessing (cast, /255, BGR->RGB, layout, per-model normalization) is baked
    into the graph, so the deploy node's contract is: resize to (height, width),
    memcpy BGR HWC uint8, run.
    """
    import torch
    import transformers

    return {
        "schema": "vision.onnx.v1",
        "tag": enc.tag,
        "hf_id": enc.mid,
        "input": {
            "name": "image_bgr_hwc",
            "height": height,
            "width": width,
            "layout": "bgr_hwc_uint8",
        },
        "patch": {
            "size": enc.patch,
            "grid_h": height // enc.patch,
            "grid_w": width // enc.patch,
        },
        "outputs": {
            "patch_tokens": {"num": num_patches, "dim": token_dim},
            "cls_token": {"dim": token_dim} if has_cls else None,
        },
        "compute_dtype": compute_dtype,
        "opset": opset,
        "versions": {
            "vibe": git_sha(Path(__file__).resolve().parents[3]),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }


def build_manifest(
    env: ManagerBasedRlEnv,
    onnx_model,
    *,
    task_id: str,
    run_path: str,
    checkpoint: str,
    model_class: str,
) -> dict:
    """Assemble the full deploy manifest for an exported policy.

    Pairs the action metadata (:func:`_action_metadata`) with per-input term tables derived
    from the model's OWN port list, so the manifest can never advertise an order the graph
    does not consume.
    """
    inputs = []
    for port in onnx_model.layout:
        group = port["groups"][0]
        entry = {
            "name": port["name"],
            "shape": list(port["shape"]),
            "groups": list(port["groups"]),
            "terms": _term_table(env, group, only=port["term"]),
        }
        inputs.append(entry)

    import rsl_rl

    return {
        "schema": _SCHEMA,
        "task_id": task_id,
        "run_path": run_path,
        "checkpoint": checkpoint,
        "model_class": model_class,
        "control": {
            "sim_timestep": env.cfg.sim.mujoco.timestep,
            "decimation": env.cfg.decimation,
            "step_dt": env.cfg.sim.mujoco.timestep * env.cfg.decimation,
        },
        "action": _action_metadata(env),
        "inputs": inputs,
        "outputs": list(onnx_model.output_names),
        "versions": {
            "vibe": git_sha(Path(__file__).resolve().parents[3]),
            "rsl_rl": git_sha(Path(rsl_rl.__file__).resolve().parent),
        },
    }
