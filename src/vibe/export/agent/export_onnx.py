"""Export a trained policy to a self-contained, deployable ONNX file + manifest, then fly it.

Loads a checkpoint exactly like `play` does (same env cfg, same runner, same
`runner.load(load_cfg={"actor": True})`), exports the actor, writes the manifest, and runs a
TWO-WORLD test in one environment:

    world 0  checkpoint agent drives; the exported graph is asked for an action on world 0's
             observation every step and that action is thrown away  -> OPEN-LOOP numerics (gated)
    world 1  exported ONNX agent drives, nothing else               -> CLOSED-LOOP stability

Only the gate compares like with like — one observation, two backends, exact. World 1 is a
stability check, not a twin of world 0: on a multi-instance task the two worlds are two
DIFFERENT instances by construction (uolm's variant table gives world 0 a suitcase and
world 1 a trashcan), and even one instance would diverge, since the sim is not bitwise
reproducible and FSQ amplifies ~1e-7 into a token flip in a few steps.

WHICH episodes run is the TASK's to say, never this file's — `vibe.export.agent.case`,
declared per task beside `register_mjlab_task`. The seed is drawn before the env is built,
so the startup events, the variant assignment and the clip draw are all pinned by it.

Artifacts are written only if the open-loop gate passes. `--viewer` then replays the same
`DualPolicy` object so what you watch is bit-identical to what was verified.

Usage:
    export-agent <task-id> --release                      # the lkrajan/vibe checkpoint
    export-agent <task-id> --checkpoint-file <ckpt.pt>
    export-agent <task-id> --wandb-run-path <e/p/run> --viewer native

The two-world test needs onnxruntime: `bash scripts/setup/sync_deps.sh --deploy`.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

import vibe  # noqa: F401 — task registration
from vibe.export.agent import case as export_case
from vibe.export.manifest import build_manifest, duplicate_group_aliases

_TORCH_WORLD, _ONNX_WORLD = 0, 1
_NUM_ENVS = 2


@dataclass(frozen=True)
class ExportConfig:
    """CLI for the ONNX exporter (checkpoint resolution mirrors `play`)."""

    release: bool = False
    """Export the released checkpoint (`play --agent release`'s), fetched on first use."""
    wandb_run_path: str | None = None
    """W&B run holding the checkpoint, e.g. '<entity>/<project>/<run>'."""
    wandb_checkpoint_name: str | None = None
    """Checkpoint within the run (default: latest)."""
    checkpoint_file: str | None = None
    """Local checkpoint, bypassing W&B."""
    output_dir: str | None = None
    """Where to write <name>.onnx / <name>.manifest.json (default: the checkpoint's folder;
    `exports/agent/<task-id>/` for `--release`, never the download cache)."""
    check: bool = True
    """Run the two-world test; artifacts are written only if the open-loop gate passes."""
    tolerance: float = 1e-5
    check_steps: int | None = None
    """Rollout length (default: the task's case)."""
    closed_loop_min_steps: int | None = None
    """Fail the export if world 1 resets before this many steps (default: the task's
    case; 0 = report only)."""
    seed: int | None = None
    """Episode seed (default: the task's case). Drawn before the env is built."""
    provider: Literal["auto", "cuda", "tensorrt", "cpu"] = "auto"
    """ORT provider for the closed-loop world. GR00T's WBC deploys ONNX->TensorRT on GPU;
    'auto' takes TensorRT/CUDA when installed and falls back to CPU. The gate is always CPU."""
    viewer: Literal["none", "native", "viser"] = "none"
    """Watch both worlds side by side after the gate passes."""
    dedupe_inputs: bool = True
    """Merge content-identical observation groups onto one ONNX input."""
    with_attn: bool = True
    """Expose the extractor attention map as a second output (free; deploy-side overlay)."""
    device: str | None = None
    log_root: str = "logs/rsl_rl"


def _onnx_agent():
    """`vibe.export.agent.onnx_agent`, which needs onnxruntime — only the check/viewer do."""
    try:
        from vibe.export.agent import onnx_agent
    except ModuleNotFoundError as exc:
        if exc.name != "onnxruntime":
            raise
        raise SystemExit("[export] onnxruntime missing: `bash scripts/setup/sync_deps.sh "
                         "--deploy` (never `pip install -e .[deploy]` — it swaps out the "
                         "rsl_rl fork). `--no-check` exports without it.") from exc
    return onnx_agent


def _released(task_id: str) -> Path:
    """The released checkpoint, routed by manifest exactly as `play --agent release` is."""
    from vibe import release

    if task_id in release.released_model_ids():
        return release.ensure_released_model(task_id)
    from orcs import release as orcs_release

    return orcs_release.ensure_released_model(task_id)


def _resolve_checkpoint(cfg: ExportConfig, task_id: str, experiment_name: str) -> Path:
    """Resolve a released, local or W&B checkpoint path (same rules as `play`)."""
    if cfg.release:
        if cfg.checkpoint_file or cfg.wandb_run_path:
            raise ValueError("--release cannot be combined with --checkpoint-file / "
                             "--wandb-run-path")
        return _released(task_id)
    if cfg.checkpoint_file is not None:
        path = Path(cfg.checkpoint_file)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return path
    if cfg.wandb_run_path is None:
        raise ValueError("name a checkpoint: --release | --checkpoint-file | --wandb-run-path")
    log_root = (Path(cfg.log_root) / experiment_name).resolve()
    path, cached = get_wandb_checkpoint_path(
        log_root, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
    print(f"[export] checkpoint {path.name} (run {path.parent.name}, "
          f"{'cached' if cached else 'downloaded'})")
    return path


def _resolve_case(task_id: str) -> export_case.ExportCase:
    """The task's declared export episode, or an unpinned fallback with a warning.

    Every `Vibe-*` task declares one (`tests/test_export_cases.py` holds that); an orcs
    twin or a brand-new row can still be exported, it just runs whatever episode the
    dataset draws.
    """
    case = export_case.get(task_id)
    if case is None:
        print(f"[export] no export case for {task_id} — running UNPINNED on whatever the "
              "dataset draws. Declare one in the task's config/g1/export_case.py.")
        return export_case.ExportCase(task_id=task_id, min_steps=0)
    return case


def _two_world_rollout(env, policy, steps: int, tolerance: float):
    """Step both worlds under one policy, gating world 0's open-loop numerics as we go."""
    stats = _onnx_agent().WorldStats((_TORCH_WORLD, _ONNX_WORLD))
    obs = env.get_observations()
    with torch.no_grad():
        for step in range(steps):
            actions = policy(obs)
            if policy.open_loop_max > tolerance:
                raise RuntimeError(
                    f"open-loop parity failed at step {step}: max|delta a| = "
                    f"{policy.open_loop_max:.3e} > {tolerance:.1e} "
                    "(cpu fp32 both sides — this is a real export defect)"
                )
            obs, rewards, dones, _ = env.step(actions)
            stats.update(step, rewards, dones, env)
    return stats


def run_export(task_id: str, cfg: ExportConfig) -> None:
    """Build the env, load the checkpoint, export, verify in two worlds, write the artifacts."""
    if cfg.check or cfg.viewer != "none":
        _onnx_agent()  # fail before the env build, not after an artifact is written
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    case = _resolve_case(task_id)
    steps = cfg.check_steps if cfg.check_steps is not None else case.steps
    min_steps = (cfg.closed_loop_min_steps if cfg.closed_loop_min_steps is not None
                 else case.min_steps)
    seed = cfg.seed if cfg.seed is not None else case.seed

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = _NUM_ENVS  # world 0 = checkpoint, world 1 = exported ONNX
    clips = export_case.apply(env_cfg, case)
    pinned = ", ".join(clips) if len(clips) <= 2 else f"{len(clips)} clips (one per bucket)"
    print(f"[export] case: seed {seed}, {steps} steps, "
          f"episode {pinned or 'unpinned (dataset already deterministic)'}")

    checkpoint = _resolve_checkpoint(cfg, task_id, agent_cfg.experiment_name)
    out_dir = (Path(cfg.output_dir) if cfg.output_dir
               else Path("exports/agent") / task_id if cfg.release
               else checkpoint.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = checkpoint.stem

    # BEFORE the env exists: startup events, the variant assignment and the first reset's
    # clip draw all happen in construction, so seeding afterwards would pin nothing.
    ManagerBasedRlEnv.seed(seed)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    model = runner.alg.get_policy()
    model.eval()

    # --- export ---
    kwargs = {"verbose": False}
    if "with_attn" in inspect.signature(model.as_onnx).parameters:
        kwargs["with_attn"] = cfg.with_attn
    onnx_model = model.as_onnx(**kwargs)
    onnx_model.to("cpu").eval()

    if cfg.dedupe_inputs:
        aliases = duplicate_group_aliases(env.unwrapped, onnx_model.layout)
        if aliases:
            onnx_model.alias_ports(aliases)
            merged = ", ".join(f"{a} -> {c}" for a, c in aliases.items())
            print(f"[export] deduped identical groups: {merged}")

    onnx_path = out_dir / f"{stem}.onnx"
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes={},
        dynamo=False,
    )
    print(f"[export] inputs : {onnx_model.input_names}")
    print(f"[export] outputs: {onnx_model.output_names}")

    # --- manifest FIRST: the deploy agent reads it back out of the .onnx to drive world 1 ---
    manifest = build_manifest(
        env.unwrapped,
        onnx_model,
        task_id=task_id,
        run_path=cfg.wandb_run_path or str(checkpoint),
        checkpoint=str(checkpoint),
        model_class=type(model).__name__,
    )
    manifest_path = out_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    attach_metadata_to_onnx(str(onnx_path), {"manifest": json.dumps(manifest)})

    # --- two-world test, then keep or delete ---
    policy = None
    if cfg.check or cfg.viewer != "none":
        oa = _onnx_agent()
        agent = oa.OnnxAgent(onnx_path, provider=cfg.provider)
        gate_agent = agent if agent.providers == ["CPUExecutionProvider"] else oa.OnnxAgent(
            onnx_path, provider="cpu")
        print(f"[export] onnxruntime providers: {agent.providers}")
        policy = oa.DualPolicy(
            model, agent, gate_agent=gate_agent,
            reference_model=copy.deepcopy(model).to("cpu").eval(),
        )

    if cfg.check:
        try:
            stats = _two_world_rollout(env, policy, steps, cfg.tolerance)
            survived = stats.first_reset[_ONNX_WORLD] or steps
            if survived < min_steps:
                raise RuntimeError(
                    f"closed-loop world reset after {survived} steps "
                    f"(< min_steps {min_steps})"
                )
        except Exception:
            onnx_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            print("[export] check FAILED — no files written")
            raise
        detail = (f"{policy.token_flips} FSQ token flip(s)" if policy.token_flips
                  else "TF32 matmul precision only")
        print(f"[export] open-loop  gate  : max|delta a| = {policy.open_loop_max:.3e} "
              f"over {steps} steps (cpu fp32 both sides)")
        print(f"[export] device gap       : {detail}")
        print(f"[export] closed-loop onnx : {stats.line(_ONNX_WORLD, steps)}")
        print(f"[export] reference  torch : {stats.line(_TORCH_WORLD, steps)}")

    print(f"[export] wrote {onnx_path}")
    print(f"[export] wrote {manifest_path}")

    if cfg.viewer != "none":
        _launch_viewer(env, policy, cfg.viewer)
    env.close()


def _launch_viewer(env, policy, backend: str) -> None:
    """Watch world 0 (checkpoint) and world 1 (exported ONNX) side by side."""
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    if backend == "native" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[export] no display — falling back to the viser viewer")
        backend = "viser"
    print("[export] viewer: world 0 = checkpoint agent, world 1 = exported ONNX agent")
    viewer = NativeMujocoViewer if backend == "native" else ViserPlayViewer
    viewer(env, policy).run()


def main() -> None:
    """Parse the task id, then the exporter flags (same two-stage CLI as `play`)."""
    import mjlab.tasks  # noqa: F401

    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
    )
    args = tyro.cli(
        ExportConfig,
        args=remaining,
        default=ExportConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
    )
    run_export(chosen_task, args)


if __name__ == "__main__":
    main()
