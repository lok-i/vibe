"""Reference deploy-side agent: manifest -> named input buffers -> ONNX -> joint targets.

This is the Python twin of the C++ ROS2 node. It reads ONLY the `.onnx` (the manifest rides
in its metadata) — never the torch model, never rsl_rl. If this class can fly the robot in
sim, the node has everything it needs.

Port assembly is deliberately term-by-term: each term is sliced out at its manifest offset and
re-concatenated in manifest order. Against a real env that is the identity, which is exactly
what makes it a test — it proves the offsets tile every group with no gap and no overlap, the
one thing a hand-written node gets wrong.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import onnxruntime as ort
import torch


def resolve_providers(preference: str = "auto") -> list[str]:
    """Pick ORT execution providers. GR00T's WBC deploys ONNX->TensorRT on GPU (x86 CUDA or
    onboard Jetson Orin); CPU is the fallback when those EPs are not installed."""
    available = ort.get_available_providers()
    wanted = {
        "auto": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }[preference]
    return [p for p in wanted if p in available] or ["CPUExecutionProvider"]


class OnnxAgent:
    """One exported policy, driven exactly as the deploy node drives it."""

    def __init__(self, onnx_path: str, provider: str = "auto") -> None:
        self.providers = resolve_providers(provider)
        self.session = ort.InferenceSession(str(onnx_path), providers=self.providers)
        self.manifest = json.loads(
            self.session.get_modelmeta().custom_metadata_map["manifest"]
        )
        self.ports = self.manifest["inputs"]
        self.action_index = self.manifest["outputs"].index("actions")

        action = self.manifest["action"]
        self.joint_names: list[str] = action["joint_names"]
        self.scale = np.asarray(action["scale"], dtype=np.float32)
        self.default_joint_pos = np.asarray(action["default_joint_pos"], dtype=np.float32)

        graph_inputs = {i.name: tuple(i.shape) for i in self.session.get_inputs()}
        for port in self.ports:
            expected = tuple([1, *port["shape"]])
            if graph_inputs.get(port["name"]) != expected:
                raise RuntimeError(
                    f"manifest/graph mismatch on '{port['name']}': manifest says {expected}, "
                    f"graph says {graph_inputs.get(port['name'])}"
                )
            width = int(np.prod(port["shape"]))
            declared = sum(t["dim"] for t in port["terms"])
            if declared != width:
                raise RuntimeError(
                    f"manifest terms for '{port['name']}' sum to {declared}, input is {width}"
                )

    def assemble(self, obs, world: int) -> dict[str, np.ndarray]:
        """Build every ONNX input for one world, term by term, in manifest order."""
        inputs = {}
        for port in self.ports:
            group = obs[port["groups"][0]]
            terms = port["terms"]
            if hasattr(group, "keys"):  # dict group: one term per input, no offsets
                buffer = group[terms[0]["name"]][world].reshape(-1)
            else:
                row = group[world]
                buffer = torch.cat(
                    [row[t["offset"] : t["offset"] + t["dim"]] for t in terms]
                )
            inputs[port["name"]] = (
                buffer.reshape(1, *port["shape"]).detach().cpu().numpy().astype(np.float32)
            )
        return inputs

    def act(self, obs, world: int) -> np.ndarray:
        """Actions for one world, shape (1, num_actions)."""
        return self.session.run(None, self.assemble(obs, world))[self.action_index]

    def joint_targets(self, actions: np.ndarray) -> np.ndarray:
        """What the PD loop consumes. The env applies this itself; the node must not forget it."""
        return self.default_joint_pos + self.scale * actions.reshape(-1)


class DualPolicy:
    """World 0 runs the checkpoint agent, world 1 runs the exported ONNX agent.

    One object for the headless check and for both viewers, so what you watch is bit-identical
    to what was verified. It also records the open-loop comparison at world 0: the exported
    graph is asked for an action on the CHECKPOINT's observation every step, and that action is
    thrown away — measuring numerics without perturbing the trajectory.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        agent: OnnxAgent,
        gate_agent: OnnxAgent | None = None,
        reference_model: torch.nn.Module | None = None,
    ) -> None:
        self.model = model
        self.agent = agent
        self.gate_agent = gate_agent or agent
        self.reference_model = reference_model
        self.open_loop_max = 0.0
        self.token_flips = 0
        self.steps = 0

    @torch.no_grad()
    def __call__(self, obs) -> torch.Tensor:
        actions = self.model(obs)
        onnx_actions = self.agent.act(obs, 1)
        actions[1] = torch.as_tensor(onnx_actions[0], device=actions.device)

        if self.reference_model is not None:
            obs_cpu = obs.to("cpu")
            reference = self.reference_model(obs_cpu)[0:1].numpy()
            exported = self.gate_agent.act(obs, 0)
            self.open_loop_max = max(
                self.open_loop_max, float(np.abs(reference - exported).max())
            )
            if hasattr(self.reference_model, "encode_tokens"):
                on_device = self.model.encode_tokens(obs)[0:1].detach().cpu()
                on_cpu = self.reference_model.encode_tokens(obs_cpu[0:1])
                self.token_flips += int((on_device - on_cpu).abs().gt(1e-6).sum())
        self.steps += 1
        return actions


class WorldStats:
    """Per-world survival and return over a rollout."""

    def __init__(self, worlds: Sequence[int]) -> None:
        self.worlds = list(worlds)
        self.steps = dict.fromkeys(self.worlds, 0)
        self.first_reset = dict.fromkeys(self.worlds, None)
        self.resets = dict.fromkeys(self.worlds, 0)
        self.returns = dict.fromkeys(self.worlds, 0.0)
        self.fired: dict[int, set[str]] = {w: set() for w in self.worlds}

    def update(self, step: int, rewards: torch.Tensor, dones: torch.Tensor, env) -> None:
        manager = env.unwrapped.termination_manager
        for world in self.worlds:
            self.returns[world] += float(rewards[world])
            self.steps[world] += 1
            if bool(dones[world]):
                self.resets[world] += 1
                if self.first_reset[world] is None:
                    self.first_reset[world] = step + 1
                for name in manager.active_terms:
                    if bool(manager.get_term(name)[world]):
                        self.fired[world].add(name)

    def line(self, world: int, total_steps: int) -> str:
        survived = self.first_reset[world] or total_steps
        terms = ",".join(sorted(self.fired[world])) or "none"
        mean_reward = self.returns[world] / max(self.steps[world], 1)
        return (f"{survived}/{total_steps} steps to first reset, {self.resets[world]} reset(s) "
                f"[{terms}], r̄ = {mean_reward:.3f}")
