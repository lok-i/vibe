"""Exact comparisons and capacity diagnostics for simulator qualification."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from itertools import zip_longest

import torch


def digest(value) -> str:
    """Hash dtype, shape and raw bits, including signed zero and bfloat16."""
    h = hashlib.sha256()

    def update(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
            h.update(str((str(v.dtype), tuple(v.shape))).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(v, Mapping) or hasattr(v, "items"):
            for key, child in sorted(v.items(), key=lambda pair: str(pair[0])):
                update(str(key))
                update(child)
        elif isinstance(v, (tuple, list)):
            h.update(str((type(v).__name__, len(v))).encode())
            for child in v:
                update(child)
        elif isinstance(v, float):
            h.update(b"float" + struct.pack("!d", v))
        else:
            h.update(json.dumps(v, sort_keys=True).encode() + b"\0")

    update(value)
    return h.hexdigest()


def first_difference(left, right):
    """Compare ordered records; report the earliest step/iteration and field."""
    for index, (a, b) in enumerate(zip_longest(left, right)):
        if a is None or b is None:
            return {"record": index, "field": "record_count"}
        for key in sorted(a.keys() | b.keys()):
            if a.get(key) != b.get(key):
                return {"record": index, "field": key, "left": a.get(key), "right": b.get(key)}
    return None


class BufferMonitor:
    """Sample every physics substep; intentionally excluded from timing runs."""

    def __init__(self, sim):
        self.sim = sim
        self.max_nacon = self.max_nefc = self.samples = 0
        self.solver_limit_samples = 0

    def sample(self):
        from mujoco_warp import OverflowType

        data = self.sim.data
        nacon = int(data.nacon.max())
        nefc = int(data.nefc.max())
        self.max_nacon = max(self.max_nacon, nacon)
        self.max_nefc = max(self.max_nefc, nefc)
        self.samples += 1
        # Iteration limits share this bitfield, but are convergence diagnostics,
        # not buffer exhaustion. Preserve their warnings and report occurrences.
        limits = int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
        self.solver_limit_samples += int(bool((data.overflow[:] & limits).any()))
        if (bool((data.overflow[:] & ~limits).any()) or nacon > self.sim.wp_data.naconmax
                or nefc > self.sim.wp_data.njmax):
            raise RuntimeError("MuJoCo Warp buffer overflow: repeatability is invalid")

    def report(self):
        d = self.sim.wp_data
        contact_margin = d.naconmax / self.max_nacon if self.max_nacon else None
        constraint_margin = d.njmax / self.max_nefc if self.max_nefc else None
        return {
            "nconmax_requested": self.sim.cfg.nconmax,
            "naconmax_total": d.naconmax, "njmax_per_world": d.njmax,
            "naccdmax_total": d.naccdmax,
            "max_nacon_total": self.max_nacon, "max_nefc_per_world": self.max_nefc,
            "contact_headroom": contact_margin, "constraint_headroom": constraint_margin,
            "headroom_pass": all(m is None or m >= 1.5 for m in (contact_margin, constraint_margin)),
            "is_sparse": self.sim.wp_model.is_sparse, "physics_samples": self.samples,
            "solver_limit_samples": self.solver_limit_samples,
        }
