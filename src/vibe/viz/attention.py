"""Attention → RGB kernels, viser-free so the take recorder never imports a GUI.

Split out of ``attention_overlay`` (2026-08-05); the panel and the recorder paint
the same pixels from the same code.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

PATCH = 16  # Theia patch16
TURBO = matplotlib.colormaps["turbo"]


def patch_grid(p: int, h: int, w: int) -> tuple[int, int]:
    """(Hp, Wp) patch grid for P tokens: camera-derived, else near-square fallback."""
    hp, wp = h // PATCH, w // PATCH
    if hp * wp == p:
        return hp, wp
    hp = max(1, int(round(p**0.5)))
    while p % hp:
        hp -= 1
    return hp, p // hp


def colorize(attn_2d: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """(Hp, Wp) attention → (H, W, 3) uint8 turbo heatmap (per-map min-max)."""
    lo, hi = float(attn_2d.min()), float(attn_2d.max())
    norm = (attn_2d - lo) / (hi - lo + 1e-8)
    up = F.interpolate(
        torch.from_numpy(norm)[None, None], size=out_hw, mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return (TURBO(up)[..., :3] * 255).astype(np.uint8)


def colorbar(width: int, height: int = 12) -> np.ndarray:
    """Horizontal turbo gradient strip, low (left) → high (right)."""
    row = (TURBO(np.linspace(0, 1, width))[..., :3] * 255).astype(np.uint8)
    return np.tile(row[None], (height, 1, 1))


def upsample(img: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour block upscale — the FPV is 112x63 and must look it."""
    if scale <= 1:
        return img
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def select_map(attn: np.ndarray, labels: list[str], sel: str) -> np.ndarray:
    """(Q, P) → the (P,) map named by ``sel`` ("mean"/"max" reduce across rows)."""
    if sel == "mean":
        return attn.mean(0)
    if sel == "max":
        return attn.max(0)
    return attn[labels.index(sel) if sel in labels else 0]


def overlay(rgb: np.ndarray, attn_p: np.ndarray, alpha: float) -> np.ndarray:
    """(H, W, 3) uint8 FPV α-blended with one (P,) attention map."""
    hp, wp = patch_grid(attn_p.shape[0], *rgb.shape[:2])
    heat = colorize(attn_p.reshape(hp, wp), rgb.shape[:2])
    return ((1 - alpha) * rgb + alpha * heat).astype(np.uint8)


def entropy(attn: np.ndarray) -> np.ndarray:
    """(Q, P) → per-row flatness, normalized by log P (1 = uniform → 0 = focused)."""
    p = attn.shape[1]
    return -(attn * np.log(attn + 1e-9)).sum(1) / np.log(p)
