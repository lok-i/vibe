"""ImgRgb's trainable vision encoder — raw pixels to z, agent-side.

The peer of rsl_rl's `CrossAttentionExtractor` (frozen tokens): same `from_obs`
contract, so `EXTRACTOR_CFGS["cnn"]` names it and nothing else changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNN
from tensordict import TensorDict

__all__ = ["CnnEncoder"]


class CnnEncoder(nn.Module):
    """Spatial-softmax CNN over an RGB group -> LayerNorm'd latent."""

    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        latent_dim: int = 128,
        layer_norm: bool = True,
        spatial_softmax_temperature: float = 1.0,
        **cnn_cfg,
    ) -> None:
        super().__init__()
        cnn_cfg.pop("spatial_softmax", None)  # implied by SpatialSoftmaxCNN
        self.cnn = SpatialSoftmaxCNN(
            input_dim=(height, width), input_channels=channels,
            temperature=spatial_softmax_temperature, **cnn_cfg,
        )
        self.proj = nn.Linear(self.cnn.output_dim, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim) if layer_norm else nn.Identity()
        self.latent_dim = latent_dim
        self.input_groups: tuple[str, ...] = ()

    @classmethod
    def from_obs(cls, obs: TensorDict, group: str, **cfg) -> CnnEncoder:
        """Build from an (B, C, H, W) observation group."""
        shape = obs[group].shape
        if len(shape) != 4:
            raise ValueError(f"CnnEncoder expects (B, C, H, W), got {shape} for '{group}'.")
        ext = cls(height=shape[2], width=shape[3], channels=shape[1], **cfg)
        ext.input_groups = (group,)
        return ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.proj(self.cnn(x)))

    def update_normalization(self, x: torch.Tensor) -> None:
        """No-op: `camera_rgb` already delivers [0, 1]."""
