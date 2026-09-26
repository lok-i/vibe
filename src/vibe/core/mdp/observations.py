"""Task-agnostic vision mdp terms — the frozen encoder, and nothing else.

Atomic obs TERMS live here; the cfg builders that compose them into groups are
`vibe.core.observation_cfgs`. Nothing in this module knows about a cube, a
terrain or a task — it reads a camera sensor and returns features.
"""

from __future__ import annotations

import inspect

import torch
from mjlab.managers.manager_base import ManagerTermBase

__all__ = ["image_feature"]


class image_feature(ManagerTermBase):
    """Frozen vision encoder -> dense tokens (B, P, C) | flat (B, P*C) | global (B, C).

    Backend selected from ``model_name`` (``_detect_backend``), and EVERY per-backend
    decision — loader, input normalization, how many leading tokens are not patches,
    what the global vector is — is resolved ONCE in ``__init__``. There is no runtime
    dispatch. A new encoder is three edits: a ``_detect_backend`` branch, a ``_load``
    branch, and an ``_encode_*`` returning ``(dense, global)``.

    | backend | ``model_name`` looks like | dense | global vector |
    |---|---|---|---|
    | ``theia`` | ``theia-*`` (bare; ``_load`` prefixes the org) | ``h[:, 1:]`` | CLS — untrained, ~noise |
    | ``clip``  | ``openai/clip-*``, ``*TinyCLIP*`` | ``h[:, 1:]`` | CLS — trained |
    | ``siglip``| ``google/siglip2-*`` | ``h`` (no CLS at all) | **MAP-head pool**, trained |
    | ``dino``  | ``facebook/dinov{2,3}-*`` | ``h[:, 1+n_reg:]`` | CLS — trained |
    | ``resnet``| ``resnet*`` | ``(B, D)`` flat | none |

    ``output="tokens"`` (default) returns dense and honors ``flatten``; ``output="cls"``
    returns the global vector (siglip has no CLS token — its MAP pool IS the trained
    global token, so the row is more meaningful there, not less). Token GEOMETRY is a
    property of the camera, not the backbone: at the 112x63 head cam every stride-16
    backbone above yields the same P = 7x3 = 21, so a swap moves ``C`` and nothing else.

    Two economies keep the vision cost at ONE forward regardless of how many terms read it:
      - backbone SHARED across term instances of the same ``model_name`` (`_MODELS`);
      - each env-step's forward CACHED (`_FWD`), so a tokens-term + a cls-term over the
        same sensor+step reuse one pass (keyed on step AND the rgb buffer ptr, so a new
        render or a new step always recomputes — no stale reuse).
    ``model_dtype`` ("float16" default | "float32" | "bfloat16") is the encoder
    COMPUTE dtype only — weights, norm tensors, and input all cast to it; the
    returned obs is always fp32, so downstream buffers/agents never change.
    """

    _MODELS: dict = {}   # (model_name, device, dtype) -> shared frozen backbone
    _FWD: dict = {}      # (id(env), sensor, model_name, dtype) -> (step, rgb_ptr, (dense, glob))

    # per-backend input normalization (ImageNet vs CLIP vs SigLIP training stats)
    _NORM = {
        "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "clip": ((0.48145466, 0.4578275, 0.40821073),
                 (0.26862954, 0.26130258, 0.27577711)),
        "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    }

    def __init__(self, cfg, env):
        super().__init__(env)
        self._sensor_name = cfg.params["sensor_name"]
        self._flatten = cfg.params.get("flatten", False)
        self._output = cfg.params.get("output", "tokens")
        self._model_name = cfg.params.get("model_name", "theia-tiny-patch16-224-cddsv")
        dev = cfg.params.get("model_device", env.device)
        self._dev = dev
        self._dtype = getattr(torch, cfg.params.get("model_dtype", "float16"))
        self._backend = self._detect_backend(self._model_name)
        # Resolve the per-backend seq-encoder + normalization ONCE (no hot-path branch).
        norm_key, self._encode, self._has_cls = {
            "theia": ("imagenet", self._encode_theia, True),
            "clip": ("clip", self._encode_clip, True),
            "siglip": ("half", self._encode_siglip, True),
            "dino": ("imagenet", self._encode_dino, True),
            "resnet": ("imagenet", self._encode_resnet, False),
        }[self._backend]
        mean, std = self._NORM[norm_key]
        self._mean = torch.tensor(mean, device=dev, dtype=self._dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=dev, dtype=self._dtype).view(1, 3, 1, 1)
        if self._output == "cls" and not self._has_cls:
            raise ValueError(f"output='cls' needs a token model, got {self._model_name}")
        self._model = self._load(self._backend, self._model_name, dev, self._dtype)
        # DINO-only, read off the LIVE model rather than hardcoded per checkpoint:
        # CLS + N register tokens sit ahead of the patches, and DINOv3 resizes via
        # RoPE (no interpolate flag) where DINOv2 takes one. Both vary by checkpoint.
        if self._backend == "dino":
            self._skip = 1 + getattr(self._model.config, "num_register_tokens", 0)
            self._interp_kw = (
                {"interpolate_pos_encoding": True}
                if "interpolate_pos_encoding"
                in inspect.signature(type(self._model).forward).parameters else {})
        self._ckey = (id(env), self._sensor_name, self._model_name, self._dtype)

    def reset(self, env_ids=None) -> None:
        """Drop the cached encode. The obs manager calls this on every env reset.

        Neither half of the cache key moves across a reset — `common_step_counter` does
        not advance, and the renderer writes the SAME buffer in place — so without this
        the first obs after an explicit `env.reset()` serves the previous episode's
        features (measured: identical pixels, features off by 4.2). Free in the stepping
        path, where the step counter would have missed anyway.
        """
        image_feature._FWD.pop(self._ckey, None)

    @staticmethod
    def _detect_backend(model_name: str) -> str:
        """Backend from the model id. Ordered: the first match wins, and "siglip"
        is tested before "clip" so a future `siglip-*-clip-*` id cannot alias."""
        n = model_name.lower()
        for token, backend in (("theia", "theia"), ("siglip", "siglip"),
                               ("dino", "dino"), ("clip", "clip"), ("resnet", "resnet")):
            if token in n:
                return backend
        raise ValueError(f"Unsupported model: {model_name}")

    @classmethod
    def _load(cls, backend: str, model_name: str, dev, dtype: torch.dtype) -> torch.nn.Module:
        key = (model_name, str(dev), str(dtype))
        if key not in cls._MODELS:
            if backend == "theia":
                from transformers import AutoModel
                cls._MODELS[key] = AutoModel.from_pretrained(
                    f"theaiinstitute/{model_name}", trust_remote_code=True,
                ).eval().to(dev, dtype)
            elif backend == "clip":
                from transformers import CLIPVisionModel
                # vision tower ONLY — CLIPModel would park the (unused) text
                # tower in GPU mem forever (~19M params for TinyCLIP-39M).
                cls._MODELS[key] = CLIPVisionModel.from_pretrained(
                    model_name).eval().to(dev, dtype)  # exposes .vision_model
            elif backend == "siglip":
                from transformers import AutoModel
                # Same economy as clip, one level deeper: siglip2-b16 is 375M total
                # for a 93M tower, so keep the tower and drop the parent BEFORE the
                # .to(dev) — `Siglip2VisionModel.from_pretrained` cannot read these
                # checkpoints (the repo ships full-model keys).
                cls._MODELS[key] = AutoModel.from_pretrained(
                    model_name).vision_model.eval().to(dev, dtype)
            elif backend == "dino":
                from transformers import AutoModel
                cls._MODELS[key] = AutoModel.from_pretrained(
                    model_name).eval().to(dev, dtype)
            elif backend == "resnet":
                from torchvision import models
                cls._MODELS[key] = getattr(models, model_name)(
                    weights=f"{model_name.title().replace('net', 'Net')}_Weights.IMAGENET1K_V1",
                ).eval().to(dev, dtype)
        return cls._MODELS[key]

    # --- per-backend seq encoders (bound in __init__ as self._encode) ---
    # Each returns (dense (B, P, C) | flat (B, D), global (B, C) | None). Splitting
    # the leading non-patch tokens HERE is what keeps `__call__` backend-blind.
    def _encode_theia(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._model.backbone.model(
            pixel_values=x, interpolate_pos_encoding=True).last_hidden_state  # (B, 1+P, C)
        return h[:, 1:], h[:, 0]

    def _encode_clip(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._model.vision_model(
            pixel_values=x, interpolate_pos_encoding=True).last_hidden_state  # (B, 1+P, C)
        return h[:, 1:], h[:, 0]

    def _encode_siglip(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # No CLS: the sequence is all patches and the global token is the MAP head's
        # pooled output (the PaliGemma usage), so `dense` takes no slice.
        o = self._model(pixel_values=x, interpolate_pos_encoding=True)
        return o.last_hidden_state, o.pooler_output

    def _encode_dino(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._model(pixel_values=x, **self._interp_kw).last_hidden_state  # (B, skip+P, C)
        return h[:, self._skip:], h[:, 0]

    def _encode_resnet(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self._model(x), None  # (B, D)

    @torch.no_grad()
    def _forward(self, env) -> tuple[torch.Tensor, torch.Tensor | None]:
        """This step's (dense, global), cached & shared across term instances."""
        rgb = env.scene[self._sensor_name].data.rgb  # (B, H, W, 3) uint8
        # (step, buffer ptr) so sibling terms sharing a sensor encode once. Neither
        # moves across a reset — `reset()` above is what invalidates that case.
        step = int(env.common_step_counter)
        ptr = rgb.data_ptr()
        hit = image_feature._FWD.get(self._ckey)
        if hit is not None and hit[0] == step and hit[1] == ptr:
            return hit[2]
        x = rgb.to(self._dev).permute(0, 3, 1, 2).to(self._dtype) / 255.0
        x = (x - self._mean) / self._std
        out = self._encode(x)  # backend-bound at init
        image_feature._FWD[self._ckey] = (step, ptr, out)
        return out

    @torch.no_grad()
    def __call__(self, env, **_kw) -> torch.Tensor:
        dense, glob = self._forward(env)
        if self._output == "cls":
            feats = glob
        else:
            feats = dense
            if self._flatten:
                feats = feats.reshape(feats.shape[0], -1)
        return feats.detach().float().to(env.device)
