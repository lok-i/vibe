"""Frozen vision backbones, one adapter per family — the `export-encoder` roster.

    enc = load("theia-tiny", device="cpu", dtype=torch.float32)
    out = enc._fwd(x)   # x: normalized (N, 3, H, W) -> {"dense": (N, P, D), "cls": (N, D) | None}

Frames run at their NATIVE resolution, as `vibe.core.mdp.observations.image_feature` runs
them in training (`interpolate_pos_encoding=True`, no resize): at the 112x63 head cam a
stride-16 backbone yields P = 7x3 = 21. Vision towers only — no text weights are loaded.
dinov3 checkpoints are HF-gated (accept the license once with your HF token).
"""

from __future__ import annotations

import inspect

import torch

_IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
_HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class _Enc:
    """Base adapter. Subclasses set `_norm` and implement `_load` (sets `_m`, `patch`)
    and `_fwd` (normalized pixels -> {"dense", "cls"})."""

    _norm: tuple = _IMNET

    def __init__(self, tag: str, mid: str, device: str, dtype: torch.dtype):
        self.tag, self.mid, self.dev, self.dtype = tag, mid, device, dtype
        self.patch: int | None = None
        self._load()
        mean, std = self._norm
        self._mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)

    def _load(self):
        raise NotImplementedError

    def _fwd(self, x: torch.Tensor) -> dict:
        raise NotImplementedError


class _TheiaEnc(_Enc):
    def _load(self):
        from transformers import AutoModel
        self._m = AutoModel.from_pretrained(self.mid, trust_remote_code=True) \
            .eval().to(self.dev, self.dtype)
        self._vit = self._m.backbone.model
        self.patch = self._vit.config.patch_size

    def _fwd(self, x):
        h = self._vit(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        return {"dense": h[:, 1:], "cls": h[:, 0]}


class _ClipEnc(_Enc):
    """HF CLIP vision tower (openai CLIP, TinyCLIP)."""

    _norm = _CLIP

    def _load(self):
        from transformers import CLIPVisionModel
        self._m = CLIPVisionModel.from_pretrained(self.mid).eval().to(self.dev, self.dtype)
        self.patch = self._m.config.patch_size

    def _fwd(self, x):
        h = self._m.vision_model(pixel_values=x,
                                 interpolate_pos_encoding=True).last_hidden_state
        return {"dense": h[:, 1:], "cls": h[:, 0]}


class _SiglipEnc(_Enc):
    """SigLIP 2 vision tower. No CLS token: every token is a patch."""

    _norm = _HALF

    def _load(self):
        from transformers import AutoModel
        self._m = AutoModel.from_pretrained(self.mid).vision_model \
            .eval().to(self.dev, self.dtype)
        self.patch = self._m.config.patch_size

    def _fwd(self, x):
        h = self._m(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        return {"dense": h, "cls": None}


class _DinoEnc(_Enc):
    """DINOv2/v3. Dense skips CLS + register tokens; DINOv3 resizes via RoPE, so the
    interpolate flag is passed only where `forward` takes it."""

    def _load(self):
        from transformers import AutoModel
        self._m = AutoModel.from_pretrained(self.mid).eval().to(self.dev, self.dtype)
        self._skip = 1 + getattr(self._m.config, "num_register_tokens", 0)
        self.patch = self._m.config.patch_size
        self._kw = ({"interpolate_pos_encoding": True}
                    if "interpolate_pos_encoding" in inspect.signature(type(self._m).forward).parameters
                    else {})

    def _fwd(self, x):
        h = self._m(pixel_values=x, **self._kw).last_hidden_state
        return {"dense": h[:, self._skip:], "cls": h[:, 0]}


ROSTER: dict[str, tuple[str, type[_Enc]]] = {
    "theia-tiny": ("theaiinstitute/theia-tiny-patch16-224-cddsv", _TheiaEnc),
    "tinyclip-39m": ("wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M", _ClipEnc),
    "clip-b32": ("openai/clip-vit-base-patch32", _ClipEnc),
    "siglip2-b16": ("google/siglip2-base-patch16-224", _SiglipEnc),
    "dinov2-s": ("facebook/dinov2-small", _DinoEnc),
    "dinov3-splus": ("facebook/dinov3-vits16plus-pretrain-lvd1689m", _DinoEnc),
}


def load(tag: str, device: str = "cuda:0", dtype: torch.dtype = torch.float16) -> _Enc:
    mid, cls = ROSTER[tag]
    return cls(tag, mid, device, dtype)
