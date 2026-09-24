"""Uniform frozen-encoder adapters for offline analysis.

One adapter per backbone family, one API:

    enc = load("siglip2-b16", device="cuda:0")
    out = enc.encode(frames_u8)   # {"dense": (N,P,D), "cls": (N,D)|None,
                                  #  "gpool": (N,D)|None, "dense_txt": (N,P,Dt)|None}
    z   = enc.text(prompts)       # (M, Dt) image-text space | None

`dense` = raw patch tokens (probe/PCA space); `dense_txt` = patches projected into
the shared image-text space (MaskCLIP-style dense CLIP), paired-text models only;
`gpool` = the model's own trained global vector (CLS projection / MAP head / conv
pool). `grid_hw` and `proc_hw` are valid after the first encode.

Frames go in at their NATIVE resolution — the deployment condition, matching
`mdp.observations.image_feature` (`interpolate_pos_encoding=True`, no resize). At
the 112x63 head cam a stride-16 backbone therefore yields P = 7x3 = 21, not the
196 a square 224 resize would give. `load(..., native=False)` restores the square
resize for an A/B.

Notes: dinov3 ckpts are HF-gated (accept the license once with your HF token);
mobileclip2 needs `pip install open_clip_torch>=2.33` and cannot run native (its
conv trunk requires a stride-multiple input) — it pads instead, see `_MobileClipEnc`.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch
import torch.nn.functional as F

_IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
_HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
_NONE = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def _accepts(fn, name: str) -> bool:
    return name in inspect.signature(fn).parameters


class _Enc:
    """Base adapter: preprocessing + batched no-grad forward.

    Subclasses set `_norm` / `has_text`, implement `_load` (which must also set
    `self.patch`) and `_fwd` (+ `_txt`).
    """

    _norm: tuple = _IMNET
    input_res: int = 224   # square fallback when native=False
    has_text: bool = False
    pad_to: int = 0        # >0: pad H,W up to a multiple of this instead of resizing

    def __init__(self, tag: str, mid: str, device: str, dtype: torch.dtype,
                 native: bool = True):
        self.tag, self.mid, self.dev, self.dtype = tag, mid, device, dtype
        self.native = native
        self.patch: int | None = None
        self.grid_hw: tuple[int, int] | None = None
        self.proc_hw: tuple[int, int] | None = None  # HW actually fed to the model
        self._load()
        mean, std = self._norm
        self._mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)

    def _load(self):
        raise NotImplementedError

    def _fwd(self, x: torch.Tensor) -> dict:
        raise NotImplementedError

    def _pre(self, frames_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(self.dev)
        x = x.permute(0, 3, 1, 2).to(self.dtype) / 255.0
        if not self.native:
            x = F.interpolate(x, size=self.input_res, mode="bilinear", antialias=True)
        elif self.pad_to:
            h, w = x.shape[-2:]
            ph, pw = (-h) % self.pad_to, (-w) % self.pad_to
            x = F.pad(x, (0, pw, 0, ph))  # bottom/right only, so pixel coords survive
        self.proc_hw = tuple(x.shape[-2:])
        return (x - self._mean) / self._std

    def _grid(self) -> tuple[int, int]:
        h, w = self.proc_hw
        return (h // self.patch, w // self.patch)

    @property
    def covered_hw(self) -> tuple[int, int]:
        """Pixel extent the token grid actually spans, origin at the top-left.

        The patch conv drops the remainder, so at 112x63 with stride 16 the bottom
        15 rows have no token. Any pixel->cell mapping must crop to THIS, not to the
        frame, or every grid metric is misregistered by half a cell.
        """
        gh, gw = self.grid_hw
        return (gh * self.patch, gw * self.patch) if self.patch else self.proc_hw

    def map_bbox(self, bbox: np.ndarray, cam_hw: tuple[int, int]) -> np.ndarray:
        """Camera-pixel bbox (x0,y0,x1,y1) -> `proc_hw` pixels. Identity when native."""
        cam_h, cam_w = cam_hw
        ph, pw = self.proc_hw
        if (ph, pw) == (cam_h, cam_w) or self.pad_to:  # pad keeps the origin
            return bbox
        s = np.array([pw / cam_w, ph / cam_h, pw / cam_w, ph / cam_h], dtype=np.float32)
        return np.where(bbox < 0, bbox, bbox * s)

    @torch.no_grad()
    def encode(self, frames_u8: np.ndarray, bs: int = 64) -> dict:
        outs: list[dict] = []
        for i in range(0, len(frames_u8), bs):
            outs.append(self._fwd(self._pre(frames_u8[i:i + bs])))
        cat = {k: (torch.cat([o[k] for o in outs]) if outs[0][k] is not None else None)
               for k in outs[0]}
        if self.grid_hw is None:
            self.grid_hw = self._grid()
        gh, gw = self.grid_hw
        assert gh * gw == cat["dense"].shape[1], (
            f"{self.tag}: grid {gh}x{gw} != P={cat['dense'].shape[1]} "
            f"(proc {self.proc_hw}, patch {self.patch})")
        return cat

    @torch.no_grad()
    def text(self, prompts: list[str]) -> torch.Tensor | None:
        return self._txt(prompts) if self.has_text else None

    def _txt(self, prompts: list[str]) -> torch.Tensor:
        raise NotImplementedError


class _TheiaEnc(_Enc):
    _norm = _IMNET

    def _load(self):
        from transformers import AutoModel
        self._m = AutoModel.from_pretrained(self.mid, trust_remote_code=True) \
            .eval().to(self.dev, self.dtype)
        self._vit = self._m.backbone.model
        self.patch = self._vit.config.patch_size

    def _fwd(self, x):
        h = self._vit(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        return {"dense": h[:, 1:], "cls": h[:, 0], "gpool": None, "dense_txt": None}


class _ClipEnc(_Enc):
    """HF CLIPModel family (openai CLIP, TinyCLIP). dense_txt = visual_projection
    after post_layernorm, the standard dense-CLIP recipe."""

    _norm = _CLIP
    has_text = True

    def _load(self):
        from transformers import AutoTokenizer, CLIPModel
        self._m = CLIPModel.from_pretrained(self.mid).eval().to(self.dev, self.dtype)
        self._tok = AutoTokenizer.from_pretrained(self.mid)
        self.patch = self._m.config.vision_config.patch_size

    def _fwd(self, x):
        h = self._m.vision_model(pixel_values=x,
                                 interpolate_pos_encoding=True).last_hidden_state
        hn = self._m.vision_model.post_layernorm(h)
        proj = self._m.visual_projection(hn)
        return {"dense": h[:, 1:], "cls": h[:, 0],
                "gpool": proj[:, 0], "dense_txt": proj[:, 1:]}

    def _txt(self, prompts):
        toks = self._tok(prompts, padding=True, return_tensors="pt").to(self.dev)
        return self._m.get_text_features(**toks).float()


class _SiglipEnc(_Enc):
    """SiglipModel (siglip2). No CLS token — all-patch sequence + MAP-head pool;
    patch tokens share the text width (the PaliGemma usage) -> dense_txt = dense."""

    _norm = _HALF
    has_text = True

    def _load(self):
        from transformers import AutoModel, AutoProcessor
        self._m = AutoModel.from_pretrained(self.mid).eval().to(self.dev, self.dtype)
        self._proc = AutoProcessor.from_pretrained(self.mid)
        self.patch = self._m.config.vision_config.patch_size

    def _fwd(self, x):
        vm = self._m.vision_model(pixel_values=x, interpolate_pos_encoding=True)
        return {"dense": vm.last_hidden_state, "cls": None,
                "gpool": vm.pooler_output, "dense_txt": vm.last_hidden_state}

    def _txt(self, prompts):
        toks = self._proc(text=prompts, padding="max_length",
                          return_tensors="pt").to(self.dev)
        return self._m.get_text_features(**toks).float()


class _DinoEnc(_Enc):
    """DINOv2/v3. Skips CLS + registers for dense; CLS is the DINO-loss-trained
    global. Position embeddings interpolate inside the model, so no flag is
    needed. dinov3 checkpoints are HF-gated."""

    _norm = _IMNET

    def _load(self):
        from transformers import AutoModel
        self._m = AutoModel.from_pretrained(self.mid).eval().to(self.dev, self.dtype)
        self._skip = 1 + getattr(self._m.config, "num_register_tokens", 0)
        self.patch = self._m.config.patch_size
        self._kw = ({"interpolate_pos_encoding": True}
                    if _accepts(type(self._m).forward, "interpolate_pos_encoding")
                    else {})

    def _fwd(self, x):
        h = self._m(pixel_values=x, **self._kw).last_hidden_state
        return {"dense": h[:, self._skip:], "cls": h[:, 0],
                "gpool": None, "dense_txt": None}


class _MobileClipEnc(_Enc):
    """MobileCLIP2 via open_clip (FastViT hybrid). dense = final-stage feature map
    from the timm trunk (conv grid, coarser than ViT-p16); gpool = encode_image.

    The trunk downsamples by 32 and rejects non-multiples, so native input is
    zero-padded bottom/right rather than resized — pixel coordinates survive, and
    `proc_hw` reports the padded extent the grid actually spans.

    `mid` is "<arch>:<pretrained_tag>" (e.g. "MobileCLIP2-S2:dfndr2b"): open_clip
    resolves the ckpt from its own registry, NOT the HF repo (apple's MobileCLIP2
    repos ship weights but no `open_clip_config.json`, so `hf-hub:apple/...` 404s).
    An "org/name" mid without ':' falls back to the hf-hub loader.
    """

    _norm = _NONE
    input_res = 256
    has_text = True
    pad_to = 32

    def _load(self):
        try:
            import open_clip
        except ImportError as e:
            raise RuntimeError(
                f"{self.tag} needs `pip install -U open_clip_torch>=2.33`") from e
        if ":" in self.mid:                      # "<arch>:<pretrained>" registry path
            name, mk = self.mid.split(":", 1)[0], dict(pretrained=self.mid.split(":", 1)[1])
        else:                                    # "org/repo" -> hf-hub config path
            name, mk = f"hf-hub:{self.mid}", {}
        self._m, _, pre = open_clip.create_model_and_transforms(name, **mk)
        self._m = self._m.eval().to(self.dev, self.dtype)
        self._tok = open_clip.get_tokenizer(name)
        for t in getattr(pre, "transforms", []):  # honor the ckpt's own norm
            if hasattr(t, "mean"):
                self._norm = (tuple(t.mean), tuple(t.std))
        if hasattr(pre, "transforms") and hasattr(pre.transforms[0], "size"):
            s = pre.transforms[0].size
            self.input_res = s if isinstance(s, int) else s[0]

    def _fwd(self, x):
        fm = self._m.visual.trunk.forward_features(x)  # (N, C, h, w)
        self.grid_hw = tuple(fm.shape[-2:])
        return {"dense": fm.flatten(2).transpose(1, 2), "cls": None,
                "gpool": self._m.encode_image(x), "dense_txt": None}

    def _grid(self):
        return self.grid_hw

    def _txt(self, prompts):
        return self._m.encode_text(self._tok(prompts).to(self.dev)).float()


ROSTER: dict[str, tuple[str, type[_Enc]]] = {
    "theia-tiny": ("theaiinstitute/theia-tiny-patch16-224-cddsv", _TheiaEnc),
    "tinyclip-39m": ("wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M", _ClipEnc),
    "clip-b32": ("openai/clip-vit-base-patch32", _ClipEnc),
    "siglip2-b16": ("google/siglip2-base-patch16-224", _SiglipEnc),
    "dinov2-s": ("facebook/dinov2-small", _DinoEnc),
    "dinov3-splus": ("facebook/dinov3-vits16plus-pretrain-lvd1689m", _DinoEnc),
    "mobileclip2-s2": ("MobileCLIP2-S2:dfndr2b", _MobileClipEnc),
}

TEXT_TAGS = [t for t, (_, c) in ROSTER.items() if c.has_text]


def load(tag: str, device: str = "cuda:0", dtype: torch.dtype = torch.float16,
         native: bool = True) -> _Enc:
    mid, cls = ROSTER[tag]
    return cls(tag, mid, device, dtype, native=native)
