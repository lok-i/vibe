"""Frozen vision encoders -> self-contained ONNX artifacts (`export-encoder`).

One graph per `vibe.encoders.zoo` tag, ALL preprocessing baked in — cast, /255,
BGR->RGB, HWC->CHW, per-model normalization (`zoo._Enc._norm`) — so the deploy
node's contract shrinks to: resize, memcpy BGR HWC uint8, run. Outputs are the
zoo's `dense` / `cls` verbatim (register/CLS handling inherited per backbone);
compute runs in `compute_dtype`, outputs always cast to fp32 (the wire contract).

Config: sibling `export_enc.yaml` — the single source of truth (resolution,
dtype, tags, gate tolerance). CLI only cherry-picks:

    export-encoder                       # every tag in the yaml
    export-encoder --tag theia-tiny      # one artifact
    export-encoder --out /path/models    # land artifacts elsewhere

Per tag: `<tag>.onnx` + `<tag>.manifest.json` (vision.onnx.v1, also embedded in
`metadata_props`), gated by a torch-vs-ORT parity check on random frames —
nothing is kept on failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from vibe.encoders import ROSTER, zoo
from vibe.export.manifest import build_encoder_manifest

_CONFIG = Path(__file__).parent / "export_enc.yaml"
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


class EncoderGraph(nn.Module):
    """The exported graph: BGR HWC uint8 frame in, fp32 tokens out.

    Wraps a `zoo._Enc`: normalization buffers come from `enc._mean/_std`, the
    forward is `enc._fwd` verbatim — parity with training/bench by construction.
    """

    def __init__(self, enc: zoo._Enc):
        super().__init__()
        self.model = enc._m  # registered so export sees parameters
        self.enc = [enc]  # plain list: keep the non-Module adapter out of state_dict
        self.compute_dtype = enc.dtype
        self.register_buffer("mean", enc._mean)
        self.register_buffer("std", enc._std)
        self.has_cls = None  # set on first forward

    def forward(self, image_bgr_hwc: torch.Tensor):
        x = image_bgr_hwc.flip(-1)  # BGR -> RGB (channel axis is size 3)
        x = x.permute(0, 3, 1, 2).to(self.compute_dtype) / 255.0
        x = (x - self.mean) / self.std
        out = self.enc[0]._fwd(x)
        self.has_cls = out["cls"] is not None
        if self.has_cls:
            return out["dense"].float(), out["cls"].float()
        return out["dense"].float()


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _embed_manifest(onnx_path: Path, manifest: dict) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    entry = model.metadata_props.add()
    entry.key = "manifest"
    entry.value = json.dumps(manifest)
    onnx.save(model, str(onnx_path))


def _parity_gate(graph: EncoderGraph, onnx_path: Path, height: int, width: int,
                 frames: int, tolerance: float, device: str) -> float:
    """Max abs err, torch (compute_dtype) vs ORT CPU, over random uint8 frames."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(frames):
        u8 = rng.integers(0, 256, size=(1, height, width, 3), dtype=np.uint8)
        with torch.no_grad():
            ref = graph(torch.from_numpy(u8).to(device))
        ref = ref if isinstance(ref, tuple) else (ref,)
        got = sess.run(None, {"image_bgr_hwc": u8})
        for r, g in zip(ref, got, strict=True):
            worst = max(worst, float(np.abs(r.cpu().numpy() - g).max()))
    if worst > tolerance:
        raise RuntimeError(f"parity gate FAILED: max|delta| = {worst:.3e} > {tolerance:.1e}")
    return worst


def export_tag(tag: str, cfg: dict, out_dir: Path) -> None:
    height, width = int(cfg["height"]), int(cfg["width"])
    dtype_name = str(cfg["compute_dtype"])
    device = str(cfg.get("device", "cpu"))
    opset = int(cfg["opset"])

    enc = zoo.load(tag, device=device, dtype=_DTYPES[dtype_name])
    graph = EncoderGraph(enc).eval()
    dummy = torch.zeros(1, height, width, 3, dtype=torch.uint8, device=device)
    with torch.no_grad():
        out = graph(dummy)
    dense = out[0] if isinstance(out, tuple) else out
    num_patches, token_dim = int(dense.shape[1]), int(dense.shape[2])
    expected = (height // enc.patch) * (width // enc.patch)
    assert num_patches == expected, (
        f"{tag}: measured P={num_patches} != grid {height // enc.patch}x{width // enc.patch}")

    onnx_path = out_dir / f"{tag}.onnx"
    manifest_path = out_dir / f"{tag}.manifest.json"
    output_names = ["patch_tokens", "cls_token"] if graph.has_cls else ["patch_tokens"]
    torch.onnx.export(
        graph, (dummy,), str(onnx_path),
        input_names=["image_bgr_hwc"], output_names=output_names,
        opset_version=opset, dynamo=False,
    )

    try:
        worst = _parity_gate(graph, onnx_path, height, width,
                             int(cfg.get("gate_frames", 8)),
                             float(cfg["tolerance"]), device)
        manifest = build_encoder_manifest(
            enc, height=height, width=width, num_patches=num_patches,
            token_dim=token_dim, has_cls=graph.has_cls,
            compute_dtype=dtype_name, opset=opset,
        )
        _embed_manifest(onnx_path, manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except Exception:
        onnx_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    mb = onnx_path.stat().st_size / 1e6
    print(f"[export] {tag:<14} P={num_patches:>3} ({height // enc.patch}x{width // enc.patch}) "
          f"D={token_dim:<4} cls={'y' if graph.has_cls else 'n'}  "
          f"gate {worst:.2e}  {mb:.1f} MB  -> {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=_CONFIG)
    parser.add_argument("--tag", action="append",
                        help="ROSTER tag (repeatable); default: every tag in the yaml")
    parser.add_argument("--out", type=Path, default=None, help="override yaml `out` dir")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    tags = args.tag or cfg["tags"]
    unknown = [t for t in tags if t not in ROSTER]
    if unknown:
        raise SystemExit(f"unknown tag(s) {unknown}; roster: {list(ROSTER)}")
    out_dir = args.out or Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for tag in tags:
        try:
            export_tag(tag, cfg, out_dir)
        except Exception as e:  # keep sweeping the zoo; summarize at the end
            failed.append(tag)
            print(f"[export] {tag:<14} FAILED: {e}")
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
