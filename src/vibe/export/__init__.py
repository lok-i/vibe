"""Export artifacts for the C++ ROS2 deploy side.

Two subpackages, one philosophy — the exporter owns all model knowledge, the
deploy node executes a self-describing artifact:

- `vibe.export.agent`   — policy graphs (`export-agent`, vibe.onnx.v1)
- `vibe.export.encoder` — frozen vision backbones (`export-encoder`, vision.onnx.v1)
"""

from vibe.export.manifest import build_manifest, duplicate_group_aliases

__all__ = ["build_manifest", "duplicate_group_aliases"]
