"""Frozen vision-encoder zoo — one adapter API per backbone family.

Consumed by `vibe.export.encoder` (the ONNX exporter) and offline analysis. See `zoo.ROSTER` for the tags.
"""

from vibe.encoders.zoo import ROSTER, TEXT_TAGS, load

__all__ = ["ROSTER", "TEXT_TAGS", "load"]
