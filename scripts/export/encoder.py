"""Thin shim — the encoder exporter lives in `vibe.export.encoder` (installed as `export-encoder`)."""

from vibe.export.encoder.export_onnx import main

if __name__ == "__main__":
    main()
