"""Thin shim — the policy exporter lives in `vibe.export.agent` (installed as `export-agent`)."""

from vibe.export.agent.export_onnx import main

if __name__ == "__main__":
    main()
