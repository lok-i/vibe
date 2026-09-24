"""The two per-RUN vision knobs, read back off the W&B run that made a checkpoint.

Rebuilding an arm from the task id alone is wrong the moment a run set
`--env.img-encoder` or `--agent.drop-query-rows`: both move TENSOR SHAPES (token
channel dim / `proj` fan-in), so a mismatch is a strict-load error at best and a
silently wrong backbone at worst (two backbones sharing C).
"""

from __future__ import annotations

AUTO = "<from-run>"
"""CLI sentinel: take this knob from the run, not from the flag."""


def run_arm(run_path: str | None) -> dict:
    """`{img_encoder, drop_query_rows}` for a W&B run; the task defaults when None."""
    if not run_path:
        return {"img_encoder": None, "drop_query_rows": ""}
    import wandb

    cfg = wandb.Api().run(run_path).config
    return {
        "img_encoder": (cfg.get("env_cfg") or {}).get("img_encoder") or None,
        "drop_query_rows": (cfg.get("train_cfg") or {}).get("drop_query_rows") or "",
    }


def apply_arm(arm: dict, env_cfg=None, agent_cfg=None) -> None:
    """Bind an arm onto built cfgs. `__post_init__` is the only definition in each
    MRO, so re-calling it is the sanctioned re-resolve: it rebinds BOTH image terms
    and prunes the query groups, asserting every named row exists."""
    if env_cfg is not None and arm.get("img_encoder"):
        if not hasattr(env_cfg, "img_encoder"):
            raise SystemExit("--img-encoder: this task carries no vision knobs.")
        env_cfg.img_encoder = arm["img_encoder"]
        env_cfg.__post_init__()
    if agent_cfg is not None and arm.get("drop_query_rows"):
        if not hasattr(agent_cfg, "drop_query_rows"):
            raise SystemExit("--drop-query-rows: this task carries no extractor knobs.")
        agent_cfg.drop_query_rows = arm["drop_query_rows"]
        agent_cfg.__post_init__()
