"""THE task-agnostic vision obs library — what every vibe task shares.

Peer of `orcs.core.obs`: the atoms a task cannot disagree about. Here that is
the whole vision channel — which encoder, which dtype, what a token group is
called and what shape it has. A task assembles these into ITS groups in
`<task>/config/g1/observation_cfgs.py`; nothing here names a cube or a terrain.

    [encoder]   IMG_ENCODER / IMG_DTYPE         which backbone, which precision
    [naming]    TOKEN_GROUP / TOKEN_TERMS / CLS_GROUP   the wiring contract
    [atoms]     img_tokens_term / img_cls_term  one ObservationTermCfg each
    [groups]    kv_tokens_group / cls_query_group

Group NAMES live here because both sides read them — the env builds the group,
the agent cfg names it in `extractor_cfg` — and a name spelled twice drifts.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from orcs.core.obs import T as _T
from orcs.core.obs import grp as _grp

from vibe.core.mdp.observations import image_feature

__all__ = [
    "IMG_ENCODER", "IMG_DTYPE",
    "TOKEN_GROUP", "TOKEN_TERMS", "CLS_GROUP", "CAMERA_GROUP",
    "CamSpec", "img_tokens_term", "img_cls_term", "img_flat_term",
    "kv_tokens_group", "cls_query_group", "camera_group",
]

# --- default frozen vision encoder --------------------------------------------
# Theia-tiny: kept over TinyCLIP for dense localization, the axis the extractor
# pools on. Swap per run with `--env.img-encoder <hf-id>` (`VibeEnvCfg`).
IMG_ENCODER = "theia-tiny-patch16-224-cddsv"  # `image_feature._load` prefixes "theaiinstitute/"
IMG_DTYPE = "float16"  # encoder COMPUTE dtype (obs still returns fp32) — bench parity, halves cost

# --- group naming (single source of truth; imported env-side AND agent-side) ---
TOKEN_GROUP = "kv_tokens"                                    # extractor K/V (dict group)
TOKEN_TERMS = ("img_tokens",)                                # (B, P, C) dense token term(s)
CLS_GROUP = "q_cls"                                          # encoder global-token query
CAMERA_GROUP = "camera"                                      # raw RGB (B, 3, H, W) — ImgRgb


@dataclass(frozen=True)
class CamSpec:
    """Which camera, read through which encoder — the vision half of an ObsCtx.

    Mixed into each task's ObsCtx alongside orcs's task context, so the three
    fields are declared once no matter how many tasks carry a camera.
    """

    sensor: str = "head_cam"
    model: str = IMG_ENCODER
    model_dtype: str = IMG_DTYPE  # encoder compute dtype; obs out is always fp32


# ---------------------------------------------------------------------------
# Vision atoms — the only term bundles vibe owns
# ---------------------------------------------------------------------------

def img_tokens_term(sensor: str, model: str, model_dtype: str = "float32") -> ObservationTermCfg:
    return _T(image_feature, {"sensor_name": sensor, "model_name": model,
                              "model_dtype": model_dtype, "output": "tokens", "flatten": False})


def img_cls_term(sensor: str, model: str, model_dtype: str = "float32") -> ObservationTermCfg:
    return _T(image_feature, {"sensor_name": sensor, "model_name": model,
                              "model_dtype": model_dtype, "output": "cls"})


def img_flat_term(sensor: str, model: str, model_dtype: str = "float32") -> ObservationTermCfg:
    """Flat features (B, P*C) — the no-extractor path, straight into a stream."""
    return _T(image_feature, {"sensor_name": sensor, "model_name": model,
                              "model_dtype": model_dtype, "flatten": True})


# ---------------------------------------------------------------------------
# The shared groups
# ---------------------------------------------------------------------------

def kv_tokens_group(c: CamSpec) -> ObservationGroupCfg:
    """Extractor K/V: dense vision tokens, kept as (B, P, C) (dict group)."""
    return _grp({"img_tokens": img_tokens_term(c.sensor, c.model, c.model_dtype)}, concat=False)


def camera_group(c: CamSpec) -> ObservationGroupCfg:
    """Raw RGB (B, 3, H, W) for the ImgRgb baseline — same sensor, no encoder."""
    from mjlab.tasks.manipulation import mdp as manipulation_mdp

    return _grp({"rgb": _T(manipulation_mdp.camera_rgb, {"sensor_name": c.sensor})})


def cls_query_group(c: CamSpec) -> ObservationGroupCfg:
    """Encoder global token as a query row — on for every backbone.

    Theia's CLS is untrained, yet the row stays: a diffuse attention row IS a
    mean-pool row, so this is the one global-pool path into z. Without it that
    path is inexpressible; with two such rows they duplicate `proj` params.
    """
    return _grp({"img_cls": img_cls_term(c.sensor, c.model, c.model_dtype)})
