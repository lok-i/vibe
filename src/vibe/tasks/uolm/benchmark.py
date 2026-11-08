"""UOLM's native repertoire benchmark with ViBe's nominal vision domain."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from orcs.tasks.uolm.benchmark import SPEC as ORCS_SPEC

from vibe.core.env_cfgs import prepare_nominal_vision


def prepare(cfg: ManagerBasedRlEnvCfg, nominal: bool) -> None:
    ORCS_SPEC.prepare(cfg, nominal)
    if nominal:
        prepare_nominal_vision(cfg)


SPEC = replace(ORCS_SPEC, prepare=prepare)
