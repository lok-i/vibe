# Simulator API migration

The pinned main revisions introduce these changes for consumers of ViBe/ORCS.
Saved checkpoint provenance describes the training stack and is not repinned.

| API or behavior | Candidate |
|---|---|
| `mjlab.managers.event_manager.RecomputeLevel` | Removed upstream; use `mujoco_warp.SetConstGroup` |
| `requires_model_fields(..., recompute=...)` / `func.recompute` | Removed upstream; `func.model_fields` remains |
| `sim.recompute_constants(level)` | Accepts `SetConstGroup` flags; combine with bitwise OR, not `max()` |
| Direct event calls | Use `recompute_groups(fields, env.sim.mj_model)` and recompute after mutations; ensure affected and derived fields are expanded per world |
| Geometry/site world rotation | Scalar-first `geom_xquat` / `site_xquat`; framework pose APIs and viewers are adapted |
| Sparse solver flag | `env.sim.wp_model.is_sparse` |
| Contact capacity | `nacon` and `naconmax` are totals across worlds; `nefc` and `njmax` are per world |
| Old global body velocity rewards | Preserved by the source patch, including original defining import paths |
| G1 ankle mechanics | Preserved by the ORCS asset factory; new upstream coupled ankle stops excluded |
| Observation manager noise | Corruption disabled on the manager's copied terms; task configuration retains its noise specification |
| Image feature cache | Still keyed by step/buffer; manual state writes/rendering must clear `image_feature._FWD` when neither key changes |
| Sensor cache | Upstream now invalidates sensor caches after forward, step and reset |
| Actuator delay/reset and Gaussian/bias DR | Upstream corrections can change trajectories even with unchanged configuration |

For an already initialized environment whose event fields were expanded by the
event manager, manual replay of selected terms can use:

```python
from mjlab.managers.event_manager import recompute_groups
from mujoco_warp import SetConstGroup

groups = SetConstGroup.NONE
for term in selected_terms:
    term.func(env, env_ids, **term.params)
    groups |= recompute_groups(tuple(getattr(term.func, "model_fields", ())),
                               env.sim.mj_model)
if groups:
    env.sim.recompute_constants(groups)
env.sim.forward()
```

Introducing new randomized fields after initialization also requires the event
manager's corresponding derived-field expansion. `set_const` alone does not
allocate per-world storage. Preserve the event manager path where possible.

Environment/runner/registry imports, viewer classes, quaternion math helpers,
ViBe release loading, task factories, runner configuration, projection metrics,
attention helpers and ORCS `MISSING_DATA` retain their existing paths. ViBe task
factories still default to one environment. Cross-version numerical parity is a
separate validation question from import/configuration compatibility.

The tracking-specific evaluation CLI remains limited to tracking tasks. ViBe's
task-independent `scripts/repeat_check.py` uses training configurations and the
released actor for repeatability evaluation; both accept the determinism flag.
