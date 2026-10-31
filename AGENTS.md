# Working on ViBe

ViBe adapts a frozen SONIC controller with camera features and a LoRA adapter.
The four task families are Repose, PerLoco (Grail/OmRe), Uolm and Dodge.
ORCS owns the privileged task mechanics; ViBe owns the vision adaptation.
Keep shared mechanics in ORCS and task-independent ViBe code in `src/vibe/core`.

## Read first

- `readme.md`: installation, training, play, export and hardware deployment link.
- `docs/setup.md`: dependency ownership, data staging and troubleshooting.
- `docs/tasks.md`: task naming and training flags.
- `docs/architecture.md`: observations, policy and render domain.
- `CONTRIBUTING.md`: validation requirements and concise PR descriptions.
- `CLAUDE.md`: additional architectural rationale and known integration traps.

## Dependency changes

- `deps.lock` owns Git dependency revisions. `pyproject.toml` owns the pinned
  simulator stack. ViBe's lock takes precedence over a dependency's own lock.
- Use the repository-local uv `.venv` and `scripts/setup/sync_deps.sh`; use
  `--deploy` for ONNX extras. Reinstalling extras independently can replace
  the `rsl_rl` fork with mjlab's PyPI dependency.
- Assess an upgrade in an isolated checkout before changing the installed
  dependency. Compare train and play configurations, observation/action order,
  resets, rewards, terminations, terrain/rendering, CLI patches and export paths.
- Passing config tests does not establish rollout or checkpoint parity. Report
  validation limits and retain the current pin when an upgrade conflicts with
  a request to preserve behavior.
- `src/vibe/release.json` describes released checkpoint provenance. Do not
  relabel its compatibility revisions merely to match a new dependency pin.

## Integration rules

- Import `mjlab` before `orcs` in standalone scripts: mjlab's entry-point scan
  can otherwise re-enter partially initialized packages and skip ViBe tasks.
- ViBe's mjlab patches stack after ORCS's. Preserve the union of
  `orcs.MULTI_CLIP_CFGS` and the parent's `--agent` choices.
- Keep saved-config import paths working through re-exports and
  `core/_legacy_paths.py`; registration must not load a module twice.
- Return copies of mutable configuration values. The critic stays privileged;
  the scene's manipulated-object entity is named `object`.
- Play factories remove training randomization. Thin wrappers inherit ORCS's
  play overrides before adding ViBe terms; their guards alone cannot detect
  an upstream behavioral change.
- Keep the training encoder (`core/mdp/observations.py`) and export encoder
  (`encoders/zoo.py`) numerically aligned.

## Validation

```bash
.venv/bin/ruff check src tests
.venv/bin/python -m pytest tests/
bash scripts/setup/sync_deps.sh --check
bash scripts/setup/sync_data.sh --check
```

Tests require synced dependencies and task data. Inspect task registration and
`SKIP_REASON` values: missing data can reduce coverage while the suite passes.
Behavioral changes also need appropriate train/play/export smoke checks with
the required GPU and checkpoints. Keep investigation logs under `logs/`;
keep usage and design guidance in `docs/`.
