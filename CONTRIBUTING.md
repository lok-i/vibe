# Contributing

- Setup and commands: [README](readme.md) and [setup guide](docs/setup.md).
- Task contracts and design: [tasks](docs/tasks.md) and [architecture](docs/architecture.md).
- PR structure: [template](.github/PULL_REQUEST_TEMPLATE.md).

## Write for review

- Lead with the problem and resulting behavior. Describe the final change.
- Use short bullets, concrete names and direct links. Use tables for comparisons.
- State the release impact, breaking changes and any required migration.
- Keep small PRs small: delete unused template sections and rows.
- Report commands and observed results, including skipped checks and their limits.
- Link larger evidence; keep development history and raw debug logs out of the description.

## Changes and dependencies

| Change | Include |
|---|---|
| Task or runtime behavior | Before/after behavior, reproduction command and relevant rollout evidence |
| Git dependency | Upstream PR, old → new SHA in `deps.lock`, setup and compatibility checks |
| Motion data or assets | Effect on selected clips, physics and policy observations |
| Released checkpoint | Immutable revision and checksum in `src/vibe/release.json` |
| Third-party code or data | Attribution, license and access conditions in [notices](THIRD_PARTY_NOTICES.md) |

- ViBe owns its dependency lock. A code upgrade does not require adopting every upstream data pin.
- Change shared mechanics in their owning repository; use published upstream commits when repinning.
- Preserve released checkpoint provenance; record newly validated compatibility separately.
- Keep datasets, checkpoints, recordings and investigation logs out of source commits.

## Validate the change

| Scope | Check |
|---|---|
| Every PR | `git diff --check`; review the final diff |
| Python | `.venv/bin/ruff check src tests`; relevant `.venv/bin/python -m pytest tests/` |
| Packaging or shared dependencies | Full contract suite; inspect task registration and missing-data skips |
| Runtime behavior | Relevant train/play checks and released-policy rollouts; export checks when affected |
| Dependency pins | `bash scripts/setup/sync_deps.sh --check`; `bash scripts/setup/sync_data.sh --check` |
| Documentation only | Verify links, commands and examples |

Add tests for meaningful contracts. Config tests do not establish rollout parity;
state what was not checked and why. Keep detailed local evidence under `logs/`.
