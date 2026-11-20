# roadmap

| # | status | item | what |
|---|---|---|---|
| 1 | ✅ | orcs on `main` | `deps.lock` pins ORCS `v0.1.0` (`49886d1`) |
| 2 | ⬜ | behavior tests | headless play of each released checkpoint, metrics held above a floor, beside the `pytest` contracts |
| 3 | 🔧 | mjlab 1.6.1 + MuJoCo Warp main | Local candidate: native visual DR and selectable Warp/OptiX; PerLoco remains Warp-only. OptiX hardware validation and full training remain pending. [Design and migration](simulator-upgrade.md). |
| 4 | ⬜ | rsl_rl upstream | upstream's native bf16 AMP (5.5.0): bump if it holds the fork's fp32-head seam, else keep the fork |
