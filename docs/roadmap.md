# roadmap

| # | item | what |
|---|---|---|
| 1 | orcs on `main` | `deps.lock` pins orcs's dev branch `setup/lean-no-smpl`; re-pin once it lands on `main` |
| 2 | behavior tests | headless play of each released checkpoint, metrics held above a floor, beside the `pytest` contracts |
| 3 | mjlab ≥ 1.6 | its native visual DR (light, textures, `mat_texid`) for the render domain. 1.6 also renders the skybox into camera sensors, so the policy sees new images: a retrain |
| 4 | rsl_rl upstream | upstream's native bf16 AMP (5.5.0): bump if it holds the fork's fp32-head seam, else keep the fork |
