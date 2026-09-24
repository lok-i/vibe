# metrics

What the extractor and the aux objectives log, under four W&B sections.

| key | good direction | range | a bad value means |
|---|---|---|---|
| `ZAttention/<row>` | ↓ | [0, 1], 1 = uniform | at 1 the row is a mean-pool: delete it or fix its query |
| `ZAttention/query_div` | ↑ | [0, 1] | → 0: the query rows attend alike, whatever their entropy |
| `ZCapacity/rankme`, `rank_frac` | ↑ | 1 … 128 | → 1: z collapsed; flat and low: wasted latent width |
| `ZCapacity/std` | ↑ | ≤ 1 | → 0: collapse RankMe cannot see |
| `ZPrediction/total`, `/<term>` | ↓ | ~1 = predicting the mean | high and flat: unlearnable at this horizon (read against `ZPrediction/ruler`) |
| `ZPrediction/gain_frac_*` | ↑ | 0 = z unused | ~0: the target was free without the image |
| `ZPrediction/floor` (`Lfd`) | — | do-nothing baseline | `total` must beat it |
| `ZGradient/frac`, `pull_frac` | — | 0.5 = parity | aux vs PPO share of the gradient on z |
| `ZGradient/cos` | — | [-1, 1] | < 0: the two objectives fight over z |
| `Episode_Metrics/object_in_fov` | ↑ | [0, 1] | low and flat: a camera-aim problem, not an extractor one |
| `Metrics/motion/at_goal_color` · `_ever` | ↑ | [0, 1] | repose success at the terminal step · at any frame |

Three rules:

1. `ZAttention` and `ZCapacity` belong to the extractor. They are the only sections
   comparable across `-Ext` / `-Sfd` / `-Lfd`.
2. `ZPrediction` depends on the target set; never compare it between variants.
3. `Metrics/motion/*` is logged at the reset step, so each is a terminal-step rate, not an
   episode mean.

Attention entropy is normalized by log P, so it survives a camera-resolution change.
`ZAttention` / `ZCapacity` sample the last minibatch of each update: read them smoothed.
