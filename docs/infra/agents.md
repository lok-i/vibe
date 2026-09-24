# Agent architecture

Agent-side only (the vision backbone is env-side — the agent receives `kv_tokens`
already encoded). Signals are single letters at the **obs-group** level; blocks are
single words. `──►` inference (deploys), `╌╌►` train-time only. Legend at the bottom.

**Obs-group vocabulary** — every figure below speaks these groups (source of truth:
`<task>/config/g1/observation_cfgs.py`; frozen-base streams owned by `mocke`):

| group | signal | contents | consumer |
|---|---|---|---|
| `policy` / `tokenizer` | x,c | proprio history / future-ref window (SONIC contract, owned by `mocke`) | frozen SONIC base |
| `critic` | — | privileged (object refs, critic-only) | critic |
| `augmentation` | c | robot-motion-cmd VALUE (SUGAR c_t) | adapter stream |
| `kv_tokens` | t | vision tokens (B,P,C), env-side encoded (ImgFeat) | extractor K/V |
| `camera` | t | raw RGB (B,3,H,W) (ImgRgb) | CNN encoder |
| `q_task_cmd` | g | `object_goal_ori` | extractor query row |
| `q_motion_cmd` | c | robot-motion-cmd | extractor query row — **PARKED**; value still routes via `augmentation` |
| `q_proprio` | x | grav + base twist + joint pos/vel | extractor query row |
| `q_cls` | — | encoder global token (off by default; on with a CLS-bearing encoder) | extractor query row |
| `prediction_target` | s | object base-frame pose + twist | aux target |
| `prediction_conditioning` | r | proprio (no object state) | aux conditioning |

**Wiring contract:** every trainable block's port is an obs **GROUP**, never a slice;
swap an experiment = add/remove a term in a group (dims flow through, rsl_rl unchanged).

## 1 · Policy augmentations

> **Def.** Freeze the SONIC WBC; train a small delta on the `augmentation` stream. vibe ships
> the **adapter** form (weight-residual ΔW); base bit-exact at init.

**I/O (groups):** in = base stream + `augmentation`; out = action `a`.

```
 adapter · weight-residual ΔW  (LoRA, injected INTO the base)
 ────────────────────────────────────────────────────────────
   base stream ──►[ base° ]──►( a )
                      ▲
                      │ ΔW  (rank r, per layer)
                      ▼
   augmentation ──►[ adapter* ]
```

| | AdaptSonic |
|---|---|
| base | SONIC WBC, frozen |
| delta | ΔW (LoRA) on the decoder |
| delta stream | `augmentation` + z |
| action std | frozen base-band |
| model class | `SonicWithAdapterModel` / `ExtractorSonicAdapterModel` |

Notes: ΔW rides the **decoder** (the policy head); a delta upstream of the FSQ front-end is
snapped away by STE rounding. Std frozen at the base band because a learnable std inflates
~3-5× within ~1k updates and wrecks tracking. The action-residual **sidecar** form and the
from-scratch **TaRa** row both live in `orcs.core.rl` — vibe's axis is the perception,
not the head.

## 2 · Auxiliary modules (PPOAux)

> **Def.** Vision only. An **extractor** turns the vision obs into a latent z that feeds the
> adapter stream; an optional **predictor** trains that extractor with an auxiliary loss.
> Predictor is train-only — dropped at deployment.

**I/O (blocks):**

```
 vision obs ─►[ extractor* ]─► z ─►( adapter stream )
                               ┆
                               ▼
            aux obs, a ╌►[ predictor* ]╌►( L ) ╌► ∇ extractor
```

| row | task suffix | extractor | predictor |
|---|---|---|---|
| vanilla | `-ImgFeat` | ✗ | — |
| ext | `-ImgFeat-Ext` | ✓ | — (PPO grad only) |
| Sfd | `-ImgFeat-Sfd` | ✓ | StateFdAux |
| Lfd | `-ImgFeat-Lfd` | ✓ | LatentFdAux |

**Training (joint mode).** Every PPO minibatch backward is followed by an accumulated
`aux_weight · aux_loss` on a fresh aux minibatch ≡ one summed loss, aux gradient at every
PPO step. LR groups: actor+critic (adaptive-KL) | extractor | predictor (fixed 1e-4).
`Loss/` logs the true losses; diagnostics under the `Z*` sections (below).

### 2.1 · Extractor

> **Def.** Map the vision obs group(s) to a fixed z (128-d, LN'd), shared by the adapter
> stream and the predictor.

**I/O (groups):** in = `kv_tokens` (K/V) + query groups `{q_task_cmd, q_motion_cmd,
q_proprio[, q_cls]}` (one group → one row); out = z.

```
 CrossAttentionExtractor   (one W_q per query group → one row)
 ────────────────────────────────────────────────────────────
   q_task_cmd  ─►[norm]─►[ W_q ]─┐          live rows today; q_motion_cmd is
   q_proprio   ─►[norm]─►[ W_q ]─┼─► Q      PARKED (see the group table above)
   q_cls       ─►[norm]─►[ W_q ]─┘   │
   kv_tokens ─►[norm]─► W_k,W_v ─► K,V ─►[ attn ]─►[ proj ]─►[ LN ]─► z
                                  A = softmax(QKᵀ/√d),  z = LN(proj(flatten(A·V)))
```

| | MlpExtractor (**retired**) | CrossAttentionExtractor (**current**) |
|---|---|---|
| idea | norm + MLP over a flat vector | query-pool dense tokens, one query group → one row |
| token input | flat (B,D), pre-pooled | dense (B,P,C), P free at runtime |
| queries | none | `q_*` groups, own W_q each (+ learned, off) |
| query values | — | NOT carried into z (pure pooling; value routes via `augmentation`) |
| params | MLP-sized | ~60k (d=64, 3 rows) |

Design: single head (`num_heads` asserts 1; multi-head reserved); per-term token norm +
per-group query norm; roles **explicit cfg** (`token_terms`, `query_groups`), never
shape-inferred.

### 2.2 · Predictor

> **Def.** Train the extractor with a K-step forward-dynamics loss (K = 10,
> `unroll_length`). Two variants by **recursion space**.

**I/O (groups):** in = z + aux groups (per variant, below) + action `a`; out = loss `L`.

| | Sfd · StateFdAux | Lfd · LatentFdAux |
|---|---|---|
| kind | supervised (SL) | self-supervised (SSL) |
| recurses in | physical state ŝ | latent z |
| head | predictor `(ŝ,z,r,a)→ŝ′` | transition `z+mlp([z,a,r])` + projector |
| target | true s (`prediction_target`) | `sg[ema_enc]` (tau 0.99) |
| cond `r` | `prediction_conditioning` | same group (→ transition only) |
| loss | L1 (normalized target) | MSE (LayerNorm'd latent) |
| parent | AnyAdapter / OpenTrack | multimodal_rl `ForwardDynamics` |

Defaults: `K=10`; `autoregress`(`ive`)`=True`; Sfd `start_with_current_step=False`.

**Matched asks on the shared extractor** — both variants train the SAME extractor, so an
unmatched ask is a confound, not a result:

| | Sfd | Lfd | matched by |
|---|---|---|---|
| head inputs | ŝ, z, **r**, a | z, **r**, a | Lfd `condition_group` on by default |
| encoder-∇ rows / aux call | `(T//K)·K/nmb · N` | same | `mini_batch_rows_per_env` (`_AUX_ROWS_PER_ENV`, per-env ⇒ `-e`-agnostic) |
| `aux_weight` | 1.0 | 1.0 | untuned: both losses already in normalized units |
| ∇ steps | 1 / PPO minibatch | same | joint mode |

Residual, unfixable without redesigning Sfd: same row count, different decorrelation (Sfd =
`N/nmb` columns × consecutive steps; Lfd = scattered `(t,env)`). Lfd's `r` carries
`object_goal_color` too (it is in the group for Sfd's reward slice) — inert for Lfd, kept for
input parity. Where we depart from the parents: OpenTrack's world model also supervises the
POLICY (`policy_supervised_loss`) — we inherit only its regression loss, so Sfd is a
representation regularizer, not model-based RL; and Lfd mean-reduces where the parent sums
(≡ parent per-term, 1/(2K−1) overall).

**Modes — signal (`──►`) vs gradient (`╌►`) flow** (`[P]` predictor · `[T]` transition; K=3 shown):

```
 TEACHER FORCING (=False) — state re-grounded each step ⇒ K independent 1-step ∇
   Sfd   s0─►[P]─►ŝ¹      s1─►[P]─►ŝ²      s2─►[P]─►ŝ³     ŝ-input = TRUE s_k
              ▲                ▲                ▲
              z0               z1               z2         fresh z ─► into [P]
              ┆                ┆                ┆           ╌► L_k trains z_k only
              L0               L1               L2
   Lfd   z0─►[T]─►ẑ¹      z1─►[T]─►ẑ²      z2─►[T]─►ẑ³     fresh z_k on the spine
         ┆                ┆                ┆               ╌► L_k trains z_k only
         L0               L1               L2

 AUTOREGRESSIVE (=True, default) — prediction fed forward ⇒ ∇ spans steps
   Sfd   s0─►[P]─►ŝ¹─►[P]─►ŝ²─►[P]─►ŝ³     ŝ CHAINED (only s0 true); z still fresh each step
              ▲        ▲        ▲
              z0       z1       z2
              └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ L₂         ╌► late ∇ reaches z0 through the ŝ chain
                                           (+ every z_k keeps its own 1-step ∇)
   Lfd   z0─►[T]─►ẑ¹─►[T]─►ẑ²─►[T]─►ẑ³     ẑ CHAINED from z0 (k≥1 not re-encoded)
         ┆                                 ╌► only z0 is fresh ⇒ 1 encoder ∇ path ([T] trains)
                                           ⇒ default Lfd = TF sum ⊕ this AR chain
```

- **Why AR** — the chain drifts ŝ/ẑ off truth, so only the fresh image can re-anchor it: AR
  is what forces z to encode current object state (TF alone is ~solvable by dynamics).
- **`start_with_current_step`** (Sfd) — `True` drops the true seed: step k=0 becomes same-step
  regression `(0,z_t,r_t,0)→s_t`, fully image-grounded. Default `False` keeps the true seed.
- **Retired Reg probe** = Sfd `K=1, start_with_current_step=True, autoregress=False` — pure
  decodability of s from z, no dynamics.

**Diagnostics.** The flat `Auxiliaries/` bucket is gone, replaced by four `Z*` sections. Every
key, its formula and its failure mode live in
**[../perception/metrics.md](../perception/metrics.md)** — not restated here.

| section | answers | owner |
|---|---|---|
| `ZAttention/` | where z looks | extractor — fires on `-Ext` too |
| `ZCapacity/` | how much z carries | extractor — fires on `-Ext` too |
| `ZPrediction/` | is the objective's target predicted | the aux objective |
| `ZGradient/` | who trains z | `PPOAux` |

## Legend

```
 signals (obs-group level)
   x  robot state (proprio)      c  robot-motion-cmd (SUGAR c_t)   g  task/goal cmd
   t  vision tokens (kv_tokens)  z  extractor latent (128, LN)
   s  aux target (object base-frame state)   r  aux conditioning (proprio)   (sg) stop-grad
   a  action    ą  base action   δa  action residual   ΔW  LoRA weight residual

 blocks   base  frozen WBC policy · adapter  trained delta · extractor  vision→z · predictor  train-only
 marks    °  frozen    *  trainable    ──►  inference    ╌╌►  train-only    (L)  loss    ∇  trains
```
