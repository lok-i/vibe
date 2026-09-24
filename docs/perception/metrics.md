# Perception metrics

Everything logged about **z** — the extracted perception signal — under the four `Z*`
sections. What each number means, how it is computed, where it lives.

**This file IS the watch list.** CLAUDE.md carries the pointer and the three reading rules,
nothing more. Design context: [encoders.md](encoders.md) · wiring:
[../infra/agents.md](../infra/agents.md).

All of it answers ONE question in different cuts: **is the image path carrying anything, and
where does it go?** The policy stream cannot answer it — PPO reward moves for a dozen reasons.

```
   img_tokens (B,P,C) ──┐
                        ├─► CrossAttentionExtractor ──► z (B,128, LayerNorm'd) ──► adapter ──► actions
   q_* query groups ────┘        │                          │
                                 │                          └─► aux predictor ──► ŝ  (train only)
                        attn (B,Q,P)                                    │
                                 │                                      │
                          §1 ZAttention/                         §2 ZPrediction/
                          §3 ZCapacity/                          §2 ZPrediction/gain_*
                                 └────────── §4 ZGradient/ ──────────┘
```

| § | section | answers | owner | fires on | offline (§5) |
|---|---|---|---|---|---|
| §1 | `ZAttention/` | where z looks | extractor | **every** run with a `CrossAttentionExtractor`, incl. `-Ext` | ✅ |
| §2 | `ZPrediction/` | is the objective's target predicted, and how much of that is z | aux objective | `-Sfd`, `-Lfd` | ❌ |
| §3 | `ZCapacity/` | how much z carries | extractor | **every** extractor run, incl. `-Ext` | ✅ |
| §4 | `ZGradient/` | who trains z | `PPOAux` | aux runs (`-Sfd`, `-Lfd`) | ❌ |

**§5 is the post-training counterpart** — the same subject measured on rollouts of a *trained*
policy, which answers a different class of question (design: *which query row earns its keep?*)
and drops the predictor entirely.

**§1 and §3 are the portable core.** They describe z itself, not any objective's fit to it, so
they are the *only* sections comparable across `-Ext` / `-Sfd` / `-Lfd`. That is why they live
on the extractor and are drained in base `PPO.update()`
([ppo.py:413-432](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/algorithms/ppo.py#L413-L432)) — the `-Ext` row
(extractor + plain PPO, no objective) reports them, and it is the baseline the aux rows are read
against. §2 is objective-specific by construction; **never cross-read it between variants.**
Across aux choices the standard of comparison is §1, §3, and task success — nothing else.

**Routing.** An aux metric keyed `loss/<x>` lands in the runner's `Loss/` section; every other
key is passed through verbatim, so the producer owns its section name
([ppo_aux.py:207-211](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/algorithms/ppo_aux.py#L207-L211)). Hence
`Loss/fd_l1` (the loss) alongside `ZPrediction/total` (the same number, mirrored so the section
reads on its own). With several extractors registered the group name folds in as
`ZAttention/<group>.<row>`; with one — today's case — the keys stay short.

---

## §1 `ZAttention/` — where z looks

### definition

**How hard each query row is looking at one place, and whether the rows look at different
places.** Each query group owns one attention row over the P image tokens. A row at the
ceiling is uniform, so `attn @ V ≈ mean(V)` — it is a **mean-pool path with extra
parameters**, and any two rows at the ceiling are the same row.

### formulation

Per query row $q$ over $P$ tokens, batch-averaged, then normalized by its own ceiling:

$$H_q = \frac{1}{B}\sum_{b}\Big[-\sum_{p} a_{bqp}\log\big(a_{bqp}+10^{-12}\big)\Big],
\qquad a_{bq\cdot}=\mathrm{softmax}\!\left(\frac{q_{bq}K_b^{\top}}{\sqrt{d}}\right)$$

$$\boxed{\texttt{ZAttention/<row>} = H_q / \log P \in [0,1]}$$

1 = diffuse (mean-pool), 0 = a single token. Normalizing is what makes the number survive a
camera or patch-size change — the raw-nats ceiling moves with $P$, this does not.

$$\boxed{\texttt{ZAttention/query\_div} = \binom{Q}{2}^{-1}\sum_{i<j}\tfrac12\big\|\bar a_i - \bar a_j\big\|_1 \in [0,1]},
\qquad \bar a_i = \tfrac1B\textstyle\sum_b a_{bi\cdot}$$

`ZAttention/query_div` = **the mean pairwise total-variation distance between the QUERY rows** —
whether the queries look at *different* places. A "row" here is one query group's attention
distribution over all $P$ tokens; it has nothing to do with image rows or patch rows. 1 = the
queries attend to disjoint tokens, 0 = every query attends identically, so the extra rows are
duplicate `proj` parameters. It is **not** an entropy and it is independent of the per-query
numbers above: three SHARP queries all aimed at the same patch score 0 here while each looks
perfectly healthy there. That case is invisible to entropy alone, which is why this key exists.

So this section carries **two different quantities**, not four entropies: one entropy per query
row (`task_cmd`, `proprio`, `cls`) and one between-query distance (`query_div`).

**$P$ is not logged** — it is constant for a run and already implied by two things the run
dumps, the camera cfg and the backbone's patch stride, so a scalar series would be noise. Today
**P = 21**: the head cam is 112×63 ([_helpers.py:326-338](../../src/vibe/core/sensors.py)),
the frozen backbone patchifies at stride 16 with `interpolate_pos_encoding=True` and no resize,
so ⌊112/16⌋ × ⌊63/16⌋ = 7 × 3. The vertical remainder (63 = 3·16 + 15) is dropped — P is a
floor, not a round. **Recompute it whenever the camera changes**: it re-bases every number in
this section, which is exactly what normalizing by $\log P$ is there to absorb.

### implementation

[cross_attention.py:152-205](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/modules/cross_attention.py#L152-L205).

- **Entropy and divergence answer different questions, and you need both.** Entropy says how
  *concentrated* one query row is; `query_div` says whether the queries concentrate on *different*
  things. Three sharp rows all pointed at the same patch are still one row's worth of
  information — low entropy would not catch it, `query_div → 0` does.
- **The design rule this section enforces:** *a diffuse row IS a mean-pool row.* Keep at most
  ONE (a global-pool path into z is otherwise inexpressible); every additional diffuse row is
  duplicate `proj` parameters. Query VALUES never enter z — the query only steers *where to
  look* — so a row that cannot steer contributes nothing.
- **Sharp — it reads the LAST forward, not the storage.** `metrics()` consumes `last_attn`, the
  debug tap set in `forward()`. In joint mode that is whichever PPO minibatch ran last — one
  sample of `mini_batch_size` rows, not a rollout statistic. Cheap (no encode, no storage pass),
  noisy per-iteration, fine smoothed.
- **Row labels strip the `q_` prefix**
  ([:89-93](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/modules/cross_attention.py#L89-L93)): `q_task_cmd` →
  `ZAttention/task_cmd`. Learned queries (none today) label as `learned<i>`.
- There is deliberately **no `attn_ent_mean`**. It averaged over rows, so dropping a query row
  changed it by construction — it moved when the *question* changed, not the answer. Compare
  per-row.
- Untrained reference (32-env smoke, 3 iterations): `task_cmd` 0.970, `proprio` 0.975,
  `cls` 0.993, `query_div` 0.061. Everything starts pinned to the ceiling and identical.

---

## §2 `ZPrediction/` — is the target predicted, and how much of that is z

### definition

**How well the auxiliary objective's target can be predicted, split by what it is made of, and
how much of that fit z actually supplies.** `total` is the headline; the per-term split says
*which part of the world*; `early`/`late` says *whether the image is doing the work or dead
reckoning is*; `gain_*` says *whether any of it is the image at all*.

### formulation — the fit (Sfd)

Window of $K$ steps from $t_0$, seeded with the **true** normalized target, autoregressed:

$$\hat s_{t_0}=s_{t_0},\qquad \hat s_{t+1}=f_\theta\big([\,\hat s_t\,\|\,z_t\,\|\,c_t\,\|\,a_t\,]\big)$$

$$L=\frac{1}{|\mathcal W|K}\sum_{w}\sum_{k=1}^{K}
\frac{\sum_b m_{b,t}\cdot\frac1D\sum_d|\hat s^{(d)}-s^{(d)}|}{\sum_b m_{b,t}},\qquad m=1-\text{done}$$

Targets are normalized, $s=(x-\mu)/(\sigma+\varepsilon)$, so **the loss is in units of target
standard deviations**: ~1.0 ≈ predicting the marginal mean, 0 ≈ perfect. Per term $T$:

$$\texttt{ZPrediction/<term>}=\frac{\sum_{\text{steps}}\sum_b m\sum_{d\in T}|e_d|}{|T|\sum_{\text{steps}}\sum_b m}
\qquad\Longrightarrow\qquad
\texttt{total}=\frac{\sum_T |T|\cdot\texttt{<term>}}{\sum_T |T|}$$

**The total is the dim-weighted mean of its terms** — verified against the logged value. That
identity is the loss-budget audit: a term's share of the gradient is $|T|\cdot\texttt{<term>}$.
`early`/`late` are the halves *within* each window: early $=k\in[0,\lfloor K/2\rfloor)$.

For **Lfd** the same section carries `total` (= `Loss/z_mse`), `tf` / `ar` (teacher-forced vs
open-loop branch) and `floor` — the do-nothing temporal-smoothness floor, $\mathrm{MSE}(z_t,z_{t+1})$
under the EMA target. A latent loss that only matches `floor` is exploiting smoothness, not dynamics.

### formulation — the z-ablation (Sfd)

Re-scan with z re-paired to the wrong environments; the rise in L1 is z's contribution.

$$\texttt{shuffled\_m}=L(z_{\pi_m}),\quad \texttt{gain\_m}=L(z_{\pi_m})-L(z),\quad
\texttt{gain\_frac\_m}=\frac{\texttt{gain\_m}}{\texttt{shuffled\_m}}$$

| $m$ | permutation | destroys | reads as |
|---|---|---|---|
| `col` | ONE, reused for every (window, step) | pairing only — z stays a coherent trajectory, from the wrong env | **conservative** bound on z's contribution |
| `step` | FRESH per (window, step) | pairing **and** temporal coherence | z's total contribution |

`gain_frac` is the number to quote: unit-free, so unlike `total` it survives a change of target
set. `1 - gain_frac` is what dead reckoning + conditioning already covered. **The gap
`gain_step − gain_col` is how much of z's value is temporal consistency rather than
instantaneous content** — the reason both are logged.

This is the *conditional* quantity, and it is the one that cannot be faked: a low `total` alone
cannot distinguish "z is informative" from "the target was already free from $(c,\hat s,a)$".

### implementation

[state_fd.py:175-216](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/extensions/aux/state_fd.py#L175-L216) (scan
setup + metrics), [:218-269](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/extensions/aux/state_fd.py#L218-L269)
(the autoregressive scan), [:272-324](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/extensions/aux/state_fd.py#L272-L324)
(shuffle + ablation), [base.py:260-272](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/extensions/aux/base.py#L260-L272)
(`_slice_metrics`). Lfd: [latent_fd.py:96-129](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/extensions/aux/latent_fd.py#L96-L129).

- **Term names are inferred, never declared.** The target group is a **dict** group, and
  `_target_layout` reads one slice per obs term — name and width both
  ([state_fd.py:103-120](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/extensions/aux/state_fd.py#L103-L120),
  [observation_cfgs.py:154-163](../../src/vibe/tasks/repose/config/g1/observation_cfgs.py)).
  So **adding a target is ONE edit** — append a term — and a term can no longer be silently
  sliced off the loss, which is what the old hand-written `(name, dim)` table allowed. The obs
  term's name IS the panel's name: rename it there to rename the panel. Today:
  `object_pose_b[9] object_twist_b[6] upface_color[6] task_reward_vec[2] bodywise_saturated_force[2]`,
  printed at init.
- **Sharp — dim count is the only loss weight there is.** No per-term weighting today, so a
  2-dim term that is 3× harder than a 15-dim one claims a disproportionate share of the
  gradient, and every target added re-weights every existing one. Read the budget as
  $|T|\cdot\texttt{<term>}$, not as `<term>`.
- **Sharp — `total` rides a moving ruler; `ruler` is logged so you can see it.**
  `EmpiricalNormalization` is *cumulative* over everything since init
  ([normalization.py:46-48](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/modules/normalization.py#L46-L48)), so
  σ keeps growing as the policy changes what it produces. Worked example on the contact term:
  raw error 0.010 against σ 0.05 logs 0.20; later raw error 0.031 against σ 0.15 logs 0.21 — the
  error **tripled** and the curve moved 5%. It runs the other way too: a rising curve can mean
  the residual grew *or* that σ shrank. `ZPrediction/ruler` is mean $(\sigma+\varepsilon)$;
  `<term> × ruler` ≈ the error in raw units. Freezing or EMA-ing the normalizer would change the
  loss, so it is not done here — this is a readability fix, not a behavior change.
- **Sharp — the last partial window is dropped.** `_window_starts = range(0, T-K, K)`; full
  coverage needs $T = nK+1$. At T=24, K=10 → starts {0,10}, targets 1..20: **stored steps 21-23
  never enter the loss.** A sampling loss (fresh data every iteration), not a correctness one —
  and it is printed at init (`2 x K=10 -> targets 1..20 of 23 stored transitions — trailing 3
  UNUSED`) so it stays a decision.
- **Sharp — resets are masked, not skipped.** Unlike the base `_valid_indices` path (used by
  Lfd), `StateFdAux` keeps reset-crossing windows: the step loss is masked by `not_done` and the
  carry is reset to the *true* post-reset state. A masked step contributes 0 to the numerator
  AND 0 to the denominator.
- **Sharp — `early` is not "the start of the episode".** It is the first half of each K-window,
  which rides the true-state seed. `early ≪ late` is expected and healthy; the two *converging*
  would mean the seed stopped helping.
- **Sharp — the ablation measures the CURRENT predictor's reliance on z, not z's information
  content.** A predictor that has learned to ignore z reports a small gain even if z is rich.
  Read it with §1 (is any row selecting?) and §3 (has z collapsed?).
- **Perf.** All K encodes of a window read *stored* obs (only $\hat s$ is sequential), so the
  extractor runs once over the whole (windows × K × mb) block
  ([`_encode_span`](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/extensions/aux/state_fd.py#L326-L335)) —
  identical math, K-fold fewer launches. Per-dim error accumulates as masked **sum + count**,
  never `err[mask.bool()]`: a boolean index has a data-dependent shape and forces a device→host
  sync per step per window per minibatch. The ablation is predictor-only (the encode is reused,
  no backward): one extra unroll per mode. `log_z_ablation` accepts `"col" | "step" | "both" |
  False` if it ever shows in wall time.
- **Batch shape.** `sample_loss` draws `num_envs // num_mini_batches` env columns, each
  contributing `(T // K) × K` encoder rows; joint mode fires one aux call per PPO minibatch
  (`num_learning_epochs × num_mini_batches` = 20 per iteration) and the logged value is their mean.

---

## §3 `ZCapacity/` — how much z carries

### definition

**How many independent directions z actually uses, and whether it has collapsed.** RankMe is
the elbow-plot metric for `latent_dim` ablations; `std` is the cheap collapse alarm.

### formulation

RankMe (Garrido et al. 2023, *RankMe: Assessing the downstream performance of pretrained
self-supervised representations by their rank*, arXiv:2210.02885) — exponentiated entropy of the
normalized singular-value spectrum of $Z\in\mathbb R^{N\times d}$:

$$p_i=\frac{\sigma_i}{\sum_j\sigma_j}+10^{-12},\qquad
\texttt{rankme}=\exp\!\Big(-\sum_i p_i\log p_i\Big)\in[1,\min(N,d)],\qquad
\texttt{rank\_frac}=\frac{\texttt{rankme}}{d}$$

$$\texttt{std}=\frac1d\sum_d\mathrm{std}_b\big(z_{bd}\big)$$

### implementation

[cross_attention.py:197-205](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/modules/cross_attention.py#L197-L205).

- **Sharp — read `rankme` and `std` as a PAIR.** RankMe normalizes the spectrum, so it is
  **scale-invariant** and structurally cannot see a uniform shrink of z. That is `std`'s job.
  Neither is sufficient alone.
- **Sharp — `std` is measured on a LayerNorm'd z.** LayerNorm normalizes per **row** (across the
  d dims), not per dim across the batch, so per-dim across-batch collapse is still detectable —
  but the scale is bounded (unit RMS per row ⇒ mean per-dim std ≤ 1). Read it as a ratio over
  training, never against an absolute target.
- **Sharp — samples the LAST minibatch**, like §1. There used to be a second, storage-sampled
  copy of these two numbers on the aux objective (`latent_rankme`/`latent_std`). It is **gone**:
  the two tracked each other to ~1%, the storage version cost an extra 4096-row encoder forward
  every iteration, and — decisively — it could not be reported by the `-Ext` row, which is the
  baseline the whole section exists to be compared against.
- Ceiling today = `latent_dim` = **128**. Untrained reference (32-env smoke): `-Sfd` 32.3
  (`rank_frac` 0.25), `-Ext` 60.0 (0.47) — z starts using a quarter to a half of its width.

---

## §4 `ZGradient/` — who trains z

### definition

**Who is actually training the extractor.** Two backward passes land on the same parameters
every minibatch. This section reports their relative strength in two currencies — **demand**
(how loud each source shouts) and **pull** (which one moved the weights) — plus whether they are
pulling the same way at all.

### formulation

Per source $s\in\{\text{ppo},\text{aux}\}$ over extractor parameters $\theta$ and the $N_s$
backward passes tagged to that source in one iteration:

$$\texttt{<s>}=\sqrt{\frac{1}{N_s}\sum_{\text{passes}}\sum_\theta g_\theta^2}=\mathrm{RMS}\big(\|g\|_2\big),
\qquad \texttt{frac}=\frac{\texttt{aux}}{\texttt{ppo}+\texttt{aux}}$$

$$\texttt{cos}=\frac{\sum\langle g_\text{ppo},g_\text{aux}\rangle}{\sqrt{\sum\|g_\text{ppo}\|^2\sum\|g_\text{aux}\|^2}}
\qquad\text{(cosine in the product space over all params and passes)}$$

$$\texttt{pull\_<s>}=\big\langle g_s,\,-\Delta\theta\big\rangle,\qquad
\texttt{pull\_frac}=\frac{\texttt{pull\_aux}}{\texttt{pull\_ppo}+\texttt{pull\_aux}}$$

where $\Delta\theta$ is the **realized** parameter delta of the step. Because
$g_\text{total}=g_\text{ppo}+g_\text{aux}$ and $\Delta\theta$ is shared, `pull` is an **exact
additive split of the first-order loss decrease along the step actually taken**, with clipping,
the Adam preconditioner and the fixed encoder LR all already inside $\Delta\theta$. Signed: a
negative entry means that source's own loss went *up* along the step.

`frac = 0.5` is demand parity; `cos < 0` means the two sources are asking the encoder to move in
opposing directions.

### implementation

Hooks + accumulators [ppo_aux.py:101-145](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/algorithms/ppo_aux.py#L101-L145),
pre-step snapshot [:163-173](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/algorithms/ppo_aux.py#L163-L173),
attribution [:176-187](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/algorithms/ppo_aux.py#L176-L187), reduction
[:213-235](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/algorithms/ppo_aux.py#L213-L235). The realized delta
needs one seam in the base algorithm,
[`PPO._post_optimizer_step`](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/algorithms/ppo.py#L438-L440).

- **This section is not comparable with runs before rsl_rl `6cee06d`.** The old `enc_grad_aux`
  measured the **total**, not the aux term: joint mode accumulates both backwards into one
  `param.grad`, so the post-accumulate hook firing in the aux phase read $g_\text{ppo}+w g_\text{aux}$.
  That floors the ratio at ≈0.5 by construction and is the entire explanation for the
  "`frac ≈ 0.54` regardless of the delta" invariant that earlier analysis flagged as suspicious.
  Snapshotting the PPO-phase gradient and subtracting recovers the true term.
- **Sharp — demand is PRE-clip.** Raw ask, not what was applied; a source can dominate `frac` and
  be clipped down to parity in the step. `pull` is the post-everything counterpart — that is the
  point of having both. They routinely disagree, and the disagreement is informative: the encoder
  group runs on a *fixed* LR (`extractor_lr`) while actor/critic ride the adaptive-KL schedule
  ([ppo_aux.py:84-91](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/algorithms/ppo_aux.py#L84-L91)), so equal
  norms do not mean equal parameter motion. Untrained smoke run: `frac` 0.58 but `pull_frac` 0.72.
- **Sharp — pass counting rides ONE marker parameter** (`extractor_params[0]`). Correct as long
  as every backward touches the whole extractor; a partially-frozen extractor would miscount.
- **Sharp — in `sequential` aux-only mode the `ppo` entry is PPO's *unapplied* demand.** The
  hooks measure gradient FLOW, not optimizer steps. Not a bug; just don't read it as influence
  there. Joint mode (what ships) has both at step parity by construction — the aux backward
  fires on **every** PPO minibatch, and the subtraction is applied only there.
- **`pull_frac` is suppressed when `pull_ppo + pull_aux ≤ 0`** — a step that raised both losses
  has no meaningful split.

---

## §5 Offline — post-training representation eval

Everything above is measured *during* training and answers "is learning working". This section is
measured on rollouts of a **frozen, trained** policy and answers a design question:
**which query row earns its keep?** Same subject, different question, so mostly different metrics.

The rollout-collection and plotting tooling is not part of this release; the definitions below
stand alone.

### what ports, and the rule that decides

> A training metric is offline-valid **iff it is a pure function of (trained weights, observed
> data)**. Anything that reads a gradient or an optimizer step is learning dynamics by
> construction and has no test-time meaning.

| §1-§4 key | offline | why |
|---|---|---|
| `ZAttention/*` | ✅ **and better** | pure fn(weights, data). Offline it spans the whole eval distribution instead of §1's last-minibatch sample |
| `ZCapacity/*` | ✅ **and better** | same — and with ~1e5 rows the *spectrum* is readable, not just its entropy |
| `ZPrediction/*` | ❌ | measures **the objective's fit**, not z. Target-set-specific by construction: 0.7.3 has 23 target dims, 0.7.10 has 25, against two different frozen normalizers — the numbers are not on one scale |
| `ZGradient/*` | ❌ | no gradients at test time |

The offline-valid set is **exactly the portable core** (§1 + §3) named at the top of this file.
That is not a coincidence: cross-variant comparability and test-time validity have the same
cause — describing z rather than something's relationship to z. **The aux predictor is a
train-time entity and appears nowhere in §5.**

### the structural fact §5 is built on

`proj` is linear over the concatenated per-query pooled blocks
([cross_attention.py:143-147](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/modules/cross_attention.py#L143-L147)),
so with $v_g = a_g V$ the pooled value of query row $g$:

$$\boxed{z=\mathrm{LN}\Big(\underbrace{\textstyle\sum_g \mathrm{proj}_g v_g}_{\text{one term per query row}}+\,b\Big)}$$

**z is an exact additive sum of per-query contributions** $c_g=\mathrm{proj}_g v_g$. Every
"which query" number below is a *decomposition* of that sum, not an estimate of it — verified to
1e-5 against the checkpoint. The collector taps it with a `proj` **pre**-hook, whose input is
that concatenation laid out as Q contiguous `attn_dim` blocks in `query_labels` order
(`ExtractorTap`); the same seam replaces one
block to get the leave-one-query-out counterfactual. No rsl_rl change.

### §5.1 `ZAttention/` — where each row looks

Per-row normalized entropy exactly as §1, plus the cut training cannot make:

$$\texttt{focus\_gain}_g = H_g\big|_{\text{object out of frame}} - H_g\big|_{\text{object in frame}}$$

**A row doing perception must sharpen when there is something to look at.** `focus_gain > 0` is
that signature; $\approx 0$ means the row is locked to something static (the hands, the floor, a
frame corner) regardless of the scene — a *sharp* row that is not a *seeing* row. Entropy alone
cannot separate those two, and §1 never could.

`query_div` is reported **per pair**, never as the bare mean: the mean is over $\binom{Q}{2}$
pairs, so it is not comparable between runs with different query-row counts (0.7.3 has 4 rows,
0.7.10 has 3 — 6 pairs vs 3).

### §5.2 `ZGrounding/` — is it looking at *the object*?

§1 says a row is concentrated; this says it is concentrated **on the right thing**. The object's
world position is projected into the image by
[`ObjectCamProjection`](../../src/vibe/core/mdp/metrics.py) — the same geometry as the
`object_in_fov` training metric, refactored so the two cannot drift — and reduced to a patch index.

$$\texttt{on\_object}_g=\mathbb E\big[a_{g,\pi(t)}\big],\qquad
\boxed{\texttt{lift}_g=P\cdot\texttt{on\_object}_g},\qquad
\texttt{top1\_hit}_g=\Pr\big[\arg\max_p a_{gp}=\pi(t)\big]$$

where $\pi(t)$ is the object's patch. **`lift` is the number to quote**: 1 = indifferent (chance
is $1/P$), >1 = the row selects the object. Unit-free, so it survives a resolution change.

### §5.3 `ZQuery/` — how much of z does each row build?

The dominance cut, exact from the decomposition above:

$$\texttt{norm\_share}_g=\frac{\mathbb E\|c_g\|}{\sum_h \mathbb E\|c_h\|},\qquad
\texttt{var\_share}_g=\frac{\mathrm{tr\,Cov}(c_g)}{\mathrm{tr\,Cov}(z_\text{pre})},\qquad
\texttt{dz}_g=\mathbb E\frac{\|z-z_{\setminus g}\|}{\|z\|}$$

$z_{\setminus g}$ replaces $c_g$ by its mean and re-normalizes — what z loses if the row is deleted.

**Read the first two as a pair.** A diffuse mean-pool row contributes a nearly *constant* vector:
high `norm_share`, near-zero `var_share`. That gap is the cleanest "this row is a bias term
wearing a query's clothes" detector there is, and it needs no attention statistic at all.

$$\texttt{overlap}=1-\sum_g \texttt{var\_share}_g = \frac{\sum_{g\neq h}\mathrm{tr\,Cov}(c_g,c_h)}{\mathrm{tr\,Cov}(z_\text{pre})}$$

the cross-covariance the shares miss: **> 0 the rows are redundant** (contributions co-vary),
**< 0 complementary** (they partly cancel, so the sum carries more than the parts).

### §5.4 `ZContent/` — what does each row carry?

**Linear CKA** between a representation and each ground-truth block:

$$\texttt{cka}(X,Y)=\frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F\,\|Y^\top Y\|_F}\in[0,1]$$

A *statistic*, not a probe — closed form, no fitting, no hyperparameter, no train/test split,
$O(Nd^2)$ with no $N\times N$ gram. Invariant to orthogonal transforms and isotropic scaling,
which is precisely what comparing two independently trained latents requires. Computed for **z**
(what the policy gets) and for each $c_g$ (what that row supplies).

The blocks are chosen by the **eval**, not by any objective — that is what makes this section
comparable across `-Ext`/`-Sfd`/`-Lfd` and across a target-set change, i.e. it is the thing
`ZPrediction/<term>` structurally cannot be. They are attached by the collector
(`_attach_repr_probe`) rather than reused
from `prediction_target`, deliberately: `-Ext` and `-ImgFeat` carry no aux group at all, and each
aux variant's target set is its own. Today: `object_pose_b(9) object_twist_b(6) upface_color(6)
goal_color(6) hand_contact(2)`.

### §5.5 `ZCapacity/` — offline

RankMe and `std` exactly as §3, over the eval distribution instead of one minibatch, plus
per-row `rankme_<g>` on $c_g$ (how many directions each query contributes — these need not sum to
z's) and the **singular-value spectrum**, which is the elbow a `latent_dim` ablation would be
chosen from and is too noisy to plot from a single training minibatch.

### §5.6 `ZUtility/` — what is a row worth, in task units?

The only section measured on the scale that decides anything. Same protocol, same seed, one
causal intervention on the policy's own forward pass; report $\Delta$ success against `none`.

| mode | seam | replaces | destroys |
|---|---|---|---|
| `q_<row>_mean` | `proj` pre-hook | that row's block → its batch mean | that row's per-env content only |
| `z_mean` | extractor output | z → batch mean | all per-env perception |
| `z_shuffle_col` | ″ | z ← one fixed *wrong* env, held for the run | pairing; z stays a coherent trajectory |
| `z_shuffle_step` | ″ | z ← a fresh wrong env each step | pairing **and** temporal coherence |
| `z_zero` | ″ | z → 0 | everything, but off-distribution — a bound, not a control |

Mean-substitution keeps the marginal and kills the per-environment content: the same "destroy the
pairing, keep the distribution" contract as §2's z-shuffle, but run through the **policy**, so
what it measures is z's contribution to *behaviour* rather than to some objective's fit. This is
the honest offline replacement for `ZPrediction/gain_frac`, and it needs no predictor.

### the verdict table

§7 of the notebook joins all of it, one row per query. The point is that the four ways a query
row can fail look **identical** if you read only one section:

| symptom | reads as |
|---|---|
| `H` ≈ 1 | **mean-pool row** — attention is uniform, the query steers nothing |
| `lift` ≈ 1 | looks *somewhere*, but not at the object |
| `norm_share` high, `var_share` ≈ 0 | contributes a **constant** — a bias term with a query attached |
| `cka_*` low everywhere | carries no ground-truth structure |
| `Δsucc` ≈ 0 | the policy does not use it, whatever the other four say |

**A row is only safe to delete when the last column says so.** The first four explain *why*; the
task delta decides. And keep at most ONE diffuse row — a global-pool path into z is otherwise
inexpressible (§1).

### §5.7 the radar — where is one variant better than another?

§7 of the notebook is the per-query table; §8 is the same content as a Kiviat plot, at two
granularities (polygons = variants; polygons = query rows, one panel per variant). The contract is
**radially outward is better on every axis**, and two rules make that true rather than asserted:

1. **Direction lives in the scale, not the metric.** Each axis carries an explicit
   `(floor → ceil)`; a lower-is-better quantity simply gets `floor > ceil` and the same linear map
   inverts it. `H` enters as `1 → 0` — never negated, never renamed, and the axis label prints the
   raw range so the transform is visible.
2. **Floors are natural; ceilings are natural only when attainable.** Floors are real zeros —
   0 (CKA, `var_share`, `rank_frac`, `focus_gain`, reliance), `lift = 1` (chance), `H = 1`
   (uniform / mean-pool), 0% success — so a polygon collapsing toward the center genuinely means
   *contributes nothing*. Ceilings split: `success` (100%) and `H` (0) are reachable and stay
   natural; `CKA`, `rank_frac`, `top1_hit`, mean-TV all have a nominal 1 that is **never
   approached** (a 128-d latent does not use 128 directions; softmax over 21 tokens does not give
   disjoint supports), so pinning the rim there would park those spokes at the center and discard
   the comparison — they fall back to the observed max, printed on the label.

Anchoring the *floor* is what keeps this from being a min-max radar. With **two** variants,
per-axis min-max puts the loser at the center and the winner at the rim on every axis — a picture
of having two runs, carrying no information.

⚠ Two readings are interpretations, not facts. **`reliance`** (success drop under `z_mean`) is
scored higher-is-better, which encodes "the perception pipeline is load-bearing" — not "a
policy that needs z is a better policy". **`diversity`** is dropped automatically when the
variants have different query-row counts, since the mean over $\binom{Q}{2}$ pairs is not
comparable across Q (§5.1).

### sharp bits

1. **`ZUtility` is not a paired test.** Actions diverge from the first intervened step, so
   episodes do not correspond between modes. Read column aggregates; never a per-episode diff.
2. **Every mode costs a full collection pass.** `Z_MODES` in
   `collect_rollouts.sh` is a wall-clock multiplier.
3. **P is read off the captured attention tensor**, then cross-checked against the recorded camera
   cfg's patch division. A camera change re-bases every §1/§5.1 number and invalidates the §5.2
   patch mapping; the notebook prints a loud mismatch rather than mapping into the wrong grid.
   The patch grid is a **floor** — at 112×63 the bottom 15 pixel rows have no patch, so a
   projection inside the frustum can still be outside the token grid (`in_grid`, not `in_fov`).
4. **repr capture is PRE-step, the task dump is POST-step.** z/attention were produced from a
   given obs and scene state, so the projection and the GT blocks are read before `env.step`.
   `repr_data.pt` therefore carries its own `time_steps` and needs no alignment with
   `rollout_data.pt` (`collect:406`).
5. **Reset frames are dropped everywhere.** A reset teleports both reference and object; its
   attention and its z belong to no episode.
6. **Linear CKA sees *linear* shared structure.** High is strong evidence the block is there; low
   means "not linearly present", not "absent".
7. **The dataset SHA is now recorded** in `metadata.json → deps`, along with the rsl_rl SHA and
   the vibe git HEAD + dirty flag. It is load-bearing and was previously unrecorded: the
   contact-matrix eps bump flipped `bodywise_contact_cmd` from identically-zero to live, which is
   a policy **input**. Two dumps taken against different dataset SHAs are not one experiment.

---

## Reading table

| key | direction | floor / ceiling | what a bad value means |
|---|---|---|---|
| `ZAttention/<row>` | ↓ = localizing | [0, 1], 1 = diffuse | at 1 ⇒ the row is a mean-pool; delete it or fix its query |
| `ZAttention/query_div` | ↑ | [0, 1] | → 0 ⇒ the query rows are duplicates, whatever their entropy |
| `ZPrediction/total` | ↓ | ~1.0 = marginal mean | not comparable across a target-set change |
| `ZPrediction/<term>` | ↓ | ~1.0 | high + flat ⇒ unlearnable at this horizon, and still spending budget |
| `ZPrediction/ruler` | — | — | context for the two above; never read them without it |
| `ZPrediction/early,late` | ↓ | late > early expected | converging ⇒ the true-state seed stopped helping |
| `ZPrediction/gain_frac_*` | ↑ | 0 = z unused | ~0 ⇒ the target was free from (cond, ŝ, a); the image path is idle |
| `ZPrediction/floor` (Lfd) | — | the do-nothing baseline | `total` must beat it or nothing is being learned |
| `ZCapacity/rankme,rank_frac` | ↑ | 1 … `latent_dim` | → 1 = collapse; flat & low = wasted latent width |
| `ZCapacity/std` | ↑ | ≤ 1 (LayerNorm'd) | → 0 = collapse RankMe cannot see |
| `ZGradient/frac` | — | 0.5 = parity | demand, pre-clip |
| `ZGradient/cos` | — | [-1, 1] | < 0 = the two objectives are fighting over z |
| `ZGradient/pull_frac` | — | 0.5 = parity | the effect counterpart of `frac`; differs from it, by design |
| **§5 offline** | | | |
| `ZAttention/focus_gain` | ↑ | 0 = scene-blind | ≈0 ⇒ sharp but locked to something static, not seeing |
| `ZGrounding/lift` | ↑ | 1 = chance | ≈1 ⇒ looks somewhere, but not at the object |
| `ZQuery/var_share` | ↑ | Σ ≈ 1 | high `norm_share` + ≈0 here ⇒ the row is a constant offset |
| `ZQuery/overlap` | → 0 | 0 = independent | >0 rows redundant; <0 complementary |
| `ZContent/cka_<block>` | ↑ | [0, 1] | low everywhere ⇒ carries no ground-truth structure |
| `ZUtility/d_succ_term` | ↓ (more negative = more used) | 0 = unused | ≈0 ⇒ delete the row; this is the column that decides |

## Sharp bits, one list

1. **§1 and §3 are portable across aux variants; §2 is not.** Across `-Ext`/`-Sfd`/`-Lfd` the
   standard of comparison is those two sections plus task success.
2. `ZAttention/*` and `ZCapacity/*` sample the **last minibatch** of the update — noisy per
   iteration, read smoothed.
3. Attention entropy is **normalized by log P**, so it survives a resolution change. P is NOT
   logged (constant, implied by the camera cfg + patch stride); it is 21 today, a **floor** of
   the patch division — recompute it whenever the camera changes.
4. A **diffuse row is a mean-pool row** — and three sharp rows on the same patch are also one
   row. Entropy alone cannot tell you; `query_div` can.
5. `ZPrediction/*` is in **target-σ units** against a **cumulative, still-moving** normalizer.
   Read it against `ruler`.
6. `total` is a **dim-weighted** mean of its terms — dim count is the only weight there is.
7. Target terms are **inferred from the obs group**; adding one is a single edit and cannot be
   silently dropped. The obs term's name is the panel's name.
8. `range(0, T-K, K)` **discards the trailing partial window** (3 of 24 steps at K=10); full
   coverage needs `T = n·K + 1`.
9. The z-ablation runs in **two modes**; `col` is the conservative bound, and the gap to `step`
   isolates temporal consistency.
10. `ZGradient/*` **breaks comparability with runs before rsl_rl `6cee06d`** — the old aux entry
    measured the total. Demand (`frac`) and effect (`pull_frac`) are different numbers on purpose.
11. **Offline (§5), only §1 and §3 port.** `ZPrediction` is objective-fit and `ZGradient` needs
    gradients; neither has a test-time meaning. The offline sections `ZGrounding`/`ZQuery`/
    `ZContent`/`ZUtility` are *not* ports — they answer a design question training cannot ask,
    and none of them involves the predictor.

## Task metrics, for contrast

Not part of this file's subject, but the trap that most often corrupts a reading of it:
`Metrics/motion/*` is published by `CommandTerm.reset`, which logs `mean(metric[env_ids])` **at
the reset step** and then zeroes. Every one of them is therefore a **terminal-step** statistic,
not an episode mean — `at_goal_color` is the terminal success rate. `at_goal_color_ever` is its
any-frame counterpart (a per-episode running max), which is the quantity the offline eval
reports, and the two have ranked runs differently before.

## See also

- [encoders.md](encoders.md) §11 — the extractor these metrics measure.
- the encoder bake-off — what the frozen backbone carries before any of this trains.
