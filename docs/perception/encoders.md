# Vision stack — idea · built form · roadmap

Why the `-ImgFeat-` path looks the way it does. Supersedes the retired `vlm_command_reward.md`
and `extractor_ahead.md`.

| question | answer lives in |
|---|---|
| which backbone carries what | the encoder bake-off — the bake-off |
| what the numbers mean at train time | [metrics.md](metrics.md) — the `Z*` sections |
| what the wiring is | [../infra/agents.md](../infra/agents.md) §2 — diagrams |
| what the renderer will feed it | [render_domain.md](render_domain.md) |

Notation: `VisEnc()` vision tower · `TxtEnc()` text tower. Q/K/V stay attention roles.

---

# Part I — the idea

## 1 · two structural suspects

Repose stalled at ~60% success. Not hyperparameters:

```
task truth: "queried face on top"      policy input: [proprio | extero | cmd(6d)]
        │                                     │
   reward was full-quat error          cmd drowned when extero is high-dim
   (over-constrained by 1 DOF: yaw)    (6 dims against 768 encoder dims)
```

## 2 · reward — the face-up kernel (shipped)

| | quat kernel (old) | face-up kernel (new) |
|---|---|---|
| constraint | full SO(3), 3-DOF | face normal vs world-z, 2-DOF |
| penalizes task-irrelevant yaw | yes | no |
| success set | one orbit point | the whole yaw circle |

`err = acos((R_obj·n_face)·z)`, gaussian std 0.5, bonus at tilt < 0.3.
Face order and colours: [`vibe/assets/repose.py`](../../src/vibe/assets/repose.py) — one source.
Old and new metrics logged side by side (`at_goal` vs `at_goal_face`), so no self-conning.

## 3 · command — the goal obs was ignorable

Zero-goal probe: feed 0 instead of `object_goal_ori_mat6d`, eval success.

| agent | aug-stream dims beside the 6-d goal | Δ success | verdict |
|---|---|---|---|
| Sonic-ObjKin | object kin ~O(10) | −13 pts | goal read |
| Txtp-ImgFeat | encoder feats ~O(10²–10³) | **−0.1 pts** | goal **stone-dead** |

Two mechanisms, both structural:

1. **fan-in share** — 6 of ~774 first-layer inputs never win gradient against 768 dense channels.
2. **redundancy** — goal ≡ clip end frame, and the dataset's clip-end orientations cluster ⇒
   **every command is effectively one word** (zero mutual information).

> Any command representation is capped by (2) until goals dissociate from clips. That is a
> dataset fix, not a representation fix.

## 4 · render resolution

Two pipelines with **different** res behaviour — do not conflate:

```
bench (offline):  render ─► [processor resize 224²] ─► 196 tokens ─► pooled   (res-independent)
env  (training):  render ─► VisEnc() at NATIVE res  ─► P(res) dense tokens    (res IS the knob)
```

| claim | verdict |
|---|---|
| env-native low res starves features | **partly** — colour is low-frequency and survives; far/small-cube detail and OOD blur die |
| higher res eats RL throughput | **yes** — the render pays most; encoder compute grows only linearly in tokens |
| a naive MLP extractor is inefficient at high res | **yes** — flattened dense grid ⇒ fan-in ∝ area, worsening §3 dilution 7× at 114×64. Fix is query-pooling, not lower res |

Render cost @ 4096 envs, RTX 3090:

| res | steps/s | peak VRAM | flat img_feat dim (theia-tiny) |
|---|---|---|---|
| 57×32 | ~21.5k | 6.5 GiB | 768 |
| 114×64 | ~11.6k | 10.2 GiB | ~5.4k — **fps sweet spot**, flatten untenable |
| 224×126 | ~4.2k | 20.3 GiB | ~19k |

Online ceiling: ViT forward ≈ `2·params·tokens` FLOPs/frame; this box calibrates to
~45 TFLOP/s effective fp16. **Tokens are the currency, params the co-factor.** At P = 28:

| class | N | GF/frame | 4096-img pass | verdict |
|---|---|---|---|---|
| ViT-T | 6M | 0.3 | 0.03 s | free |
| ViT-S | 21–29M | 1.2 | 0.11 s | comfortable |
| ViT-B | 86–93M | 4.8 | 0.44 s | ~halves fps — payable iff features earn it |
| so400m | 400M | 22 | 2.0 s | dead online; offline scorer only |

> **~90M ViT-B is the hard ceiling; comfort zone 5–40M. >100M is offline-only.**

## 5 · the extractor — language-queried attention pooling

The unification of §3 (dilution), §4 (tokens grow with res): a Q-Former-style pooler where
**the command asks and the image answers**.

```
X = VisEnc(frame)    X: (P, D_x)   P patch tokens, P grows with res
c = TxtEnc(prompt)   c: (D_c,)     command embedding, offline, episode-constant

Q = [ c·W_q ; q_1 … q_m ]    K = X·W_k    V = X·W_v
A = softmax(QKᵀ/√d)          Z = A·V      z = flatten(Z) ──► adapter stream
```

Each row of Q is a *question*; row i of A is its attention over P patches; row i of Z is its
*answer*. **`z` is `(m+1)·d` for ANY P** — the whole scalability claim in one line.

| property | why |
|---|---|
| kills fan-in dilution structurally | the command multiplicatively gates *which patches get read*; the policy cannot ignore the query that produced its input |
| meaningful at init | same CLIP pair for VisEnc/TxtEnc ⇒ `QKᵀ` logits calibrated zero-shot (MaskCLIP-style) |
| scalable | res ↑ ⇒ P ↑, `z` shape fixed; backbone swap = new `W_k/W_v`; open vocab = new `c`, zero code |
| probe-safe | `c` is an episode constant — nothing to dead-reckon from |

**Why the reverse wiring fails:** put `c` on the K/V side and each patch attends over ONE key →
softmax over a single logit ≡ 1 → Z is a constant copy of `c·W_v`, the image never mixes in.
A single-token command can only sit on the Q side.

## 6 · design space

**What is actually constrained** (batched RL: memory-bound, sample-rich):

| cost | at P = 4…98 | verdict |
|---|---|---|
| attention FLOPs | µs — this is not NLP | free |
| module params | 10⁴–10⁵ | free |
| **rollout storage of X** (24 steps × 8k envs × P·D_x × 4B) | 768-d → 0.6 GB · 5.4k → **4.2 GB** · 19k → 15 GB | **the bill** |

Pooling cannot shrink storage — the aux needs gradients *through* the pooler, so raw X sits in
rollout storage. Mitigation is fp16, not architecture.
⇒ **pick architecture on optimization signal, not on compute.**

**Fusion variants.** Load-bearing fact: X already passed ~12 self-attn layers inside the frozen
ViT, so patches arrive globally contextualized. The extractor's job is **selection, not
re-mixing**.

| # | variant | pro | con |
|---|---|---|---|
| a | pure cross-attn (§5) | fixed z; weakest-signal-friendly; zero-shot init with a CLIP pair | queries cannot coordinate → may ask redundant questions |
| b | concat `[c;X]` + self-attn (ViLT) | richest fusion | re-does the frozen ViT's mixing; still P tokens out |
| c | Q-Former block (a + self-attn among queries) | queries divide labor | one more sublayer to train |

> **Call: ship (a); upgrade to (c) if query maps come out redundant. Skip (b).**
> FiLM rejected — its pitch is cheapness, which the table above says buys nothing here, and it
> has no spatial selection.

**CLS as a query row** is a *content-adaptive* query: the m learned queries ask the same thing
every frame, a CLS query changes with the image.

| backbone class | CLS status | verdict |
|---|---|---|
| Theia | void — per-patch distillation, CLS untrained | noise in, noise out |
| CLIP / TinyCLIP / SigLIP2 | THE trained global token | meaningful; `CLS·Kᵀ` ≈ saliency at init |

**Reading list:** RT-1 (TokenLearner, closest robotics precedent) · BLIP-2 (Q-Former = variant c)
· Flamingo (Perceiver Resampler) · Perceiver IO (the general theory) · MaskCLIP (the
"meaningful at init" evidence) · VIMA · VC-1 / Parisi 2022 (CLS-vs-dense: spatial wins
manipulation).

---

# Part II — what we measured

## 7 · reward from cosine similarity — **NO** (for now)

23-ep siglip2 fleet, trained-policy rollouts, neutral render:

- margin = `sim(cmd) − max(sim(others))` > 0 on ~85% of steps, but
  **corr(margin, task reward) ≈ −0.05** — positive as often in failure steps as in success.
- CLIP-family models are **bags of concepts**: they report *which face is visible to the
  camera*, not which face is **up** (the "on top" clause is ignored — ARO/Winoground failure).
- the psychometric curve (margin vs commanded-face tilt) IS monotone-ish ⇒ shapeable later, but
  raw cos is a **presence detector, not a reward**.

## 8 · command as centered text embedding — **YES**

Text geometry of the 6 colour prompts, any template, any tested text tower:

| | raw pairwise cos | after subtracting the vocab mean |
|---|---|---|
| separation | 0.80–0.95 (85–90% shared boilerplate direction) | −0.47…+0.48, eff-rank ~4/5, worst pair ~65° |

`z_cmd = normalize(TxtEnc(prompt) − vocab_mean)`, optionally PCA → 8–16 d.
**RAW embeddings recreate the dilution problem.** With a closed 6-word vocabulary this equals a
learned embedding table — the value is the open-vocab / omni-object future, nothing sooner.

## 9 · cross-model task scoring — and a metric trap

> **Trap:** `margin > 0` unconditionally assumes successful rollouts. In a failure episode a
> NEGATIVE margin is *correct detection*. Metric = **balanced accuracy**.

| model | acc\|succ | acc\|fail | note |
|---|---|---|---|
| TinyCLIP-8M | 0.90 | **0.61** | the only model that says "no" |
| clip-b32 | 0.94 | 0.03 | few fail steps in its sample |
| siglip2-base | 0.89 | 0.12 | |
| TinyCLIP-39M | 0.92 | 0.11 | biggest absolute margins |

Early sweep, 4 eps/model, **different rollouts per model ⇒ confounded**. The fix is the
collect/eval split that the encoder bake-off runs on: record frames once with no VL model in the
loop, score every model offline on the identical stream. Paired by construction.

## 10 · the encoder bake-off

Moved out — it is a notebook conclusion, not a design note: **the encoder bake-off**.

---

# Part III — built form and roadmap

## 11 · `CrossAttentionExtractor`

```
[base]      policy stream: robot-motion-cmd + state  ─► tracks the motion    (frozen WBC)
[adapter]   z + robot-motion-cmd                     ─► corrects the base    (LoRA)
[extractor] kv_tokens ⟨queried by⟩ query groups      ─► z                    (trainable)
```

`Q = [W_q^g · x_g for g in query_groups]`, `K,V = f·W_{k,v}`,
`z = LayerNorm(proj(flatten(softmax(QKᵀ/√d)·V)))`. **One query group → one row.**

| query group | signal | nature | role |
|---|---|---|---|
| `q_task_cmd` | `object_goal_color` (repose) / `object_goal_{ori,pos}` (uolm) | episode-static | *what to achieve* — **the language slot** |
| `q_motion_cmd` | `{l,v,w}_cmd_t` | time-varying | **PARKED** — value still reaches control via the adapter stream |
| `q_proprio` | grav + base twist + joint pos/vel | time-varying | *where am I now* |
| `q_cls` | encoder global token | — | the sanctioned mean-pool path |
| `kv_tokens` | dense patch tokens (B,P,C) | — | K/V |

Query VALUES are **not** carried into `z` (pure pooling) — a query only steers *where to look*.
Learned queries removed (`num_learned_queries=0`, API kept): they collapsed to mean-pooling.

**Where it lives — change one place:**

| what | file |
|---|---|
| the module | [`rsl_rl/modules/cross_attention.py`](https://github.com/lok-i/rsl_rl/blob/215b6f4ceda569d2ca39df64dd679b9af15375cb/rsl_rl/modules/cross_attention.py) |
| vision obs atoms (shared by every task) | [`vibe/core/observation_cfgs.py`](../../src/vibe/core/observation_cfgs.py) |
| one task's groups + query rows | `vibe/<task>/config/g1/observation_cfgs.py` |
| extractor cfg, names only | [`EXTRACTOR_CFGS` in `vibe/core/rl.py`](../../src/vibe/core/rl.py) |
| the shared 1-forward encoder | [`image_feature`](../../src/vibe/core/mdp/observations.py) |

**Two axes vary per RUN, not per task id** — a bake-off is manifest lines, not registrations:

| flag | does | lives |
|---|---|---|
| `--env.img-encoder <hf-id>` | swap the frozen backbone, rebinding EVERY `image_feature` term at once | `VibeEnvCfg` in [`vibe/core/env_cfgs.py`](../../src/vibe/core/env_cfgs.py) |
| `--agent.drop-query-rows q_proprio` | drop attention rows (comma/space separated) | `VibeRunnerCfg` in [`vibe/core/rl.py`](../../src/vibe/core/rl.py) |

Both are inert when unset. `img_encoder` is ONE flag because the name lives in two obs
terms (`kv_tokens.img_tokens`, `q_cls.img_cls`), both already tyro-reachable — setting one
and not the other loads TWO backbones and queries the wrong global token, silently.
`drop_query_rows` leaves the env alone: the group is still built, only unread, so two arms
differ in the ACTOR and nothing else — and `z` stays 128-d (`proj: m*attn_dim -> latent_dim`),
so the ablation measures **routing, not capacity**.

DRY, load-bearing: `robot_motion_cmd_terms` defines the signal ONCE for both the adapter stream
and `q_motion_cmd`; `proprio_terms` feeds both `q_proprio` and `prediction_conditioning`.

**Backends `image_feature` speaks** — token GEOMETRY is the camera's, not the backbone's, so
at the 112x63 head cam every stride-16 row below lands on the SAME P = 7x3 = 21 grid and a
swap moves `C` alone. That is what keeps `ZAttention`'s normalizer and the viser overlay
grid fixed across a bake-off.

| backend | id | C | global vector |
|---|---|---|---|
| `theia` | `theia-tiny-patch16-224-cddsv` (bare; org prefixed in `_load`) | 192 | CLS, untrained |
| `clip` | `openai/clip-vit-base-patch32`, `wkcn/TinyCLIP-*` | 768 / 512 | CLS, trained |
| `siglip` | `google/siglip2-base-patch16-224` | 768 | **MAP pool** — no CLS token exists |
| `dino` | `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 | CLS (dense skips 1 CLS + 4 registers) |
| `resnet` | `resnet18`… | — | none (flat, no `output="cls"`) |

siglip's `q_cls` row reads the MAP head, which is a *trained* global token — so the row is
more meaningful there than on Theia, not less. Adding a backend is three edits: a
`_detect_backend` branch, a `_load` branch, and an `_encode_*` returning `(dense, global)`.

**Known properties — log, do not fix:**

1. **goal is query-only** ⇒ `q_task_cmd` is purely vision-grounded (the adapter never sees the
   goal value). Intended. Revisit if exact-orientation success plateaus.
2. **proprio-as-query deviates from §5/§6**, which banned it as a "dynamics leak". Defensible:
   pure pooling carries no query value into `z`, and the aux head already gets proprio via `r`.
   Residual risk is a mildly confounded decodability probe.

## 12 · staged extensions

**Step 0 — encoder + language conditioning**

| | change | gate |
|---|---|---|
| 0a | **BUILT** (2026-08-17) — `--env.img-encoder`, no registration. `siglip` + `dino` branches landed in `image_feature`; `mobileclip2-s2` is still out (conv trunk cannot run native — it needs a padding path — and it has no `dense_txt`, so it blocks 0b anyway) | — |
| 0b | swap `q_task_cmd`'s terms → a `text_embedding` term. One group def; every other channel unchanged | **blocked on dataset v2** (§13.1) or it means nothing |
| 0c | activate `q_cls` | real once the backbone's global token carries signal, i.e. after 0a lands a CLIP-family model |

**Step 1 — multi-head attention.** `num_heads` is plumbed (asserts 1). Implement the head split
in `CrossAttentionExtractor.forward`; store `last_attn` as mean-over-heads so the overlay keeps
working.

**Step 2 — decide the twist question**, forced by the encoder bake-off §3 finding 2. Either add
`img_feat` history (H frames, stride = camera rate) so velocity becomes representable, or drop
twist from the Sfd target. Today the target includes something the features provably lack.

## 13 · open items

1. **dataset v2** — dissociate goal face from clip endpoint. Kills the one-word vocabulary.
   **Blocks 0b and every command-conditioned result.**
2. wire `z_cmd` as the ImgFeat-family goal obs; re-run the §3 zero-goal probe as the acceptance
   test — the −13 pt drop must reappear.
3. reward shaping over the psychometric curve — parked until (1).
4. camera viewpoint does as much work as the model ("visible" ≈ "up" only under top-down-ish
   views) — revisit when judging reward use.
5. cross-seed / cross-camera repeat of the bake-off (the encoder bake-off confound #9).
6. **"Est" baseline** — a detached supervised estimator `z ≡ ŝ_obj → policy`: a hard bottleneck
   at the engineered state. A separate row, not a variant of one.

## 14 · sharp bits

- **Checkpoints do not carry over** across query-layout changes — `w_q` structure changed, old
  Sfd ckpts will not load. Fresh runs intended.
- **One forward, shared.** `image_feature` shares the backbone across term instances and caches
  the per-step forward (keyed on step **and** rgb buffer ptr). Adding CLS costs ~0.
- **z scale parity.** `z` exits LayerNorm; robot-motion-cmd rides the adapter's
  `EmpiricalNormalization`. Neither swamps the other regardless of 128-vs-18 dims. If `z`
  empirically dominates, the knob is `latent_dim`, not a structural patch.
