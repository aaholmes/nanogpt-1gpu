# Bug & Confound Log — nanogpt architecture search

An internal, running record of every bug and confound we've caught, the checks that
would have caught them, and hypotheses about what else might be lurking. Kept local
(working doc, not part of the public writeup).

---

## 1. High-level summary (start here)

*Plain-language version. Each row is a thing we got wrong, in the order we found it.*

| # | What went wrong | One-line description | The lesson |
|---|---|---|---|
| B1 | **Learning-rate confound** | An "activation A beats B" result was really "A was given a better learning rate." Matching the LR made the effect vanish. | Tune (or match) the LR *per variant* before believing an architecture win. |
| B2 | **Unpaired comparison** | Compared the *average* of 5 baseline runs to the *average* of 2 variant runs. Random seeds are shared/correlated, so this inflated significance. | Compare the *same seed* head-to-head (paired), not group means. |
| B3 | **Gaming the softcap** | A conv branch added near the output, *after* the final normalization, inflated the logit magnitude and "won" by exploiting the logit soft-cap — not by predicting better. | Re-normalize anything you add before the output cap; a suspicious win near the head is usually a scale artifact. |
| B4 | **Crippled head (no logit scale)** | The output head couldn't make confident predictions (logits were tiny), so the model got stuck just above random. We added a learnable logit scale to fix it. | If a model can't express confidence, you're measuring the head's handicap, not the architecture. |
| B5 | **Sub-unigram regime** | The whole full-architecture setup was performing *worse than a context-free unigram model* — so every comparison in it was meaningless. We'd been screening in a broken regime. | Always check you beat the unigram floor (7.66) before trusting any comparison. |
| B6 | **Frozen embedding + head** | The token embedding and output head were **never trained** — silently excluded from the optimizer — for every full-architecture run. | Verify *every* weight actually changes during training. This is the root cause beneath B3–B5. |

**The meta-pattern:** B3, B4, B5, and B6 are *the same underlying problem viewed from different angles* — the output head was broken (frozen, B6), which made it unable to express confidence (B4), which kept us below the unigram floor (B5), which made a re-norm trick look like a win (B3). **We kept treating symptoms (add a logit scale, add a unigram bias) instead of finding the root cause (a frozen weight).** When several independent "fixes" all point at the same subsystem, stop patching and audit that subsystem directly.

---

## 2. Standing checks (run these to catch this whole class of bug)

These are cheap. Run them on any new model/harness before trusting a single number.

### C1 — Optimizer coverage (would have caught B6 immediately)
Every trainable parameter must be in *some* optimizer group. `named_parameters()`
deduplicates tied weights to one name, so a "skip this tied copy" line can silently
drop the *only* copy.
```python
opt_ids = {id(p) for o in optimizers for g in o.param_groups for p in g["params"]}
missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in opt_ids]
assert not missing, f"NOT being optimized: {missing}"
```

### C2 — Every weight actually moves (the universal frozen-param check)
```python
before = {n: p.detach().clone() for n, p in model.named_parameters()}
# ... run K training steps ...
frozen = [n for n, p in model.named_parameters()
          if p.requires_grad and torch.equal(p, before[n])]
assert not frozen, f"these params never changed: {frozen}"
```
Run for K=20 steps. Anything that *should* train but didn't is a B6-class bug.
(Conversely: things that should *not* train — buffers like the unigram bias — should
appear unchanged.)

### C3 — Loss at initialization matches theory
- Plain random model: loss ≈ `ln(vocab)` = **10.83**.
- With the unigram-log-prob bias: loss ≈ unigram entropy = **7.66** (on real data).
- Anything materially lower at step 0 ⇒ a label leak or a bias you didn't intend.

### C4 — Unigram floor (would have caught B5)
Final loss **must** beat the unigram entropy (7.66 here; recompute `−Σ pᵢ ln pᵢ`
for any new data). Above it = the model isn't using context; the comparison is void.

### C5 — Paired stats with correlated seeds (would have caught B2)
Use the *same* seed for variant and baseline; report the mean of per-seed **differences**
and a paired t-test — never the difference of group means across unequal n.

### C6 — Matched hyperparameters (would have caught B1)
Before claiming an architecture effect, give each variant its own tuned LR (or sweep a
small grid). A "win" that disappears under a matched/own-best LR was an LR confound.

### C7 — Head can express confidence (would have caught B4)
Check the logit spread early in training: `logits.std()` / `logits.abs().max()`. If it's
tiny (≈0.1), the head is magnitude-limited and can't be confident — fix the head before
comparing anything downstream of it.

### C8 — Re-normalize before the cap (would have caught B3)
Any residual/branch added near the output must pass through the same normalization as the
main path before the (soft)cap. Otherwise it can lower the loss by inflating logit scale.

### C9 — Environment variables took effect
`PYTORCH_CUDA_ALLOC_CONF` is the name that engages `expandable_segments`; the "new"
`PYTORCH_ALLOC_CONF` silently *didn't* and caused OOMs. Verify the behavior you wanted
actually happened (memory curve, warnings), don't assume the var was read.

### C10 — Don't mistake buffering for a hang
`nohup` buffers stdout; an empty log ≠ a stuck run. Check GPU utilization and whether
params are updating before concluding anything hung.

---

## 3. Detailed bug entries (full root-cause for future reference)

### B1 — Learning-rate confound in the activation study
- **Symptom:** `sniqu` appeared to *lose* to `relu2` by ~0.078 under NorMuon.
- **Root cause:** the activations were compared at a single shared LR; `sniqu` prefers a
  ~2× higher LR. The "loss" was an LR mismatch, not an activation effect.
- **Found by:** re-running each activation at its own best LR (matched comparison) — the
  gap dissolved.
- **Avoid:** C6. Never compare architectures at a single LR unless you've shown the LR
  optimum doesn't move with the change.
- **Affected:** the (now-removed) NorMuon "terminal result."

### B2 — Unpaired comparison with correlated seeds
- **Symptom:** the per-layer conv branch looked like a significant win at small n.
- **Root cause:** I compared baseline (n=5) mean vs conv (n=2) mean as if independent.
  Seeds are shared across variants → outcomes are correlated → group-mean comparison
  overstates significance and is the wrong test.
- **Found by:** the user flagging seed correlation; re-running as a paired comparison
  (same seeds) collapsed the effect to null at n=5.
- **Avoid:** C5. Pair by seed; test the per-seed differences.
- **Affected:** the conv-branch "win" (retracted; confirmed null).

### B3 — Shared conv-tower gamed the softcap
- **Symptom:** `tower_shared` reached ~7.95 loss — suspiciously good.
- **Root cause:** in shared mode the conv tower was added to the residual stream **after**
  the final RMS-norm, with no re-normalization. That inflated the logit magnitude, which
  exploited the logit soft-cap (`23·σ((logits+5)/7.5)`) to lower CE *without better
  predictions*. Causality was clean; the scale was the cheat.
- **Found by:** the number being too good + auditing where the tower entered the stream.
- **Fix:** `x = norm(x + conv_mix * conv_tower(x_emb))` — re-normalize before the head.
- **Avoid:** C8, and treat any near-head win as a scale artifact until proven otherwise.
- **Affected:** the quarantined `tower_shared` JSON.

### B4 — Crippled head: tied + normed + no learnable logit scale
- **Symptom:** under plain CE the baseline stalled at ~9.6 (just below random 10.83);
  under softcap it floored at ~10.
- **Root cause:** RMS-norm before a tied head with no learnable output scale → logits
  ≈ ±0.14 → near-uniform softmax → can't express confidence. (The record avoids this via
  untie + asymmetric logit rescale, which we'd omitted.)
- **Found by:** the softcap "floor" investigation; confirmed by adding a global logit
  scale → loss dropped to ~5.8.
- **Fix (partial / symptomatic):** a learnable global logit scale.
- **Avoid:** C7, C3.
- **Affected:** every full-arch screen — and note this was a *symptom* of B6, not the root.

### B5 — Screening below the unigram floor
- **Symptom:** none, until we computed it — that *was* the problem.
- **Root cause:** the full-arch runs (softcapped ~10, crippled plain-CE ~9.6) sat **above**
  the unigram entropy (7.66), i.e. worse than a context-free model. Architecture changes
  in that regime are masked by the head bottleneck, so comparisons were near-meaningless.
- **Found by:** the user asking us to compute `−Σ pᵢ ln pᵢ` and compare.
- **Fix/Check:** C4 — require beating 7.66; discard sub-unigram results. (We removed a
  block of results because of this.)
- **Affected:** the budget BO search and the NorMuon replication (removed).

### B6 — Frozen tied embedding + output head (the root cause)
- **Symptom:** none visible — the model still trained (to ~5.8) via its *other* trainable
  parameters (value embeddings, bigram embeddings, body, and the added logit scale).
- **Root cause:** `self.lm_head.weight = self.embed.weight` (standard tie). PyTorch's
  `named_parameters()` deduplicates the shared tensor to the **first** registered name,
  `embed.weight`. `build_optimizers` then did `if name == "embed.weight": skip  # lm_head
  carries it` — but `lm_head.weight` is the deduped name and **never appears**. Result:
  the shared token-embedding **and** output head were excluded from the optimizer and
  **frozen at init (std 0.005)** for the entire phase2 line.
- **Found by:** implementing `--head-rank` forced an audit of the optimizer routing; the
  param-count math (tied vs untie) didn't add up, and C1 confirmed the shared weight's id
  was in no optimizer group.
- **Fix:** route the (deduped) `embed.weight` into the optimizer unconditionally; in the
  untied case it's the standalone embedding, in the tied case it's the shared weight.
- **Avoid:** C1 and C2 — both would have caught this on day one.
- **Affected:** **all** phase2 results, including the published unigram-bias headline.
  Why the unigram bias looked so strong: a frozen *random* head can't learn the token
  marginal, so handing it the marginal is worth a lot. With a trainable head the gap
  should shrink — re-measurement pending on the corrected baseline.
- **Relationship to others:** B6 → causes B4 (frozen head can't make confident logits) →
  causes B5 (stuck below unigram) → enables B3 (scale tricks look like wins). One root,
  three symptoms we patched separately.

### Operational / near-misses (not result-affecting, but cost time)
- **`PYTORCH_ALLOC_CONF` silent no-op:** the renamed env var didn't enable
  `expandable_segments`; runs OOM'd. Reverted to `PYTORCH_CUDA_ALLOC_CONF`. (C9)
- **`nohup` stdout buffering looked like a hang.** GPU was at 97%; it was just unflushed
  output. (C10)
- **Multiple competing drivers** left orphaned `phase2` processes contending for the GPU
  after `pkill` killed a driver but not its child. Always verify the GPU is clear after
  stopping a run.
- **Disk filled → inductor cache mkdir failures** silently killed grid runs. Monitor disk.

---

## 4. Hypotheses — other bugs worth checking

Ordered roughly by suspicion. The top group is the *same class* as B6 (silent optimizer /
gradient coverage) and should be checked first, now that we know that failure mode exists.

**Same class as B6 (highest priority — run C1/C2 on the full real model):**
- [ ] Are **value_embeds**, **bigram_embed**, the **logit scale**, **temp_head**, and all
      conv params actually in the optimizer and changing? (C1 + C2 on the full model.)
- [ ] Other aliased/tied tensors: do **kv_tied** / **v_identity** create shared params that
      get frozen *or* double-counted in the optimizer?
- [ ] Is the **unigram bias** correctly a non-trained *buffer* (changes under C2 = bug),
      and does it move to the right device/dtype with the model?
- [ ] Does the low-rank head offset actually receive gradients to **both** factors once
      `up` leaves zero (the `down` factor is frozen until then by construction)?

**Numerics / autograd:**
- [ ] **mm8 / FP8 custom autograd**: does its straight-through bf16 backward match an eager
      reference within tolerance? A wrong backward would silently bias every run that uses it.
- [ ] **Checkpointed chunked CE**: do the logit scale and unigram bias get *correct*
      gradients through `torch.utils.checkpoint` (recompute path)? Compare to non-chunked.
- [ ] **compile vs eager**: does `torch.compile` change the loss vs eager beyond fp noise?

**Correctness of the task itself:**
- [ ] **Causal/off-by-one**: confirm targets are token *t+1* given context *≤t*, with no leak
      (e.g., the adaptive temperature/head can't see the target). Verify with C3-style init loss.
- [ ] **Identity-init convs**: are the depthwise convs actually identity at step 0 (output ==
      input)? An init that isn't identity changes the baseline silently.
- [ ] **Data shard reading**: header offset (256 int32) and uint16 dtype correct; no token
      misalignment; batches actually random and non-repeating across steps.

**Schedule / budget:**
- [ ] **Time-budget cooldown**: if the step count is hit before `max_seconds`, does the LR
      schedule still *complete* its cooldown, or does it get truncated (un-annealed end)?
- [ ] **Warmup vs cooldown units**: warmup is in steps, cooldown is in time-fraction — does
      their interaction ever produce a non-monotone or truncated schedule at odd budgets?

**Statistical / process:**
- [ ] **Seed independence**: are train and val generators actually distinct and reproducible,
      and are "different seeds" truly producing different data orders?
- [ ] **Silent truncation**: any top-N / no-retry / sampling cap that we don't `log()` —
      could read as "covered everything" when it didn't.

---

## 5. How to extend this log
When a new bug is found: add a one-liner to §1, a `C#` check to §2 if it's catchable, a full
entry to §3, and prune anything in §4 that it confirms or rules out. Keep §1 readable by
someone who's never seen the codebase; keep §3 precise enough to re-derive the fix.
