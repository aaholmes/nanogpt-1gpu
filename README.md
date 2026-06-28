# nanogpt-1gpu
<!-- After pushing, enable the CI badge (replace OWNER):
[![checks](https://github.com/OWNER/nanogpt-1gpu/actions/workflows/checks.yml/badge.svg)](https://github.com/OWNER/nanogpt-1gpu/actions/workflows/checks.yml) -->

A single-GPU adaptation of the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) speedrun, for local experimentation and architecture screening on **one consumer GPU (16 GB)**. The speedrun trains a 124M-parameter GPT on FineWeb to 3.28 validation loss as fast as possible on 8×H100 — hardware beyond a typical home setup. This is a faithful-where-it-can-be re-implementation that runs on a single GPU, paired with a methodology built to keep comparisons trustworthy.

It's a research **harness, not a benchmark** — a tool I use to decide what's worth validating on the real thing, not a stable reference for others to adopt. Absolute numbers here don't transfer to the record; the value is in the *rankings* and the *method*.

**Lineage & credit.** This descends from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) / [llm.c](https://github.com/karpathy/llm.c) (the 3.28-on-FineWeb target and the original code lineage) and from Keller Jordan and the modded-nanogpt contributors (the speedrun architecture and the Muon / NorMuon optimizer). Their work is the foundation — see `LICENSE`. The single-GPU harness and the methodology are mine.

## What it is

One file, [`harness.py`](harness.py) (~950 lines, in the single-file nanoGPT tradition): the record's architecture — RoPE/YaRN, value + bigram embeddings, paired-head attention with skip/backout connections, always-on QK-norm, a relu-squared MLP, and a tied output head — trained with **NorMuon** (Polar-Express orthogonalization) under a **trapezoidal/WSD learning-rate schedule** on a wall-clock budget, plain cross-entropy. No experimental sprawl: one MLP, one activation, one head — the validated configuration.

The single-GPU simplifications (SDPA instead of FlashAttention-3, bf16 instead of FP8, a pure-PyTorch orthogonalizer, fixed batch/sequence length) are deliberate and documented below.

## Methodology

The point of the repo is a fast screening loop that doesn't lie to you. Cheap, sharp checks — run `python checks.py`:

- **Every weight actually trains.** The harness asserts that every trainable parameter is routed to an optimizer and changes during training. (A silently-frozen weight is a subtle, devastating bug; this makes that class impossible.)
- **Beat the unigram floor.** A run only counts if it beats a context-free unigram model: `−Σ pᵢ ln pᵢ ≈ 7.66` on this data (vs. `ln(50304) ≈ 10.83` for random). Below 7.66 the model isn't using context and the comparison is meaningless.
- **Paired comparisons.** Same seed for variant and baseline; test the per-seed differences, not group means.
- **Matched hyperparameters.** Tune the learning rate per variant before believing an architecture effect — an LR confound is easily mistaken for a win.
- **Budget stability.** Re-run apparent winners at longer budgets; early leads often fade.

CI runs `python checks.py` on every push (see [`.github/workflows/checks.yml`](.github/workflows/checks.yml)), so the self-verification can't silently regress.

## How it differs from the speedrun

| | |
|---|---|
| **Attention / precision** | SDPA instead of FlashAttention-3, bf16 instead of FP8 — the record's fused kernels need a setup I don't run locally. |
| **Optimizer kernel** | pure-PyTorch Polar-Express orthogonalizer instead of the Triton kernel. |
| **LR schedule** | *matched* — trapezoidal/WSD (constant, then linear cooldown to 0.15× over the final 60%), on a wall-clock budget so iso-compute comparisons work. |
| **Batch / seq-len** | fixed — the record ramps both, but its batch sizes need 8×H100 and its seq-len schedule is tied to the FA3 windowing. |
| **Output head / loss** | plain tied head + plain cross-entropy (the record unties at 2/3 and softcaps). |
| **Scale** | one GPU (16 GB) vs 8×H100, shorter budgets. |

## Quickstart

```bash
python data/cached_fineweb10B.py 9          # download ~900M training tokens
python checks.py                            # self-verification (CPU, seconds)
python harness.py --config speedrun --compile \
                  --batch-size 12 --max-seconds 1800 --seeds 3 --out results/run.json
```
