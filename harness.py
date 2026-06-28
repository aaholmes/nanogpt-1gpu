"""
Phase 2: Activation study using the full NanoGPT speedrun architecture (single GPU).

Starts from record-holder train_gpt.py. Key changes for single-GPU use:
  - Remove distributed/DDP (world_size=1 assumed)
  - Replace FA3 (flash_attn_varlen_func) with F.scaled_dot_product_attention
  - Replace FP8 CastedLinearT with standard nn.Linear
  - Replace Triton polar_express with pure-PyTorch orthogonalizer (from phase1.py)
  - Replace ReLUSqrdMLP with swappable activation classes (from phase1.py)
  - Use simple random batch sampler instead of BOS-aligned distributed loader
  - Single-stage cosine LR schedule (no multi-stage batch/seqlen ramp)

Architecture preserved from train_gpt.py:
  - RoPE / YaRN (half-truncated, half-stationary)
  - Paired head layers (0, 2, 5, 9)
  - Key offset on long-window layers (3, 10) for induction
  - Value embeddings (5 banks) with learned per-layer gates
  - Bigram embeddings with learned per-layer lambdas
  - Learnable residual scalars (resid_lambdas, post_lambdas, x0_lambdas)
  - Learnable SA lambdas (per-layer Q/K and O scaling)
  - Smear gate (shift token embed forward 1 position)
  - Skip connection layer 3 → layer 6
  - Backout (layer 7 contribution subtracted from final x)
  - QK-norm always-on (F.rms_norm, matching speedrun)
  - Attention gate and VE gate
  - Softcapped cross-entropy (23*sigmoid((logits+5)/7.5))

Experimental note: phase1 vs phase2 baselines are themselves informative.
Any phase1 result should be re-validated here before claiming relevance to the speedrun.
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use TF32 for any fp32 matmuls (cheap throughput win on tensor cores; the bf16
# autocast already covers most of the model, this catches the rest).
torch.set_float32_matmul_precision("high")


# -----------------------------------------------------------------------------
# All large matmuls go through mm8 (a thin F.linear wrapper). The name is kept as the
# single seam where a custom (e.g. FP8) GEMM could be slotted in later.

def mm8(x, weight):
    """x @ weight.T, weight is (out, in)."""
    return F.linear(x, weight)


# -----------------------------------------------------------------------------
# Activations (from phase1.py)

class IQU(nn.Module):
    def forward(self, x):
        pos = F.relu(x)
        neg = x - pos
        return pos + pos * pos + neg / (1.0 - neg)


class SNIQU(nn.Module):
    BETA = 0.70638
    LAM = 0.56004
    def forward(self, x):
        pos = F.relu(x)
        neg = x - pos
        h = pos + pos * pos + neg / (1.0 - neg)
        return self.LAM * (h - self.BETA)


class ReLU2(nn.Module):
    def __init__(self, slope=0.0, shift=0.0, curv=1.0):
        super().__init__()
        self.slope, self.shift, self.curv = slope, shift, curv

    def forward(self, x):
        z = F.relu(x - self.shift) if self.shift else F.relu(x)
        out = self.curv * z.square() if self.curv != 1.0 else z.square()
        if self.slope:
            out = out + self.slope * z
        return out


class XAbsX(nn.Module):
    def __init__(self, slope=0.0, curv=1.0):
        super().__init__()
        self.slope, self.curv = slope, curv

    def forward(self, x):
        out = self.curv * x * x.abs() if self.curv != 1.0 else x * x.abs()
        if self.slope:
            out = out + self.slope * x
        return out


def _inv_softplus(y):
    return math.log(math.expm1(y)) if y < 20 else y


_INIT_VALUES = {
    "gelu_minimax": dict(L=1.910, R=1.910, alpha=0.247),
    "gelu_rms":     dict(L=1.749, R=1.749, alpha=0.275),
    "c1":           dict(L=1.000, R=1.000, alpha=0.250),
}


class TriLU(nn.Module):
    def __init__(self, asymmetric=True, init="gelu_minimax"):
        super().__init__()
        v = _INIT_VALUES[init]
        if asymmetric:
            self.theta_L = nn.Parameter(torch.tensor(_inv_softplus(v["L"])))
            self.theta_R = nn.Parameter(torch.tensor(_inv_softplus(v["R"])))
        else:
            self.theta_a = nn.Parameter(torch.tensor(_inv_softplus(v["L"])))
        self.theta_alpha = nn.Parameter(torch.tensor(_inv_softplus(v["alpha"])))
        self.asymmetric = asymmetric

    def get_params(self):
        if self.asymmetric:
            L = -F.softplus(self.theta_L)
            R = F.softplus(self.theta_R)
        else:
            a = F.softplus(self.theta_a)
            L, R = -a, a
        return L, R, F.softplus(self.theta_alpha)

    def forward(self, x):
        L, R, alpha = self.get_params()
        denom = (R - L).clamp(min=1e-6)
        beta = R / denom - alpha * (L + R)
        gamma = alpha * L * R - L * R / denom
        q = alpha * x * x + beta * x + gamma
        return torch.where(x <= L, torch.zeros_like(x), torch.where(x >= R, x, q))


def make_activation(name, init="gelu_minimax", slope=0.0, shift=0.0, curv=1.0):
    if name in ("relu2", "reglu"):   return ReLU2(slope=slope, shift=shift, curv=curv)
    if name in ("gelu", "geglu"):    return nn.GELU()
    if name == "swiglu":             return nn.SiLU()
    if name == "trilu_sym":          return TriLU(asymmetric=False, init=init)
    if name in ("trilu_asym", "triglu"): return TriLU(asymmetric=True, init=init)
    if name in ("xabsx", "xglu"):    return XAbsX(slope=slope, curv=curv)
    if name == "bilinear":           return nn.Identity()
    if name == "iqu":                return IQU()
    if name in ("sniqu", "iqu_sn"):  return SNIQU()
    if name == "selu":               return nn.SELU()
    raise ValueError(f"Unknown activation: {name}")


GATED = {"swiglu", "geglu", "triglu", "xglu", "reglu", "bilinear"}


class MLP(nn.Module):
    def __init__(self, model_dim, act):
        super().__init__()
        hidden = 4 * model_dim
        self.fc  = nn.Linear(model_dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.act = act

    def forward(self, x):
        return mm8(self.act(mm8(x, self.fc.weight)), self.proj.weight)


class GatedMLP(nn.Module):
    def __init__(self, model_dim, act):
        super().__init__()
        hidden = max(64, round(8 * model_dim / 3 / 64) * 64)
        self.fc      = nn.Linear(model_dim, hidden, bias=False)
        self.fc_gate = nn.Linear(model_dim, hidden, bias=False)
        self.proj    = nn.Linear(hidden, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.act = act

    def forward(self, x):
        return mm8(self.act(mm8(x, self.fc.weight)) * mm8(x, self.fc_gate.weight), self.proj.weight)


def make_mlp(model_dim, act_name, init="gelu_minimax", slope=0.0, shift=0.0, curv=1.0):
    act = make_activation(act_name, init, slope, shift, curv)
    if act_name in GATED:
        return GatedMLP(model_dim, act)
    return MLP(model_dim, act)


# -----------------------------------------------------------------------------
# Orthogonalizer (pure PyTorch, from phase1.py — replaces Triton polar_express)

_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461,  -22.48329292557795,   15.878769915207462),
    (4.042929935166739,   -2.808917465908714,   0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,   0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376,  0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081,  0.42323551169305323),
]
_NEWTON_SCHULZ_COEFFS = [(3.4445, -4.7750, 2.0315)] * 5


@torch.no_grad()
def orthogonalize(G, method="polar_express"):
    assert G.ndim == 2
    coeffs = _POLAR_EXPRESS_COEFFS if method == "polar_express" else _NEWTON_SCHULZ_COEFFS
    X = G.bfloat16()
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() * (1 + 2e-2) + 1e-7)
    for a, b, c in coeffs:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


@torch.no_grad()
def normuon_variance_reduce(v, buf, beta2):
    red_dim = -1 if v.size(-2) >= v.size(-1) else -2
    n = v.size(red_dim)
    v_mean = v.float().square().mean(dim=red_dim, keepdim=True)
    v_norm = (v_mean.sum() * n).sqrt()
    buf.lerp_(v_mean.to(buf.dtype), 1 - beta2)
    step = buf.clamp_min(1e-10).rsqrt()
    v_norm_new = ((v_mean * n) * step.float().square()).sum().sqrt().clamp_min(1e-10)
    return v.mul_((step * (v_norm / v_norm_new)).type_as(v))


# -----------------------------------------------------------------------------
# Norm helper (RMSNorm, no learnable params — matches train_gpt.py exactly)

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# -----------------------------------------------------------------------------
# YaRN rotary embeddings (from train_gpt.py, device-parameterised)

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, device, paired=False):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.paired = paired
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        f1 = self.factor1[None, :x_BTHD.size(-3), None, :]
        f2 = self.factor2[None, :x_BTHD.size(-3), None, :]
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return f1 * x_BTHD + f2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim // 4,
                                                     dtype=torch.float32, device=self.device)
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 2)])
        t = torch.arange(2 * self.max_seq_len, dtype=torch.float32, device=self.device)
        if not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
            self.factor2 = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        else:
            t_even, t_odd = 2 * t, 2 * t + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd,  angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16), persistent=False)
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16), persistent=False)
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        self.attn_scale = 0.1  # from train_gpt.py, inspired by @leloykun


# Speedrun layer topology — the legacy L=11 record structure lives in
# topology.legacy_topology(); topology.build_topology() generates depth/skip
# variants for architecture search. The GPT model derives all per-layer indexing
# from a topology dict (default = legacy), so num_layers and the skip/backout
# pattern are no longer hardcoded.
from topology import legacy_topology, build_topology, validate_topology  # noqa: E402

# Legacy constants kept for reference / back-compat with older scripts.
KEY_OFFSET_LAYERS  = {3, 10}
PAIRED_HEAD_LAYERS = {0, 2, 5, 9}
ATTN_SKIP_LAYER    = 6
VE_LAYERS          = {1, 2, 8, 9, 10}


# -----------------------------------------------------------------------------
# Causal Self-Attention (SDPA-based, replaces FA3)

class DepthwiseCausalConv1d(nn.Module):
    """Per-channel (depthwise) causal 1-D conv over the time axis, shape (B, T, C).

    Each channel gets its own length-k kernel, left-padded by k-1 so position t sees
    only t-k+1 .. t (strictly causal). Identity-init (last tap = 1) so it starts as a
    no-op. Applied to q/k/v BEFORE QK-norm and RoPE — conv-before-rotary, the sane
    ordering (conv-after-rotary would mix rotated coordinates across time)."""

    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, -1] = 1.0

    def forward(self, x):
        x = x.transpose(1, 2)                          # (B, C, T)
        x = F.pad(x, (self.kernel_size - 1, 0))        # causal left-pad
        # contiguous: the paired-head path downstream does strided .view()s that
        # reject the transpose's non-contiguous output.
        return self.conv(x).transpose(1, 2).contiguous()   # (B, T, C)


class ConvBranch(nn.Module):
    """Parallel local-mixing branch, run alongside attention and summed into the residual:
        out = scale * fc_out( relu2( dwconv( fc_in(x) ) ) )
    Unlike the in-series q/k/v conv (which can only act THROUGH attention's value
    aggregation), this is an independent path: even with attention removed it can learn
    local structure — n-gram-like "which word-types follow which" — leaving attention to
    handle global meaning. The pointwise fc_in/fc_out give the cross-channel mixing that
    a pure depthwise conv lacks (needed to relate word-types, not just smooth channels).

    Init = LayerScale, not ReZero: fc_in/fc_out are variance-matched (normal init) and the
    branch is gated by a small learnable per-channel `scale` (init 0.1). So at step 0 the
    branch is a *small* well-scaled perturbation (~2% of the residual), not dead zero —
    every weight gets gradient immediately (no one-step starvation), while the gain still
    lets the model dial the branch up or down. Matrices are Muon-trained; `scale` is a 1D
    AdamW param. The depthwise kernel (C,1,K) isn't a 2D matrix, so it goes to AdamW."""

    def __init__(self, model_dim, kernel, hidden, layerscale_init=0.1):
        super().__init__()
        b_in  = (3 ** 0.5) * 0.5 * model_dim ** -0.5
        b_out = (3 ** 0.5) * 0.5 * hidden ** -0.5
        self.fc_in  = nn.Linear(model_dim, hidden, bias=False)
        self.dwconv = DepthwiseCausalConv1d(hidden, kernel)
        self.fc_out = nn.Linear(hidden, model_dim, bias=False)
        with torch.no_grad():
            self.fc_in.weight.uniform_(-b_in, b_in)
            self.fc_out.weight.uniform_(-b_out, b_out)   # variance-matched, NOT zero
        self.scale = nn.Parameter(layerscale_init * torch.ones(model_dim))

    def forward(self, x):
        return self.scale * mm8(F.relu(self.dwconv(mm8(x, self.fc_in.weight))).square(), self.fc_out.weight)


class ConvTower(nn.Module):
    """Transformer-free local-only body: embedded tokens -> a stack of residual causal-conv
    blocks -> final hidden state. The causal conv gives it position implicitly (no RoPE).
    Used as a parallel product-of-experts track alongside the transformer, or standalone
    (the conv-only baseline). Blocks use full-strength residuals (LayerScale 1.0); the whole
    tower's contribution is gated by an outer learnable mix when combined with the transformer."""

    def __init__(self, model_dim, n_blocks, kernel, hidden):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBranch(model_dim, kernel, hidden, layerscale_init=1.0) for _ in range(n_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(norm(x))
        return norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, qk_layernorm=False,
                 kv_tied=False, v_identity=False, drop_o=False, qkv_conv=0, qkv_conv_where="qkv"):
        super().__init__()
        assert not (kv_tied and v_identity), "kv_tied and v_identity are mutually exclusive"
        self.num_heads  = num_heads
        self.head_dim   = head_dim
        self.hdim       = num_heads * head_dim
        self.kv_tied    = kv_tied
        self.v_identity = v_identity
        self.drop_o     = drop_o
        # Short depthwise causal conv before norm/RoPE (kernel qkv_conv; 0=off). qkv_conv_where
        # selects which projections get a conv: 'qkv' all, 'v' value-only, 'qk' query/key-only.
        # V-only is the cleanest test of content-smoothing: RoPE never touches V, and conv on V
        # avoids blurring the Q/K addressing. A projection's conv is None when not selected;
        # v_conv is also None when V is derived (kv_tied/v_identity) and so can't be convolved.
        self.qkv_conv = qkv_conv
        mk = lambda sel: DepthwiseCausalConv1d(self.hdim, qkv_conv) if (qkv_conv and sel) else None
        self.q_conv = mk("q" in qkv_conv_where)
        self.k_conv = mk("k" in qkv_conv_where)
        self.v_conv = mk(("v" in qkv_conv_where) and not (kv_tied or v_identity))
        std   = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        if kv_tied or v_identity:
            # Only Q and K projections; V comes from K (kv_tied) or input x (v_identity)
            self.qk = nn.Linear(model_dim, 2 * self.hdim, bias=False)
            with torch.no_grad():
                self.qk.weight.uniform_(-bound, bound)
        else:
            self.qkv = nn.Linear(model_dim, 3 * self.hdim, bias=False)
            with torch.no_grad():
                self.qkv.weight.uniform_(-bound, bound)
        if drop_o:
            # No output projection: concatenated head outputs go straight to the
            # residual stream (only a learnable scalar scale via sa_lambdas[1]).
            # Requires hdim == model_dim so dims line up with the residual stream.
            assert self.hdim == model_dim, \
                f"drop_o requires hdim ({self.hdim}) == model_dim ({model_dim})"
            self.out = None
        else:
            self.out = nn.Linear(self.hdim, model_dim, bias=False)
            with torch.no_grad():
                self.out.weight.uniform_(-bound, bound)
        self.q_norm = nn.LayerNorm(head_dim, bias=False) if qk_layernorm else None
        self.k_norm = nn.LayerNorm(head_dim, bias=False) if qk_layernorm else None

    def forward(self, x, yarn, sa_lambdas, paired,
                attn_gate_w=None, ve=None, ve_gate_w=None, key_offset=False):
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # QKV (or QK only) with learnable scale (sa_lambdas[0])
        if self.kv_tied or self.v_identity:
            qk = mm8(x, sa_lambdas[0] * self.qk.weight)
            q, k = qk.chunk(2, dim=-1)
            if self.q_conv is not None: q = self.q_conv(q)
            if self.k_conv is not None: k = self.k_conv(k)
            q = q.view(B, T, H, D)
            k = k.view(B, T, H, D)
            # V comes from K (before QK-norm/RoPE; inherits the K conv) or from the input
            v = k.clone() if self.kv_tied else x.view(B, T, H, D)
        else:
            qkv = mm8(x, sa_lambdas[0] * self.qkv.weight)
            q, k, v = qkv.chunk(3, dim=-1)
            if self.q_conv is not None: q = self.q_conv(q)
            if self.k_conv is not None: k = self.k_conv(k)
            if self.v_conv is not None: v = self.v_conv(v)
            q = q.view(B, T, H, D)
            k = k.view(B, T, H, D)
            v = v.view(B, T, H, D)

        # QK-norm: RMSNorm always-on (matches train_gpt.py); LayerNorm if --qk-layernorm
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = norm(q), norm(k)

        if not paired:
            q, k = yarn.rotary(q), yarn.rotary(k)
            if key_offset:
                # Shift keys forward for stationary head dims — enables 1-layer induction
                k[:, 1:, :, D // 2:] = k[:, :-1, :, D // 2:]
            if ve is not None:
                # Gate: concat first 6 dims of x and ve
                gate_in = torch.cat([x[..., :6], ve[..., :6]], dim=-1)   # (B,T,12)
                ve_gate = 2 * torch.sigmoid(F.linear(gate_in, ve_gate_w)) \
                              .view(B, T, H, 1)                            # (B,T,H,1)
                v = v + ve_gate * ve.view(B, T, H, D)

            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, scale=yarn.attn_scale
            ).transpose(1, 2)   # (B, T, H, D)
        else:
            # Paired heads: adjacent head pairs attend over doubled sequence length
            q = q.view(B, T, H // 2, D * 2)
            k = k.view(B, T, H // 2, D * 2)
            v = v.reshape(B, T * 2, H // 2, D)    # interleave heads→positions

            q, k = yarn.rotary(q), yarn.rotary(k)

            q = q.view(B, T * 2, H // 2, D)
            k = k.view(B, T * 2, H // 2, D)

            if ve is not None:
                # Use first 12 dims of x for gate (viewed as 2×(H//2) positions)
                ve_gate = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w)) \
                              .view(B, T * 2, H // 2, 1)
                v = v + ve_gate * ve.view(B, T * 2, H // 2, D)

            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, scale=yarn.attn_scale
            ).transpose(1, 2)   # (B, T*2, H//2, D)

        # Reshape back to (B, T, H, D) regardless of paired/regular
        y = y.contiguous().view(B, T, H, D)

        # Attention gate applied uniformly in (B, T, H) space
        if attn_gate_w is not None:
            gate = torch.sigmoid(F.linear(x[..., :12], attn_gate_w))  # (B, T, H)
            y = y * gate.unsqueeze(-1)                                  # (B, T, H, 1) broadcast

        y = y.contiguous().view(B, T, self.hdim)
        if self.drop_o:
            # No O matrix: concatenated heads go straight out (scalar scale only)
            return sa_lambdas[1] * y
        # Output projection with learnable O scale (sa_lambdas[1])
        return mm8(y, sa_lambdas[1] * self.out.weight)


# -----------------------------------------------------------------------------
# Main GPT model

def next_multiple_of_n(v, n):
    return math.ceil(v / n) * n


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, head_dim, model_dim,
                 max_seq_len, device, act_name="relu2", act_init="gelu_minimax",
                 act_slope=0.0, act_shift=0.0, act_curv=1.0, qk_layernorm=False,
                 paired_heads=False, kv_tied=False, v_identity=False, drop_o=False,
                 qkv_conv=0, qkv_conv_where="qkv", conv_branch=0, conv_branch_hidden=0,
                 conv_tower=0, conv_tower_kernel=4, conv_tower_width=0, conv_tower_mode="off",
                 skip_conv=0, logit_temp="none", unigram_bias=False, head_rank=0, topology=None):
        super().__init__()
        # Topology: default = exact legacy L=11 record structure (parity-preserving).
        # A passed-in topology dict (from topology.build_topology) enables depth/skip
        # search. All per-layer indexing below is derived from it.
        topo = topology if topology is not None else legacy_topology()
        validate_topology(topo)
        assert topo["num_layers"] == num_layers, \
            f"topology num_layers {topo['num_layers']} != requested {num_layers}"
        self.topo          = topo
        self.skips         = dict(topo["skips"])              # dst -> src
        self.skip_dsts     = set(topo["skips"].keys())
        self.skip_srcs     = set(topo["skips"].values())
        self.backout_src   = topo["backout_src"]
        self.backout_mode  = topo["backout_mode"]
        self.key_offset_layers = set(topo["key_offset_layers"])
        ve_layers          = list(topo["ve_layers"])         # ordered → bank index
        self.ve_layers_set = set(ve_layers)
        self._ve_gate_idx  = {layer: j for j, layer in enumerate(ve_layers)}
        attn_layers_topo   = list(topo["attn_layers"])
        self._attn_gate_idx = {layer: j for j, layer in enumerate(attn_layers_topo)}
        self.n_ve          = len(ve_layers)
        self.num_layers = num_layers
        self.vocab_size = next_multiple_of_n(vocab_size, 128)
        self.model_dim  = model_dim
        self.num_heads  = num_heads
        self.head_dim   = head_dim

        std   = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std

        # Embeddings
        self.embed        = nn.Embedding(self.vocab_size, model_dim)
        self.bigram_embed = nn.Embedding(BIGRAM_VOCAB_SIZE, model_dim)
        nn.init.zeros_(self.bigram_embed.weight)

        # Value embeddings: n_ve banks × vocab × model_dim (small init, matches speedrun).
        # n_ve == 5 for the legacy topology, preserving the RNG draw for parity.
        self.value_embeds = nn.Parameter(0.01 * torch.randn(self.n_ve * self.vocab_size, model_dim))

        # lm_head tied to embed (both (vocab_size, model_dim), standard tie)
        self.lm_head = nn.Linear(model_dim, self.vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.005)
        self.lm_head.weight = self.embed.weight  # tied
        # Head parameterization: 0 = tied (above); r>0 = tied base + rank-r LoRA-style offset
        # (W_out = embed + up@down, init offset 0 so it starts tied); -1 = full untie.
        self.head_rank = head_rank
        if head_rank and head_rank > 0:
            _rng = torch.get_rng_state()   # restore after, so the body init is identical across head modes
            self.head_lr_down = nn.Parameter(torch.empty(head_rank, model_dim).normal_(std=0.02))
            torch.set_rng_state(_rng)
            self.head_lr_up   = nn.Parameter(torch.zeros(self.vocab_size, head_rank))  # offset=0 at init
        elif head_rank == -1:
            self.lm_head.weight = nn.Parameter(self.embed.weight.detach().clone())  # untie; starts equal

        # Smear and skip gates (zero-init → neutral at start)
        self.smear_gate = nn.Linear(12, 1, bias=False)
        self.skip_gate  = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)
        nn.init.zeros_(self.skip_gate.weight)

        # Per-layer attention gate bank: one gate per attention layer (== num_layers
        # minus the number of skip-destination layers that drop attention).
        # VE gate bank: one gate per value-embedding layer.
        n_attn = len(attn_layers_topo)
        self.attn_gate_bank = nn.Parameter(torch.zeros(n_attn, num_heads, 12))
        self.ve_gate_bank   = nn.Parameter(torch.zeros(self.n_ve, num_heads, 12))

        # Learnable per-layer scalars packed into one parameter (matches train_gpt.py):
        #   indices [2i, 2i+1] = sa_lambdas[i] for layer i (QKV scale, O scale)
        #   index [2*L]   = smear_lambda
        #   index [2*L+1] = backout_lambda
        #   index [2*L+2] = skip_lambda
        L = num_layers
        self.scalars = nn.Parameter(torch.cat([
            *[torch.tensor([0.5, 1.0]) for _ in range(L)],
            torch.zeros(1),       # smear_lambda
            0.5 * torch.ones(1),  # backout_lambda
            -1.5 * torch.ones(1), # skip_lambda → σ(-1.5) ≈ 0.18
        ]))

        self.post_lambdas    = nn.Parameter(torch.ones(L, 2))
        self.resid_lambdas   = nn.Parameter(torch.full((L, 2), 1.1 ** 0.5))
        self.x0_lambdas      = nn.Parameter(torch.zeros(L))
        self.bigram_lambdas  = nn.Parameter(0.05 * torch.ones(L))

        # Attention modules (one shared regular + one shared paired, weights are per-layer)
        # Per-layer attention modules with their own weight matrices
        self.attn_modules = nn.ModuleList([
            CausalSelfAttention(model_dim, num_heads, head_dim,
                                qk_layernorm=qk_layernorm,
                                kv_tied=kv_tied, v_identity=v_identity, drop_o=drop_o,
                                qkv_conv=qkv_conv, qkv_conv_where=qkv_conv_where)
            for _ in attn_layers_topo
        ])
        self._attn_layer_idx = {layer: j for j, layer in enumerate(attn_layers_topo)}

        # Parallel conv branch (Conformer-style), one per attention layer, summed into the
        # residual alongside attention. conv_branch = kernel size (0 = off); hidden defaults
        # to model_dim. Independent of attention; LayerScale-gated (small at init, not dead).
        # Skip-layer conv: at layers where attention is SKIPPED (no token mixing of their
        # own), add a depthwise-conv local-mixing block — fills the gap cheaply rather than
        # duplicating attention. One ConvBranch per skip-destination layer (kernel = skip_conv).
        self.skip_conv = skip_conv
        if skip_conv:
            self.skip_convs = nn.ModuleDict({
                str(layer): ConvBranch(model_dim, skip_conv, model_dim) for layer in self.skip_dsts
            })
        # Optional learned global logit temperature (one scalar). RMS-norm removes per-token
        # magnitude, so the model can't scale its confidence; this restores that scalar DOF.
        # exp() keeps it positive; init -> 1 (no-op).
        self.logit_temp_mode = logit_temp
        if logit_temp == "global":
            self.logit_temp = nn.Parameter(torch.zeros(()))
        # Fixed unigram-log-prob bias on the logits: the model starts at the unigram
        # distribution (loss ~7.66) and learns the contextual residual. Precomputed offline
        # (legitimate for the local proxy; a real speedrun would track it online instead).
        if unigram_bias:
            lp = np.load("unigram_logprobs.npy")
            assert lp.shape[0] == self.vocab_size, f"bias len {lp.shape[0]} != vocab {self.vocab_size}"
            self.register_buffer("unigram_bias", torch.from_numpy(lp.astype(np.float32)))
        else:
            self.unigram_bias = None
        # Parallel conv tower (embeddings -> conv stack -> hidden), combined with the
        # transformer per conv_tower_mode: 'shared' adds hidden states (one head),
        # 'separate' is a true product-of-experts (own head), 'alone' skips the transformer.
        self.conv_tower_blocks = conv_tower
        self.conv_tower_mode = conv_tower_mode
        if conv_tower:
            self.conv_tower = ConvTower(model_dim, conv_tower, conv_tower_kernel,
                                        conv_tower_width or model_dim)
            self.conv_mix = nn.Parameter(torch.tensor(0.1))   # learnable track mix (shared/separate)
            if conv_tower_mode == "separate":
                self.conv_head = nn.Linear(model_dim, self.vocab_size, bias=False)
                with torch.no_grad():
                    self.conv_head.weight.uniform_(-bound, bound)
        self.conv_branch = conv_branch
        if conv_branch:
            cb_hidden = conv_branch_hidden or model_dim
            self.conv_branches = nn.ModuleList([
                ConvBranch(model_dim, conv_branch, cb_hidden) for _ in attn_layers_topo
            ])

        # Per-layer MLP modules
        self.mlp_modules = nn.ModuleList([
            make_mlp(model_dim, act_name, act_init, act_slope, act_shift, act_curv)
            for _ in range(L)
        ])

        # Which layers use paired-head attention (empty = disabled for speed)
        self.paired_head_layers = set(topo["paired_layers"]) if paired_heads else set()

        self.ce_chunk = 0  # set externally after construction; 0 = full logits
        self.softcap  = 23.0  # logit softcap scale (set externally); 0 = plain CE

        # RoPE — only allocate paired Yarn if paired heads are enabled
        self.yarn        = Yarn(head_dim, max_seq_len, device, paired=False)
        self.yarn_paired = Yarn(head_dim, max_seq_len, device, paired=True) if paired_heads else None

    def forward(self, input_ids, targets, bigram_ids):
        B, T = input_ids.shape
        L = self.num_layers
        H, D = self.num_heads, self.head_dim

        # Unpack scalars
        sa_lambdas_all    = self.scalars[:2 * L].view(L, 2)
        smear_lambda      = self.scalars[2 * L]
        backout_lambda    = self.scalars[2 * L + 1]
        skip_lambda       = self.scalars[2 * L + 2]
        rl_attn = self.resid_lambdas[:, 0].bfloat16().unbind(0)
        rl_mlp  = self.resid_lambdas[:, 1].bfloat16().unbind(0)
        pl_attn = self.post_lambdas[:, 0].bfloat16().unbind(0)
        pl_mlp  = self.post_lambdas[:, 1].bfloat16().unbind(0)

        # Embeddings + smear
        x_emb     = self.embed(input_ids)                  # (B, T, model_dim)
        # conv-only baseline: skip the transformer entirely, predict from the conv tower.
        if self.conv_tower_blocks and self.conv_tower_mode == "alone":
            xc = self.conv_tower(x_emb)
            return self._head_ce(xc.reshape(-1, xc.size(-1)), targets.reshape(-1))
        x0_bigram = self.bigram_embed(bigram_ids)             # (B, T, model_dim)
        smear_out = smear_lambda * torch.sigmoid(
            self.smear_gate(x_emb[:, 1:, :12])            # (B, T-1, 1)
        )
        x = torch.cat([
            x_emb[:, :1, :],
            x_emb[:, 1:, :] + smear_out * x_emb[:, :-1, :]
        ], dim=1)
        x = x0 = norm(x)

        # Layer-0 bigram injection (before the loop)
        x = x + x0_bigram * self.bigram_lambdas[0]

        # Precompute per-layer x0/bigram injections.
        x0_inject = [x0 * self.x0_lambdas[0]] + [
            x0 * self.x0_lambdas[i] + x0_bigram * self.bigram_lambdas[i]
            for i in range(1, L)
        ]

        # Skip gate (applied at layer 6 in place of attention)
        skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(
            self.skip_gate(x0[:, :, :12])   # (B, T, 1)
        )

        # Value embeddings: (n_ve, vocab, model_dim) → index by input_ids → (n_ve, B, T, D)
        ve_all = self.value_embeds.view(self.n_ve, self.vocab_size, self.model_dim)[:, input_ids]

        x_backout = None
        skip_store = {}   # src layer -> saved activation, for forward skips

        for i in range(L):
            paired     = (i in self.paired_head_layers)
            key_off    = (i in self.key_offset_layers)
            sa_lam     = sa_lambdas_all[i].bfloat16()

            # Value embedding for this layer (if any)
            ve_i    = ve_all[self._ve_gate_idx[i]] if i in self.ve_layers_set else None
            ve_gw   = self.ve_gate_bank[self._ve_gate_idx[i]] if i in self.ve_layers_set else None
            attn_gw = self.attn_gate_bank[self._attn_gate_idx[i]] if i not in self.skip_dsts else None
            yarn    = self.yarn_paired if paired else self.yarn

            if i in self.skip_dsts:
                # No attention; inject the saved skip-source activation in its place
                x = x + skip_gate_out * skip_store[self.skips[i]]
                if self.skip_conv:
                    # Fill the no-token-mixing gap with cheap local convolution.
                    x = x + self.skip_convs[str(i)](norm(x))
            else:
                attn_mod  = self.attn_modules[self._attn_layer_idx[i]]
                # Backout: once frozen, later attention layers read the frozen state
                attn_in   = x_backout if x_backout is not None else x
                n_in      = norm(attn_in)
                attn_out  = attn_mod(
                    n_in, yarn, sa_lam, paired,
                    attn_gate_w=attn_gw, ve=ve_i, ve_gate_w=ve_gw, key_offset=key_off
                )
                x = rl_attn[i] * x + pl_attn[i] * attn_out + x0_inject[i]
                if self.conv_branch:
                    # Parallel local-mixing branch on the SAME normed input as attention,
                    # added independently to the residual (LayerScale-gated, small at init).
                    x = x + self.conv_branches[self._attn_layer_idx[i]](n_in)

            mlp_out = self.mlp_modules[i](norm(x))
            x = rl_mlp[i] * x + pl_mlp[i] * mlp_out

            if i in self.skip_srcs:
                skip_store[i] = x
            # Freeze the backout state (used by later attention if mode != 'none')
            if i == self.backout_src and self.backout_mode != "none":
                x_backout = x

        # Backout final op: subtract the frozen state (only in 'freeze_subtract' mode)
        if self.backout_mode == "freeze_subtract":
            x = x - backout_lambda * x_backout
        x = norm(x)

        # Combine the conv tower with the transformer (if enabled): 'shared' adds the
        # conv hidden state (one head); 'separate' is a product of experts via its own head.
        t_flat = targets.reshape(-1)
        if self.conv_tower_blocks and self.conv_tower_mode == "shared":
            # Re-normalize after adding the tower so the head input is scale-matched to the
            # baseline. Without this, the un-normed sum lets the model inflate logit magnitude
            # and game the fixed softcap (an artifact, not a real gain).
            x = norm(x + self.conv_mix * self.conv_tower(x_emb))
        if self.conv_tower_blocks and self.conv_tower_mode == "separate":
            xc = self.conv_tower(x_emb)
            return self._ce_two_head(x.reshape(-1, x.size(-1)), xc.reshape(-1, xc.size(-1)), t_flat)
        return self._head_ce(x.reshape(-1, x.size(-1)), t_flat)

    def _logit_scale(self, x_flat):
        # None = off; a positive scalar multiplier on the logits when 'global'.
        if self.logit_temp_mode == "global":
            return torch.exp(self.logit_temp)
        return None

    def _head_logits(self, x):
        # Base (tied or untied) logits, plus the rank-r head offset when head_rank > 0.
        logits = mm8(x, self.lm_head.weight)
        if self.head_rank and self.head_rank > 0:
            logits = logits + (x @ self.head_lr_down.t()) @ self.head_lr_up.t()
        return logits

    def _head_ce(self, x_flat, t_flat):
        # Softcapped cross-entropy through the (shared) lm_head; chunked to avoid the full
        # (B*T, vocab) logit tensor (~1.65 GB at batch 16).
        s = self._logit_scale(x_flat)
        if self.ce_chunk and self.ce_chunk > 0:
            return self._chunked_softcap_ce(x_flat, t_flat, self.ce_chunk, s)
        logits = self._head_logits(x_flat)
        if s is not None:
            logits = s * logits
        if self.unigram_bias is not None:
            logits = logits + self.unigram_bias
        if self.softcap and self.softcap > 0:
            logits = self.softcap * torch.sigmoid((logits + 5) / 7.5)
        return F.cross_entropy(logits, t_flat)

    def _ce_two_head(self, xt_flat, xc_flat, t_flat):
        # Product of experts: logits = lm_head(h_T) + mix * conv_head(h_C), softcapped, chunked.
        chunk = self.ce_chunk if (self.ce_chunk and self.ce_chunk > 0) else xt_flat.size(0)
        n = t_flat.size(0)
        total = xt_flat.new_zeros(())
        def _chunk_loss(xt_c, xc_c, t_c):
            logits = mm8(xt_c, self.lm_head.weight) + self.conv_mix * mm8(xc_c, self.conv_head.weight)
            if self.softcap and self.softcap > 0:
                logits = self.softcap * torch.sigmoid((logits + 5) / 7.5)
            return F.cross_entropy(logits, t_c, reduction="sum")
        for i in range(0, n, chunk):
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_loss, xt_flat[i:i+chunk], xc_flat[i:i+chunk], t_flat[i:i+chunk],
                use_reentrant=False)
        return total / n

    def _chunked_softcap_ce(self, x_flat, targets_flat, chunk, s=None):
        per_pos = s is not None and s.dim() > 0 and s.shape[0] == x_flat.shape[0]
        def _chunk_loss(x_c, t_c, s_c):
            logits = self._head_logits(x_c)
            if s_c is not None:
                logits = s_c * logits
            if self.unigram_bias is not None:
                logits = logits + self.unigram_bias
            if self.softcap and self.softcap > 0:
                logits = self.softcap * torch.sigmoid((logits + 5) / 7.5)
            return F.cross_entropy(logits, t_c, reduction="sum")
        n = x_flat.size(0)
        total = x_flat.new_zeros(())
        for i in range(0, n, chunk):
            s_c = (s[i:i+chunk] if per_pos else s)
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_loss, x_flat[i:i+chunk], targets_flat[i:i+chunk], s_c,
                use_reentrant=False)
        return total / n


# -----------------------------------------------------------------------------
# Optimizer

class Muon(torch.optim.Optimizer):
    """NorMuon: momentum SGD + Polar-Express orthogonalization + neuron-wise variance normalization."""
    def __init__(self, params, lr=0.023, momentum=0.95, beta2=0.9, ortho="polar_express"):
        super().__init__(params, dict(lr=lr, momentum=momentum, beta2=beta2, ortho=ortho))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, beta2 = group["lr"], group["momentum"], group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)  # Nesterov
                g = orthogonalize(g, method=group["ortho"])
                if beta2 is not None:
                    if "v2" not in state:
                        shape = list(g.shape)
                        shape[-1 if g.size(-2) >= g.size(-1) else -2] = 1
                        state["v2"] = torch.zeros(shape, device=g.device, dtype=torch.float32)
                    g = normuon_variance_reduce(g, state["v2"], beta2)
                    p.add_(g, alpha=-lr)
                else:
                    p.add_(g, alpha=-lr * max(1.0, p.size(0) / p.size(1)) ** 0.5)


def build_optimizers(model, config):
    opt_name = config.get("optimizer", "normuon")

    # Partition parameters
    # mlp_gate_params: fc_gate weights in GatedMLP (the selection branch)
    # matrix_params:   all other 2D attention/MLP weights (value/projection branches)
    matrix_params, mlp_gate_params, scalar_params, gate_params = [], [], [], []
    ve_params, bigram_params, lmhead_params, other_params, trilu_params = [], [], [], [], []

    for name, p in model.named_parameters():
        if name == "embed.weight":
            # The token weight (shared with lm_head when tied) is deduped by named_parameters to
            # THIS name, so it must be optimized here -- previously it was skipped, which left the
            # tied embedding+head frozen at init. Untied: this is the standalone embedding.
            lmhead_params.append(p)
        elif "theta_" in name:
            trilu_params.append(p)
        elif p.ndim == 2 and "mlp_modules" in name and "fc_gate" in name:
            mlp_gate_params.append(p)  # gating/selection branch of GatedMLP
        elif p.ndim == 2 and any(k in name for k in ("attn_modules", "mlp_modules", "conv_branches", "conv_tower", "skip_convs")):
            matrix_params.append(p)    # value/projection branches + attention + conv-branch/tower/skip pointwise
        elif name in ("scalars", "post_lambdas", "resid_lambdas", "x0_lambdas", "bigram_lambdas"):
            scalar_params.append(p)
        elif any(k in name for k in ("smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank")):
            gate_params.append(p)
        elif name == "value_embeds":
            ve_params.append(p)
        elif "bigram_embed" in name:
            bigram_params.append(p)
        elif name == "lm_head.weight":
            lmhead_params.append(p)
        else:
            other_params.append(p)

    if opt_name in ("normuon", "muon", "hybrid", "hybrid-muon"):
        # ── NorMuon / Muon path ──────────────────────────────────────────────
        # Matrix weights → Muon/NorMuon. Everything else → AdamW with
        # per-group lr_muls calibrated to train_gpt.py's param_table
        # (where adam_lr ≈ 0.008 is the sub-LR for non-matrix params).
        alr = config["adam_lr"]
        awd = config["adam_wd"]

        def ag(params, lr_mul=1.0, betas=(0.9, 0.95), wd_mul=1.0):
            if params:
                return {"params": params, "base_lr": alr * lr_mul, "lr": alr * lr_mul,
                        "betas": betas, "weight_decay": awd * wd_mul}

        adam_groups = list(filter(None, [
            ag(scalar_params,  lr_mul=5.0,  betas=(0.9, 0.99), wd_mul=0.0),
            ag(gate_params,    lr_mul=0.05, betas=(0.9, 0.99), wd_mul=0.0),
            ag(ve_params,      lr_mul=75.0, betas=(0.75, 0.95), wd_mul=5.0),
            ag(bigram_params,  lr_mul=75.0, betas=(0.75, 0.95), wd_mul=5.0),
            ag(lmhead_params,  lr_mul=1.0,  betas=(0.5, 0.95),  wd_mul=150.0),
            ag(other_params,   lr_mul=1.0,  betas=(0.9, 0.95)),
            ag(trilu_params,   lr_mul=0.1,  betas=(0.9, 0.95)),
        ]))

        # Hybrid mode: gate branch (fc_gate) → AdamW; value/proj + attention → Muon.
        # For non-hybrid modes, mlp_gate_params fold into matrix_params (Muon).
        is_hybrid = opt_name in ("hybrid", "hybrid-muon")
        if is_hybrid:
            # Gate matrices join the Adam groups at base adam_lr (no special lr_mul —
            # in AdamW mode these are the "body" weights, not the sub-LR scalars)
            if mlp_gate_params:
                adam_groups.append({"params": mlp_gate_params,
                                    "base_lr": alr, "lr": alr,
                                    "betas": (0.9, 0.95), "weight_decay": awd})
        else:
            matrix_params = matrix_params + mlp_gate_params  # all matrices → Muon

        mlr   = config.get("muon_lr", 0.023)
        beta2 = config.get("muon_beta2", 0.9) if opt_name in ("normuon", "hybrid") else None
        muon_opt = Muon(
            [{"params": matrix_params, "base_lr": mlr, "lr": mlr}],
            lr=mlr, beta2=beta2, ortho=config.get("muon_ortho", "polar_express"),
        )
        return [muon_opt, torch.optim.AdamW(adam_groups, lr=alr,
                                             betas=(0.9, 0.95), weight_decay=awd)]

    else:
        # ── Pure AdamW path ──────────────────────────────────────────────────
        # All parameters get Adam. The lr_muls from the NorMuon regime
        # (75× for embeddings, 5× for scalars) are designed to compensate
        # for Adam's sub-LR role there and are wrong here. Use uniform LR.
        alr = config["adam_lr"]
        awd = config["adam_wd"]
        all_params = (matrix_params + scalar_params + gate_params +
                      ve_params + bigram_params + lmhead_params + other_params)
        groups = []
        if all_params:
            groups.append({"params": all_params, "base_lr": alr, "lr": alr,
                           "betas": (0.9, 0.95), "weight_decay": awd})
        if trilu_params:
            groups.append({"params": trilu_params, "base_lr": alr * 0.1, "lr": alr * 0.1,
                           "betas": (0.9, 0.95), "weight_decay": awd})
        return [torch.optim.AdamW(groups, lr=alr, betas=(0.9, 0.95), weight_decay=awd)]


# -----------------------------------------------------------------------------
# Bigram hash (from train_gpt.py)

BIGRAM_VOCAB_SIZE = 50304 * 5  # matches train_gpt.py args.bigram_vocab_size

def get_bigram_ids(input_ids):
    """Compute bigram hash for each position. input_ids: (B, T) int64 → (B, T) int64."""
    rand1, rand2 = 36313, 27191
    mod = BIGRAM_VOCAB_SIZE - 1
    x = input_ids.to(torch.int64)
    out = torch.full_like(x, mod)            # position 0 → reserved index
    # positions 1..T-1: XOR hash of (curr, prev) tokens
    out[:, 1:] = (rand1 * x[:, 1:] ^ rand2 * x[:, :-1]) % mod
    return out


# -----------------------------------------------------------------------------
# Data loading

def _load_data_shard(path):
    import numpy as np
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520
    assert header[1] == 1
    n = int(header[2])
    with open(path, "rb") as f:
        f.seek(256 * 4)
        tokens = np.fromfile(f, dtype=np.uint16, count=n)
    assert tokens.size == n
    return torch.from_numpy(tokens.astype(np.int64)).clone()


def load_data(data_dir="data/fineweb10B", train_cap=0, val_cap=0):
    data_dir = Path(data_dir)
    train_files = sorted(data_dir.glob("fineweb_train_*.bin"))
    val_files   = sorted(data_dir.glob("fineweb_val_*.bin"))
    if not train_files or not val_files:
        print(f"ERROR: no .bin files in {data_dir.resolve()}")
        print("Run: python data/cached_fineweb10B.py 1")
        sys.exit(1)
    train = _load_data_shard(train_files[0])
    val   = _load_data_shard(val_files[0])
    if train_cap: train = train[:train_cap]
    if val_cap:   val   = val[:val_cap]
    return train, val


def sample_batch(data, batch_size, seq_len, device, gen):
    high = len(data) - seq_len - 1
    ix = torch.randint(0, high, (batch_size,), generator=gen).tolist()
    buf = torch.stack([data[i:i + seq_len + 1] for i in ix]).to(device, non_blocking=True)
    return buf[:, :-1], buf[:, 1:]   # (B, T) inputs, (B, T) targets


# -----------------------------------------------------------------------------
# LR schedule
# Trapezoidal / WSD shape matching the speedrun (train_gpt.py get_lr): hold a constant LR,
# then LINEARLY cool to LR_FLOOR over the final LR_COOLDOWN_FRAC of training. We keep it
# budget-fraction based (rather than the record's step-based form) so iso-compute still works.
LR_COOLDOWN_FRAC = 0.60   # cooldown spans the last 60% of the budget (matches cooldown_frac=0.60)
LR_FLOOR = 0.15           # cooldown target, matches the record's 0.15x floor

def wsd_mult(frac):
    """Multiplier for budget-fraction `frac` in [0,1]: constant 1.0, then linear decay to LR_FLOOR."""
    cd_start = 1.0 - LR_COOLDOWN_FRAC
    if frac < cd_start:
        return 1.0
    t = (frac - cd_start) / max(1.0 - cd_start, 1e-9)
    return 1.0 * (1.0 - t) + LR_FLOOR * t

def lr_mult(step, total_steps, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return wsd_mult((step - warmup) / max(total_steps - warmup, 1))


# -----------------------------------------------------------------------------
# Training

@contextlib.contextmanager
def _nullctx():
    yield


@torch.no_grad()
def evaluate(model, val_data, batch_size, seq_len, device, gen, num_batches=20):
    model.eval()
    losses = []
    amp = (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else _nullctx
    for _ in range(num_batches):
        x, y = sample_batch(val_data, batch_size, seq_len, device, gen)
        bigram = get_bigram_ids(x)
        torch.compiler.cudagraph_mark_step_begin()   # no-op unless reduce-overhead/CUDA graphs
        with amp():
            loss = model(x, y, bigram)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(config, train_data, val_data, seed, device):
    torch.manual_seed(seed)
    gen_tr  = torch.Generator().manual_seed(seed)
    gen_val = torch.Generator().manual_seed(seed + 10_000)

    model = GPT(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        head_dim=config["head_dim"],
        model_dim=config["model_dim"],
        max_seq_len=config["seq_len"],
        device=device,
        act_name=config["activation"],
        act_init=config.get("act_init", "gelu_minimax"),
        act_slope=config.get("act_slope", 0.0),
        act_shift=config.get("act_shift", 0.0),
        act_curv=config.get("act_curv", 1.0),
        qk_layernorm=config.get("qk_layernorm", False),
        paired_heads=config.get("paired_heads", False),
        kv_tied=config.get("kv_tied", False),
        v_identity=config.get("v_identity", False),
        drop_o=config.get("drop_o", False),
        qkv_conv=config.get("qkv_conv", 0),
        qkv_conv_where=config.get("qkv_conv_where", "qkv"),
        conv_branch=config.get("conv_branch", 0),
        conv_branch_hidden=config.get("conv_branch_hidden", 0),
        conv_tower=config.get("conv_tower", 0),
        conv_tower_kernel=config.get("conv_tower_kernel", 4),
        conv_tower_width=config.get("conv_tower_width", 0),
        conv_tower_mode=config.get("conv_tower_mode", "off"),
        skip_conv=config.get("skip_conv", 0),
        logit_temp=config.get("logit_temp", "none"),
        unigram_bias=config.get("unigram_bias", False),
        head_rank=config.get("head_rank", 0),
        topology=config.get("topology"),
    ).to(device)
    model.ce_chunk = config.get("ce_chunk", 0)
    model.softcap  = config.get("softcap", 23.0)

    cmodel = torch.compile(model) if config.get("compile") else model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}  act={config['activation']}  opt={config.get('optimizer','normuon')}"
          f"{'  compile' if config.get('compile') else ''}")

    optimizers = build_optimizers(model, config)
    total_steps = config["total_steps"]
    warmup      = config["warmup"]
    max_seconds = config.get("max_seconds")  # wall-clock budget; LR decays on time
    log = {"val_loss": [], "wallclock": [], "config": config, "seed": seed}

    t0 = time.time()
    for step in range(total_steps):
        elapsed = time.time() - t0
        if max_seconds:
            # Warmup by step, then trapezoidal (WSD) decay over the TIME budget so the LR
            # schedule completes regardless of how many steps fit in the budget.
            if step < warmup:
                mult = (step + 1) / warmup
            else:
                mult = wsd_mult(min(1.0, elapsed / max_seconds))
        else:
            mult = lr_mult(step, total_steps, warmup)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult

        x, y = sample_batch(train_data, config["batch_size"], config["seq_len"], device, gen_tr)
        bigram = get_bigram_ids(x)
        amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else _nullctx()
        torch.compiler.cudagraph_mark_step_begin()   # no-op unless reduce-overhead/CUDA graphs
        with amp:
            loss = cmodel(x, y, bigram)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        if step % config["eval_every"] == 0 or step == total_steps - 1:
            # Read the train loss BEFORE evaluate() runs the model again — under CUDA
            # graphs the loss is a reused graph buffer that eval would overwrite.
            train_loss_val = loss.item()
            val_loss = evaluate(cmodel, val_data, config["batch_size"], config["seq_len"], device, gen_val)
            elapsed  = time.time() - t0
            log["val_loss"].append([step, val_loss])
            log["wallclock"].append([step, elapsed])
            print(f"  step {step:5d}  lr={optimizers[0].param_groups[0]['lr']:.5f}"
                  f"  train={train_loss_val:.4f}  val={val_loss:.4f}  t={elapsed:.1f}s")
            # Opt-in early stop: once the target is crossed, the time-to-target is
            # already recorded — no need to keep training (used by the BO search).
            if config.get("stop_at_target") and config.get("target_loss") \
                    and val_loss <= config["target_loss"]:
                print(f"  -> reached target {config['target_loss']} at step {step}, stopping early")
                break
            # Wall-clock budget: stop once the time budget is exhausted (iso-compute
            # architecture comparison — objective is the loss reached in the budget).
            if max_seconds and elapsed >= max_seconds:
                print(f"  -> reached time budget {max_seconds}s at step {step}, stopping")
                break

    vals = [v for _, v in log["val_loss"]]
    k = min(5, len(vals))
    log["final_loss"]          = vals[-1]
    log["final_loss_smoothed"] = sum(vals[-k:]) / k
    log["auc"]                 = sum(vals) / len(vals)
    target = config.get("target_loss")
    log["steps_to_target"] = next((s for s, v in log["val_loss"] if v <= target), None) if target else None
    print(f"  -> final={log['final_loss']:.4f}  smoothed={log['final_loss_smoothed']:.4f}"
          f"  auc={log['auc']:.4f}  steps_to_target={log['steps_to_target']}")
    return log


# -----------------------------------------------------------------------------
# Configs

CONFIGS = {
    # phase2.py requires num_layers=11 for the speedrun topology.
    # "tiny" uses small model_dim for fast sanity checks on a laptop.
    "tiny": dict(
        num_layers=11, model_dim=64, num_heads=4, head_dim=16,
        seq_len=256, batch_size=4,
        total_steps=100, warmup=10, eval_every=20,
        vocab_size=50304,
        adam_lr=8e-3, adam_wd=0.005,
    ),
    "default": dict(
        num_layers=11, model_dim=384, num_heads=6, head_dim=64,
        seq_len=1024, batch_size=8,
        total_steps=1000, warmup=80, eval_every=50,
        vocab_size=50304,
        adam_lr=8e-3, adam_wd=0.005,
    ),
    # Matches train_gpt.py's architecture exactly (124M params)
    "speedrun": dict(
        num_layers=11, model_dim=768, num_heads=6, head_dim=128,
        seq_len=1024, batch_size=16,
        total_steps=4000, warmup=256, eval_every=100,
        vocab_size=50304,
        adam_lr=8e-3, adam_wd=0.005,
    ),
}

ALL_ACTIVATIONS = ["relu2", "gelu", "trilu_sym", "trilu_asym", "swiglu", "geglu", "triglu",
                   "xabsx", "xglu", "reglu", "bilinear", "iqu", "sniqu", "selu"]
DEFAULT_ACTIVATIONS = ["relu2", "sniqu", "xglu", "reglu"]


# -----------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="default")
    ap.add_argument("--tiny",       action="store_true")
    ap.add_argument("--seeds",      type=int,   default=3)
    ap.add_argument("--seed-start", type=int,   default=0)
    ap.add_argument("--steps",      type=int,   default=None)
    ap.add_argument("--batch-size", type=int,   default=None)
    ap.add_argument("--activations",type=str,   default=",".join(DEFAULT_ACTIVATIONS))
    ap.add_argument("--init",       type=str,   default="gelu_minimax", choices=list(_INIT_VALUES))
    ap.add_argument("--act-slope",  type=float, default=0.0)
    ap.add_argument("--act-shift",  type=float, default=0.0)
    ap.add_argument("--act-curv",   type=float, default=1.0)
    ap.add_argument("--optimizer",  type=str,   default="normuon",
                    choices=["normuon", "muon", "adamw", "hybrid", "hybrid-muon"],
                    help="hybrid/hybrid-muon: gate branch (fc_gate) → AdamW, "
                         "value/proj + attention → NorMuon/Muon. Tests whether "
                         "splitting the optimizer by parameter role captures the "
                         "best of gating (AdamW) and orthogonalization (Muon).")
    ap.add_argument("--muon-lr",    type=float, default=0.023)
    ap.add_argument("--muon-beta2", type=float, default=0.9)
    ap.add_argument("--muon-ortho", type=str,   default="polar_express",
                    choices=["polar_express", "newton_schulz"])
    ap.add_argument("--adam-lr",    type=float, default=None)
    ap.add_argument("--kv-tied", action="store_true",
                    help="K=V: share the key projection for values too. Attention output is a "
                         "weighted sum of key vectors. Saves one projection matrix; tests whether "
                         "coupling K and V through one matrix helps Muon conditioning.")
    ap.add_argument("--v-identity", action="store_true",
                    help="V=identity: skip the V projection entirely, use the normalised input "
                         "x directly as values. Attention becomes pure routing over the residual "
                         "stream. Saves one projection matrix.")
    ap.add_argument("--drop-o", action="store_true",
                    help="Drop the output projection O: concatenated head outputs go straight to "
                         "the residual stream. Tests whether the MLP can absorb cross-head mixing. "
                         "Each head is confined to write into its own coordinate block. Requires "
                         "hdim == model_dim.")
    ap.add_argument("--softcap-scale", type=float, default=23.0,
                    help="logit softcap scale: logits = s*sigmoid((logits+5)/7.5). "
                         "23 = speedrun default. Larger s → less compression (closer "
                         "to plain CE, sharper achievable softmax); smaller s → more "
                         "compression (flatter, more regularised).")
    ap.add_argument("--plain-ce", action="store_true",
                    help="bypass the softcap entirely (plain cross-entropy). Equivalent "
                         "to --softcap-scale 0. Reaches a meaningful loss regime at low "
                         "compute, at the cost of fidelity to the record's eval path.")
    ap.add_argument("--ce-chunk", type=int, default=0,
                    help="chunk size for cross-entropy (tokens). 0 = full logits (~1.65 GB at "
                         "batch 16). Use 4096 to avoid OOM on 16 GB GPUs with paired heads.")
    ap.add_argument("--paired-heads", action="store_true",
                    help="enable paired-head attention (layers 0,2,5,9 attend over doubled "
                         "sequence length). Matches train_gpt.py exactly but ~1.5x slower. "
                         "Off by default for local experiments; the activation ranking is "
                         "unlikely to change since paired heads affect attention, not the MLP.")
    ap.add_argument("--qkv-conv", type=int, default=0,
                    help="kernel size of a depthwise causal conv on q/k/v, applied before "
                         "QK-norm and RoPE (0 = off; 3-4 typical). Conformer/Mamba-style local "
                         "mixing on top of attention; identity-init no-op. NOTE: tiny FLOPs but a "
                         "non-fused memory-bound op — judge it on loss-at-budget here, not wallclock "
                         "(the record is wallclock-bound and may not keep a per-step win).")
    ap.add_argument("--qkv-conv-where", type=str, default="qkv", choices=["qkv", "v", "qk"],
                    help="which projections the conv is applied to: 'qkv' all (default), "
                         "'v' value-only (RoPE-free pathway, no addressing blur — the cleanest "
                         "content-smoothing test), 'qk' query/key-only.")
    ap.add_argument("--conv-branch", type=int, default=0,
                    help="kernel size of a PARALLEL conv branch (Conformer-style: pointwise -> "
                         "depthwise causal conv -> relu2 -> pointwise) run alongside attention and "
                         "summed into the residual (0 = off). Independent of attention — learns "
                         "local n-gram-like structure on its own path. No-op at init. Bigger/costlier "
                         "than --qkv-conv; judge on loss-at-budget.")
    ap.add_argument("--conv-branch-hidden", type=int, default=0,
                    help="inner width of the parallel conv branch (0 = model_dim).")
    ap.add_argument("--conv-tower", type=int, default=0,
                    help="number of residual conv blocks in a parallel conv tower (0 = off). "
                         "A transformer-free local-only track combined with the transformer.")
    ap.add_argument("--conv-tower-kernel", type=int, default=4, help="conv tower kernel size.")
    ap.add_argument("--conv-tower-width", type=int, default=0, help="conv tower inner width (0 = model_dim).")
    ap.add_argument("--conv-tower-mode", type=str, default="off",
                    choices=["off", "shared", "separate", "alone"],
                    help="how the conv tower combines with the transformer: 'shared' adds hidden "
                         "states through one lm_head; 'separate' is a product of experts with its "
                         "own head; 'alone' skips the transformer (conv-only baseline).")
    ap.add_argument("--skip-conv", type=int, default=0,
                    help="kernel size of a depthwise conv added at attention-SKIP layers (0 = off). "
                         "Those layers have no token mixing; the conv fills that gap cheaply rather "
                         "than duplicating attention.")
    ap.add_argument("--logit-temp", type=str, default="none", choices=["none", "global"],
                    help="learned global logit temperature (one scalar) before the head, "
                         "restoring the confidence-magnitude DOF that RMS-norm removes. Init = no-op.")
    ap.add_argument("--unigram-bias", action="store_true",
                    help="add a fixed log-unigram bias to the logits, so the model starts at the "
                         "unigram distribution (~7.66) and learns the contextual residual. "
                         "Loads unigram_logprobs.npy.")
    ap.add_argument("--head-rank", type=int, default=0,
                    help="output-head parameterization: 0 = tied to embedding (default); r>0 = tied "
                         "base + rank-r learned offset (cheap partial untie, starts tied); "
                         "-1 = full untie (separate lm_head, ~38M extra params).")
    ap.add_argument("--qk-layernorm", action="store_true",
                    help="replace the always-on RMSNorm QK-norm with LayerNorm(bias=False), "
                         "which also removes the mean (zero-centered). Tests whether the DC "
                         "direction matters beyond scale normalization for Muon conditioning.")
    # Generative topology (architecture search). When any is given, phase2 builds
    # a topology via topology.build_topology() instead of the legacy default.
    ap.add_argument("--num-layers",     type=int,   default=None,
                    help="override depth (number of transformer layers).")
    ap.add_argument("--num-skips",      type=int,   default=None,
                    help="number of forward skip connections (each dst drops attention).")
    ap.add_argument("--skip-src-frac",  type=float, default=None,
                    help="first skip source as a fraction of depth (legacy 0.30).")
    ap.add_argument("--skip-span-frac", type=float, default=None,
                    help="skip span (dst-src) as a fraction of depth (legacy 0.30).")
    ap.add_argument("--backout-src-frac", type=float, default=None,
                    help="backout source layer as a fraction of depth (legacy 0.70).")
    ap.add_argument("--backout-mode",   type=str,   default=None,
                    choices=["none", "freeze_only", "freeze_subtract"],
                    help="backout behaviour (legacy freeze_subtract).")
    ap.add_argument("--model-dim",  type=int,   default=None,
                    help="override model/residual width. num_heads*head_dim must equal it.")
    ap.add_argument("--num-heads",  type=int,   default=None,
                    help="override attention head count (head_dim auto-adjusts only if also "
                         "given). num_heads*head_dim must equal model_dim.")
    ap.add_argument("--head-dim",   type=int,   default=None,
                    help="override per-head dimension. Use with --num-heads to sweep head "
                         "count at fixed model_dim (e.g. 4x192, 6x128, 8x96, 12x64).")
    ap.add_argument("--data-dir",   type=str,   default="data/fineweb10B")
    ap.add_argument("--out",        type=str,   default="experiments/arch_search/results_phase2.json")
    ap.add_argument("--device",     type=str,   default=None)
    ap.add_argument("--compile",    action="store_true")
    ap.add_argument("--target-loss",type=float, default=None)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="wall-clock training budget in seconds. Cosine LR decays over "
                         "the budget so the schedule completes; training stops when the "
                         "budget is exhausted. Use a large --steps so time is the binding "
                         "constraint. For iso-compute architecture comparison.")
    ap.add_argument("--stop-at-target", action="store_true",
                    help="stop training as soon as val loss crosses --target-loss "
                         "(time-to-target is already recorded). Used by the BO search "
                         "to avoid wasting compute after the objective is measured.")
    args = ap.parse_args()

    if args.tiny:
        args.config = "tiny"

    cfg = dict(CONFIGS[args.config])
    if args.steps:      cfg["total_steps"] = args.steps
    if args.batch_size: cfg["batch_size"]  = args.batch_size
    if args.adam_lr:    cfg["adam_lr"]     = args.adam_lr
    if args.model_dim:  cfg["model_dim"]   = args.model_dim
    if args.num_heads:  cfg["num_heads"]   = args.num_heads
    if args.head_dim:   cfg["head_dim"]    = args.head_dim

    # Generative topology: build one when any topology arg is provided.
    _topo_args = (args.num_layers, args.num_skips, args.skip_src_frac,
                  args.skip_span_frac, args.backout_src_frac, args.backout_mode)
    if any(a is not None for a in _topo_args):
        if args.num_layers:
            cfg["num_layers"] = args.num_layers
        tparams = {}
        if args.num_skips      is not None: tparams["num_skips"]       = args.num_skips
        if args.skip_src_frac  is not None: tparams["skip_src_frac"]   = args.skip_src_frac
        if args.skip_span_frac is not None: tparams["skip_span_frac"]  = args.skip_span_frac
        if args.backout_src_frac is not None: tparams["backout_src_frac"] = args.backout_src_frac
        if args.backout_mode   is not None: tparams["backout_mode"]    = args.backout_mode
        cfg["topology"] = build_topology(cfg["num_layers"], tparams)
    assert cfg["num_heads"] * cfg["head_dim"] == cfg["model_dim"], \
        f"num_heads*head_dim ({cfg['num_heads']}*{cfg['head_dim']}) must equal " \
        f"model_dim ({cfg['model_dim']}); head_dim must be divisible by 4 for RoPE"
    assert cfg["head_dim"] % 4 == 0, "head_dim must be divisible by 4 (RoPE frequency bands)"
    cfg.update(dict(
        optimizer=args.optimizer,
        muon_lr=args.muon_lr, muon_beta2=args.muon_beta2, muon_ortho=args.muon_ortho,
        act_slope=args.act_slope, act_shift=args.act_shift, act_curv=args.act_curv,
        act_init=args.init, compile=args.compile,
        target_loss=args.target_loss,
        stop_at_target=args.stop_at_target,
        qk_layernorm=args.qk_layernorm,
        paired_heads=args.paired_heads,
        kv_tied=args.kv_tied,
        v_identity=args.v_identity,
        drop_o=args.drop_o,
        qkv_conv=args.qkv_conv,
        qkv_conv_where=args.qkv_conv_where,
        conv_branch=args.conv_branch,
        conv_branch_hidden=args.conv_branch_hidden,
        conv_tower=args.conv_tower,
        conv_tower_kernel=args.conv_tower_kernel,
        conv_tower_width=args.conv_tower_width,
        conv_tower_mode=args.conv_tower_mode,
        skip_conv=args.skip_conv,
        logit_temp=args.logit_temp,
        unigram_bias=args.unigram_bias,
        head_rank=args.head_rank,
        ce_chunk=args.ce_chunk,
        softcap=(0.0 if args.plain_ce else args.softcap_scale),
        max_seconds=args.max_seconds,
    ))

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}  Config: {args.config}  steps={cfg['total_steps']}  seeds={args.seeds}")

    activations = args.activations.split(",")
    for a in activations:
        if a not in ALL_ACTIVATIONS:
            print(f"Unknown activation: {a}")
            sys.exit(1)

    train_cap = 5_000_000 if args.config == "tiny" else 0
    val_cap   = 1_000_000 if args.config == "tiny" else 0
    train_data, val_data = load_data(args.data_dir, train_cap, val_cap)

    # Load existing results for incremental runs (seeds-outer loops accumulate here)
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for act in activations:
        run_cfg = dict(cfg, activation=act)
        key = act
        if args.act_slope or args.act_shift:
            key += f"_s{args.act_slope:g}t{args.act_shift:g}"
        if args.act_curv != 1.0:
            key += f"_c{args.act_curv:g}"
        if args.optimizer != "normuon":
            key += f"_{args.optimizer}"
        if args.qkv_conv:
            key += f"_qc{args.qkv_conv}{args.qkv_conv_where if args.qkv_conv_where != 'qkv' else ''}"
        if args.conv_branch:
            key += f"_cb{args.conv_branch}"
        if args.conv_tower:
            key += f"_ct{args.conv_tower}{args.conv_tower_mode[0]}"
        if args.skip_conv:
            key += f"_sc{args.skip_conv}"
        if args.logit_temp != "none":
            key += f"_lt{args.logit_temp[0]}"
        if args.unigram_bias:
            key += "_ub"
        if args.head_rank:
            key += f"_hr{args.head_rank}"
        if args.qk_layernorm:
            key += "_qkln"
        if args.kv_tied:
            key += "_kv"
        if args.v_identity:
            key += "_vI"
        if args.drop_o:
            key += "_noO"
        if args.num_heads or args.head_dim:
            key += f"_h{cfg['num_heads']}x{cfg['head_dim']}"
        if key not in all_results:
            all_results[key] = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            print(f"\n=== act={act}  seed={seed} ===")
            log = train_one(run_cfg, train_data, val_data, seed, device)
            all_results[key].append(log)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                # topology dicts contain sets (paired/key-offset layers) → sorted lists
                json.dump(all_results, f, indent=2,
                          default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
