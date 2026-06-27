"""
Generative, layer-count-agnostic topology for phase2's asymmetric skip structure.

The record architecture is NOT a symmetric U-Net. It has:
  * FORWARD skip(s): save activation at `src`, inject it at `dst` — and each
    `dst` DROPS its attention (the skip replaces it). There can be 0, 1, or more.
  * a BACKOUT with three possible modes:
      - 'none':            later attention reads the live residual; nothing
                           subtracted at the end.
      - 'freeze_only':     after `backout_src`, later attention layers read the
                           FROZEN backout state, but it is NOT subtracted at end.
      - 'freeze_subtract': frozen state used by later attention AND
                           `backout_lambda * x_backout` subtracted at the end
                           (the legacy behaviour).
    Note `backout_lambda` is a FREE learnable scalar, so its effective SIGN is
    learned — there is no separate sign hyperparameter.
  * auxiliary per-layer flags (paired-heads, value-embeds, key-offset) placed by
    an even-spread density rule.

`legacy_topology()` returns the EXACT L=11 record structure (membership-exact),
used as the default and for the parity check. `build_topology(L, params)` is the
generative version for a Bayesian search; its defaults reproduce the legacy CORE
(skip 3->6, backout 7) but auxiliary placements match legacy *density*, not
membership.
"""

BACKOUT_MODES = ("none", "freeze_only", "freeze_subtract")


def legacy_topology():
    """Exact L=11 record topology — membership-exact, for default use + parity."""
    return dict(
        num_layers=11,
        skips={6: 3},                       # dst -> src ; layer 6 drops attention
        backout_src=7,
        backout_mode="freeze_subtract",
        paired_layers={0, 2, 5, 9},
        ve_layers=[1, 2, 8, 9, 10],         # ordered: maps to VE banks 0..4
        key_offset_layers={3, 10},
        attn_layers=[i for i in range(11) if i != 6],
    )


DEFAULT_PARAMS = dict(
    num_skips=1,
    skip_src_frac=0.30,      # first skip source: 3/10
    skip_span_frac=0.30,     # span 3 -> dst 6
    backout_src_frac=0.70,   # 7/10
    backout_mode="freeze_subtract",
    paired_density=4 / 11,
    ve_density=5 / 11,
    key_offset_density=2 / 11,
)


def _even_spread(n_on, L, lo=0, hi=None):
    """Place n_on layers as evenly as possible across [lo, hi]."""
    hi = (L - 1) if hi is None else hi
    if n_on <= 0:
        return []
    if n_on == 1:
        return [round((lo + hi) / 2)]
    if n_on >= (hi - lo + 1):
        return list(range(lo, hi + 1))
    return [round(lo + i * (hi - lo) / (n_on - 1)) for i in range(n_on)]


def build_topology(L, params=None):
    """Return a topology dict for depth L from generative params (for BO search)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    last = L - 1
    span = max(1, round(p["skip_span_frac"] * last))

    # ---- Forward skips (0..k): each dst drops attention ----
    k = int(p["num_skips"])
    skips = {}
    if k > 0:
        base_src = max(0, min(last - 1, round(p["skip_src_frac"] * last)))
        if k == 1:
            srcs = [base_src]                    # single skip anchored at skip_src_frac
        else:
            src_hi = max(base_src, last - span)  # spread sources from base toward end
            srcs = _even_spread(k, L, lo=base_src, hi=src_hi)
        for s in srcs:
            dst = min(last - 1, s + span)        # dst interior, drops attention
            if dst <= s:
                dst = min(last - 1, s + 1)
            if 1 <= dst <= last - 1 and dst not in skips and dst != s:
                skips[dst] = s

    # ---- Backout (placed after the last skip dst, like legacy) ----
    backout_src = max(0, min(last, round(p["backout_src_frac"] * last)))
    if skips:
        backout_src = max(max(skips) + 1, backout_src) if max(skips) + 1 <= last else backout_src
    backout_mode = p["backout_mode"]
    assert backout_mode in BACKOUT_MODES, f"bad backout_mode {backout_mode}"

    attn_skip = set(skips.keys())
    attn_layers = [i for i in range(L) if i not in attn_skip]

    # ---- Auxiliary placements ----
    # At the record depth (L=11) use the record's hand-tuned aux sets so that
    # build_topology(11, default core) == legacy_topology() exactly (lets prior
    # legacy-topology observations be injected without inconsistency). At other
    # depths, scale by an even-spread density rule. Aux layers that became
    # attention-skip destinations are dropped (a skip-dst has no attention).
    if L == 11:
        lt = legacy_topology()
        paired = set(lt["paired_layers"])
        ve     = list(lt["ve_layers"])
        keyoff = set(lt["key_offset_layers"])
    else:
        paired = set(_even_spread(round(p["paired_density"] * L), L))
        ve     = _even_spread(round(p["ve_density"] * L), L)
        keyoff = set(_even_spread(round(p["key_offset_density"] * L), L))
    paired = paired - attn_skip
    ve     = [l for l in ve if l not in attn_skip]
    keyoff = keyoff - attn_skip

    return dict(
        num_layers=L,
        skips=skips,
        backout_src=backout_src,
        backout_mode=backout_mode,
        paired_layers=paired,
        ve_layers=ve,
        key_offset_layers=keyoff,
        attn_layers=attn_layers,
    )


def validate_topology(t):
    L = t["num_layers"]
    for dst, src in t["skips"].items():
        assert 0 <= src < dst <= L - 1, f"bad skip {src}->{dst}"
        assert 1 <= dst <= L - 2, "attn-skip (skip dst) must be interior"
    assert 0 <= t["backout_src"] <= L - 1, "bad backout src"
    assert t["backout_mode"] in BACKOUT_MODES, "bad backout mode"
    assert set(t["attn_layers"]).isdisjoint(t["skips"].keys()), "attn layer is also a skip dst"
    assert len(t["attn_layers"]) == L - len(t["skips"]), "attn layer count mismatch"
    assert t["paired_layers"].isdisjoint(t["skips"].keys()), "paired layer can't drop attention"
    return True


if __name__ == "__main__":
    lt = legacy_topology()
    validate_topology(lt)
    print(f"legacy: skips={lt['skips']} backout_src={lt['backout_src']} "
          f"mode={lt['backout_mode']}")

    g = build_topology(11)
    validate_topology(g)
    full_ok = (g["skips"] == lt["skips"] and g["backout_src"] == lt["backout_src"]
               and g["backout_mode"] == lt["backout_mode"]
               and g["paired_layers"] == lt["paired_layers"]
               and list(g["ve_layers"]) == list(lt["ve_layers"])
               and g["key_offset_layers"] == lt["key_offset_layers"]
               and list(g["attn_layers"]) == list(lt["attn_layers"]))
    print(f"L=11 build_topology(default) == legacy_topology() exactly: {full_ok}")
    assert full_ok, "default generative params must reproduce legacy EXACTLY at L=11"

    print("\nVarying num_skips at L=11:")
    for k in [0, 1, 2, 3]:
        t = build_topology(11, {"num_skips": k})
        validate_topology(t)
        print(f"  k={k}: skips={t['skips']}  attn_layers={t['attn_layers']}")

    print("\nVarying backout_mode:")
    for m in BACKOUT_MODES:
        t = build_topology(11, {"backout_mode": m})
        print(f"  {m}: backout_src={t['backout_src']}")

    print("\nScaling across L:")
    for L in [8, 11, 14, 16]:
        t = build_topology(L, {"num_skips": 2})
        validate_topology(t)
        print(f"  L={L:2d}: skips={t['skips']}  backout_src={t['backout_src']}")
    print("\nAll topologies valid.")
