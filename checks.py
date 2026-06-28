"""Self-verification for the harness (run on CPU, tiny model).

Guards against the bug classes in BUGLOG.md:
  C1  every trainable parameter is in the optimizer (catches the frozen-head bug)
  C2  every trainable parameter actually changes during training
plus a build+step smoke test across every supported configuration.
"""
import sys; sys.path.insert(0, '.')
import torch, harness as h

CFG_OPT = {'optimizer':'normuon','muon_lr':0.02,'adam_lr':0.008,'adam_wd':0.0,'muon_wd':0.0,'adam_betas':(0.9,0.95)}

ARMS = {
    'baseline (relu2, tied)': dict(act_name='relu2'),
    'sniqu':                  dict(act_name='sniqu'),
    'reglu (gated)':          dict(act_name='reglu'),
    'logit-temp global':      dict(logit_temp='global'),
    'unigram-bias':           dict(unigram_bias=True),
    'head-rank 64 (low-rank)':dict(head_rank=64),
    'head-rank -1 (untie)':   dict(head_rank=-1),
}

def build(**kw):
    torch.manual_seed(0)
    m = h.GPT(vocab_size=50304, num_layers=11, num_heads=4, head_dim=16, model_dim=64,
              max_seq_len=64, device='cpu', act_name=kw.pop('act_name','relu2'), **kw).to('cpu')
    m.ce_chunk = 0
    return m

def check_arm(name, kw):
    m = build(**kw)
    opts = h.build_optimizers(m, CFG_OPT)
    opt_ids = {id(p) for o in opts for g in o.param_groups for p in g['params']}
    missing = [n for n,p in m.named_parameters() if p.requires_grad and id(p) not in opt_ids]
    assert not missing, f"C1 FAIL [{name}]: not in optimizer: {missing[:3]}"
    before = {n: p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
    g = torch.Generator().manual_seed(1)
    last = None
    for _ in range(6):
        x = torch.randint(0,50304,(2,48),generator=g); y = torch.randint(0,50304,(2,48),generator=g)
        loss = m(x, y, h.get_bigram_ids(x))
        for o in opts: o.zero_grad()
        loss.backward()
        for o in opts: o.step()
        last = float(loss)
    assert last == last, f"[{name}] loss is NaN"   # finite check
    frozen = [n for n,p in m.named_parameters() if p.requires_grad and torch.equal(p, before[n])]
    assert not frozen, f"C2 FAIL [{name}]: never changed: {frozen[:3]}"
    return last

if __name__ == "__main__":
    ok = True
    for name, kw in ARMS.items():
        try:
            loss = check_arm(name, kw); print(f"  PASS  {name:26} final_loss={loss:.4f}")
        except Exception as e:
            ok = False; print(f"  FAIL  {name:26} {e}")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
