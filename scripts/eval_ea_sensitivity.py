#!/usr/bin/env python3
"""Evolutionary search sensitivity: how does the final policy quality scale
with (population × generation) budget?"""
import sys, os, argparse, time, pickle, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.graph import load_graph
import warnings; warnings.filterwarnings('ignore')


def evo_path(kes_dkt, cands, hc, hr, targets, L, num_c, POP, GEN, device):
    """Evolutionary search with configurable POP/GEN."""
    with torch.no_grad():
        es = kes_dkt(torch.tensor([hc[-200:]], dtype=torch.long, device=device),
                      torch.tensor([hr[-200:]], dtype=torch.long, device=device))[0, -1, targets].cpu().numpy()

    bl = min(len(hc), 200)
    pops = np.zeros((POP, L), dtype=np.int64)
    for p in range(POP):
        if len(cands) >= L:
            pops[p] = np.random.choice(cands, L, replace=False)
        else:
            pops[p, :len(cands)] = cands

    total_len = bl + L
    hist_q = torch.zeros(POP, total_len, dtype=torch.long, device=device)
    hist_r = torch.zeros(POP, total_len, dtype=torch.long, device=device)
    if bl > 0:
        hist_q[:, :bl] = torch.tensor(hc[-200:], dtype=torch.long, device=device).unsqueeze(0).expand(POP, -1)
        hist_r[:, :bl] = torch.tensor(hr[-200:], dtype=torch.long, device=device).unsqueeze(0).expand(POP, -1)
    tgt_t = torch.tensor(targets, dtype=torch.long, device=device)

    def eval_fit(pops):
        hist_q[:, bl:] = 0; hist_r[:, bl:] = 0
        pops_t = torch.tensor(pops, dtype=torch.long, device=device)
        for step in range(L):
            cur = bl + step
            with torch.no_grad():
                out = kes_dkt(hist_q[:, :cur], hist_r[:, :cur]) if cur > 0 else None
            if out is not None:
                pc = out[torch.arange(POP, device=device), cur-1, pops_t[:, step]]
            else:
                pc = torch.full((POP,), 0.5, device=device)
            hist_q[:, cur] = pops_t[:, step]
            hist_r[:, cur] = (pc > 0.5).long()
        with torch.no_grad():
            out_f = kes_dkt(hist_q[:, :total_len], hist_r[:, :total_len])
        ee = out_f[:, total_len-1, tgt_t]
        es_t = torch.tensor(es, dtype=torch.float32, device=device).unsqueeze(0)
        denom = (1.0 - es_t).sum(1).clamp(min=1e-6)
        ep = (ee - es_t).sum(1).div(denom).cpu().numpy()
        return ep

    fit = eval_fit(pops)
    for gen in range(GEN):
        i1 = np.random.randint(POP, size=POP); i2 = np.random.randint(POP, size=POP)
        new = pops[np.where(fit[i1] > fit[i2], i1, i2)].copy()
        for i in range(0, POP-1, 2):
            if np.random.random() < 0.8 and L > 1:
                cx = np.random.randint(1, L)
                ch = list(new[i, :cx])
                for c in new[i+1]:
                    if c not in ch and len(ch) < L: ch.append(c)
                while len(ch) < L:
                    c = np.random.choice(cands)
                    if c not in ch: ch.append(c)
                new[i] = ch[:L]
        for i in range(POP):
            if np.random.random() < 0.3:
                pos = np.random.randint(L); nc = np.random.choice(cands)
                if nc not in new[i]: new[i, pos] = nc
        new[0] = pops[np.argmax(fit)]
        pops = new; fit = eval_fit(pops)

    return pops[int(np.argmax(fit))].tolist()


def get_candidates(targets, mastery, graph, num_c, cap=30):
    r = set()
    for t in targets:
        r.add(int(t))
        for p in graph.get_prerequisites(t): r.add(int(p))
        for s in graph.get_similar(t, top_k=5): r.add(int(s))
    if len(r) < cap:
        for c in sorted(range(num_c), key=lambda c: abs(mastery[c] - 0.5)):
            r.add(int(c))
            if len(r) >= cap: break
    return list(r)[:cap]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--POP', type=int, required=True)
    p.add_argument('--GEN', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_students', type=int, default=200)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    cfg = get_dataset_config('junyi')
    dev = f'cuda:{args.gpu}'
    dkt, _ = load_dkt(cfg, dev)
    kes = KES(dkt, cfg['num_c'], dev)
    graph = load_graph(cfg, 'dkt')
    _, _, test_data = load_data(cfg, seed=args.seed)
    test_data = test_data[:args.n_students]
    L = 10

    np.random.seed(args.seed)
    t0 = time.time(); eps = []
    for i, (hc, hr, tgts) in enumerate(test_data):
        m0 = kes.mastery(hc, hr)
        cands = get_candidates(tgts, m0, graph, cfg['num_c'])
        path = evo_path(dkt, cands, hc, hr, tgts, L, cfg['num_c'], args.POP, args.GEN, dev)
        eps.append(kes.evaluate(hc, hr, path, tgts))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_data)} | mean={np.mean(eps):+.4f} | {time.time()-t0:.0f}s", flush=True)

    print(f"\nEA POP={args.POP} GEN={args.GEN} (budget={args.POP*args.GEN}) | seed={args.seed}")
    print(f"  Test EP: {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
