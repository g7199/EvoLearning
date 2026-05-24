#!/usr/bin/env python3
"""Compare evolutionary search vs (a) greedy 1-step lookahead and (b) random search
with equal compute budget. Addresses reviewer concern: why evolutionary specifically?
"""
import sys, os, argparse, time, pickle, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.graph import load_graph
import warnings; warnings.filterwarnings('ignore')


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


def greedy_path(kes, graph, hc, hr, targets, L, num_c):
    """1-step greedy: at each step, try all candidates, pick the one that maximises
    immediate EP-style improvement (Ee - Es) / (1 - Es). Myopic, single-step lookahead."""
    cands = get_candidates(targets, kes.mastery(hc, hr), graph, num_c)
    es = kes.mastery(hc, hr)[targets]
    sc, sr = list(hc), list(hr)
    chosen = []
    for step in range(L):
        best_score, best_c = -1e9, None
        for c in cands:
            if c in chosen: continue
            # Simulate one step
            cur_m = kes.mastery(sc, sr)
            r_pred = 1 if cur_m[c] > 0.5 else 0
            sc_try = sc + [c]
            sr_try = sr + [r_pred]
            ee = kes.mastery(sc_try, sr_try)[targets]
            # Immediate improvement
            denom = (1 - es).sum()
            score = (ee - es).sum() / max(denom, 1e-6)
            if score > best_score:
                best_score = score; best_c = c
        if best_c is None:
            # Pick random non-used candidate
            avail = [c for c in cands if c not in chosen]
            best_c = avail[0] if avail else 0
        chosen.append(best_c)
        cur_m = kes.mastery(sc, sr)
        sc.append(best_c)
        sr.append(1 if cur_m[best_c] > 0.5 else 0)
    return chosen


def random_search_path(kes, graph, hc, hr, targets, L, num_c, n_samples=3000):
    """Random search baseline: sample N random distinct-concept paths, evaluate each,
    pick the best. n_samples=3000 matches our evolutionary budget (60×50)."""
    cands = get_candidates(targets, kes.mastery(hc, hr), graph, num_c)
    if len(cands) < L:
        cands = cands + [c for c in range(num_c) if c not in cands][:L - len(cands)]
    best_ep, best_path = -1e9, None
    es = kes.mastery(hc, hr)[targets]
    for _ in range(n_samples):
        path = list(np.random.choice(cands, L, replace=False))
        ep = kes.evaluate(hc, hr, path, targets)
        if ep > best_ep:
            best_ep, best_path = ep, path
    return best_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='junyi')
    p.add_argument('--L', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_students', type=int, default=200)
    p.add_argument('--method', choices=['greedy', 'random_search'], required=True)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    cfg = get_dataset_config(args.dataset)
    dev = f'cuda:{args.gpu}'
    dkt, _ = load_dkt(cfg, dev)
    kes = KES(dkt, cfg['num_c'], dev)
    graph = load_graph(cfg, 'dkt')
    _, _, test_data = load_data(cfg, seed=args.seed)
    test_data = test_data[:args.n_students]

    np.random.seed(args.seed)
    t0 = time.time(); eps = []
    for i, (hc, hr, tgts) in enumerate(test_data):
        if args.method == 'greedy':
            path = greedy_path(kes, graph, hc, hr, tgts, args.L, cfg['num_c'])
        else:
            path = random_search_path(kes, graph, hc, hr, tgts, args.L, cfg['num_c'])
        eps.append(kes.evaluate(hc, hr, path, tgts))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_data)} | mean={np.mean(eps):+.4f} | {time.time()-t0:.0f}s", flush=True)

    print(f"\n{args.method} | {args.dataset} L={args.L} seed={args.seed} | N={len(eps)}")
    print(f"  Test EP: {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
