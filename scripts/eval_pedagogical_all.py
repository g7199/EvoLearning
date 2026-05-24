#!/usr/bin/env python3
"""Pedagogical coherence analysis on Junyi / ASSIST15 / EdNet (L=10).
RL-based methods only: EVOL-BC/AWR/DAPG, PPO-vanilla, CSEAL, GEHRL, DLELP, KnowLP.
Metrics:
- Target coverage: |targets ∩ path| / |targets|
- Prerequisite violation rate: fraction of in-path prereq edges that appear in wrong order
- Difficulty smoothness: avg |mastery jump| between consecutive concepts
- Concept diversity: unique / L
"""
import sys, os, argparse, pickle, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.graph import load_graph
from simpath.eval.methods.base import PolicyNet, make_state_standard
import warnings; warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--L', type=int, default=10)
parser.add_argument('--n_students', type=int, default=200)
parser.add_argument('--gpu', default='0')
parser.add_argument('--prereq_thr', type=float, default=None)
args = parser.parse_args()

cfg = get_dataset_config(args.dataset)
dev = f'cuda:{args.gpu}'
NC = cfg['num_c']; L = args.L
dkt, _ = load_dkt(cfg, dev)
kes = KES(dkt, NC, dev)
graph = load_graph(cfg, 'dkt')

# Adaptive prereq threshold (per-dataset, percentile-based)
with open(cfg['graph_dkt_path'], 'rb') as f:
    raw = pickle.load(f)
inf_matrix = raw['influence']
if args.prereq_thr is None:
    # target ~100 edges per dataset
    target_edges = 100
    flat = inf_matrix.flatten()
    flat.sort()
    args.prereq_thr = flat[-(target_edges + NC)]  # top target+diagonal
prereq_mask = inf_matrix > args.prereq_thr
np.fill_diagonal(prereq_mask, False)
print(f"[{args.dataset}] prereq threshold={args.prereq_thr:.4f}, edges={int(prereq_mask.sum())}")


def get_prereqs(c):
    return [int(i) for i in np.where(prereq_mask[:, c])[0]]


def get_path_pn(policy, mastery, targets):
    policy.eval()
    path, used = [], set()
    for step in range(L):
        s = torch.tensor(make_state_standard(mastery, targets, step, L, NC),
                         dtype=torch.float32, device=dev)
        vm = torch.ones(NC, device=dev)
        for c in used: vm[c] = 0
        with torch.no_grad(): lo, _ = policy(s, None, vm)
        a = lo.argmax().item(); path.append(a); used.add(a)
    return path


def make_predictor(method_name, ckpt_dir):
    if method_name == 'GEHRL':
        from simpath.eval.methods.gehrl import GEHRLMethod
        m = GEHRLMethod(NC, L, cfg['hidden'], dev)
        ck = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=False, map_location=dev)
        m.high.load_state_dict(ck['high']); m.low.load_state_dict(ck['low']); m.graph = graph
        return lambda mast, tgts: m.predict(mast, tgts)
    if method_name in ('DLELP', 'KnowLP'):
        from simpath.eval.methods.dlelp import DLELPMethod
        m = DLELPMethod(NC, L, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.policy.load_state_dict(st); m.graph = graph
        return lambda mast, tgts: m.predict(mast, tgts)
    if method_name == 'CSEAL':
        from simpath.eval.methods.cseal import CSEALMethod
        m = CSEALMethod(NC, L, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.policy.load_state_dict(st); m.graph = graph
        return lambda mast, tgts: m.predict(mast, tgts)
    policy = PolicyNet(NC*2+1, NC, cfg['hidden']).to(dev)
    policy.load_state_dict(torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev))
    return lambda mast, tgts: get_path_pn(policy, mast, tgts)


def metrics(path, targets, mastery_init):
    coverage = len(set(path) & set(targets)) / max(len(targets), 1)
    v, total = 0, 0
    for i, c in enumerate(path):
        for pr in get_prereqs(c):
            if pr in path:
                total += 1
                if path.index(pr) > i: v += 1
    pv = v / total if total > 0 else None
    div = len(set(path)) / len(path)
    jumps = [abs(mastery_init[path[i+1]] - mastery_init[path[i]]) for i in range(len(path)-1)]
    smooth = float(np.mean(jumps))
    return coverage, pv, smooth, div


METHODS = ['EvoLearning-BC', 'EvoLearning-AWR', 'EvoLearning-DAPG',
           'PPO-vanilla', 'CSEAL', 'GEHRL', 'DLELP', 'KnowLP']

print(f"\n=== Pedagogical metrics on {args.dataset.upper()} L={L} (3 seeds, {args.n_students} students) ===")
print(f"{'Method':<22} {'Coverage':>10} {'PrereqViol':>12} {'Smooth':>10} {'Diversity':>11} {'PV_n':>6}")
print("-"*75)

# Preload data and masteries (avoid redundant work)
print("[setup] loading data + computing init masteries...", flush=True)
seed_data = {}
for seed in [42, 123, 7]:
    _, _, test_data = load_data(cfg, seed=seed)
    test_data = test_data[:args.n_students]
    masteries = [kes.mastery(hc, hr) for hc, hr, _ in test_data]
    seed_data[seed] = (test_data, masteries)

for method in METHODS:
    all_c, all_pv, all_s, all_d = [], [], [], []
    pv_n = 0
    for seed in [42, 123, 7]:
        ckpt_dir = f'outputs/experiments/{args.dataset}_L{L}_seed{seed}/{method}'
        if not os.path.exists(f'{ckpt_dir}/best_model.pt'): continue
        try:
            predictor = make_predictor(method, ckpt_dir)
        except Exception as e:
            print(f"  {method} err: {e}"); continue
        test_data, masteries = seed_data[seed]
        for (hc, hr, tgts), m_init in zip(test_data, masteries):
            path = predictor(m_init, tgts)
            c, pv, s, d = metrics(path, tgts, m_init)
            all_c.append(c); all_s.append(s); all_d.append(d)
            if pv is not None: all_pv.append(pv); pv_n += 1
    if not all_c:
        print(f"{method:<22} skipped"); continue
    pv_str = f"{np.mean(all_pv):.3f}" if all_pv else "n/a"
    print(f"{method:<22} {np.mean(all_c):>10.3f} {pv_str:>12} {np.mean(all_s):>10.3f} "
          f"{np.mean(all_d):>11.3f} {pv_n:>6}", flush=True)
