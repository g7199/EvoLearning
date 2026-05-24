#!/usr/bin/env python3
"""Pedagogical coherence analysis on ASSIST15 L=10.
Metrics:
- Target coverage: fraction of targets that appear in path
- Prerequisite violation rate: prereq-edges in path where prereq comes AFTER dependent
- Difficulty smoothness: average |mastery_jump| between consecutive concepts (lower = smoother)
- Concept diversity: unique concepts / L
"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.graph import load_graph
from simpath.eval.methods.base import PolicyNet, make_state_standard
import warnings; warnings.filterwarnings('ignore')

cfg = get_dataset_config('assist15')
dev = 'cuda:0'
NC = cfg['num_c']; L = 10
dkt, _ = load_dkt(cfg, dev)
kes = KES(dkt, NC, dev)
graph = load_graph(cfg, 'dkt')
# ASSIST15: rebuild prereq edges with smaller threshold (default 0.25 too strict; influence p95=0.06)
import pickle
with open(cfg['graph_dkt_path'], 'rb') as f:
    raw = pickle.load(f)
inf_matrix = raw['influence']
PREREQ_THR = 0.05
ped_prereq = inf_matrix > PREREQ_THR
np.fill_diagonal(ped_prereq, False)


def get_prereqs_ped(c):
    return [int(i) for i in np.where(ped_prereq[:, c])[0]]


def get_path(policy, mastery, targets, L, NC, device):
    policy.eval()
    path, used = [], set()
    for step in range(L):
        s = torch.tensor(make_state_standard(mastery, targets, step, L, NC),
                         dtype=torch.float32, device=device)
        vm = torch.ones(NC, device=device)
        for c in used: vm[c] = 0
        with torch.no_grad(): lo, _ = policy(s, None, vm)
        a = lo.argmax().item(); path.append(a); used.add(a)
    return path


def metrics(path, targets, mastery_initial):
    coverage = len(set(path) & set(targets)) / max(len(targets), 1)
    v, total = 0, 0
    for i, c in enumerate(path):
        for prereq in get_prereqs_ped(c):
            if prereq in path:
                total += 1
                if path.index(prereq) > i:
                    v += 1
    pv = v / max(total, 1) if total > 0 else None
    div = len(set(path)) / len(path)
    jumps = [abs(mastery_initial[path[i+1]] - mastery_initial[path[i]]) for i in range(len(path)-1)]
    smooth = np.mean(jumps)
    return coverage, pv, smooth, div


# Method-specific predict functions
def make_predictor(method_name, ckpt_dir, cfg, dev):
    NC = cfg['num_c']; L_ = 10
    if method_name == 'GEHRL':
        from simpath.eval.methods.gehrl import GEHRLMethod
        m = GEHRLMethod(NC, L_, cfg['hidden'], dev)
        ck = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=False, map_location=dev)
        m.high.load_state_dict(ck['high']); m.low.load_state_dict(ck['low'])
        # GEHRL needs graph for predict
        from simpath.eval.graph import load_graph as lg
        m.graph = lg(cfg, 'dkt')
        return lambda mast, tgts: m.predict(mast, tgts)
    if method_name in ('DLELP', 'KnowLP'):
        from simpath.eval.methods.dlelp import DLELPMethod
        m = DLELPMethod(NC, L_, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.policy.load_state_dict(st); m.graph = graph
        return lambda mast, tgts: m.predict(mast, tgts)
    if method_name == 'CSEAL':
        from simpath.eval.methods.cseal import CSEALMethod
        m = CSEALMethod(NC, L_, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.policy.load_state_dict(st); m.graph = graph
        return lambda mast, tgts: m.predict(mast, tgts)
    if method_name == 'GRU4Rec':
        from simpath.eval.methods.gru4rec import GRU4RecMethod
        m = GRU4RecMethod(NC, L_, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.model.load_state_dict(st)
        return lambda mast, tgts: m.predict(mast, tgts)
    if method_name == 'SASRec':
        from simpath.eval.methods.sasrec import SASRecMethod
        m = SASRecMethod(NC, L_, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.model.load_state_dict(st)
        return lambda mast, tgts: m.predict(mast, tgts)
    # Default: PolicyNet-based
    policy = PolicyNet(NC*2+1, NC, cfg['hidden']).to(dev)
    policy.load_state_dict(torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev))
    return lambda mast, tgts: get_path(policy, mast, tgts, L_, NC, dev)


METHODS = ['EvoLearning-BC', 'EvoLearning-AWR', 'EvoLearning-DAPG',
           'PPO-vanilla', 'CSEAL', 'GEHRL', 'DLELP', 'KnowLP', 'GRU4Rec', 'SASRec']

print(f"{'Method':<20} {'Coverage':>10} {'PrereqViol':>11} {'Smooth':>10} {'Diversity':>11}")
print("-" * 65)
for method in METHODS:
    all_cov, all_pv, all_smooth, all_div = [], [], [], []
    skipped = False
    for seed in [42, 123, 7]:
        ckpt_dir = f'outputs/experiments/assist15_L{L}_seed{seed}/{method}'
        if not os.path.exists(f'{ckpt_dir}/best_model.pt'):
            skipped = True; continue
        try:
            predictor = make_predictor(method, ckpt_dir, cfg, dev)
        except Exception as e:
            print(f"{method} skip: {e}"); skipped = True; break
        _, _, test_data = load_data(cfg, seed=seed)
        for hc, hr, tgts in test_data[:200]:
            m_init = kes.mastery(hc, hr)
            path = predictor(m_init, tgts)
            c, p, s, d = metrics(path, tgts, m_init)
            all_cov.append(c); all_pv.append(p); all_smooth.append(s); all_div.append(d)
    if not all_cov:
        print(f"{method:<20} skipped")
        continue
    print(f"{method:<20} {np.mean(all_cov):>10.3f} {np.mean(all_pv):>11.3f} "
          f"{np.mean(all_smooth):>10.3f} {np.mean(all_div):>11.3f}")
