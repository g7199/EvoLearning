#!/usr/bin/env python3
"""Comprehensive pedagogical analysis. Computes everything; pick what's useful.

Metrics computed:
  Path-level:
    1. Target coverage: |targets ∩ path| / |targets|
    2. Concept diversity: unique(path) / L
    3. Prereq violation rate: in-path prereq edges in wrong order
  Per-step mastery analysis (when concept chosen):
    4. ZPD ratio: % steps where chosen-concept mastery ∈ [0.3, 0.7]
    5. Easy ratio: % steps where mastery > 0.8 (wasted)
    6. Hard ratio: % steps where mastery < 0.2 (likely fail)
    7. Avg selected-concept mastery (overall difficulty)
  Target-level:
    8. Mean target mastery improvement
    9. Fraction of targets that improved (>0.01)
    10. Per-target initial mastery vs improvement correlation
  Learner-stratified:
    11. Mean EP for low/mid/high initial mastery learners
    12. Path adaptiveness: 1 - mean Jaccard(path_i, path_j) across learner pairs
  Mastery trajectory:
    13. Mean mastery at each step t (selected concept's mastery)
    14. Mean target mastery at each step t
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
args = parser.parse_args()

cfg = get_dataset_config(args.dataset)
dev = f'cuda:{args.gpu}'
NC = cfg['num_c']; L = args.L
dkt, _ = load_dkt(cfg, dev)
kes = KES(dkt, NC, dev)
graph = load_graph(cfg, 'dkt')

# Top-100 edges prereq threshold
with open(cfg['graph_dkt_path'], 'rb') as f:
    raw = pickle.load(f)
inf = raw['influence']
flat = sorted(inf.flatten(), reverse=True)
thr = flat[min(100 + NC, len(flat) - 1)]
prereq_mask = inf > thr
np.fill_diagonal(prereq_mask, False)
print(f"[{args.dataset}] prereq edges: {int(prereq_mask.sum())}, threshold {thr:.4f}")


def get_prereqs(c):
    return [int(i) for i in np.where(prereq_mask[:, c])[0]]


def make_predictor(method, ckpt_dir):
    if method == 'GEHRL':
        from simpath.eval.methods.gehrl import GEHRLMethod
        m = GEHRLMethod(NC, L, cfg['hidden'], dev)
        ck = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=False, map_location=dev)
        m.high.load_state_dict(ck['high']); m.low.load_state_dict(ck['low']); m.graph = graph
        return lambda mast, tgts: m.predict(mast, tgts)
    if method in ('DLELP', 'KnowLP'):
        from simpath.eval.methods.dlelp import DLELPMethod
        m = DLELPMethod(NC, L, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.policy.load_state_dict(st); m.graph = graph
        return lambda mast, tgts: m.predict(mast, tgts)
    if method == 'CSEAL':
        from simpath.eval.methods.cseal import CSEALMethod
        m = CSEALMethod(NC, L, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.policy.load_state_dict(st); m.graph = graph
        return lambda mast, tgts: m.predict(mast, tgts)
    policy = PolicyNet(NC*2+1, NC, cfg['hidden']).to(dev)
    policy.load_state_dict(torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev))
    policy.eval()
    def pn_predict(mast, tgts):
        path, used = [], set()
        for step in range(L):
            s = torch.tensor(make_state_standard(mast, tgts, step, L, NC),
                             dtype=torch.float32, device=dev)
            vm = torch.ones(NC, device=dev)
            for c in used: vm[c] = 0
            with torch.no_grad(): lo, _ = policy(s, None, vm)
            a = lo.argmax().item(); path.append(a); used.add(a)
        return path
    return pn_predict


def jaccard(p1, p2):
    s1, s2 = set(p1), set(p2)
    return len(s1 & s2) / max(len(s1 | s2), 1)


def analyze(predictor, test_data, init_masteries):
    """Compute all per-student metrics. Returns dict of lists."""
    per_st = {'cov': [], 'div': [], 'pv': [], 'pv_n': 0,
              'zpd': [], 'easy': [], 'hard': [], 'avg_mast': [],
              'tgt_improve': [], 'tgt_improved_frac': [],
              'paths': [], 'init_mast_mean': [], 'ep': []}
    step_masteries = [[] for _ in range(L)]
    target_traj = [[] for _ in range(L+1)]  # 0..L, mean target mastery at each step

    for (hc, hr, tgts), m_init in zip(test_data, init_masteries):
        path = predictor(m_init, tgts)
        per_st['paths'].append(path)
        per_st['init_mast_mean'].append(float(np.mean(m_init)))

        # 1-3
        cov = len(set(path) & set(tgts)) / max(len(tgts), 1)
        div = len(set(path)) / len(path)
        v, total = 0, 0
        for i, c in enumerate(path):
            for pr in get_prereqs(c):
                if pr in path:
                    total += 1
                    if path.index(pr) > i: v += 1
        per_st['cov'].append(cov); per_st['div'].append(div)
        if total > 0:
            per_st['pv'].append(v / total); per_st['pv_n'] += 1

        # 4-7: per-step mastery analysis (use mastery AT TIME OF SELECTION, which is init_mast for actor)
        zpd = sum(1 for c in path if 0.3 <= m_init[c] <= 0.7) / L
        easy = sum(1 for c in path if m_init[c] > 0.8) / L
        hard = sum(1 for c in path if m_init[c] < 0.2) / L
        avg_m = float(np.mean([m_init[c] for c in path]))
        per_st['zpd'].append(zpd); per_st['easy'].append(easy); per_st['hard'].append(hard)
        per_st['avg_mast'].append(avg_m)

        # Step-wise selected concept mastery (averaged across learners later)
        for t, c in enumerate(path):
            step_masteries[t].append(m_init[c])

        # 8-10: target improvement (need KES simulation)
        es = m_init[tgts]
        sc, sr = list(hc), list(hr)
        for c in path:
            cur = kes.mastery(sc, sr)
            sc.append(c); sr.append(1 if cur[c] > 0.5 else 0)
        ee = kes.mastery(sc, sr)[tgts]
        improve = ee - es
        per_st['tgt_improve'].append(float(np.mean(improve)))
        per_st['tgt_improved_frac'].append(float(np.mean(improve > 0.01)))
        denom = (1 - es).sum()
        per_st['ep'].append(0.0 if denom < 1e-6 else float(improve.sum() / denom))

        # Target mastery trajectory: re-simulate, sample at each step
        sc2, sr2 = list(hc), list(hr)
        target_traj[0].append(float(np.mean(m_init[tgts])))
        for t, c in enumerate(path):
            cur = kes.mastery(sc2, sr2)
            sc2.append(c); sr2.append(1 if cur[c] > 0.5 else 0)
            new = kes.mastery(sc2, sr2)
            target_traj[t+1].append(float(np.mean(new[tgts])))

    # Adaptiveness: avg pairwise Jaccard of paths (lower = more diverse / adaptive)
    sample_paths = per_st['paths'][:50]
    n_pairs = 0; jacc_sum = 0
    for i in range(len(sample_paths)):
        for j in range(i+1, len(sample_paths)):
            jacc_sum += jaccard(sample_paths[i], sample_paths[j]); n_pairs += 1
    adapt = 1 - (jacc_sum / max(n_pairs, 1))

    # Stratify by initial mean mastery
    inits = np.array(per_st['init_mast_mean'])
    eps = np.array(per_st['ep'])
    lo = inits < np.percentile(inits, 33)
    hi = inits > np.percentile(inits, 67)
    md = (~lo) & (~hi)
    ep_lo = float(eps[lo].mean()) if lo.sum() > 0 else 0
    ep_md = float(eps[md].mean()) if md.sum() > 0 else 0
    ep_hi = float(eps[hi].mean()) if hi.sum() > 0 else 0

    return {
        'cov': np.mean(per_st['cov']),
        'div': np.mean(per_st['div']),
        'pv': np.mean(per_st['pv']) if per_st['pv'] else None,
        'zpd': np.mean(per_st['zpd']),
        'easy': np.mean(per_st['easy']),
        'hard': np.mean(per_st['hard']),
        'avg_mast': np.mean(per_st['avg_mast']),
        'tgt_improve': np.mean(per_st['tgt_improve']),
        'tgt_improved_frac': np.mean(per_st['tgt_improved_frac']),
        'adapt': adapt,
        'ep_low_init': ep_lo,
        'ep_mid_init': ep_md,
        'ep_high_init': ep_hi,
        'step_mastery': [np.mean(sm) if sm else 0 for sm in step_masteries],
        'target_traj': [np.mean(tt) if tt else 0 for tt in target_traj],
    }


METHODS = ['EvoLearning-BC', 'EvoLearning-AWR', 'EvoLearning-DAPG',
           'PPO-vanilla', 'CSEAL', 'GEHRL', 'DLELP', 'KnowLP']

# Preload
print("[setup] preloading data + masteries...", flush=True)
seed_data = {}
for seed in [42, 123, 7]:
    _, _, td = load_data(cfg, seed=seed)
    td = td[:args.n_students]
    masteries = [kes.mastery(hc, hr) for hc, hr, _ in td]
    seed_data[seed] = (td, masteries)
print("[setup] done", flush=True)

# Run
results = {}
for method in METHODS:
    print(f"\n>>> {method}", flush=True)
    per_seed = []
    for seed in [42, 123, 7]:
        ckpt_dir = f'outputs/experiments/{args.dataset}_L{L}_seed{seed}/{method}'
        if not os.path.exists(f'{ckpt_dir}/best_model.pt'):
            print(f"  seed {seed}: no checkpoint"); continue
        try:
            predictor = make_predictor(method, ckpt_dir)
        except Exception as e:
            print(f"  seed {seed} err: {e}"); continue
        td, masteries = seed_data[seed]
        per_seed.append(analyze(predictor, td, masteries))
    if not per_seed: continue
    # Average across seeds
    agg = {}
    for k in per_seed[0].keys():
        if isinstance(per_seed[0][k], list):
            agg[k] = [float(np.mean([s[k][i] for s in per_seed])) for i in range(len(per_seed[0][k]))]
        elif per_seed[0][k] is None:
            agg[k] = None
        else:
            agg[k] = float(np.mean([s[k] for s in per_seed if s[k] is not None]))
    results[method] = agg

# Print summary table
print(f"\n\n=== {args.dataset.upper()} L={L} Pedagogical Comprehensive ===\n")
hdr = f"{'Method':<18} {'EP':>7} {'Cov':>6} {'ZPD':>6} {'Easy':>6} {'Hard':>6} {'AvgM':>6} {'TgtImp':>8} {'ImpFrac':>8} {'Adapt':>7} {'EP_lo':>7} {'EP_mid':>7} {'EP_hi':>7} {'PV':>6}"
print(hdr)
print("-" * len(hdr))
for m, r in results.items():
    pv_str = f"{r['pv']:.3f}" if r['pv'] is not None else "n/a"
    # Compute EP from individual fields
    ep = (r['ep_low_init'] + r['ep_mid_init'] + r['ep_high_init']) / 3
    print(f"{m:<18} {ep:>+7.3f} {r['cov']:>6.3f} {r['zpd']:>6.3f} {r['easy']:>6.3f} {r['hard']:>6.3f} "
          f"{r['avg_mast']:>6.3f} {r['tgt_improve']:>+8.3f} {r['tgt_improved_frac']:>8.3f} "
          f"{r['adapt']:>7.3f} {r['ep_low_init']:>+7.3f} {r['ep_mid_init']:>+7.3f} {r['ep_high_init']:>+7.3f} {pv_str:>6}")

# Save full results
import json
out = f'outputs/pedagogical_{args.dataset}_L{L}.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {out}")
