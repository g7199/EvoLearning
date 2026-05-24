#!/usr/bin/env python3
"""Mechanism analysis: why does EVOL achieve higher EP?
- Target Influence per step: mean increase in target mastery per chosen concept
- Sequencing impact: EP drop when path is shuffled
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
parser.add_argument('--n_shuffles', type=int, default=5)
parser.add_argument('--gpu', default='0')
args = parser.parse_args()

cfg = get_dataset_config(args.dataset)
dev = f'cuda:{args.gpu}'
NC = cfg['num_c']; L = args.L
dkt, _ = load_dkt(cfg, dev)
kes = KES(dkt, NC, dev)
graph = load_graph(cfg, 'dkt')


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
    def pn(mast, tgts):
        path, used = [], set()
        for step in range(L):
            s = torch.tensor(make_state_standard(mast, tgts, step, L, NC),
                             dtype=torch.float32, device=dev)
            vm = torch.ones(NC, device=dev)
            for c in used: vm[c] = 0
            with torch.no_grad(): lo, _ = policy(s, None, vm)
            a = lo.argmax().item(); path.append(a); used.add(a)
        return path
    return pn


def eval_path(hc, hr, path, tgts):
    """Compute EP and per-step target mastery trajectory."""
    es = kes.mastery(hc, hr)[tgts]
    sc, sr = list(hc), list(hr)
    target_traj = [float(np.mean(kes.mastery(sc, sr)[tgts]))]
    for c in path:
        cur = kes.mastery(sc, sr)
        sc.append(c); sr.append(1 if cur[c] > 0.5 else 0)
        target_traj.append(float(np.mean(kes.mastery(sc, sr)[tgts])))
    ee = kes.mastery(sc, sr)[tgts]
    denom = (1 - es).sum()
    ep = 0.0 if denom < 1e-6 else float((ee - es).sum() / denom)
    # per-step target lift
    target_lifts = [target_traj[i+1] - target_traj[i] for i in range(L)]
    return ep, target_lifts


def analyze(predictor, test_data, rng):
    """For each student: target influence, sequencing drop."""
    all_lifts = []
    ep_orig = []
    ep_shuffled = []
    for hc, hr, tgts in test_data:
        m_init = kes.mastery(hc, hr)
        path = predictor(m_init, tgts)
        ep, lifts = eval_path(hc, hr, path, tgts)
        ep_orig.append(ep)
        all_lifts.extend(lifts)
        # n_shuffles random shuffles
        shuf_eps = []
        for _ in range(args.n_shuffles):
            p_shuf = list(path)
            rng.shuffle(p_shuf)
            ep_s, _ = eval_path(hc, hr, p_shuf, tgts)
            shuf_eps.append(ep_s)
        ep_shuffled.append(np.mean(shuf_eps))
    seq_drop = np.mean(np.array(ep_orig) - np.array(ep_shuffled))
    return {
        'ep_orig': float(np.mean(ep_orig)),
        'ep_shuf': float(np.mean(ep_shuffled)),
        'seq_drop': float(seq_drop),
        'target_lift_step_mean': float(np.mean(all_lifts)),
        'target_lift_step_std': float(np.std(all_lifts)),
    }


METHODS = ['EvoLearning-BC', 'EvoLearning-AWR', 'EvoLearning-DAPG',
           'PPO-vanilla', 'CSEAL', 'GEHRL', 'DLELP']

print(f"[setup] loading {args.dataset} L={L}...", flush=True)
seed_data = {}
for seed in [42, 123, 7]:
    _, _, td = load_data(cfg, seed=seed)
    seed_data[seed] = td[:args.n_students]
print(f"[setup] done", flush=True)

print(f"\n=== {args.dataset.upper()} L={L} Mechanism Analysis ({args.n_shuffles} shuffles/path) ===")
print(f"{'Method':<18} {'EP_orig':>9} {'EP_shuf':>9} {'SeqDrop':>9} {'StepLift':>10}")
print("-" * 65)
import json
all_results = {}
for method in METHODS:
    per_seed = []
    for seed in [42, 123, 7]:
        ckpt_dir = f'outputs/experiments/{args.dataset}_L{L}_seed{seed}/{method}'
        if not os.path.exists(f'{ckpt_dir}/best_model.pt'): continue
        try:
            pred = make_predictor(method, ckpt_dir)
        except Exception as e:
            print(f"  {method} seed{seed} err: {e}"); continue
        rng = np.random.default_rng(seed)
        per_seed.append(analyze(pred, seed_data[seed], rng))
    if not per_seed: continue
    r = {k: np.mean([s[k] for s in per_seed]) for k in per_seed[0]}
    all_results[method] = r
    print(f"{method:<18} {r['ep_orig']:>+9.4f} {r['ep_shuf']:>+9.4f} "
          f"{r['seq_drop']:>+9.4f} {r['target_lift_step_mean']:>+10.5f}",
          flush=True)

with open(f'outputs/mechanism_{args.dataset}_L{L}.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved: outputs/mechanism_{args.dataset}_L{L}.json")
