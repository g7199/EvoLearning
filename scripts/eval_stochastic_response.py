#!/usr/bin/env python3
"""Evaluate trained policies on a CALIBRATION-NOISY version of original DKT.
Response: r = 1[h_c > 0.5] flipped with probability epsilon (per step).
This models the deployment gap: real students don't perfectly follow DKT.
- epsilon=0: original deterministic (baseline)
- epsilon=0.05, 0.10, 0.20: progressively more deviation
Cross-architecture/cross-DKT는 절대값이 너무 다르므로, 같은 DKT의 미세 perturbation이 더 의미 있음."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.methods.base import PolicyNet
import warnings; warnings.filterwarnings('ignore')


def eval_noisy(policy, kes, data, L, NC, dev, epsilon, n_rollouts=5, rng=None):
    """Evaluate with stochastic response flip (ε prob)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    B = len(data)
    hc_list = [d[0] for d in data]; hr_list = [d[1] for d in data]; tgts_list = [d[2] for d in data]
    init_m = kes.batch_mastery(hc_list, hr_list)
    es_list = [init_m[i][tgts_list[i]].copy() for i in range(B)]
    tmask_np = np.zeros((B, NC), dtype=np.float32)
    for i in range(B): tmask_np[i, tgts_list[i]] = 1.0
    tmask_t = torch.from_numpy(tmask_np).to(dev)
    init_m_t = torch.from_numpy(init_m.astype(np.float32)).to(dev)

    # Generate path once (deterministic policy)
    paths = [[] for _ in range(B)]
    sim_m = init_m.copy()
    sc_b = [list(hc_list[i]) for i in range(B)]
    sr_b = [list(hr_list[i]) for i in range(B)]
    used = [set() for _ in range(B)]
    policy.eval()
    for step in range(L):
        sf = torch.full((B, 1), step / L, dtype=torch.float32, device=dev)
        state = torch.cat([init_m_t, tmask_t, sf], dim=1)
        vm_np = np.ones((B, NC), dtype=np.float32)
        for i in range(B):
            for c in used[i]: vm_np[i, c] = 0
        vm = torch.from_numpy(vm_np).to(dev)
        with torch.no_grad(): logits, _ = policy(state, None, vm)
        actions = logits.argmax(dim=-1).cpu().numpy()
        for i in range(B):
            a = int(actions[i]); used[i].add(a)
            paths[i].append(a)
            sc_b[i].append(a); sr_b[i].append(1 if sim_m[i][a] > 0.5 else 0)
        sim_m = kes.batch_mastery(sc_b, sr_b)

    # Now multi-rollout under stochastic response
    all_student_eps = []
    for i in range(B):
        ep_per_rollout = []
        for _ in range(n_rollouts):
            sc, sr = list(hc_list[i]), list(hr_list[i])
            for c in paths[i]:
                m_cur = kes.mastery(sc, sr)
                det = 1 if m_cur[c] > 0.5 else 0
                if rng.random() < epsilon:
                    det = 1 - det  # flip
                sc.append(c); sr.append(det)
            ee = kes.mastery(sc, sr)[tgts_list[i]]
            es = es_list[i]
            denom = (1 - es).sum()
            ep_per_rollout.append(0.0 if denom < 1e-6 else float((ee - es).sum() / denom))
        all_student_eps.append(np.mean(ep_per_rollout))
    return all_student_eps


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--L', type=int, default=10)
    p.add_argument('--epsilon', type=float, default=0.05)
    p.add_argument('--n_rollouts', type=int, default=5)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    cfg = get_dataset_config('assist15')
    dev = f'cuda:{args.gpu}'; NC = cfg['num_c']
    dkt, _ = load_dkt(cfg, dev)
    kes = KES(dkt, NC, dev)
    policy = PolicyNet(NC*2+1, NC, cfg['hidden']).to(dev)
    ckpt = f'outputs/experiments/assist15_L{args.L}_seed{args.seed}/{args.method}/best_model.pt'
    policy.load_state_dict(torch.load(ckpt, weights_only=True, map_location=dev))

    _, _, test_data = load_data(cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed * 1000 + int(args.epsilon * 1000))
    eps = eval_noisy(policy, kes, test_data, args.L, NC, dev, args.epsilon, args.n_rollouts, rng)
    print(f"{args.method} | seed={args.seed} | epsilon={args.epsilon} | n_rollouts={args.n_rollouts} | N={len(eps)}")
    print(f"  EP (response flip prob={args.epsilon}): {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
