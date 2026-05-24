#!/usr/bin/env python3
"""Stochastic-response robustness on ASSIST15 L=10 for all 10 RL/sequence methods.
The deterministic-trained checkpoint is evaluated under three alternative response models
in addition to the standard deterministic threshold:
  - Deterministic (reference):   r = 1[h_c > 0.5]            (EduSim/CSEAL default)
  - Bernoulli:                   r ~ Bern(h_c)
  - Slip-Guess (mild):           P(r=1) = (1-s)h_c + g(1-h_c),  s=g=0.10
  - Slip-Guess (strong):         same,  s=g=0.20
Paths are deterministic (h_0-only actors); only the response generation is randomised.
N stochastic rollouts per student to reduce variance.
"""
import sys, os, argparse, torch, numpy as np, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.graph import load_graph
from simpath.eval.methods.base import PolicyNet, make_state_standard
import warnings; warnings.filterwarnings('ignore')

# Reuse the predictor factory used by the cross-DKT evaluation
from eval_cross_sim_all import make_predictor


def response(p, mode, rng, slip=0.1, guess=0.1):
    if mode == 'deterministic':
        return 1 if p > 0.5 else 0
    if mode == 'bernoulli':
        return int(rng.random() < p)
    # slip_guess
    p_obs = (1.0 - slip) * p + guess * (1.0 - p)
    return int(rng.random() < p_obs)


def stoch_eval(predictor, kes, data, L, NC, mode, n_rollouts, rng_seed, slip=0.1, guess=0.1):
    rng = np.random.default_rng(rng_seed)
    eps_per_student = []
    for hc, hr, tgts in data:
        m_init = kes.mastery(hc, hr)
        path = predictor(m_init, tgts)  # deterministic path (h_0-only)
        es = kes.mastery(hc, hr)[tgts]
        denom = (1 - es).sum()
        rolls = 1 if mode == 'deterministic' else n_rollouts
        ep_rolls = []
        for _ in range(rolls):
            sc, sr = list(hc), list(hr)
            for c in path:
                m_cur = kes.mastery(sc, sr)
                r = response(float(m_cur[c]), mode, rng, slip, guess)
                sc.append(c); sr.append(r)
            ee = kes.mastery(sc, sr)[tgts]
            ep_rolls.append(0.0 if denom < 1e-6 else float((ee - es).sum() / denom))
        eps_per_student.append(float(np.mean(ep_rolls)))
    return float(np.mean(eps_per_student)), float(np.std(eps_per_student))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--L', type=int, default=10)
    p.add_argument('--gpu', default='0')
    p.add_argument('--n_students', type=int, default=200)
    p.add_argument('--n_rollouts', type=int, default=20)
    args = p.parse_args()

    cfg = get_dataset_config('assist15')
    dev = f'cuda:{args.gpu}'; NC = cfg['num_c']

    from simpath.eval.methods import METHOD_REGISTRY
    import simpath.eval.methods.knowlp
    method_cls = METHOD_REGISTRY[args.method]
    graph_type = getattr(method_cls, 'graph_type', 'dkt')
    graph = load_graph(cfg, graph_type)

    ckpt_dir = f'outputs/experiments/assist15_L{args.L}_seed{args.seed}/{args.method}'
    if not os.path.exists(f'{ckpt_dir}/best_model.pt'):
        print(f"NO CHECKPOINT: {args.method} seed={args.seed}"); sys.exit(0)
    predictor = make_predictor(args.method, ckpt_dir, cfg, dev, graph)

    dkt, _ = load_dkt(cfg, dev)
    kes = KES(dkt, NC, dev)

    _, _, test_data = load_data(cfg, seed=args.seed)
    test_data = test_data[:args.n_students]

    out = {'method': args.method, 'seed': args.seed, 'L': args.L,
           'n_students': len(test_data), 'n_rollouts': args.n_rollouts}

    settings = [
        ('deterministic', {}, 'Determ.'),
        ('bernoulli',     {}, 'Bernoulli'),
        ('slip_guess',    dict(slip=0.10, guess=0.10), 'Slip/Guess s=g=0.10'),
        ('slip_guess',    dict(slip=0.20, guess=0.20), 'Slip/Guess s=g=0.20'),
    ]
    print(f"=== {args.method} | seed={args.seed} | N={len(test_data)} | rollouts={args.n_rollouts} ===", flush=True)
    for i, (mode, params, label) in enumerate(settings):
        mu, sd = stoch_eval(predictor, kes, test_data, args.L, NC, mode,
                            n_rollouts=args.n_rollouts,
                            rng_seed=args.seed * 1000 + i, **params)
        key = label.replace(' ', '_').replace('/', '_').replace('.', '').replace('=', '')
        out[key] = {'ep_mean': mu, 'ep_std': sd}
        print(f"  {label:<22}: EP={mu:+.4f} ± {sd:.4f}", flush=True)

    out_dir = f'outputs/stochastic_response/assist15_L{args.L}_seed{args.seed}'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/{args.method}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_dir}/{args.method}.json")
