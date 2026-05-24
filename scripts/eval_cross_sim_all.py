#!/usr/bin/env python3
"""Cross-simulator v2 across all RL/sequential methods × 5 alt DKTs × 3 seeds.
State input from ORIGINAL DKT (in-distribution); path execution on ALT DKT."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pykt.models import init_model
from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.graph import load_graph
from simpath.eval.methods.base import PolicyNet, make_state_standard
import warnings; warnings.filterwarnings('ignore')


def load_alt_dkt(path, num_c, device):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    dkt = init_model('dkt', {'emb_size': ckpt['emb_size'], 'dropout': 0.2},
                     {'num_q': num_c, 'num_c': num_c, 'emb_path': ''}, 'qid').to(device)
    dkt.load_state_dict(ckpt['model']); dkt.eval()
    return dkt


def make_predictor(method_name, ckpt_dir, cfg, dev, graph):
    """Return a function predict(mastery, targets) -> path (length L)."""
    NC = cfg['num_c']; L = 10
    if method_name == 'GEHRL':
        from simpath.eval.methods.gehrl import GEHRLMethod
        m = GEHRLMethod(NC, L, cfg['hidden'], dev)
        ck = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=False, map_location=dev)
        m.high.load_state_dict(ck['high']); m.low.load_state_dict(ck['low'])
        m.graph = graph
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
    if method_name == 'GRU4Rec':
        from simpath.eval.methods.gru4rec import GRU4RecMethod
        m = GRU4RecMethod(NC, L, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.model.load_state_dict(st)
        return lambda mast, tgts: m.predict(mast, tgts)
    if method_name == 'SASRec':
        from simpath.eval.methods.sasrec import SASRecMethod
        m = SASRecMethod(NC, L, cfg['hidden'], dev)
        st = torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev)
        m.model.load_state_dict(st)
        return lambda mast, tgts: m.predict(mast, tgts)
    # PolicyNet-based: EvoLearning-BC/AWR/DAPG, PPO-vanilla
    policy = PolicyNet(NC*2+1, NC, cfg['hidden']).to(dev)
    policy.load_state_dict(torch.load(f'{ckpt_dir}/best_model.pt', weights_only=True, map_location=dev))
    policy.eval()
    def predict_fn(mast, tgts):
        path, used = [], set()
        for step in range(L):
            s = torch.tensor(make_state_standard(mast, tgts, step, L, NC),
                             dtype=torch.float32, device=dev)
            vm = torch.ones(NC, device=dev)
            for c in used: vm[c] = 0
            with torch.no_grad(): lo, _ = policy(s, None, vm)
            a = lo.argmax().item(); path.append(a); used.add(a)
        return path
    return predict_fn


def cross_sim_eval(predictor, orig_kes, alt_kes, data, L, NC):
    """For each student: get state from orig, predict path, execute on alt, compute EP."""
    eps = []
    for hc, hr, tgts in data:
        # State from original DKT (in-distribution input)
        m_state = orig_kes.mastery(hc, hr)
        # Path from policy
        path = predictor(m_state, tgts)
        # Execute on alt DKT
        es = alt_kes.mastery(hc, hr)[tgts]
        sc, sr = list(hc), list(hr)
        for c in path:
            m_cur = alt_kes.mastery(sc, sr)
            sc.append(c); sr.append(1 if m_cur[c] > 0.5 else 0)
        ee = alt_kes.mastery(sc, sr)[tgts]
        denom = (1 - es).sum()
        eps.append(0.0 if denom < 1e-6 else float((ee - es).sum() / denom))
    return eps


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--alt_ckpt', required=True)
    p.add_argument('--alt_name', default='alt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--L', type=int, default=10)
    p.add_argument('--gpu', default='0')
    p.add_argument('--n_students', type=int, default=200)
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
        print(f"NO CHECKPOINT: {args.method} seed={args.seed}"); exit(0)
    predictor = make_predictor(args.method, ckpt_dir, cfg, dev, graph)

    orig_dkt, _ = load_dkt(cfg, dev)
    orig_kes = KES(orig_dkt, NC, dev)
    alt_dkt = load_alt_dkt(args.alt_ckpt, NC, dev)
    alt_kes = KES(alt_dkt, NC, dev)

    _, _, test_data = load_data(cfg, seed=args.seed)
    test_data = test_data[:args.n_students]

    eps = cross_sim_eval(predictor, orig_kes, alt_kes, test_data, args.L, NC)
    print(f"{args.method} | {args.alt_name} | seed={args.seed} | N={len(eps)}")
    print(f"  EP (state=orig, exec={args.alt_name}): {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
