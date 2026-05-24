#!/usr/bin/env python3
"""Cross-simulator v2: state from ORIGINAL DKT (in-distribution input),
path execution on alt DKT (different ground truth)."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pykt.models import init_model
from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.methods.base import PolicyNet
import warnings; warnings.filterwarnings('ignore')


def load_alt_dkt(path, num_c, device):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    dkt = init_model('dkt', {'emb_size': ckpt['emb_size'], 'dropout': 0.2},
                     {'num_q': num_c, 'num_c': num_c, 'emb_path': ''}, 'qid').to(device)
    dkt.load_state_dict(ckpt['model']); dkt.eval()
    return dkt


def eval_v2(policy, orig_kes, alt_kes, data, L, NC, device):
    """State from orig DKT, execute path on alt DKT."""
    B = len(data)
    hc_list = [d[0] for d in data]; hr_list = [d[1] for d in data]; tgts_list = [d[2] for d in data]
    init_m_orig = orig_kes.batch_mastery(hc_list, hr_list)
    init_m_alt = alt_kes.batch_mastery(hc_list, hr_list)
    es_list = [init_m_alt[i][tgts_list[i]].copy() for i in range(B)]

    tmask_np = np.zeros((B, NC), dtype=np.float32)
    for i in range(B): tmask_np[i, tgts_list[i]] = 1.0
    tmask_t = torch.from_numpy(tmask_np).to(device)
    init_m_t = torch.from_numpy(init_m_orig.astype(np.float32)).to(device)

    sc_b = [list(hc_list[i]) for i in range(B)]
    sr_b = [list(hr_list[i]) for i in range(B)]
    sim_m = init_m_alt.copy()
    used = [set() for _ in range(B)]

    policy.eval()
    for step in range(L):
        sf = torch.full((B, 1), step / L, dtype=torch.float32, device=device)
        state = torch.cat([init_m_t, tmask_t, sf], dim=1)
        vm_np = np.ones((B, NC), dtype=np.float32)
        for i in range(B):
            for c in used[i]: vm_np[i, c] = 0
        vm = torch.from_numpy(vm_np).to(device)
        with torch.no_grad(): logits, _ = policy(state, None, vm)
        actions = logits.argmax(dim=-1).cpu().numpy()
        for i in range(B):
            a = int(actions[i]); used[i].add(a)
            sc_b[i].append(a); sr_b[i].append(1 if sim_m[i][a] > 0.5 else 0)
        sim_m = alt_kes.batch_mastery(sc_b, sr_b)

    eps = []
    for i in range(B):
        es = es_list[i]; ee = sim_m[i][tgts_list[i]]
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
    args = p.parse_args()

    cfg = get_dataset_config('assist15')
    dev = f'cuda:{args.gpu}'; NC = cfg['num_c']
    policy = PolicyNet(NC*2+1, NC, cfg['hidden']).to(dev)
    pck = f'outputs/experiments/assist15_L{args.L}_seed{args.seed}/{args.method}/best_model.pt'
    policy.load_state_dict(torch.load(pck, weights_only=True, map_location=dev))

    orig_dkt, _ = load_dkt(cfg, dev)
    orig_kes = KES(orig_dkt, NC, dev)
    alt_dkt = load_alt_dkt(args.alt_ckpt, NC, dev)
    alt_kes = KES(alt_dkt, NC, dev)

    _, _, test_data = load_data(cfg, seed=args.seed)
    eps = eval_v2(policy, orig_kes, alt_kes, test_data, args.L, NC, dev)
    print(f"{args.method} | {args.alt_name} | seed={args.seed} | N={len(eps)}")
    print(f"  EP (state=orig, exec={args.alt_name}): {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
