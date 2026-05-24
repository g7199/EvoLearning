#!/usr/bin/env python3
"""Cross-simulator evaluation on ASSIST15: train on DKT(emb=200,seed=42), evaluate on DKT(emb=128,seed=999)."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pykt.models import init_model
from simpath.eval.kes import KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.methods.base import PolicyNet
import warnings; warnings.filterwarnings('ignore')


def load_alt_dkt(path, num_c, device):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    emb = ckpt.get('emb_size', 128)
    dkt = init_model('dkt', {'emb_size': emb, 'dropout': 0.2},
                     {'num_q': num_c, 'num_c': num_c, 'emb_path': ''}, 'qid').to(device)
    dkt.load_state_dict(ckpt['model'])
    dkt.eval()
    return dkt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--L', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    cfg = get_dataset_config('assist15')
    dev = f'cuda:{args.gpu}'
    NC = cfg['num_c']

    state_dim = NC * 2 + 1
    policy = PolicyNet(state_dim, NC, cfg['hidden']).to(dev)
    ckpt_path = f'outputs/experiments/assist15_L{args.L}_seed{args.seed}/{args.method}/best_model.pt'
    policy.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=dev))
    policy.eval()

    alt_dkt = load_alt_dkt('outputs/checkpoints/pykt_dkt_alt_assist15.pt', NC, dev)
    alt_kes = KES(alt_dkt, NC, dev)

    _, _, test_data = load_data(cfg, seed=args.seed)
    eps = alt_kes.evaluate_policy_batch(test_data, policy, args.L)
    print(f"{args.method} | assist15 L={args.L} seed={args.seed} | N={len(eps)}")
    print(f"  EP on ALT-DKT: {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
