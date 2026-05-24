#!/usr/bin/env python3
"""Cross-simulator evaluation: train policy on DKT-A (emb=256), evaluate on DKT-B (emb=128).
Addresses concern that gains might be artifacts of a specific simulator.
If EVOL > PPO on DKT-B too, the comparison is not simulator-specific."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pykt.models import init_model
from simpath.eval.kes import KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.methods.base import PolicyNet
import warnings; warnings.filterwarnings('ignore')


def load_alt_dkt(path, num_c, device):
    """Load alternative DKT (emb=128)."""
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
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    cfg = get_dataset_config('junyi')
    dev = f'cuda:{args.gpu}'
    NC = cfg['num_c']; L = 10

    # Load policy trained on ORIGINAL DKT
    state_dim = NC * 2 + 1
    policy = PolicyNet(state_dim, NC, cfg['hidden']).to(dev)
    ckpt_path = f'outputs/experiments/junyi_L{L}_seed{args.seed}/{args.method}/best_model.pt'
    policy.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=dev))
    policy.eval()

    # Build KES with ALT DKT
    alt_dkt = load_alt_dkt('outputs/checkpoints/pykt_dkt_alt_junyi.pt', NC, dev)
    alt_kes = KES(alt_dkt, NC, dev)

    # Load same test split
    _, _, test_data = load_data(cfg, seed=args.seed)

    # Evaluate via fully batched
    eps = alt_kes.evaluate_policy_batch(test_data, policy, L)
    print(f"{args.method} | seed={args.seed} | N={len(eps)}")
    print(f"  EP on ALT-DKT (emb=128): {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
