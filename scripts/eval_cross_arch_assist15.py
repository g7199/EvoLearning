#!/usr/bin/env python3
"""Cross-ARCHITECTURE cross-simulator: DKT-trained policy evaluated on DKVMN simulator.
DKVMN outputs P(correct) per-step for the question at that position. To extract per-concept
mastery, we batch-query each candidate concept appended as the next question."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pykt.models import init_model
from simpath.eval.kes import KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.methods.base import PolicyNet
import warnings; warnings.filterwarnings('ignore')


class DKVMNasKES:
    """Wrapper that exposes a KES-compatible interface backed by DKVMN."""
    def __init__(self, dkvmn, num_c, device, max_hist=200):
        self.dkvmn = dkvmn
        self.num_c = num_c
        self.device = device
        self.max_hist = max_hist

    def _mastery_vector(self, hc, hr):
        """Get mastery vector [NC] by querying DKVMN with each concept appended."""
        NC = self.num_c
        hc_t = hc[-self.max_hist+1:]
        hr_t = hr[-self.max_hist+1:]
        L = len(hc_t)
        # Build batch: NC sequences, each = (hc_t + [c]) for c in 0..NC-1
        q = torch.zeros(NC, L+1, dtype=torch.long, device=self.device)
        r = torch.zeros(NC, L+1, dtype=torch.long, device=self.device)
        if L > 0:
            q[:, :L] = torch.tensor(hc_t, dtype=torch.long, device=self.device)
            r[:, :L] = torch.tensor(hr_t, dtype=torch.long, device=self.device)
        # Last position = each concept c (different per row)
        q[:, L] = torch.arange(NC, device=self.device)
        with torch.no_grad():
            out = self.dkvmn(q, r)  # [NC, L+1]
        return out[:, L].cpu().numpy()  # P(correct | history, next_q=c) for each c

    def mastery(self, hc, hr):
        if not hc:
            return np.full(self.num_c, 0.5, dtype=np.float32)
        return self._mastery_vector(hc, hr).astype(np.float32)

    def batch_mastery(self, hc_list, hr_list):
        # Sequential per-student (DKVMN already batches over concepts internally)
        B = len(hc_list)
        res = np.zeros((B, self.num_c), dtype=np.float32)
        for i in range(B):
            res[i] = self.mastery(hc_list[i], hr_list[i])
        return res

    def evaluate(self, hc, hr, path, targets):
        es = self.mastery(hc, hr)[targets]
        sc, sr = list(hc), list(hr)
        for c in path:
            m = self.mastery(sc, sr)
            sc.append(c); sr.append(1 if m[c] > 0.5 else 0)
        ee = self.mastery(sc, sr)[targets]
        denom = (1 - es).sum()
        if denom < 1e-6: return 0.0
        return float((ee - es).sum() / denom)

    def evaluate_policy_batch(self, data_split, policy, L):
        """Same interface as KES; uses h0 (init mastery) for actor state."""
        import torch.nn.functional as F
        B = len(data_split); NC = self.num_c
        hc_list = [d[0] for d in data_split]
        hr_list = [d[1] for d in data_split]
        tgts_list = [d[2] for d in data_split]

        init_m = self.batch_mastery(hc_list, hr_list)
        es_list = [init_m[i][tgts_list[i]].copy() for i in range(B)]

        tmask_np = np.zeros((B, NC), dtype=np.float32)
        for i in range(B): tmask_np[i, tgts_list[i]] = 1.0
        tmask_t = torch.from_numpy(tmask_np).to(self.device)
        init_m_t = torch.from_numpy(init_m.astype(np.float32)).to(self.device)

        sc_b = [list(hc_list[i]) for i in range(B)]
        sr_b = [list(hr_list[i]) for i in range(B)]
        sim_m = init_m.copy()
        used = [set() for _ in range(B)]

        policy.eval()
        for step in range(L):
            sf = torch.full((B, 1), step / L, dtype=torch.float32, device=self.device)
            state = torch.cat([init_m_t, tmask_t, sf], dim=1)
            vm_np = np.ones((B, NC), dtype=np.float32)
            for i in range(B):
                for c in used[i]: vm_np[i, c] = 0
            vm = torch.from_numpy(vm_np).to(self.device)
            with torch.no_grad():
                logits, _ = policy(state, None, vm)
            actions = logits.argmax(dim=-1).cpu().numpy()
            for i in range(B):
                a = int(actions[i])
                used[i].add(a)
                sc_b[i].append(a)
                sr_b[i].append(1 if sim_m[i][a] > 0.5 else 0)
            sim_m = self.batch_mastery(sc_b, sr_b)

        eps = []
        for i in range(B):
            es = es_list[i]; ee = sim_m[i][tgts_list[i]]
            denom = (1 - es).sum()
            eps.append(0.0 if denom < 1e-6 else float((ee - es).sum() / denom))
        return eps


def load_dkvmn(path, num_c, device):
    ckpt = torch.load(path, weights_only=False, map_location=device)
    dkvmn = init_model('dkvmn', {'dim_s': ckpt['dim_s'], 'size_m': ckpt['size_m'], 'dropout': ckpt['dropout']},
                       {'num_q': num_c, 'num_c': num_c, 'emb_path': ''}, 'qid').to(device)
    dkvmn.load_state_dict(ckpt['model'])
    dkvmn.eval()
    return dkvmn


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--L', type=int, default=10)
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

    dkvmn = load_dkvmn('outputs/checkpoints/dkvmn_assist15.pt', NC, dev)
    dkvmn_kes = DKVMNasKES(dkvmn, NC, dev)

    _, _, test_data = load_data(cfg, seed=args.seed)
    eps = dkvmn_kes.evaluate_policy_batch(test_data, policy, args.L)
    print(f"{args.method} | assist15 L={args.L} seed={args.seed} | N={len(eps)}")
    print(f"  EP on DKVMN (cross-architecture): {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
