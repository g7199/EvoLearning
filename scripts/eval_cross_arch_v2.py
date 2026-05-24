#!/usr/bin/env python3
"""Cross-architecture v2: state from ORIGINAL DKT, path execution on cross-arch simulator."""
import sys, os, argparse, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pykt.models import init_model
from simpath.eval.kes import load_dkt, KES
from simpath.eval.config import get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.methods.base import PolicyNet
import warnings; warnings.filterwarnings('ignore')


class AltKES:
    """Generic alternative KES wrapper. Supports DKT, DKVMN, AKT via per-concept query."""
    def __init__(self, model, arch, num_c, device, max_hist=200):
        self.model = model; self.arch = arch; self.num_c = num_c
        self.device = device; self.max_hist = max_hist

    def _fwd(self, q, r):
        if self.arch == 'akt':
            out = self.model(q, r, q)
            if isinstance(out, tuple): out = out[0]
        elif self.arch == 'dkvmn':
            out = self.model(q, r)
        else:  # dkt
            out = self.model(q, r)
        return out

    def _mastery_vec(self, hc, hr):
        NC = self.num_c
        if self.arch == 'dkt':
            # DKT outputs [B, T, NC] - read at last position
            q = torch.tensor([hc[-self.max_hist:]], dtype=torch.long, device=self.device) if hc else torch.zeros(1, 1, dtype=torch.long, device=self.device)
            r = torch.tensor([hr[-self.max_hist:]], dtype=torch.long, device=self.device) if hr else torch.zeros(1, 1, dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self.model(q, r)
            return out[0, -1].cpu().numpy()
        # DKVMN/AKT: query each concept
        hc_t = hc[-self.max_hist+1:]; hr_t = hr[-self.max_hist+1:]
        L = len(hc_t)
        q = torch.zeros(NC, L+1, dtype=torch.long, device=self.device)
        r = torch.zeros(NC, L+1, dtype=torch.long, device=self.device)
        if L > 0:
            q[:, :L] = torch.tensor(hc_t, dtype=torch.long, device=self.device)
            r[:, :L] = torch.tensor(hr_t, dtype=torch.long, device=self.device)
        q[:, L] = torch.arange(NC, device=self.device)
        with torch.no_grad():
            out = self._fwd(q, r)
        return out[:, L].cpu().numpy()

    def mastery(self, hc, hr):
        if not hc: return np.full(self.num_c, 0.5, dtype=np.float32)
        return self._mastery_vec(hc, hr).astype(np.float32)

    def batch_mastery(self, hc_list, hr_list):
        B = len(hc_list)
        res = np.zeros((B, self.num_c), dtype=np.float32)
        for i in range(B):
            res[i] = self.mastery(hc_list[i], hr_list[i])
        return res


def eval_v2(policy, orig_kes, alt_kes, data_split, L, NC, device):
    """State from orig, execute on alt."""
    B = len(data_split)
    hc_list = [d[0] for d in data_split]; hr_list = [d[1] for d in data_split]; tgts_list = [d[2] for d in data_split]
    init_m_orig = orig_kes.batch_mastery(hc_list, hr_list)
    init_m_alt = alt_kes.batch_mastery(hc_list, hr_list)
    es_list = [init_m_alt[i][tgts_list[i]].copy() for i in range(B)]
    tmask_np = np.zeros((B, NC), dtype=np.float32)
    for i in range(B): tmask_np[i, tgts_list[i]] = 1.0
    tmask_t = torch.from_numpy(tmask_np).to(device)
    init_m_t = torch.from_numpy(init_m_orig.astype(np.float32)).to(device)
    sc_b = [list(hc_list[i]) for i in range(B)]
    sr_b = [list(hr_list[i]) for i in range(B)]
    sim_m_alt = init_m_alt.copy()
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
            sc_b[i].append(a); sr_b[i].append(1 if sim_m_alt[i][a] > 0.5 else 0)
        sim_m_alt = alt_kes.batch_mastery(sc_b, sr_b)
    eps = []
    for i in range(B):
        es = es_list[i]; ee = sim_m_alt[i][tgts_list[i]]
        denom = (1 - es).sum()
        eps.append(0.0 if denom < 1e-6 else float((ee - es).sum() / denom))
    return eps


def load_kt(arch, num_c, device):
    if arch == 'dkvmn':
        ckpt = torch.load('outputs/checkpoints/dkvmn_assist15.pt', weights_only=False, map_location=device)
        m = init_model('dkvmn', {'dim_s': ckpt['dim_s'], 'size_m': ckpt['size_m'], 'dropout': ckpt['dropout']},
                       {'num_q': num_c, 'num_c': num_c, 'emb_path': ''}, 'qid').to(device)
    elif arch == 'akt':
        ckpt = torch.load('outputs/checkpoints/akt_pykt_assist15.pt', weights_only=False, map_location=device)
        m = init_model('akt', {'d_model': ckpt['d_model'], 'n_blocks': ckpt['n_blocks'], 'dropout': ckpt['dropout'],
                                'd_ff': 256, 'kq_same': 1, 'num_attn_heads': 8, 'final_fc_dim': 512, 'l2': 1e-5},
                       {'num_q': num_c, 'num_c': num_c, 'emb_path': ''}, 'qid').to(device)
    m.load_state_dict(ckpt['model']); m.eval()
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--arch', choices=['dkvmn', 'akt'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--L', type=int, default=10)
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    cfg = get_dataset_config('assist15')
    dev = f'cuda:{args.gpu}'; NC = cfg['num_c']
    state_dim = NC*2+1
    policy = PolicyNet(state_dim, NC, cfg['hidden']).to(dev)
    ckpt_p = f'outputs/experiments/assist15_L{args.L}_seed{args.seed}/{args.method}/best_model.pt'
    policy.load_state_dict(torch.load(ckpt_p, weights_only=True, map_location=dev))

    orig_dkt, _ = load_dkt(cfg, dev)
    orig_kes = KES(orig_dkt, NC, dev)
    alt_model = load_kt(args.arch, NC, dev)
    alt_kes = AltKES(alt_model, args.arch, NC, dev)

    _, _, test_data = load_data(cfg, seed=args.seed)
    eps = eval_v2(policy, orig_kes, alt_kes, test_data, args.L, NC, dev)
    print(f"{args.method} | {args.arch} | seed={args.seed} | N={len(eps)}")
    print(f"  EP (state=orig DKT, exec={args.arch}): {np.mean(eps):+.4f} ± {np.std(eps):.4f}")
