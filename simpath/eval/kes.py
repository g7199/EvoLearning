"""Knowledge Evolution Simulator — DKT-based environment."""
import torch
import numpy as np
from typing import List


class KES:
    """KES (Knowledge Evolution Simulator) wrapping a DKT model.
    DKT output is already sigmoid [0,1] — NEVER apply sigmoid again.
    Deterministic threshold: P > 0.5 → correct.
    """

    def __init__(self, dkt, num_c: int, device: str, max_hist: int = 200):
        self.dkt = dkt
        self.num_c = num_c
        self.device = device
        self.max_hist = max_hist

    def mastery(self, hc: List[int], hr: List[int]) -> np.ndarray:
        """Single student mastery vector [num_c]."""
        if not hc:
            return np.full(self.num_c, 0.5, dtype=np.float32)
        q = torch.tensor([hc[-self.max_hist:]], dtype=torch.long, device=self.device)
        r = torch.tensor([hr[-self.max_hist:]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.dkt(q, r)[0, -1, :].cpu().numpy()

    def batch_mastery(self, hc_list: List[List[int]], hr_list: List[List[int]]) -> np.ndarray:
        """Batch mastery [B, num_c]. Single CPU→GPU transfer."""
        B = len(hc_list)
        trimmed_hc = [h[-self.max_hist:] for h in hc_list]
        trimmed_hr = [h[-self.max_hist:] for h in hr_list]
        lens = np.array([len(h) for h in trimmed_hc], dtype=np.int64)
        ml = max(int(lens.max()), 1)

        q_np = np.zeros((B, ml), dtype=np.int64)
        r_np = np.zeros((B, ml), dtype=np.int64)
        for i in range(B):
            Li = lens[i]
            if Li > 0:
                q_np[i, :Li] = trimmed_hc[i]
                r_np[i, :Li] = trimmed_hr[i]

        q = torch.from_numpy(q_np).to(self.device)
        r = torch.from_numpy(r_np).to(self.device)
        with torch.no_grad():
            out = self.dkt(q, r)

        idx = torch.from_numpy(np.maximum(lens - 1, 0)).to(self.device)
        res = out[torch.arange(B, device=self.device), idx].cpu().numpy()
        for i in range(B):
            if lens[i] == 0:
                res[i] = 0.5
        return res

    def evaluate(self, hc: List[int], hr: List[int],
                 path: List[int], targets: List[int]) -> float:
        """Evaluate path via deterministic KES. Returns EP score.
        Aggregate EP (KnowLP): sum(Ee-Es) / sum(1-Es).
        Naturally down-weights near-ceiling targets.
        """
        es = self.mastery(hc, hr)[targets]
        sc, sr = list(hc), list(hr)
        for c in path:
            m = self.mastery(sc, sr)
            sc.append(c)
            sr.append(1 if m[c] > 0.5 else 0)
        ee = self.mastery(sc, sr)[targets]
        denom = (1 - es).sum()
        if denom < 1e-6:
            return 0.0
        return float((ee - es).sum() / denom)

    def evaluate_batch(self, data_split, predict_fn) -> List[float]:
        """Evaluate predict_fn on a data split. Returns list of EP scores.
        predict_fn signature: (mastery, targets, kes, hc, hr) -> path
        """
        eps = []
        for hc, hr, tgts in data_split:
            m = self.mastery(hc, hr)
            path = predict_fn(m, tgts, self, hc, hr)
            eps.append(self.evaluate(hc, hr, path, tgts))
        return eps

    def evaluate_policy_batch(self, data_split, policy, L: int) -> List[float]:
        """Fully GPU-batched evaluation: all students processed in parallel.
        Single batch_mastery for init, L policy forwards on full batch, single batch_mastery for final.
        Used for fast validation during BC/AWR training (no per-student loops).
        """
        import torch.nn.functional as F
        B = len(data_split)
        NC = self.num_c

        hc_list = [d[0] for d in data_split]
        hr_list = [d[1] for d in data_split]
        tgts_list = [d[2] for d in data_split]

        init_m = self.batch_mastery(hc_list, hr_list)
        es_list = [init_m[i][tgts_list[i]].copy() for i in range(B)]

        tmask_np = np.zeros((B, NC), dtype=np.float32)
        for i in range(B):
            tmask_np[i, tgts_list[i]] = 1.0
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
                for c in used[i]:
                    vm_np[i, c] = 0
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
            es = es_list[i]
            ee = sim_m[i][tgts_list[i]]
            denom = (1 - es).sum()
            eps.append(0.0 if denom < 1e-6 else float((ee - es).sum() / denom))
        return eps


class FastRollout:
    """Pre-allocated GPU rollout for RL training. Eliminates per-step Python→GPU overhead."""

    def __init__(self, dkt, num_c: int, device: str, max_hist: int = 200):
        self.dkt = dkt
        self.num_c = num_c
        self.device = device
        self.max_hist = max_hist

    def init_batch(self, hc_list, hr_list, tgts_list, L):
        """Load initial histories onto GPU. Returns init_mastery [B, NC]."""
        B = len(hc_list)
        self.B = B
        self.L = L

        trimmed_hc = [h[-self.max_hist:] for h in hc_list]
        trimmed_hr = [h[-self.max_hist:] for h in hr_list]
        self.init_lens = np.array([len(h) for h in trimmed_hc], dtype=np.int64)
        max_len = int(self.init_lens.max()) + L

        q_np = np.zeros((B, max_len), dtype=np.int64)
        r_np = np.zeros((B, max_len), dtype=np.int64)
        for i in range(B):
            Li = self.init_lens[i]
            if Li > 0:
                q_np[i, :Li] = trimmed_hc[i]
                r_np[i, :Li] = trimmed_hr[i]

        self.hist_q = torch.from_numpy(q_np).to(self.device)
        self.hist_r = torch.from_numpy(r_np).to(self.device)
        self.cur_lens = self.init_lens.copy()
        self._arange = torch.arange(B, device=self.device)

        # Target mask (reused every step)
        tmask = np.zeros((B, self.num_c), dtype=np.float32)
        for i in range(B):
            tmask[i, tgts_list[i]] = 1.0
        self.tmask_t = torch.from_numpy(tmask).to(self.device)

        # Initial mastery
        max_cur = max(int(self.init_lens.max()), 1)
        with torch.no_grad():
            out = self.dkt(self.hist_q[:, :max_cur], self.hist_r[:, :max_cur])
        idx = torch.from_numpy(np.maximum(self.init_lens - 1, 0)).to(self.device)
        init_m = out[self._arange, idx].cpu().numpy()
        for i in range(B):
            if self.init_lens[i] == 0:
                init_m[i] = 0.5
        self.init_mastery = init_m
        self.sim_mastery = init_m.copy()
        return init_m

    def make_states(self, step):
        """Build actor(h0) and critic(ht) state tensors on GPU. Returns (s_actor, s_critic)."""
        sf = torch.full((self.B, 1), step / self.L, dtype=torch.float32, device=self.device)
        s_actor = torch.cat([
            torch.from_numpy(self.init_mastery.astype(np.float32)).to(self.device),
            self.tmask_t, sf], dim=1)
        s_critic = torch.cat([
            torch.from_numpy(self.sim_mastery.astype(np.float32)).to(self.device),
            self.tmask_t, sf], dim=1)
        return s_actor, s_critic

    def step(self, actions_np):
        """Add actions to history, compute responses, advance DKT. Returns new sim_mastery."""
        responses = (self.sim_mastery[np.arange(self.B), actions_np] > 0.5).astype(np.int64)

        pos = torch.from_numpy(self.cur_lens).to(self.device)
        self.hist_q[self._arange, pos] = torch.from_numpy(actions_np.astype(np.int64)).to(self.device)
        self.hist_r[self._arange, pos] = torch.from_numpy(responses).to(self.device)
        self.cur_lens += 1

        max_cur = int(self.cur_lens.max())
        with torch.no_grad():
            out = self.dkt(self.hist_q[:, :max_cur], self.hist_r[:, :max_cur])
        idx = torch.from_numpy(self.cur_lens - 1).to(self.device)
        self.sim_mastery = out[self._arange, idx].cpu().numpy()
        return self.sim_mastery


def load_dkt(dataset_config: dict, device: str):
    """Load pykt DKT model."""
    from pykt.models import init_model
    ckpt = torch.load(dataset_config['dkt_path'], weights_only=False, map_location=device)
    num_c = dataset_config['num_c']
    emb = dataset_config['dkt_emb']
    dkt = init_model('dkt', {'emb_size': emb, 'dropout': 0.2},
                     {'num_q': num_c, 'num_c': num_c, 'emb_path': ''}, 'qid').to(device)
    dkt.load_state_dict(ckpt['model'])
    dkt.eval()
    return dkt, ckpt.get('skill_map', {})
