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
        """Batch mastery [B, num_c]. Properly trims to max_hist."""
        B = len(hc_list)
        trimmed_hc = [h[-self.max_hist:] for h in hc_list]
        trimmed_hr = [h[-self.max_hist:] for h in hr_list]
        ml = max((len(h) for h in trimmed_hc), default=1)
        ml = max(ml, 1)
        q = torch.zeros(B, ml, dtype=torch.long, device=self.device)
        r = torch.zeros(B, ml, dtype=torch.long, device=self.device)
        for i in range(B):
            Li = len(trimmed_hc[i])
            if Li > 0:
                q[i, :Li] = torch.tensor(trimmed_hc[i], dtype=torch.long)
                r[i, :Li] = torch.tensor(trimmed_hr[i], dtype=torch.long)
        with torch.no_grad():
            out = self.dkt(q, r)
        res = np.zeros((B, self.num_c), dtype=np.float32)
        for i in range(B):
            t = len(trimmed_hc[i]) - 1
            res[i] = out[i, t].cpu().numpy() if t >= 0 else 0.5
        return res

    def evaluate(self, hc: List[int], hr: List[int],
                 path: List[int], targets: List[int]) -> float:
        """Evaluate path via deterministic KES. Returns EP score.
        EP = mean((Ee-Es) / max(1-Es, 0.01)) over all targets.
        """
        es = self.mastery(hc, hr)[targets]
        sc, sr = list(hc), list(hr)
        for c in path:
            m = self.mastery(sc, sr)
            sc.append(c)
            sr.append(1 if m[c] > 0.5 else 0)
        ee = self.mastery(sc, sr)[targets]
        vals = [(ee[i] - es[i]) / (1 - es[i])
                for i in range(len(targets)) if 1 - es[i] > 0.01]
        return float(np.mean(vals)) if vals else 0.0

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
