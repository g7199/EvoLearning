"""Dataset loading with seed-controlled splits."""
import numpy as np
import pickle
from typing import List, Tuple, Optional


DATA_SEED = 42  # Fixed seed for data split — NEVER changes across experiments


def load_data(dataset_config: dict, seed: int = None, max_hist: int = 200):
    """
    Load and split dataset. Returns (train_data, val_data, test_data).
    Each item: (history_concepts, history_responses, target_concepts).

    Data split uses FIXED seed (DATA_SEED=42) — same train/val/test across
    all experiments. The `seed` parameter is ignored for splitting (kept for
    backward compatibility).

    Split: 50% dataSim (DKT) / 50% dataOff (RL)
    dataOff: 80% train / 20% test
    test: 50% val / 50% test
    """
    with open(dataset_config['data_path'], 'rb') as f:
        data = pickle.load(f)

    # Build skill mapping
    num_c = dataset_config['num_c']
    if 'skill_map' in dataset_config:
        skill_map = dataset_config['skill_map']
    else:
        # Load from DKT checkpoint
        import torch
        ckpt = torch.load(dataset_config['dkt_path'], weights_only=False, map_location='cpu')
        skill_map = ckpt['skill_map']

    # kc_to_c mapping: kc_ids in student data are strings, skill_map keys vary
    kc_list = data['kc_list']
    kc_to_c = {}
    for kc in kc_list:
        # Try direct match first (for Junyi where skill_map keys are strings)
        if kc in skill_map:
            kc_to_c[kc] = skill_map[kc]
        # Try int conversion (for ASSIST09 where skill_map keys are numpy.int64)
        else:
            try:
                ik = int(kc)
                if ik in skill_map:
                    kc_to_c[kc] = skill_map[ik]
            except (ValueError, TypeError):
                pass

    # CSEAL split: 50% dataSim, 50% dataOff (FIXED seed for reproducibility)
    np.random.seed(DATA_SEED)
    perm = np.random.permutation(len(data['students']))
    off = [data['students'][i] for i in perm[len(perm) // 2:]]

    # 80/20 train/test
    rl_train_raw = off[:int(len(off) * 0.8)]
    rl_test_raw = off[int(len(off) * 0.8):]

    # Split test into val/test (FIXED seed — same split for ALL experiments)
    np.random.seed(DATA_SEED + 81)  # = 123
    tp = np.random.permutation(len(rl_test_raw))
    val_size = len(rl_test_raw) // 2
    val_raw = [rl_test_raw[i] for i in tp[:val_size]]
    test_raw = [rl_test_raw[i] for i in tp[val_size:]]

    def prep(s):
        tset = set()
        for h in s['held_out']:
            for kc in h['kc_ids']:
                if kc in kc_to_c:
                    tset.add(kc_to_c[kc])
        targets = list(tset)[:5]
        if not targets:
            return None
        hc, hr = [], []
        for h in (s['train'] + s['rec_input'])[-max_hist:]:
            for kc in h['kc_ids']:
                if kc in kc_to_c:
                    hc.append(kc_to_c[kc])
                    hr.append(h['correct'])
                    break
        return hc, hr, targets

    train_data = [p for s in rl_train_raw if (p := prep(s)) is not None]
    val_data = [p for s in val_raw if (p := prep(s)) is not None]
    test_data = [p for s in test_raw if (p := prep(s)) is not None]

    return train_data, val_data, test_data
