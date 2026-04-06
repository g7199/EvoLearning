"""PyTorch Dataset for KT model training."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict


class KTDataset(Dataset):
    """
    Sequences of (question_idx, kc_idx, correct) for KT training.
    Pads/truncates to max_seq_len.
    """

    def __init__(self, students: List[dict], split: str = "train",
                 max_seq_len: int = 200, n_questions: int = 1, n_kcs: int = 1):
        self.max_seq_len = max_seq_len
        self.n_questions = n_questions
        self.n_kcs = n_kcs
        self.sequences = []

        for s in students:
            interactions = s[split]
            if len(interactions) < 3:
                continue

            q_ids = [h["question_idx"] for h in interactions]
            # Use first KC index for simplicity (multi-KC handled via sum)
            kc_ids = [h["kc_idxs"][0] if h["kc_idxs"] else 0 for h in interactions]
            corrects = [h["correct"] for h in interactions]

            # Chunk into max_seq_len windows (sliding)
            for start in range(0, len(q_ids), max_seq_len):
                end = min(start + max_seq_len, len(q_ids))
                if end - start < 3:
                    continue
                self.sequences.append({
                    "q_ids": q_ids[start:end],
                    "kc_ids": kc_ids[start:end],
                    "corrects": corrects[start:end],
                    "length": end - start,
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        L = seq["length"]

        q = np.zeros(self.max_seq_len, dtype=np.int64)
        kc = np.zeros(self.max_seq_len, dtype=np.int64)
        r = np.zeros(self.max_seq_len, dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)

        q[:L] = seq["q_ids"]
        kc[:L] = seq["kc_ids"]
        r[:L] = seq["corrects"]
        mask[:L] = 1.0

        return {
            "q_ids": torch.tensor(q),
            "kc_ids": torch.tensor(kc),
            "corrects": torch.tensor(r),
            "mask": torch.tensor(mask),
            "length": L,
        }
