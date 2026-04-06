import hashlib
import random
import numpy as np
import torch

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def simulation_seed(student_id, path_idx: int, persona_idx: int, sim_idx: int) -> int:
    key = f"{student_id}_{path_idx}_{persona_idx}_{sim_idx}"
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16)
