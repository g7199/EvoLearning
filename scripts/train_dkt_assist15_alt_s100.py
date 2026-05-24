#!/usr/bin/env python3
"""Train pykt DKT on ASSIST15 — pyKT standard config."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from pykt.models import init_model
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=100)
args = parser.parse_args()

DEVICE = args.device
MAX_LEN = 200
EMB_SIZE = 200
DROPOUT = 0.1
LR = 1e-3
BATCH = 256
EPOCHS = 200
PATIENCE = 10

print(f"{'='*60}")
print(f"Training pykt DKT on ASSIST15 (pyKT standard config)")
print(f"  emb={EMB_SIZE}, dropout={DROPOUT}, lr={LR}, batch={BATCH}")
print(f"  device={DEVICE}")
print(f"{'='*60}")

with open('data/processed/assistments/assistments_processed.pkl', 'rb') as f:
    data = pickle.load(f)

ckpt_path = 'outputs/checkpoints/pykt_dkt_alt_s100_assist15.pt'

# Build skill mapping
kc_list = data['kc_list']
skill_map = {}
idx = 0
seen = set()
for kc in sorted(set(kc_list)):
    try:
        ik = int(kc)
    except:
        ik = kc
    if ik not in seen:
        skill_map[ik] = idx
        idx += 1
        seen.add(ik)

kc_to_c = {}
for kc in kc_list:
    try:
        ik = int(kc)
    except:
        ik = kc
    if ik in skill_map:
        kc_to_c[kc] = skill_map[ik]

NUM_C = len(skill_map)
print(f"  KCs: {NUM_C}")

# CSEAL split
np.random.seed(args.seed)
perm = np.random.permutation(len(data['students']))
sim_students = [data['students'][i] for i in perm[:len(perm)//2]]

def build_sequences(students, max_len=MAX_LEN):
    seqs = []
    for s in students:
        concepts, corrects = [], []
        for h in s['train'] + s.get('val', []) + s.get('rec_input', []):
            for kc in h['kc_ids']:
                if kc in kc_to_c:
                    concepts.append(kc_to_c[kc])
                    corrects.append(h['correct'])
                    break
        if len(concepts) < 3:
            continue
        for start in range(0, len(concepts), max_len):
            end = min(start + max_len, len(concepts))
            if end - start >= 3:
                seqs.append((concepts[start:end], corrects[start:end]))
    return seqs

train_seqs = build_sequences(sim_students[:int(len(sim_students) * 0.85)])
val_seqs = build_sequences(sim_students[int(len(sim_students) * 0.85):])
print(f"  Train seqs: {len(train_seqs)}, Val seqs: {len(val_seqs)}")

dkt = init_model('dkt', {'emb_size': EMB_SIZE, 'dropout': DROPOUT},
                 {'num_q': NUM_C, 'num_c': NUM_C, 'emb_path': ''}, 'qid').to(DEVICE)
n_params = sum(p.numel() for p in dkt.parameters())
print(f"  Model params: {n_params:,}")

optimizer = torch.optim.Adam(dkt.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.BCELoss()

best_auc = 0
best_state = None
wait = 0

print(f"\n{'─'*60}")
for epoch in range(EPOCHS):
    dkt.train()
    np.random.shuffle(train_seqs)
    total_loss, n_batch = 0, 0

    for i in range(0, len(train_seqs), BATCH):
        batch = train_seqs[i:i + BATCH]
        if not batch: continue
        max_len = max(len(b[0]) for b in batch)
        if max_len < 2: continue

        c_b = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
        r_b = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
        m_b = torch.zeros(len(batch), max_len, dtype=torch.float, device=DEVICE)
        for j, (c, r) in enumerate(batch):
            Li = len(c)
            c_b[j, :Li] = torch.tensor(c, dtype=torch.long)
            r_b[j, :Li] = torch.tensor(r, dtype=torch.long)
            m_b[j, :Li] = 1.0

        out = dkt(c_b, r_b)
        pred = out[:, :-1, :].gather(2, c_b[:, 1:].unsqueeze(-1)).squeeze(-1)
        target = r_b[:, 1:].float()
        mask = m_b[:, 1:]

        loss = criterion(pred * mask, target * mask)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dkt.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1

    scheduler.step()

    dkt.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for i in range(0, len(val_seqs), BATCH):
            batch = val_seqs[i:i + BATCH]
            max_len = max(len(b[0]) for b in batch)
            if max_len < 2: continue
            c_b = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
            r_b = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
            m_b = torch.zeros(len(batch), max_len, dtype=torch.float, device=DEVICE)
            for j, (c, r) in enumerate(batch):
                Li = len(c)
                c_b[j, :Li] = torch.tensor(c, dtype=torch.long)
                r_b[j, :Li] = torch.tensor(r, dtype=torch.long)
                m_b[j, :Li] = 1.0
            out = dkt(c_b, r_b)
            pred = out[:, :-1, :].gather(2, c_b[:, 1:].unsqueeze(-1)).squeeze(-1)
            target = r_b[:, 1:].float()
            mask = m_b[:, 1:]
            for j in range(len(batch)):
                v = mask[j] > 0
                if v.sum() > 0:
                    all_pred.extend(pred[j][v].cpu().numpy())
                    all_label.extend(target[j][v].cpu().numpy())

    auc = roc_auc_score(all_label, all_pred) if len(set(all_label)) > 1 else 0.5
    lr_now = optimizer.param_groups[0]['lr']
    print(f"  Epoch {epoch+1}/{EPOCHS} | Loss={total_loss/max(n_batch,1):.4f} | "
          f"AUC={auc:.4f} | lr={lr_now:.6f}", flush=True)

    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.clone() for k, v in dkt.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

dkt.load_state_dict(best_state)
dkt.eval()
print(f"\n  Best Val AUC: {best_auc:.4f}")

os.makedirs('outputs/checkpoints', exist_ok=True)
torch.save({
    'model': dkt.state_dict(),
    'num_c': NUM_C,
    'skill_map': skill_map,
    'best_auc': best_auc,
    'emb_size': EMB_SIZE,
    'dropout': DROPOUT,
    'config': f'pyKT standard: emb={EMB_SIZE}, dropout={DROPOUT}, lr={LR}',
    'dataset': 'assist15',
}, ckpt_path)
print(f"  Saved: {ckpt_path}")
