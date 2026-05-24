#!/usr/bin/env python3
"""
Train pykt DKT on Junyi dataset + build DKT influence graph.
CSEAL split: 50% dataSim (DKT training), 50% dataOff (RL).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
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
args = parser.parse_args()
DEVICE = args.device
NUM_C = 39  # Junyi has 39 KCs
MAX_LEN = 200
BATCH = 128
EPOCHS = 100
PATIENCE = 10

print(f"{'='*60}")
print(f"Training pykt DKT on Junyi (NUM_C={NUM_C})")
print(f"{'='*60}")

# ═══ Load Junyi data ═══
with open('data/processed/junyi/junyi_processed.pkl', 'rb') as f:
    data = pickle.load(f)

kc_list = data['kc_list']  # 39 KC names
kc_to_idx = data['kc_to_idx']  # KC name → index (0-38)
# skill_map: maps KC name → concept index (for compatibility with base.py format)
skill_map = {kc: idx for kc, idx in kc_to_idx.items()}

print(f"  Students: {len(data['students'])}, KCs: {len(kc_list)}")
print(f"  KC examples: {kc_list[:5]}")

# ═══ CSEAL split: 50% dataSim, 50% dataOff ═══
np.random.seed(999)
perm = np.random.permutation(len(data['students']))
sim_students = [data['students'][i] for i in perm[:len(perm)//2]]
off_students = [data['students'][i] for i in perm[len(perm)//2:]]
print(f"  dataSim: {len(sim_students)}, dataOff: {len(off_students)}")

# ═══ Build training sequences ═══
def build_sequences(students, max_len=MAX_LEN):
    """Extract concept-response sequences for DKT training."""
    seqs = []
    for s in students:
        concepts, corrects = [], []
        for h in s['train'] + s.get('val', []) + s.get('rec_input', []):
            for kc in h['kc_ids']:
                if kc in kc_to_idx:
                    concepts.append(kc_to_idx[kc])
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

# ═══ Init pykt DKT ═══
dkt = init_model('dkt', {'emb_size': 128, 'dropout': 0.2},
                 {'num_q': NUM_C, 'num_c': NUM_C, 'emb_path': ''}, 'qid').to(DEVICE)
n_params = sum(p.numel() for p in dkt.parameters())
print(f"  Model params: {n_params:,}")

optimizer = torch.optim.Adam(dkt.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
criterion = nn.BCELoss()

best_auc = 0
best_state = None
wait = 0

# ═══ Training ═══
print(f"\n{'─'*60}")
for epoch in range(EPOCHS):
    dkt.train()
    np.random.shuffle(train_seqs)
    total_loss, n_batch = 0, 0

    for i in range(0, len(train_seqs), BATCH):
        batch = train_seqs[i:i + BATCH]
        if not batch:
            continue
        max_len = max(len(b[0]) for b in batch)
        if max_len < 2:
            continue

        c_b = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
        r_b = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
        m_b = torch.zeros(len(batch), max_len, dtype=torch.float, device=DEVICE)
        for j, (c, r) in enumerate(batch):
            Li = len(c)
            c_b[j, :Li] = torch.tensor(c, dtype=torch.long)
            r_b[j, :Li] = torch.tensor(r, dtype=torch.long)
            m_b[j, :Li] = 1.0

        # pykt DKT forward: output is already sigmoid [B, T, NUM_C]
        out = dkt(c_b, r_b)
        # Predict t+1 from state at t
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

    # Validation
    dkt.eval()
    all_pred, all_label = [], []
    with torch.no_grad():
        for i in range(0, len(val_seqs), BATCH):
            batch = val_seqs[i:i + BATCH]
            max_len = max(len(b[0]) for b in batch)
            if max_len < 2:
                continue
            c_b = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
            r_b = torch.zeros(len(batch), max_len, dtype=torch.long, device=DEVICE)
            m_b = torch.zeros(len(batch), max_len, dtype=torch.float, device=DEVICE)
            for j, (c, r) in enumerate(batch):
                Li = len(c)
                c_b[j, :Li] = torch.tensor(c, dtype=torch.long)
                r_b[j, :Li] = torch.tensor(r, dtype=torch.long)
                m_b[j, :Li] = 1.0
            out = dkt(c_b, r_b)  # already sigmoid
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

# Save checkpoint (same format as ASSIST09)
os.makedirs('outputs/checkpoints', exist_ok=True)
torch.save({
    'model': dkt.state_dict(),
    'num_c': NUM_C,
    'skill_map': {kc: kc_to_idx[kc] for kc in kc_list},
    'best_auc': best_auc,
    'emb_size': 128,
    'dataset': 'junyi',
}, 'outputs/checkpoints/pykt_dkt_alt_junyi.pt')
print(f"  Saved: outputs/checkpoints/pykt_dkt_alt_junyi.pt")

# ═══ Sanity Check ═══
print(f"\n{'='*60}")
print("KES Sanity Check (Junyi)")
print(f"{'='*60}")

def kes_mastery(hc, hr):
    if not hc:
        return np.full(NUM_C, 0.5)
    q = torch.tensor([hc[-MAX_LEN:]], dtype=torch.long, device=DEVICE)
    r = torch.tensor([hr[-MAX_LEN:]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return dkt(q, r)[0, -1, :].cpu().numpy()  # already sigmoid

# Test responsiveness on concept 5
print("Concept 5 — correct answers:")
hc, hr = [], []
for step in range(8):
    m = kes_mastery(hc, hr)
    print(f"  Step {step}: P(c5)={m[5]:.4f}, mean={m.mean():.4f}")
    hc.append(5); hr.append(1)

print("\nConcept 5 — wrong answers:")
hc, hr = [], []
for step in range(8):
    m = kes_mastery(hc, hr)
    print(f"  Step {step}: P(c5)={m[5]:.4f}, mean={m.mean():.4f}")
    hc.append(5); hr.append(0)

# ═══ Build DKT Influence Graph ═══
print(f"\n{'='*60}")
print("Building DKT Influence Graph (Junyi)")
print(f"{'='*60}")

# Measure cross-concept influence: practice concept A → effect on concept B
influence = np.zeros((NUM_C, NUM_C), dtype=np.float32)

# For each concept pair, simulate practice and measure mastery change
n_test = min(200, len(off_students))
test_students_sample = off_students[:n_test]

print(f"  Computing influence matrix ({NUM_C}x{NUM_C}) on {n_test} students...")

for ci in range(NUM_C):
    if (ci + 1) % 10 == 0:
        print(f"    Concept {ci+1}/{NUM_C}...", flush=True)

    for si, s in enumerate(test_students_sample):
        # Get base history
        hc, hr = [], []
        for h in (s['train'] + s.get('rec_input', []))[-MAX_LEN:]:
            for kc in h['kc_ids']:
                if kc in kc_to_idx:
                    hc.append(kc_to_idx[kc])
                    hr.append(h['correct'])
                    break

        # Get baseline mastery
        m_before = kes_mastery(hc, hr)

        # Practice concept ci (correct answer)
        hc2 = hc + [ci]
        hr2 = hr + [1]
        m_after = kes_mastery(hc2, hr2)

        # Record influence on all concepts
        for cj in range(NUM_C):
            influence[ci, cj] += (m_after[cj] - m_before[cj])

# Average over students
influence /= n_test

# Get concept names
names = {i: kc_list[i] for i in range(NUM_C)}

# Save influence graph
graph_data = {
    'influence': influence,
    'names': names,
    'dataset': 'junyi',
    'n_concepts': NUM_C,
    'n_students_sampled': n_test,
}
with open('outputs/concept_graph_dkt_junyi.pkl', 'wb') as f:
    pickle.dump(graph_data, f)

# Stats
np.fill_diagonal(influence, 0)
thr = 0.25
prereq_count = (influence > thr).sum()
sim_count = 0
for a in range(NUM_C):
    for b in range(a + 1, NUM_C):
        if influence[a, b] > thr * 0.5 and influence[b, a] > thr * 0.5:
            sim_count += 1

print(f"\n  Influence matrix saved: outputs/concept_graph_dkt_junyi.pkl")
print(f"  Threshold={thr}: {prereq_count} prereq edges, {sim_count} sim edges")
print(f"  Influence range: [{influence.min():.4f}, {influence.max():.4f}]")
print(f"  Mean abs influence: {np.abs(influence).mean():.4f}")

# Try different thresholds
for t in [0.05, 0.10, 0.15, 0.20, 0.25]:
    n_p = (influence > t).sum()
    n_s = sum(1 for a in range(NUM_C) for b in range(a+1, NUM_C)
              if influence[a, b] > t*0.5 and influence[b, a] > t*0.5)
    print(f"  thr={t:.2f}: {n_p} prereq, {n_s} sim")

print(f"\n{'='*60}")
print("Done! Next: run EvoLearning pipeline on Junyi")
print(f"{'='*60}")
