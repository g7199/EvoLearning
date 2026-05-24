#!/usr/bin/env python3
"""Train SAINT on ASSIST15."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn, numpy as np, pickle, warnings
warnings.filterwarnings('ignore')
from pykt.models import init_model
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
DEV = args.device
MAX_LEN = 200
EMB = 256; N_HEADS = 8; N_BLOCKS = 2; DROPOUT = 0.1; LR = 1e-3; BATCH = 128; EPOCHS = 60; PATIENCE = 8

print(f"Training SAINT on ASSIST15 (emb={EMB})")

with open('data/processed/assistments/assistments_processed.pkl', 'rb') as f:
    data = pickle.load(f)
kc_list = data['kc_list']; skill_map = {}; idx = 0; seen = set()
for kc in sorted(set(kc_list)):
    try: ik = int(kc)
    except: ik = kc
    if ik not in seen: skill_map[ik] = idx; idx += 1; seen.add(ik)
kc_to_c = {kc: skill_map[int(kc) if str(kc).lstrip('-').isdigit() else kc]
           for kc in kc_list if (int(kc) if str(kc).lstrip('-').isdigit() else kc) in skill_map}
NUM_C = len(skill_map)
print(f"  KCs: {NUM_C}")

np.random.seed(args.seed)
perm = np.random.permutation(len(data['students']))
sim_students = [data['students'][i] for i in perm[:len(perm)//2]]

def build(students, max_len=MAX_LEN):
    seqs = []
    for s in students:
        c, r = [], []
        for h in s['train'] + s.get('val', []) + s.get('rec_input', []):
            for kc in h['kc_ids']:
                if kc in kc_to_c:
                    c.append(kc_to_c[kc]); r.append(h['correct']); break
        if len(c) < 3: continue
        for st in range(0, len(c), max_len):
            en = min(st + max_len, len(c))
            if en - st >= 3: seqs.append((c[st:en], r[st:en]))
    return seqs

tr = build(sim_students[:int(len(sim_students) * 0.85)])
va = build(sim_students[int(len(sim_students) * 0.85):])
print(f"  Train seqs: {len(tr)}, Val seqs: {len(va)}")

model = init_model('saint', {'seq_len': MAX_LEN, 'emb_size': EMB, 'num_attn_heads': N_HEADS,
                              'n_blocks': N_BLOCKS, 'dropout': DROPOUT},
                   {'num_q': NUM_C, 'num_c': NUM_C, 'emb_path': ''}, 'qid').to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
crit = nn.BCELoss()
best_auc = 0; best_state = None; wait = 0


def pad_to(c, r, m, target_len=MAX_LEN):
    """Pad each sample to target_len (SAINT requires fixed seq_len)."""
    B, T = c.shape
    if T >= target_len:
        return c[:, :target_len], r[:, :target_len], m[:, :target_len]
    pad = target_len - T
    c2 = torch.cat([c, torch.zeros(B, pad, dtype=c.dtype, device=c.device)], dim=1)
    r2 = torch.cat([r, torch.zeros(B, pad, dtype=r.dtype, device=r.device)], dim=1)
    m2 = torch.cat([m, torch.zeros(B, pad, dtype=m.dtype, device=m.device)], dim=1)
    return c2, r2, m2


for ep in range(EPOCHS):
    model.train(); np.random.shuffle(tr); total_loss = 0; n = 0
    for i in range(0, len(tr), BATCH):
        batch = tr[i:i+BATCH]
        if not batch: continue
        ml = max(len(b[0]) for b in batch)
        if ml < 2: continue
        c_b = torch.zeros(len(batch), ml, dtype=torch.long, device=DEV)
        r_b = torch.zeros(len(batch), ml, dtype=torch.long, device=DEV)
        m_b = torch.zeros(len(batch), ml, dtype=torch.float, device=DEV)
        for j, (c, r) in enumerate(batch):
            Li = len(c)
            c_b[j, :Li] = torch.tensor(c, dtype=torch.long)
            r_b[j, :Li] = torch.tensor(r, dtype=torch.long)
            m_b[j, :Li] = 1.0
        c_b, r_b, m_b = pad_to(c_b, r_b, m_b)
        out = model(c_b, c_b, r_b)  # SAINT: (in_ex, in_cat, in_res)
        if isinstance(out, tuple): out = out[0]
        pred = out[:, :-1]
        target = r_b[:, 1:].float()
        mask = m_b[:, 1:]
        loss = crit(pred * mask, target * mask)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total_loss += loss.item(); n += 1
    sched.step()

    model.eval()
    all_pred, all_lab = [], []
    with torch.no_grad():
        for i in range(0, len(va), BATCH):
            batch = va[i:i+BATCH]
            ml = max(len(b[0]) for b in batch)
            if ml < 2: continue
            c_b = torch.zeros(len(batch), ml, dtype=torch.long, device=DEV)
            r_b = torch.zeros(len(batch), ml, dtype=torch.long, device=DEV)
            m_b = torch.zeros(len(batch), ml, dtype=torch.float, device=DEV)
            for j, (c, r) in enumerate(batch):
                Li = len(c)
                c_b[j, :Li] = torch.tensor(c, dtype=torch.long)
                r_b[j, :Li] = torch.tensor(r, dtype=torch.long)
                m_b[j, :Li] = 1.0
            c_b, r_b, m_b = pad_to(c_b, r_b, m_b)
            out = model(c_b, c_b, r_b)
            if isinstance(out, tuple): out = out[0]
            pred = out[:, :-1]; target = r_b[:, 1:].float(); mask = m_b[:, 1:]
            for j in range(len(batch)):
                v = mask[j] > 0
                if v.sum() > 0:
                    all_pred.extend(pred[j][v].cpu().numpy())
                    all_lab.extend(target[j][v].cpu().numpy())
    auc = roc_auc_score(all_lab, all_pred) if len(set(all_lab)) > 1 else 0.5
    print(f"  Ep {ep+1}/{EPOCHS} Loss={total_loss/max(n,1):.4f} AUC={auc:.4f}", flush=True)
    if auc > best_auc:
        best_auc = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stop"); break

model.load_state_dict(best_state)
os.makedirs('outputs/checkpoints', exist_ok=True)
torch.save({'model': model.state_dict(), 'num_c': NUM_C, 'skill_map': skill_map,
            'best_auc': best_auc, 'emb_size': EMB, 'num_attn_heads': N_HEADS,
            'n_blocks': N_BLOCKS, 'dropout': DROPOUT, 'seq_len': MAX_LEN,
            'dataset': 'assist15', 'arch': 'saint'},
           'outputs/checkpoints/saint_pykt_assist15.pt')
print(f"Saved with AUC={best_auc:.4f}")
