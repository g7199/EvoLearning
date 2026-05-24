"""GRU4Rec: GRU autoregressive model trained via behavioral cloning on experts."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from simpath.eval.methods import register_method
from simpath.eval.methods.base import BaseMethod


class GRU4RecModel(nn.Module):
    def __init__(self, num_c, L, embed_dim=64, hidden_dim=256):
        super().__init__()
        self.num_c = num_c
        self.L = L
        self.start_token_id = num_c
        self.concept_emb = nn.Embedding(num_c + 1, embed_dim)
        input_dim = embed_dim + num_c + num_c + 1
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_c)

    def forward(self, action_ids, mastery, target_mask, step_fracs):
        B, T = action_ids.shape
        emb = self.concept_emb(action_ids)
        m_exp = mastery.unsqueeze(1).expand(B, T, -1)
        t_exp = target_mask.unsqueeze(1).expand(B, T, -1)
        sf = step_fracs.unsqueeze(-1)
        inp = torch.cat([emb, m_exp, t_exp, sf], dim=-1)
        out, _ = self.gru(inp)
        return self.head(out)

    def generate(self, mastery, targets, L):
        self.eval()
        device = next(self.parameters()).device
        m_t = torch.tensor(mastery, dtype=torch.float32, device=device).unsqueeze(0)
        tm = torch.zeros(1, self.num_c, dtype=torch.float32, device=device)
        for t in targets:
            tm[0, t] = 1.0
        path, used, h = [], set(), None
        action_id = self.start_token_id
        for step in range(L):
            emb = self.concept_emb(torch.tensor([[action_id]], dtype=torch.long, device=device))
            sf = torch.tensor([[[step / L]]], dtype=torch.float32, device=device)
            inp = torch.cat([emb, m_t.unsqueeze(1), tm.unsqueeze(1), sf], dim=-1)
            out, h = self.gru(inp, h)
            logits = self.head(out[0, 0])
            for c in used:
                logits[c] = -1e9
            action_id = logits.argmax().item()
            path.append(action_id)
            used.add(action_id)
        return path


@register_method
class GRU4RecMethod(BaseMethod):
    name = "GRU4Rec"
    needs_experts = True
    needs_training = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GRU4RecModel(self.num_c, self.L).to(self.device)

    def train(self, train_data, val_data, kes, graph, experts, out_dir=None, **kwargs):
        model = self.model
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        L = self.L
        NC = self.num_c

        all_input, all_mastery, all_tmask, all_target = [], [], [], []
        for mastery, tgts, path, ep, *_ in experts:
            if len(path) < L:
                continue
            all_input.append([NC] + list(path[:L - 1]))
            all_target.append(list(path[:L]))
            tm = np.zeros(NC, dtype=np.float32)
            for t in tgts:
                tm[t] = 1.0
            all_mastery.append(np.array(mastery, dtype=np.float32))
            all_tmask.append(tm)

        N = len(all_input)
        print(f"  [{self.name}] {N} trajectories for training", flush=True)
        gi = torch.tensor(all_input, dtype=torch.long, device=self.device)
        gm = torch.tensor(np.array(all_mastery), dtype=torch.float32, device=self.device)
        gt = torch.tensor(np.array(all_tmask), dtype=torch.float32, device=self.device)
        ga = torch.tensor(all_target, dtype=torch.long, device=self.device)
        gsf = torch.tensor([[s / L for s in range(L)]] * N, dtype=torch.float32, device=self.device)

        n_epochs = 2000
        best_val = -999; best_state = None
        for epoch in range(n_epochs):
            model.train()
            idx = np.random.permutation(N)
            for i in range(0, N, 64):
                bi = idx[i:i + 64]
                logits = model(gi[bi], gm[bi], gt[bi], gsf[bi])
                loss = F.cross_entropy(logits.reshape(-1, NC), ga[bi].reshape(-1))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if (epoch + 1) % 5 == 0:
                self._update_progress(out_dir, epoch+1, n_epochs, reward=loss.item())
            if (epoch + 1) % 200 == 0:
                model.eval()
                eps = kes.evaluate_batch(val_data,
                    lambda m, t, k, hc, hr: self.predict(m, t, k, hc, hr))
                vep = np.mean(eps); mk = ''
                if vep > best_val:
                    best_val = vep
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    mk = ' ***'
                print(f"    [{self.name}] Epoch {epoch+1}/{n_epochs} | Val={vep:+.4f}{mk}", flush=True)
                self._update_progress(out_dir, epoch+1, n_epochs, val=vep)

        if best_state:
            model.load_state_dict(best_state)
            self._best_state = best_state
        print(f"  [{self.name}] Best Val = {best_val:+.4f}", flush=True)

    def predict(self, mastery, targets, kes=None, hc=None, hr=None):
        return self.model.generate(mastery, targets, self.L)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
