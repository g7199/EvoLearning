"""
DLELP — P-Agent (PPO) + S-Agent (inference-only plateau detection).
Paper: arXiv 2506.22303.

From paper:
- State: st = ht-1 ⊕ G (mastery + target one-hot, NC*2 dims)
- Action: softmax over ALL concepts (no action masking, only used-concept exclusion)
- S-Agent: inference only, τ=0.001, inserts similar concept on plateau
- PPO: lr=0.001, clip=0.2
- Reward: sparse terminal EP only
- "mask" does not appear in the paper — no A*, no graph masking during training
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from simpath.eval.methods import register_method
from simpath.eval.methods.base import (
    BaseMethod, PolicyNet, make_state_dlelp, run_ppo_epoch, compute_ep_reward
)

S_THRESHOLD = 0.001  # paper Section 6.3: τ = 0.001


@register_method
class DLELPMethod(BaseMethod):
    name = "DLELP"
    needs_graph = True  # needed for S-Agent at inference
    needs_training = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dim = self.num_c * 2  # paper: st = ht-1 ⊕ G
        self.policy = PolicyNet(state_dim, self.num_c, self.hidden).to(self.device)
        self.graph = None

    def train(self, train_data, val_data, kes, graph, experts,
              n_episodes=30000, batch_size=128, val_interval=2000, out_dir=None, **kwargs):
        self.graph = graph
        policy = self.policy
        NC = self.num_c; L = self.L; dev = self.device
        # Paper: lr=0.001, clip=0.2. Entropy not specified — use 0.01 (conservative).
        lr = 1e-3; clip = 0.2; ent_coef = 0.01; gamma = 0.99
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)
        bv = -999; t0 = time.time()

        for ep_i in range(n_episodes):
            indices = np.random.randint(0, len(train_data), size=batch_size)
            batch = [train_data[i] for i in indices]
            hc_b = [s[0] for s in batch]; hr_b = [s[1] for s in batch]; tgts_b = [s[2] for s in batch]
            init_mastery_b = kes.batch_mastery(hc_b, hr_b)
            es_b = [init_mastery_b[i][tgts_b[i]].copy() for i in range(batch_size)]
            sc_b = [list(hc_b[i]) for i in range(batch_size)]
            sim_mastery_b = init_mastery_b.copy()
            sr_b = [list(hr_b[i]) for i in range(batch_size)]
            used_b = [set() for _ in range(batch_size)]

            all_s, all_sc, all_a, all_lp, all_v, all_vm, all_rew = [], [], [], [], [], [], []

            for step in range(L):
                sb, scb, vmb = [], [], []
                for i in range(batch_size):
                    state = make_state_dlelp(init_mastery_b[i], tgts_b[i], NC)
                    state_c = make_state_dlelp(sim_mastery_b[i], tgts_b[i], NC)
                    vm = np.ones(NC, dtype=np.float32)
                    for c in used_b[i]: vm[c] = 0
                    sb.append(torch.tensor(state, dtype=torch.float32, device=dev))
                    scb.append(torch.tensor(state_c, dtype=torch.float32, device=dev))
                    vmb.append(torch.tensor(vm, dtype=torch.float32, device=dev))

                st_b = torch.stack(sb); sct_b = torch.stack(scb); vm_b = torch.stack(vmb)
                logits, vals = policy(st_b, sct_b, vm_b)
                probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample(); lps = dist.log_prob(actions)
                anp = actions.cpu().numpy()

                for i in range(batch_size):
                    a = anp[i]; sc_b[i].append(a)
                    sr_b[i].append(1 if sim_mastery_b[i][a] > 0.5 else 0); used_b[i].add(a)
                sim_mastery_b = kes.batch_mastery(sc_b, sr_b)

                rewards = np.zeros(batch_size, dtype=np.float32)
                if step == L - 1:
                    rewards = compute_ep_reward(sim_mastery_b, tgts_b, es_b, batch_size)

                all_s.append(st_b); all_sc.append(sct_b); all_a.append(actions)
                all_lp.append(lps); all_v.append(vals); all_vm.append(vm_b); all_rew.append(rewards)

            G = np.zeros(batch_size, dtype=np.float32); rets = [None] * L
            for t in reversed(range(L)): G = all_rew[t] + gamma * G; rets[t] = G.copy()
            st = torch.cat(all_s); sct = torch.cat(all_sc); at = torch.cat(all_a)
            olp = torch.cat(all_lp).detach()
            ret = torch.tensor(np.concatenate(rets), dtype=torch.float32, device=dev)
            ov = torch.cat(all_v).detach(); vm_t = torch.cat(all_vm)
            run_ppo_epoch(policy, st, at, olp, ret, ov, vm_t, opt, clip, ent_coef, critic_states=sct)
            sched.step()

            if (ep_i + 1) % 5 == 0:
                self._update_progress(out_dir, ep_i+1, n_episodes, reward=all_rew[-1].mean())
            if (ep_i + 1) % val_interval == 0:
                policy.eval()
                eps = kes.evaluate_batch(val_data, lambda m, t, k, hc, hr: self.predict(m, t, k, hc, hr))
                vep = np.mean(eps); mk = ''
                if vep > bv:
                    bv = vep; self._best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                    mk = ' *** SAVED'
                print(f"  [{self.name}] Ep {ep_i+1}/{n_episodes} | "
                      f"Val({len(val_data)})={vep:+.4f}{mk} | {time.time()-t0:.0f}s", flush=True)
                self._update_progress(out_dir, ep_i+1, n_episodes, val=vep)
                if out_dir:
                    torch.save(policy.state_dict(),
                               os.path.join(out_dir, f'checkpoint_ep{ep_i+1}.pt'))
                policy.train()

        if hasattr(self, '_best_state') and self._best_state:
            policy.load_state_dict(self._best_state)
        print(f"  [{self.name}] Best Val = {bv:+.4f}")

    def predict(self, mastery, targets, kes=None, hc=None, hr=None):
        """Inference with S-Agent (h0 mode: no KES during path generation)."""
        self.policy.eval()
        path, used = [], set()
        for step in range(self.L):
            if len(path) >= self.L:
                break

            # S-Agent: check plateau based on initial mastery
            s_agent_fired = False
            if step > 0 and len(path) > 0:
                last_c = path[-1]
                if mastery[last_c] < 0.5 + S_THRESHOLD:
                    sims = self.graph.get_similar(last_c, top_k=10)
                    for s in sims:
                        if s not in used and mastery[s] < 0.7:
                            path.append(s)
                            used.add(s)
                            s_agent_fired = True
                            break

            if s_agent_fired:
                continue

            if len(path) >= self.L:
                break

            # P-Agent: softmax over all concepts (only exclude used)
            state = make_state_dlelp(mastery, targets, self.num_c)
            vm = np.ones(self.num_c, dtype=np.float32)
            for c in used: vm[c] = 0
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            vm_t = torch.tensor(vm, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                lo, _ = self.policy(s_t, None, vm_t)
            a = lo.argmax().item()
            path.append(a)
            used.add(a)
        return path[:self.L]

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
