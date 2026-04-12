"""EvoLearning — Evo expert generation → BC → PPO fine-tune."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle, os, time
from simpath.eval.methods import register_method
from simpath.eval.methods.base import (
    BaseMethod, PolicyNet, make_state_standard, run_ppo_epoch, compute_ep_reward
)


@register_method
class EvoLearningMethod(BaseMethod):
    name = "EvoLearning"
    needs_experts = True
    needs_training = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dim = self.num_c * 2 + 1
        self.policy = PolicyNet(state_dim, self.num_c, self.hidden).to(self.device)

    def train(self, train_data, val_data, kes, graph, experts,
              n_episodes=30000, batch_size=128, val_interval=2000, out_dir=None, **kwargs):
        policy = self.policy
        NC = self.num_c; L = self.L; dev = self.device

        # ═══ Stage 1: BC on expert trajectories (h0: initial mastery) ═══
        print(f"  [{self.name}] BC on {len(experts)} experts...", flush=True)
        bc_s, bc_a = [], []
        for mastery, tgts, path, ep, *_ in experts:
            for step, action in enumerate(path[:L]):
                bc_s.append(make_state_standard(mastery, tgts, step, L, NC))
                bc_a.append(action)
        bc_s_t = torch.tensor(np.array(bc_s), dtype=torch.float32, device=dev)
        bc_a_t = torch.tensor(bc_a, dtype=torch.long, device=dev)
        bc_opt = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-4)

        # Total = BC(5000) + PPO(n_episodes), so tqdm shows combined progress
        bc_epochs = 5000
        total_steps = bc_epochs + n_episodes

        best_bc = -999; best_bc_state = None
        for epoch in range(bc_epochs):
            idx = np.random.permutation(len(bc_s))
            for i in range(0, len(idx), 512):
                bi = idx[i:i + 512]
                lo, _ = policy(bc_s_t[bi])
                loss = F.cross_entropy(lo, bc_a_t[bi])
                bc_opt.zero_grad(); loss.backward(); bc_opt.step()
            if (epoch + 1) % 5 == 0:
                self._update_progress(out_dir, epoch+1, total_steps, reward=loss.item())
            if (epoch + 1) % 500 == 0:
                policy.eval()
                eps = kes.evaluate_batch(val_data, lambda m, t, k, hc, hr: self.predict(m, t, k, hc, hr))
                vep = np.mean(eps); mk = ''
                if vep > best_bc:
                    best_bc = vep; best_bc_state = {k: v.clone() for k, v in policy.state_dict().items()}; mk = ' ***'
                print(f"    BC Ep {epoch+1}/{bc_epochs} | Val={vep:+.4f}{mk}", flush=True)
                self._update_progress(out_dir, epoch+1, total_steps, val=vep)
                policy.train()

        if best_bc_state:
            policy.load_state_dict(best_bc_state)
        print(f"  [{self.name}] BC Best Val = {best_bc:+.4f}", flush=True)

        # ═══ Stage 2: PPO fine-tune ═══
        print(f"  [{self.name}] PPO fine-tune ({n_episodes} ep)...", flush=True)
        lr = 1e-4; clip = 0.15; ent = 0.03; gamma = 0.99
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-6)
        bv = -999; t0 = time.time()

        for ep_i in range(n_episodes):
            indices = np.random.randint(0, len(train_data), size=batch_size)
            batch = [train_data[i] for i in indices]
            hc_b = [s[0] for s in batch]; hr_b = [s[1] for s in batch]; tgts_b = [s[2] for s in batch]
            init_mastery_b = kes.batch_mastery(hc_b, hr_b)
            es_b = [init_mastery_b[i][tgts_b[i]].copy() for i in range(batch_size)]
            sc_b = [list(hc_b[i]) for i in range(batch_size)]
            sr_b = [list(hr_b[i]) for i in range(batch_size)]
            sim_mastery_b = init_mastery_b.copy()
            used_b = [set() for _ in range(batch_size)]
            all_s, all_sc, all_a, all_lp, all_v, all_vm, all_rew = [], [], [], [], [], [], []

            for step in range(L):
                sb, scb, vmb = [], [], []
                for i in range(batch_size):
                    s = torch.tensor(make_state_standard(init_mastery_b[i], tgts_b[i], step, L, NC),
                                     dtype=torch.float32, device=dev)
                    sc = torch.tensor(make_state_standard(sim_mastery_b[i], tgts_b[i], step, L, NC),
                                      dtype=torch.float32, device=dev)
                    vm = torch.ones(NC, device=dev)
                    for c in used_b[i]: vm[c] = 0
                    sb.append(s); scb.append(sc); vmb.append(vm)
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
            run_ppo_epoch(policy, st, at, olp, ret, ov, vm_t, opt, clip, ent, critic_states=sct)
            sched.step()

            if (ep_i + 1) % 5 == 0:
                self._update_progress(out_dir, bc_epochs + ep_i+1, total_steps, reward=all_rew[-1].mean())
            if (ep_i + 1) % val_interval == 0:
                policy.eval()
                eps = kes.evaluate_batch(val_data, lambda m, t, k, hc, hr: self.predict(m, t, k, hc, hr))
                vep = np.mean(eps); mk = ''
                if vep > bv:
                    bv = vep; self._best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                    mk = ' *** SAVED'
                print(f"  [{self.name}] Ep {ep_i+1}/{n_episodes} | "
                      f"Val({len(val_data)})={vep:+.4f}{mk} | {time.time()-t0:.0f}s", flush=True)
                self._update_progress(out_dir, bc_epochs + ep_i+1, total_steps, val=vep)
                if out_dir:
                    torch.save(policy.state_dict(),
                               os.path.join(out_dir, f'checkpoint_ep{ep_i+1}.pt'))
                policy.train()

        if hasattr(self, '_best_state') and self._best_state:
            policy.load_state_dict(self._best_state)
        print(f"  [{self.name}] Best Val = {bv:+.4f}")

    def predict(self, mastery, targets, kes=None, hc=None, hr=None):
        self.policy.eval()
        path, used = [], set()
        for step in range(self.L):
            s = torch.tensor(make_state_standard(mastery, targets, step, self.L, self.num_c),
                             dtype=torch.float32, device=self.device)
            vm = torch.ones(self.num_c, device=self.device)
            for c in used: vm[c] = 0
            with torch.no_grad(): lo, _ = self.policy(s, None, vm)
            a = lo.argmax().item(); path.append(a); used.add(a)
        return path

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
