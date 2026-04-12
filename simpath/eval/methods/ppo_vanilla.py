"""PPO vanilla — standard PPO from random init."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from simpath.eval.methods import register_method
from simpath.eval.methods.base import (
    BaseMethod, PolicyNet, make_state_standard, run_ppo_epoch, compute_ep_reward
)


@register_method
class PPOVanillaMethod(BaseMethod):
    name = "PPO-vanilla"
    needs_training = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dim = self.num_c * 2 + 1
        self.policy = PolicyNet(state_dim, self.num_c, self.hidden).to(self.device)

    def train(self, train_data, val_data, kes, graph, experts,
              n_episodes=30000, batch_size=128, val_interval=2000, out_dir=None, **kwargs):
        policy = self.policy
        NC = self.num_c; L = self.L; dev = self.device
        lr = 3e-4; clip = 0.2; ent = 0.05; gamma = 0.99
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)
        bv = -999; t0 = time.time()

        for ep_i in range(n_episodes):
            indices = np.random.randint(0, len(train_data), size=batch_size)
            batch = [train_data[i] for i in indices]
            hc_b = [s[0] for s in batch]; hr_b = [s[1] for s in batch]; tgts_b = [s[2] for s in batch]
            init_mastery_b = kes.batch_mastery(hc_b, hr_b)  # h0: fixed for state
            es_b = [init_mastery_b[i][tgts_b[i]].copy() for i in range(batch_size)]
            sc_b = [list(hc_b[i]) for i in range(batch_size)]
            sr_b = [list(hr_b[i]) for i in range(batch_size)]
            sim_mastery_b = init_mastery_b.copy()  # ht: evolves for reward
            used_b = [set() for _ in range(batch_size)]
            all_s, all_a, all_lp, all_v, all_vm, all_rew = [], [], [], [], [], []

            for step in range(L):
                sb, vmb = [], []
                for i in range(batch_size):
                    # State uses h0 (initial mastery, no peeking)
                    s = torch.tensor(make_state_standard(init_mastery_b[i], tgts_b[i], step, L, NC),
                                     dtype=torch.float32, device=dev)
                    vm = torch.ones(NC, device=dev)
                    for c in used_b[i]: vm[c] = 0
                    sb.append(s); vmb.append(vm)
                st_b = torch.stack(sb); vm_b = torch.stack(vmb)
                logits, vals = policy(st_b, vm_b)
                probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample(); lps = dist.log_prob(actions)
                anp = actions.cpu().numpy()
                for i in range(batch_size):
                    a = anp[i]; sc_b[i].append(a)
                    sr_b[i].append(1 if sim_mastery_b[i][a] > 0.5 else 0); used_b[i].add(a)
                sim_mastery_b = kes.batch_mastery(sc_b, sr_b)  # ht for reward only
                rewards = np.zeros(batch_size, dtype=np.float32)
                if step == L - 1:
                    rewards = compute_ep_reward(sim_mastery_b, tgts_b, es_b, batch_size)
                all_s.append(st_b); all_a.append(actions); all_lp.append(lps)
                all_v.append(vals); all_vm.append(vm_b); all_rew.append(rewards)

            G = np.zeros(batch_size, dtype=np.float32); rets = [None] * L
            for t in reversed(range(L)): G = all_rew[t] + gamma * G; rets[t] = G.copy()
            st = torch.cat(all_s); at = torch.cat(all_a); olp = torch.cat(all_lp).detach()
            ret = torch.tensor(np.concatenate(rets), dtype=torch.float32, device=dev)
            ov = torch.cat(all_v).detach(); vm_t = torch.cat(all_vm)
            run_ppo_epoch(policy, st, at, olp, ret, ov, vm_t, opt, clip, ent)
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
                # Save periodic checkpoint
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
            with torch.no_grad(): lo, _ = self.policy(s, vm)
            a = lo.argmax().item(); path.append(a); used.add(a)
        return path

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
