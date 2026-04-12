"""
DLELP — P-Agent (PPO) + S-Agent (plateau detection).
Paper: arXiv 2506.22303.
State: st = ht-1 ⊕ G (mastery + target_mask, NC*2 dims, no step).
Action space: A*-based candidate generation.
S-Agent: τ=0.001 plateau threshold, inserts similar concept (replaces P-Agent turn).
LR=1e-3, clip=0.2, ent=0.05.
FIXED: S-Agent steps excluded from PPO loss, path length enforced.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from simpath.eval.methods import register_method
from simpath.eval.methods.base import (
    BaseMethod, PolicyNet, make_state_dlelp, run_ppo_epoch_masked, compute_ep_reward
)
from simpath.eval.graph import astar_candidates

S_THRESHOLD = 0.001  # paper: τ = 0.001


@register_method
class DLELPMethod(BaseMethod):
    name = "DLELP"
    needs_graph = True
    needs_training = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_dim = self.num_c * 2  # paper: st = ht-1 ⊕ G
        self.policy = PolicyNet(state_dim, self.num_c, self.hidden).to(self.device)
        self.graph = None

    def _make_valid_mask(self, candidates, used):
        vm = np.zeros(self.num_c, dtype=np.float32)
        for c in candidates:
            if c not in used: vm[c] = 1.0
        if vm.sum() == 0:
            for c in range(self.num_c):
                if c not in used: vm[c] = 1.0
        return vm

    def train(self, train_data, val_data, kes, graph, experts,
              n_episodes=30000, batch_size=128, val_interval=2000, out_dir=None, **kwargs):
        self.graph = graph
        policy = self.policy
        NC = self.num_c; L = self.L; dev = self.device
        lr = 1e-3; clip = 0.2; ent_coef = 0.05; gamma = 0.99
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)
        bv = -999; t0 = time.time()

        for ep_i in range(n_episodes):
            indices = np.random.randint(0, len(train_data), size=batch_size)
            batch = [train_data[i] for i in indices]
            hc_b = [s[0] for s in batch]; hr_b = [s[1] for s in batch]; tgts_b = [s[2] for s in batch]
            mastery_b = kes.batch_mastery(hc_b, hr_b)
            es_b = [mastery_b[i][tgts_b[i]].copy() for i in range(batch_size)]
            sc_b = [list(hc_b[i]) for i in range(batch_size)]
            sr_b = [list(hr_b[i]) for i in range(batch_size)]
            used_b = [set() for _ in range(batch_size)]
            prev_mastery_b = [None] * batch_size

            all_s, all_a, all_lp, all_v, all_vm, all_rew = [], [], [], [], [], []
            all_ppo_mask = []  # 1.0 for P-Agent steps, 0.0 for S-Agent overrides

            for step in range(L):
                sb, vmb = [], []
                ppo_mask_step = np.ones(batch_size, dtype=np.float32)
                final_actions = np.zeros(batch_size, dtype=np.int64)

                for i in range(batch_size):
                    # S-Agent: check plateau
                    s_agent_action = None
                    if step > 0 and prev_mastery_b[i] is not None:
                        last_added = [c for c in sc_b[i][len(hc_b[i]):]]
                        if last_added:
                            last_c = last_added[-1]
                            imp = mastery_b[i][last_c] - prev_mastery_b[i][last_c]
                            if imp < S_THRESHOLD:
                                sims = graph.get_similar(last_c, top_k=10)
                                for sc in sims:
                                    if sc not in used_b[i] and mastery_b[i][sc] < 0.7:
                                        s_agent_action = sc
                                        break

                    if s_agent_action is not None:
                        # S-Agent takes over: insert similar concept
                        final_actions[i] = s_agent_action
                        ppo_mask_step[i] = 0.0  # exclude from PPO loss
                        # Still need state/vmask for tensor shape
                        state = make_state_dlelp(mastery_b[i], tgts_b[i], NC)
                        cands = astar_candidates(tgts_b[i], mastery_b[i], graph, NC, used_b[i])
                        vm = self._make_valid_mask(cands, used_b[i])
                    else:
                        # P-Agent selects
                        state = make_state_dlelp(mastery_b[i], tgts_b[i], NC)
                        cands = astar_candidates(tgts_b[i], mastery_b[i], graph, NC, used_b[i])
                        vm = self._make_valid_mask(cands, used_b[i])

                    sb.append(torch.tensor(state, dtype=torch.float32, device=dev))
                    vmb.append(torch.tensor(vm, dtype=torch.float32, device=dev))

                st_b = torch.stack(sb); vm_b = torch.stack(vmb)
                logits, vals = policy(st_b, vm_b)
                probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
                dist = torch.distributions.Categorical(probs)
                p_actions = dist.sample(); lps = dist.log_prob(p_actions)

                # Apply S-Agent overrides
                anp = p_actions.cpu().numpy()
                for i in range(batch_size):
                    if ppo_mask_step[i] == 0.0:
                        anp[i] = final_actions[i]
                    else:
                        final_actions[i] = anp[i]

                prev_mastery_b = [mastery_b[i].copy() for i in range(batch_size)]
                for i in range(batch_size):
                    a = anp[i]; sc_b[i].append(a)
                    sr_b[i].append(1 if mastery_b[i][a] > 0.5 else 0); used_b[i].add(a)
                mastery_b = kes.batch_mastery(sc_b, sr_b)

                rewards = np.zeros(batch_size, dtype=np.float32)
                if step == L - 1:
                    rewards = compute_ep_reward(mastery_b, tgts_b, es_b, batch_size)

                all_s.append(st_b)
                all_a.append(torch.tensor(anp, dtype=torch.long, device=dev))
                all_lp.append(lps); all_v.append(vals); all_vm.append(vm_b)
                all_rew.append(rewards); all_ppo_mask.append(ppo_mask_step)

            # PPO update with mask
            G = np.zeros(batch_size, dtype=np.float32); rets = [None] * L
            for t in reversed(range(L)): G = all_rew[t] + gamma * G; rets[t] = G.copy()
            st = torch.cat(all_s); at = torch.cat(all_a); olp = torch.cat(all_lp).detach()
            ret = torch.tensor(np.concatenate(rets), dtype=torch.float32, device=dev)
            ov = torch.cat(all_v).detach(); vm_t = torch.cat(all_vm)
            ppo_m = torch.tensor(np.concatenate(all_ppo_mask), dtype=torch.float32, device=dev)
            run_ppo_epoch_masked(policy, st, at, olp, ret, ov, vm_t, ppo_m, opt, clip, ent_coef)
            sched.step()

            if (ep_i + 1) % 5 == 0:
                self._update_progress(out_dir, ep_i+1, n_episodes, reward=all_rew[-1].mean())
            if (ep_i + 1) % val_interval == 0:
                policy.eval()
                eps = kes.evaluate_batch(val_data, lambda m, t: self.predict(m, t))
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

    def predict(self, mastery, targets):
        """Inference with S-Agent: plateau → insert similar, skip P-Agent that step."""
        self.policy.eval()
        path, used = [], set()
        prev_m = None
        for step in range(self.L):
            if len(path) >= self.L:
                break

            # S-Agent: check plateau on previous concept
            s_agent_fired = False
            if step > 0 and prev_m is not None and len(path) > 0:
                last_c = path[-1]
                if (mastery[last_c] - prev_m[last_c]) < S_THRESHOLD:
                    sims = self.graph.get_similar(last_c, top_k=10)
                    for sc in sims:
                        if sc not in used and mastery[sc] < 0.7:
                            path.append(sc)
                            used.add(sc)
                            s_agent_fired = True
                            break

            if s_agent_fired:
                continue  # S-Agent replaced P-Agent this step

            if len(path) >= self.L:
                break

            # P-Agent selects
            prev_m = mastery.copy()
            cands = astar_candidates(targets, mastery, self.graph, self.num_c, used)
            state = make_state_dlelp(mastery, targets, self.num_c)
            vm = self._make_valid_mask(cands, used)
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            vm_t = torch.tensor(vm, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                lo, _ = self.policy(s_t, vm_t)
            a = lo.argmax().item()
            path.append(a)
            used.add(a)
        return path[:self.L]

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))
