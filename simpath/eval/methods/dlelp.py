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
from simpath.eval.kes import FastRollout

S_THRESHOLD = 0.001  # paper Section 6.3: τ = 0.001


def build_graph_candidate_mask(graph, targets, num_c, min_cands=None):
    """Graph-aware candidate set: targets ∪ prereq(targets) ∪ sim(targets) ∪ 2-hop prereq.
    Returns a length-num_c float32 mask (1 if candidate, 0 otherwise).
    If candidates are too few (< min_cands), pads with concepts in order of
    proximity (any graph edge) to ensure feasibility for paths of length L."""
    cands = set(int(t) for t in targets)
    for t in targets:
        # 1-hop prereqs (incoming) and posts (outgoing)
        cands.update(np.where(graph.prereq[:, t] > 0)[0].tolist())
        cands.update(np.where(graph.prereq[t, :] > 0)[0].tolist())
        # similar
        cands.update(np.where(graph.sim[t] > 0)[0].tolist())
    # 2-hop prereqs (prereqs of prereqs)
    one_hop = list(cands)
    for c in one_hop:
        cands.update(np.where(graph.prereq[:, c] > 0)[0].tolist())
    # Pad if still too small
    if min_cands is not None and len(cands) < min_cands:
        # add concepts ranked by total graph degree
        deg = (graph.prereq.sum(0) + graph.prereq.sum(1) + graph.sim.sum(1))
        order = np.argsort(-deg)
        for c in order:
            if c not in cands:
                cands.add(int(c))
                if len(cands) >= min_cands: break
    mask = np.zeros(num_c, dtype=np.float32)
    for c in cands: mask[c] = 1.0
    return mask


@register_method
class DLELPMethod(BaseMethod):
    name = "DLELP"
    needs_graph = True  # graph-aware candidate masking (P-Agent) under our h_0 adaptation
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
        lr = 1e-3; clip = 0.2; ent_coef = 0.01; gamma = 0.99
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-5)
        bv = -999; t0 = time.time()
        rollout = FastRollout(kes.dkt, NC, dev, kes.max_hist)

        # Pre-build target mask template (reuse across episodes for same targets)
        for ep_i in range(n_episodes):
            indices = np.random.randint(0, len(train_data), size=batch_size)
            batch = [train_data[i] for i in indices]
            hc_b = [s[0] for s in batch]; hr_b = [s[1] for s in batch]; tgts_b = [s[2] for s in batch]

            init_mastery_b = rollout.init_batch(hc_b, hr_b, tgts_b, L)
            es_b = [init_mastery_b[i][tgts_b[i]].copy() for i in range(batch_size)]
            used_b = [set() for _ in range(batch_size)]

            # DLELP target mask (NC dims, no step)
            tmask_np = np.zeros((batch_size, NC), dtype=np.float32)
            for i in range(batch_size):
                for t in tgts_b[i]: tmask_np[i, t] = 1.0
            tmask_t = torch.from_numpy(tmask_np).to(dev)

            # Graph-aware candidate mask per learner (constant across the L steps)
            gmask_np = np.zeros((batch_size, NC), dtype=np.float32)
            for i in range(batch_size):
                gmask_np[i] = build_graph_candidate_mask(self.graph, tgts_b[i], NC, min_cands=max(L * 3, 16))

            all_s, all_sc, all_a, all_lp, all_v, all_vm, all_rew = [], [], [], [], [], [], []

            for step in range(L):
                # DLELP state: mastery(NC) + target_mask(NC) = NC*2 (no step)
                s_actor = torch.cat([
                    torch.from_numpy(rollout.init_mastery.astype(np.float32)).to(dev),
                    tmask_t], dim=1)
                s_critic = torch.cat([
                    torch.from_numpy(rollout.sim_mastery.astype(np.float32)).to(dev),
                    tmask_t], dim=1)

                # Visit mask = (graph-related candidates) AND (not yet used).
                # Fallback to all-but-used if every graph candidate has been visited.
                vm_np = gmask_np.copy()
                for i in range(batch_size):
                    for c in used_b[i]: vm_np[i, c] = 0
                    if vm_np[i].sum() < 1:
                        vm_np[i] = 1.0
                        for c in used_b[i]: vm_np[i, c] = 0
                vm = torch.from_numpy(vm_np).to(dev)

                logits, vals = policy(s_actor, s_critic, vm)
                probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample(); lps = dist.log_prob(actions)
                anp = actions.cpu().numpy()

                for i in range(batch_size):
                    used_b[i].add(anp[i])

                sim_mastery_b = rollout.step(anp)

                rewards = np.zeros(batch_size, dtype=np.float32)
                if step == L - 1:
                    rewards = compute_ep_reward(sim_mastery_b, tgts_b, es_b, batch_size)

                all_s.append(s_actor); all_sc.append(s_critic); all_a.append(actions)
                all_lp.append(lps); all_v.append(vals); all_vm.append(vm); all_rew.append(rewards)

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
        """Inference: P-Agent with graph-aware candidate mask (S-Agent disabled in
        h_0-only mode because mastery is fixed across steps, so plateau detection
        triggers falsely; instead, the concept graph constrains the action set)."""
        self.policy.eval()
        gmask = build_graph_candidate_mask(self.graph, targets, self.num_c,
                                           min_cands=max(self.L * 3, 16))
        path, used = [], set()
        for step in range(self.L):
            state = make_state_dlelp(mastery, targets, self.num_c)
            vm = gmask.copy()
            for c in used: vm[c] = 0
            if vm.sum() < 1:
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
