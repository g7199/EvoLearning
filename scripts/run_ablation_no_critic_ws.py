#!/usr/bin/env python3
"""Ablation: skip critic warm-start (Stage 2.5).
Run EVOL-BC with Stage 1.5 disabled to quantify its contribution.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Monkey-patch to skip critic warm-start
import simpath.eval.methods.evolearning_bc as ebc
_orig_train = ebc.EvoLearningBCMethod.train

def train_no_cw(self, train_data, val_data, kes, graph, experts,
                n_episodes=30000, batch_size=128, val_interval=2000, out_dir=None, **kwargs):
    # Save original critic warm-start, replace with no-op
    import types
    # Need to inline-patch the BC method to skip the critic warm-start block
    # Simplest: edit experts to be empty for the cw section by setting a flag
    self._skip_cw = True
    return _orig_train(self, train_data, val_data, kes, graph, experts,
                       n_episodes=n_episodes, batch_size=batch_size,
                       val_interval=val_interval, out_dir=out_dir, **kwargs)

# Better: directly edit the method to skip cw if flag set. But it's not flag-driven yet.
# Implement via patching the F.mse_loss call to skip? Too invasive.
# Cleanest: re-export a patched copy of the method below.

import torch, torch.nn.functional as F, numpy as np, time
from simpath.eval.methods import register_method
from simpath.eval.methods.base import (
    BaseMethod, PolicyNet, make_state_standard, run_ppo_epoch, compute_ep_reward
)
from simpath.eval.kes import FastRollout

@register_method
class EvoLearningBCNoCWMethod(BaseMethod):
    name = "EvoLearning-BC-NoCW"
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
        gamma = 0.99

        # Stage 1: BC (identical to EvoLearning-BC)
        print(f"  [{self.name}] BC on {len(experts)} experts...", flush=True)
        bc_s, bc_a = [], []
        for mastery, tgts, path, ep, *_ in experts:
            for step, action in enumerate(path[:L]):
                bc_s.append(make_state_standard(mastery, tgts, step, L, NC))
                bc_a.append(action)
        bc_s_t = torch.tensor(np.array(bc_s), dtype=torch.float32, device=dev)
        bc_a_t = torch.tensor(bc_a, dtype=torch.long, device=dev)
        bc_opt = torch.optim.Adam(
            list(policy.actor_net.parameters()) + list(policy.pi.parameters()),
            lr=1e-3, weight_decay=1e-4)

        bc_epochs = 2000
        total_steps = bc_epochs + n_episodes
        best_bc = -999; best_bc_state = None
        N_bc = len(bc_s); BC_BATCH = 4096
        for epoch in range(bc_epochs):
            idx = torch.randperm(N_bc, device=dev)
            for i in range(0, N_bc, BC_BATCH):
                bi = idx[i:i + BC_BATCH]
                h = policy.actor_net(bc_s_t[bi])
                lo = policy.pi(h)
                loss = F.cross_entropy(lo, bc_a_t[bi])
                bc_opt.zero_grad(); loss.backward(); bc_opt.step()
            if (epoch + 1) % 200 == 0:
                eps = kes.evaluate_policy_batch(val_data, policy, L)
                vep = np.mean(eps); mk = ''
                if vep > best_bc:
                    best_bc = vep; best_bc_state = {k: v.clone() for k, v in policy.state_dict().items()}; mk = ' ***'
                print(f"    BC Ep {epoch+1}/{bc_epochs} | Val={vep:+.4f}{mk}", flush=True)
                self._update_progress(out_dir, epoch+1, total_steps, val=vep)
                policy.train()
        if best_bc_state:
            policy.load_state_dict(best_bc_state)
        print(f"  [{self.name}] BC Best Val = {best_bc:+.4f}", flush=True)

        # *** Stage 2.5 (Critic warm-start) SKIPPED ***
        print(f"  [{self.name}] Critic warm-start SKIPPED (ablation)", flush=True)

        # Stage 3: PPO (identical)
        print(f"  [{self.name}] PPO fine-tune ({n_episodes} ep)...", flush=True)
        lr = 1e-4; clip = 0.15; ent = 0.03
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_episodes, eta_min=1e-6)
        bv = -999; t0 = time.time()
        rollout = FastRollout(kes.dkt, NC, dev, kes.max_hist)

        for ep_i in range(n_episodes):
            indices = np.random.randint(0, len(train_data), size=batch_size)
            batch = [train_data[i] for i in indices]
            hc_b = [s[0] for s in batch]; hr_b = [s[1] for s in batch]; tgts_b = [s[2] for s in batch]

            init_mastery_b = rollout.init_batch(hc_b, hr_b, tgts_b, L)
            es_b = [init_mastery_b[i][tgts_b[i]].copy() for i in range(batch_size)]
            used_b = [set() for _ in range(batch_size)]
            all_s, all_sc, all_a, all_lp, all_v, all_vm, all_rew = [], [], [], [], [], [], []

            for step in range(L):
                s_actor, s_critic = rollout.make_states(step)
                vm_np = np.ones((batch_size, NC), dtype=np.float32)
                for i in range(batch_size):
                    for c in used_b[i]: vm_np[i, c] = 0
                vm = torch.from_numpy(vm_np).to(dev)
                logits, vals = policy(s_actor, s_critic, vm)
                probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample(); lps = dist.log_prob(actions)
                anp = actions.cpu().numpy()
                for i in range(batch_size): used_b[i].add(anp[i])
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
            run_ppo_epoch(policy, st, at, olp, ret, ov, vm_t, opt, clip, ent, critic_states=sct)
            sched.step()

            if (ep_i + 1) % val_interval == 0:
                eps = kes.evaluate_policy_batch(val_data, policy, L)
                vep = np.mean(eps); mk = ''
                if vep > bv:
                    bv = vep; self._best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                    mk = ' *** SAVED'
                print(f"  [{self.name}] Ep {ep_i+1}/{n_episodes} | Val={vep:+.4f}{mk} | {time.time()-t0:.0f}s", flush=True)
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--n_episodes', type=int, default=10000)
    args = p.parse_args()
    from simpath.eval.config import ExperimentConfig
    from simpath.eval.runner import run_single_method
    config = ExperimentConfig(
        dataset=args.dataset, method='EvoLearning-BC-NoCW',
        L=args.L, seed=args.seed, gpu=args.gpu,
        save_dir='outputs/ablation_no_cw',
        n_episodes=args.n_episodes
    )
    run_single_method(config, 'EvoLearning-BC-NoCW', f'cuda:{args.gpu}')
