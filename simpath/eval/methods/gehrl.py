"""
GEHRL (Chen et al., CIKM 2023) — Hierarchical RL.
High-level: goal-planning (which target to focus on).
Low-level: goal-achieving (which concept to practice).
Graph-based candidate filtering per target.
Test-based internal reward: mastery improvement on focused goal via DKT.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from simpath.eval.methods import register_method
from simpath.eval.methods.base import BaseMethod, compute_ep_reward
from simpath.eval.graph import get_goal_candidates
from simpath.eval.kes import FastRollout

MAX_TARGETS = 5


class HighLevelAgent(nn.Module):
    """Goal planner. State: mastery + target_mastery + target_gap + step."""
    def __init__(self, num_c, hidden=256):
        super().__init__()
        sd = num_c + MAX_TARGETS * 2 + 1
        self.net = nn.Sequential(nn.Linear(sd, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, MAX_TARGETS)
        self.v = nn.Linear(hidden, 1)

    def forward(self, s, vm=None):
        h = self.net(s); lo = self.pi(h)
        if vm is not None: lo = lo + (1 - vm) * (-1e9)
        return lo, self.v(h).squeeze(-1)


class LowLevelAgent(nn.Module):
    """Goal achiever. State: mastery + goal_onehot + step."""
    def __init__(self, num_c, hidden):
        super().__init__()
        sd = num_c * 2 + 1
        self.net = nn.Sequential(nn.Linear(sd, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, num_c)
        self.v = nn.Linear(hidden, 1)

    def forward(self, s, vm=None):
        h = self.net(s); lo = self.pi(h)
        if vm is not None: lo = lo + (1 - vm) * (-1e9)
        return lo, self.v(h).squeeze(-1)


def _ppo_update(agent, opt, all_s, all_a, all_lp, all_v, all_vm, all_rew,
                batch_size, L, gamma, clip, ent_coef, device):
    G = np.zeros(batch_size, dtype=np.float32); rets = [None] * L
    for t in reversed(range(L)): G = all_rew[t] + gamma * G; rets[t] = G.copy()
    st = torch.cat(all_s); at = torch.cat(all_a); olp = torch.cat(all_lp).detach()
    ret = torch.tensor(np.concatenate(rets), dtype=torch.float32, device=device)
    ov = torch.cat(all_v).detach(); vm_t = torch.cat(all_vm)
    adv = ret - ov; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(4):
        lo, vals = agent(st, vm_t)
        probs = F.softmax(lo, dim=-1).clamp(min=1e-8)
        dist = torch.distributions.Categorical(probs); nlp = dist.log_prob(at); ent = dist.entropy()
        ratio = torch.exp(nlp - olp); s1 = ratio * adv; s2 = torch.clamp(ratio, 1-clip, 1+clip) * adv
        loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(vals, ret) - ent_coef * ent.mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step()


@register_method
class GEHRLMethod(BaseMethod):
    name = "GEHRL"
    needs_graph = True
    needs_training = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.high = HighLevelAgent(self.num_c, min(self.hidden, 256)).to(self.device)
        self.low = LowLevelAgent(self.num_c, self.hidden).to(self.device)
        self.graph = None

    def train(self, train_data, val_data, kes, graph, experts,
              n_episodes=30000, batch_size=128, val_interval=2000, out_dir=None, **kwargs):
        self.graph = graph
        NC = self.num_c; L = self.L; dev = self.device
        lr = 3e-4; clip = 0.2; gamma = 0.99
        ent_h = 0.05; ent_l = 0.03
        opt_h = torch.optim.Adam(self.high.parameters(), lr=lr)
        opt_l = torch.optim.Adam(self.low.parameters(), lr=lr)
        sch_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=n_episodes, eta_min=1e-5)
        sch_l = torch.optim.lr_scheduler.CosineAnnealingLR(opt_l, T_max=n_episodes, eta_min=1e-5)
        bv = -999; t0 = time.time()
        rollout = FastRollout(kes.dkt, NC, dev, kes.max_hist)

        for ep_i in range(n_episodes):
            indices = np.random.randint(0, len(train_data), size=batch_size)
            batch = [train_data[i] for i in indices]
            hc_b = [s[0] for s in batch]; hr_b = [s[1] for s in batch]; tgts_b = [s[2] for s in batch]

            init_mastery_b = rollout.init_batch(hc_b, hr_b, tgts_b, L)
            es_b = [init_mastery_b[i][tgts_b[i]].copy() for i in range(batch_size)]
            used_b = [set() for _ in range(batch_size)]

            tgts_pad = np.zeros((batch_size, MAX_TARGETS), dtype=np.int64)
            tgts_mask = np.zeros((batch_size, MAX_TARGETS), dtype=np.float32)
            for i in range(batch_size):
                nt = min(len(tgts_b[i]), MAX_TARGETS)
                tgts_pad[i, :nt] = tgts_b[i][:nt]; tgts_mask[i, :nt] = 1.0

            ah_s, ah_a, ah_lp, ah_v, ah_vm = [], [], [], [], []
            al_s, al_a, al_lp, al_v, al_vm = [], [], [], [], []
            ah_rew, al_rew = [], []

            for step in range(L):
                # High-level state: mastery + target_mastery + target_gap + step
                init_m = rollout.init_mastery
                tm = init_m[np.arange(batch_size)[:, None], tgts_pad]
                tg = 1.0 - tm
                sf_h = np.full((batch_size, 1), step / L, dtype=np.float32)
                hs_np = np.concatenate([init_m.astype(np.float32), tm.astype(np.float32),
                                        tg.astype(np.float32), sf_h], axis=1)
                hs_b = torch.from_numpy(hs_np).to(dev)
                hvm_b = torch.from_numpy(tgts_mask).to(dev)

                h_lo, h_v = self.high(hs_b, hvm_b)
                h_p = F.softmax(h_lo, dim=-1).clamp(min=1e-8)
                h_d = torch.distributions.Categorical(h_p)
                h_a = h_d.sample(); h_lp = h_d.log_prob(h_a)
                gi = h_a.cpu().numpy()
                goals = [tgts_pad[i, min(gi[i], min(len(tgts_b[i])-1, MAX_TARGETS-1))]
                         for i in range(batch_size)]

                # Low-level state: mastery + goal_onehot + step
                goh = np.zeros((batch_size, NC), dtype=np.float32)
                for i in range(batch_size):
                    goh[i, goals[i]] = 1.0
                sf_l = np.full((batch_size, 1), step / L, dtype=np.float32)
                ls_np = np.concatenate([init_m.astype(np.float32), goh, sf_l], axis=1)
                ls_b = torch.from_numpy(ls_np).to(dev)

                # Low-level valid mask: graph candidates per goal (built on CPU)
                lvm_np = np.zeros((batch_size, NC), dtype=np.float32)
                for i in range(batch_size):
                    cands = get_goal_candidates(goals[i], init_m[i], self.graph, NC, used_b[i], cap=20)
                    if cands:
                        lvm_np[i, list(cands)] = 1.0
                    if lvm_np[i].sum() == 0:
                        all_c = [c for c in range(NC) if c not in used_b[i]]
                        if all_c:
                            lvm_np[i, all_c] = 1.0
                lvm_b = torch.from_numpy(lvm_np).to(dev)

                l_lo, l_v = self.low(ls_b, lvm_b)
                l_p = F.softmax(l_lo, dim=-1).clamp(min=1e-8)
                l_d = torch.distributions.Categorical(l_p)
                l_a = l_d.sample(); l_lp = l_d.log_prob(l_a)

                anp = l_a.cpu().numpy()
                old_m = rollout.sim_mastery.copy()

                for i in range(batch_size):
                    used_b[i].add(anp[i])

                sim_mastery_b = rollout.step(anp)

                # Test-based internal reward (low-level)
                lr_arr = np.array([sim_mastery_b[i][goals[i]] - old_m[i][goals[i]]
                                   for i in range(batch_size)], dtype=np.float32)
                # External reward (high-level)
                hr_arr = np.zeros(batch_size, dtype=np.float32)
                if step == L - 1:
                    hr_arr = compute_ep_reward(sim_mastery_b, tgts_b, es_b, batch_size)

                ah_s.append(hs_b); ah_a.append(h_a); ah_lp.append(h_lp)
                ah_v.append(h_v); ah_vm.append(hvm_b); ah_rew.append(hr_arr)
                al_s.append(ls_b); al_a.append(l_a); al_lp.append(l_lp)
                al_v.append(l_v); al_vm.append(lvm_b); al_rew.append(lr_arr)

            _ppo_update(self.high, opt_h, ah_s, ah_a, ah_lp, ah_v, ah_vm, ah_rew,
                        batch_size, L, gamma, clip, ent_h, dev)
            _ppo_update(self.low, opt_l, al_s, al_a, al_lp, al_v, al_vm, al_rew,
                        batch_size, L, gamma, clip, ent_l, dev)
            sch_h.step(); sch_l.step()

            if (ep_i + 1) % 5 == 0:
                self._update_progress(out_dir, ep_i+1, n_episodes, reward=ah_rew[-1].mean())
            if (ep_i + 1) % val_interval == 0:
                self.high.eval(); self.low.eval()
                eps = kes.evaluate_batch(val_data, lambda m, t, k, hc, hr: self.predict(m, t, k, hc, hr))
                vep = np.mean(eps); mk = ''
                if vep > bv:
                    bv = vep
                    self._best_h = {k: v.clone() for k, v in self.high.state_dict().items()}
                    self._best_l = {k: v.clone() for k, v in self.low.state_dict().items()}
                    mk = ' *** SAVED'
                print(f"  [{self.name}] Ep {ep_i+1}/{n_episodes} | "
                      f"Val({len(val_data)})={vep:+.4f}{mk} | {time.time()-t0:.0f}s", flush=True)
                self._update_progress(out_dir, ep_i+1, n_episodes, val=vep)
                if out_dir:
                    torch.save({'high': self.high.state_dict(), 'low': self.low.state_dict()},
                               os.path.join(out_dir, f'checkpoint_ep{ep_i+1}.pt'))
                self.high.train(); self.low.train()

        if hasattr(self, '_best_h'):
            self.high.load_state_dict(self._best_h)
            self.low.load_state_dict(self._best_l)
        print(f"  [{self.name}] Best Val = {bv:+.4f}")

    def predict(self, mastery, targets, kes=None, hc=None, hr=None):
        self.high.eval(); self.low.eval()
        NC = self.num_c; L = self.L; dev = self.device
        tgts_pad = np.zeros(MAX_TARGETS, dtype=np.int64)
        tgts_m = np.zeros(MAX_TARGETS, dtype=np.float32)
        nt = min(len(targets), MAX_TARGETS)
        tgts_pad[:nt] = targets[:nt]; tgts_m[:nt] = 1.0
        path, used = [], set()
        for step in range(L):
            tm = mastery[tgts_pad]; tg = 1.0 - tm
            hs = torch.tensor(np.concatenate([mastery, tm, tg, [step/L]]),
                              dtype=torch.float32, device=dev)
            hvm = torch.tensor(tgts_m, dtype=torch.float32, device=dev)
            with torch.no_grad(): h_lo, _ = self.high(hs, hvm)
            gi = h_lo.argmax().item()
            goal = tgts_pad[min(gi, nt - 1)]
            goh = np.zeros(NC, dtype=np.float32); goh[goal] = 1.0
            ls = torch.tensor(np.concatenate([mastery, goh, [step/L]]),
                              dtype=torch.float32, device=dev)
            cands = get_goal_candidates(goal, mastery, self.graph, NC, used, cap=20)
            vm = np.zeros(NC, dtype=np.float32)
            for c in cands: vm[c] = 1.0
            if vm.sum() == 0:
                for c in range(NC):
                    if c not in used: vm[c] = 1.0
            lvm = torch.tensor(vm, dtype=torch.float32, device=dev)
            with torch.no_grad(): l_lo, _ = self.low(ls, lvm)
            a = l_lo.argmax().item(); path.append(a); used.add(a)
        return path

    def save(self, path):
        torch.save({'high': self.high.state_dict(), 'low': self.low.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, weights_only=False, map_location=self.device)
        self.high.load_state_dict(ck['high']); self.low.load_state_dict(ck['low'])
