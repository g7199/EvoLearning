"""Base method class and shared PolicyNet."""
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from abc import ABC, abstractmethod


class BaseMethod(ABC):
    name: str = "base"
    needs_experts: bool = False
    needs_graph: bool = False
    needs_training: bool = True

    def __init__(self, num_c: int, L: int, hidden: int, device: str):
        self.num_c = num_c
        self.L = L
        self.hidden = hidden
        self.device = device

    @abstractmethod
    def train(self, train_data, val_data, kes, graph, experts, out_dir=None, **kwargs): ...

    @abstractmethod
    def predict(self, mastery: np.ndarray, targets: List[int],
                kes=None, hc=None, hr=None) -> List[int]:
        """Generate path. If kes/hc/hr provided, update mastery at each step."""
        ...

    def save(self, path: str): ...
    def load(self, path: str): ...

    def _update_progress(self, out_dir, ep, n_episodes, reward=None, val=None):
        """Write progress to JSON for tqdm + append val to val_history.csv."""
        if out_dir is None:
            return
        progress_path = os.path.join(out_dir, 'progress.json')
        # Load existing history
        try:
            with open(progress_path) as f:
                data = json.load(f)
            history = data.get('history', [])
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        entry = {'ep': ep}
        if reward is not None:
            entry['reward'] = round(float(reward), 4)
        if val is not None:
            entry['val'] = round(float(val), 4)

        if not history or history[-1].get('ep') != ep:
            history.append(entry)

        p = {
            'method': self.name, 'ep': ep, 'total': n_episodes,
            'reward': float(reward) if reward is not None else None,
            'val': float(val) if val is not None else None,
            'status': 'running' if ep < n_episodes else 'done',
            'history': history,
        }
        try:
            with open(progress_path, 'w') as f:
                json.dump(p, f)
        except Exception:
            pass

        # Append val results to CSV (easy to monitor)
        if val is not None:
            csv_path = os.path.join(out_dir, 'val_history.csv')
            write_header = not os.path.exists(csv_path)
            try:
                with open(csv_path, 'a') as f:
                    if write_header:
                        f.write('ep,val,reward\n')
                    r_str = f'{reward:.4f}' if reward is not None else ''
                    f.write(f'{ep},{val:.4f},{r_str}\n')
            except Exception:
                pass


class PolicyNet(nn.Module):
    """Asymmetric actor-critic. Actor sees h0, Critic sees ht (privileged).
    forward(s_actor, s_critic=None, valid_mask=None)
    - s_actor: state with initial mastery (h0) — used by policy head
    - s_critic: state with current mastery (ht) — used by value head (training only)
    - If s_critic is None, uses s_actor for both (symmetric mode)
    """
    def __init__(self, state_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, s_actor, s_critic=None, valid_mask=None):
        if s_critic is None:
            s_critic = s_actor
        h_a = self.actor_net(s_actor)
        h_c = self.critic_net(s_critic)
        logits = self.pi(h_a)
        if valid_mask is not None:
            logits = logits + (1 - valid_mask) * (-1e9)
        return logits, self.v(h_c).squeeze(-1)


def make_state_standard(mastery: np.ndarray, targets: List[int],
                        step: int, L: int, num_c: int) -> np.ndarray:
    """Standard state: mastery(NC) + target_mask(NC) + step/L(1) = NC*2+1."""
    tm = np.zeros(num_c, dtype=np.float32)
    for t in targets:
        tm[t] = 1.0
    return np.concatenate([mastery.astype(np.float32), tm, [np.float32(step / L)]])


def make_state_dlelp(mastery: np.ndarray, targets: List[int], num_c: int) -> np.ndarray:
    """DLELP state: mastery(NC) + target_mask(NC) = NC*2. No step."""
    tm = np.zeros(num_c, dtype=np.float32)
    for t in targets:
        tm[t] = 1.0
    return np.concatenate([mastery.astype(np.float32), tm])


def run_ppo_epoch(policy, states, actions, old_lps, returns, values, vmasks,
                  optimizer, clip_eps, ent_coef, n_epochs=4, critic_states=None):
    """Asymmetric PPO update. critic_states uses ht (privileged), states uses h0."""
    adv = returns - values.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(n_epochs):
        logits, vals = policy(states, critic_states, vmasks)
        probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
        dist = torch.distributions.Categorical(probs)
        nlp = dist.log_prob(actions)
        ent = dist.entropy()
        ratio = torch.exp(nlp - old_lps)
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(vals, returns) - ent_coef * ent.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()


def run_ppo_epoch_masked(policy, states, actions, old_lps, returns, values, vmasks,
                         ppo_mask, optimizer, clip_eps, ent_coef, n_epochs=4):
    """PPO update with per-sample mask (for DLELP S-Agent exclusion)."""
    adv = returns - values.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv = adv * ppo_mask  # zero out S-Agent overridden steps
    for _ in range(n_epochs):
        logits, vals = policy(states, vmasks)
        probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
        dist = torch.distributions.Categorical(probs)
        nlp = dist.log_prob(actions)
        ent = dist.entropy()
        ratio = torch.exp(nlp - old_lps)
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(vals * ppo_mask, returns * ppo_mask) - ent_coef * ent.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()


def compute_ep_reward(mastery_b, targets_b, es_b, batch_size):
    """Compute terminal EP reward for a batch. Skip targets where 1-Es <= 0.01."""
    rewards = np.zeros(batch_size, dtype=np.float32)
    for i in range(batch_size):
        ee = mastery_b[i][targets_b[i]]
        es = es_b[i]
        vals = [(ee[j] - es[j]) / (1 - es[j])
                for j in range(len(targets_b[i])) if 1 - es[j] > 0.01]
        rewards[i] = np.mean(vals) if vals else 0.0
    return rewards
