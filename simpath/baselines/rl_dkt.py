"""
B3: RL-DKT — DQN agent for learning path recommendation.
State = knowledge mastery vector, Action = select KC (then sample exercise).
Reward = delta mastery + ZPD bonus.

Following CSEAL (KDD 2019) design:
- KC-level actions (not exercise-level) for tractable action space
- Reward shaping with ZPD alignment
- 50K training episodes
- ε=0.1 at test time
"""

import random
import numpy as np
from typing import List, Dict, Tuple
from collections import deque


class DQNAgent:
    """DQN for KC-level exercise recommendation."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256,
                 lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 0.3,
                 epsilon_min: float = 0.1, epsilon_decay: float = 0.9999,
                 buffer_size: int = 50000, batch_size: int = 128):
        import torch
        import torch.nn as nn

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = deque(maxlen=buffer_size)
        self.steps = 0

    def select_action(self, state: np.ndarray, valid_actions: List[int] = None) -> int:
        import torch
        if random.random() < self.epsilon:
            return random.choice(valid_actions) if valid_actions else random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state).unsqueeze(0))[0]

        if valid_actions:
            mask = torch.full((self.n_actions,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0
            q_values = q_values + mask

        return int(q_values.argmax())

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        import torch
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % 200 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()


def train_rl_dkt(
    students: List[dict],
    pool_questions: List[dict],
    kc_list: List[str],
    n_episodes: int = 50000,
    L: int = 8,
    target_p: int = 5,
    save_path: str = None,
) -> DQNAgent:
    """
    Train DQN agent with KC-level actions (CSEAL-style).
    Action = select KC index → sample exercise from that KC.
    """
    n_kcs = len(kc_list)
    kc_to_idx = {kc: i for i, kc in enumerate(kc_list)}

    # Build KC → exercises mapping
    kc_to_exercises = {i: [] for i in range(n_kcs)}
    for q in pool_questions:
        for kc in q["kc_ids"]:
            if kc in kc_to_idx:
                kc_to_exercises[kc_to_idx[kc]].append(q)

    # Remove empty KCs from valid actions
    valid_kc_indices = [i for i in range(n_kcs) if kc_to_exercises[i]]

    agent = DQNAgent(state_dim=n_kcs, n_actions=n_kcs)
    print(f"  RL-DKT: {n_episodes} eps, {len(valid_kc_indices)} valid KCs, {n_kcs} total KCs")

    total_rewards = []
    for ep in range(n_episodes):
        student = random.choice(students)
        mastery = dict(student["mastery"])
        state = _mastery_to_vec(mastery, kc_list)

        target_kcs = sorted(mastery.items(), key=lambda x: x[1])[:target_p]
        target_kcs = [kc for kc, _ in target_kcs]

        used_kcs = set()
        ep_reward = 0

        for step in range(L):
            valid = [i for i in valid_kc_indices if i not in used_kcs]
            if not valid:
                valid = valid_kc_indices  # allow revisit if exhausted

            action = agent.select_action(state, valid)
            used_kcs.add(action)

            # Sample exercise from selected KC
            exercises = kc_to_exercises.get(action, [])
            if not exercises:
                continue
            q = random.choice(exercises)

            # Simulate + shaped reward
            reward, new_mastery = _simulate_with_reward(mastery, q, target_kcs, kc_list)

            mastery = new_mastery
            next_state = _mastery_to_vec(mastery, kc_list)
            done = (step == L - 1)

            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            ep_reward += reward

        total_rewards.append(ep_reward)

        if ep % 10000 == 0:
            avg_r = np.mean(total_rewards[-1000:]) if total_rewards else 0
            print(f"    Ep {ep}/{n_episodes} | avg_reward={avg_r:.4f} | eps={agent.epsilon:.3f}")

    if save_path:
        agent.save(save_path)
    return agent


def recommend_rl_dkt(
    agent: DQNAgent,
    mastery: Dict[str, float],
    pool: List[dict],
    kc_list: List[str],
    L: int = 8,
) -> List[dict]:
    """Use trained DQN agent. Test-time ε=0.1 (per CSEAL convention)."""
    kc_to_idx = {kc: i for i, kc in enumerate(kc_list)}
    kc_to_exercises = {}
    for q in pool:
        for kc in q["kc_ids"]:
            if kc in kc_to_idx:
                kc_to_exercises.setdefault(kc_to_idx[kc], []).append(q)

    state = _mastery_to_vec(mastery, kc_list)
    used_qids = set()
    path = []

    # ε=0.1 at test time (proposal spec)
    old_eps = agent.epsilon
    agent.epsilon = 0.1

    for step in range(L):
        valid = [i for i in kc_to_exercises if kc_to_exercises[i]]
        if not valid:
            break
        action = agent.select_action(state, valid)

        exercises = kc_to_exercises.get(action, [])
        # Pick best exercise from KC (lowest difficulty gap to mastery)
        kc = kc_list[action] if action < len(kc_list) else None
        m = mastery.get(kc, 0.5) if kc else 0.5
        exercises = [q for q in exercises if q["question_id"] not in used_qids]
        if not exercises:
            continue

        exercises.sort(key=lambda q: abs(q["difficulty"] - m - 0.1))
        q = exercises[0]
        path.append(q)
        used_qids.add(q["question_id"])

        # Update state
        _, new_mastery = _simulate_with_reward(mastery, q, list(mastery.keys())[:5], kc_list)
        mastery = new_mastery
        state = _mastery_to_vec(mastery, kc_list)

    agent.epsilon = old_eps
    return path


def _mastery_to_vec(mastery: dict, kc_list: list) -> np.ndarray:
    return np.array([mastery.get(kc, 0.5) for kc in kc_list], dtype=np.float32)


def _simulate_with_reward(mastery, question, target_kcs, kc_list):
    """Simulate + compute shaped reward."""
    kc_ids = question["kc_ids"]
    diff = question["difficulty"]

    avg_m = np.mean([mastery.get(kc, 0.5) for kc in kc_ids]) if kc_ids else 0.5
    logit = 3.0 * (avg_m - diff)
    p_correct = 1.0 / (1.0 + np.exp(-logit))
    correct = random.random() < p_correct

    new_mastery = dict(mastery)
    for kc in kc_ids:
        cur = new_mastery.get(kc, 0.5)
        if correct:
            new_mastery[kc] = cur + 0.12 * (1 - cur)
        else:
            new_mastery[kc] = max(0.0, cur - 0.08 * cur)

    # Shaped reward
    reward = 0.0

    # (1) EP gain on target KCs
    for kc in target_kcs:
        old = mastery.get(kc, 0.5)
        new = new_mastery.get(kc, 0.5)
        denom = 1.0 - old
        if denom > 1e-6:
            reward += (new - old) / denom
    reward /= max(len(target_kcs), 1)

    # (2) ZPD bonus: reward for choosing appropriately difficult questions
    zpd_match = max(0, 1.0 - abs(diff - avg_m - 0.1) * 3)
    reward += 0.01 * zpd_match

    # (3) Penalty for too-easy questions (mastery already high)
    if avg_m > 0.8:
        reward -= 0.005

    return reward, new_mastery
