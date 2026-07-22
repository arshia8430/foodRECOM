"""Adaptive policy engines for recommender parameters.

The module intentionally keeps the public policy surface small: LinUCB chooses
one of a finite set of Layer-A parameters, while SAC learns a continuous
parameter in ``[0.05, 0.95]`` from the six-dimensional simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Sequence, Tuple
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LinUCBPolicy:
    """Contextual bandit over a discrete set of Layer-A parameters."""

    def __init__(self, context_dim: int, actions: Sequence[float], alpha: float = 0.8):
        self.actions = list(actions)
        self.alpha = alpha
        self.A = [np.eye(context_dim, dtype=np.float64) for _ in self.actions]
        self.b = [np.zeros(context_dim, dtype=np.float64) for _ in self.actions]
        self.last_action_idx = 0

    def select(self, context: np.ndarray) -> float:
        context64 = context.astype(np.float64)
        scores = []
        for A, b in zip(self.A, self.b):
            inv = np.linalg.inv(A)
            theta = inv @ b
            scores.append(float(theta @ context64 + self.alpha * np.sqrt(context64 @ inv @ context64)))
        self.last_action_idx = int(np.argmax(scores))
        return self.actions[self.last_action_idx]

    def update(self, context: np.ndarray, reward: float) -> None:
        context64 = context.astype(np.float64)
        a = self.last_action_idx
        self.A[a] += np.outer(context64, context64)
        self.b[a] += reward * context64


class GaussianActor(nn.Module):
    """Squashed Gaussian SAC actor implemented in PyTorch."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(states)
        return self.mean(x), torch.clamp(self.log_std(x), -5.0, 1.0)

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        squashed = torch.tanh(z)
        action = 0.5 * (squashed + 1.0)
        log_prob = normal.log_prob(z) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        return action.clamp(0.05, 0.95), log_prob.sum(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    """State-action critic used by SAC."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1))


@dataclass
class Transition:
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class SACPolicy:
    """Compact PyTorch Soft Actor-Critic policy for stateful adaptation."""

    def __init__(self, state_dim: int, seed: int, gamma: float = 0.97, tau: float = 0.02, alpha: float = 0.15, min_replay_size: int = 2):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.min_replay_size = max(1, int(min_replay_size))
        self.gradient_updates = 0
        self.actor = GaussianActor(state_dim)
        self.q1 = QNetwork(state_dim)
        self.q2 = QNetwork(state_dim)
        self.target_q1 = QNetwork(state_dim)
        self.target_q2 = QNetwork(state_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=3e-4)
        self.memory: Deque[Transition] = deque(maxlen=10_000)
        self.last_loss: float | None = None

    def select(self, state: np.ndarray) -> float:
        with torch.no_grad():
            action, _ = self.actor.sample(torch.tensor(state[None, :], dtype=torch.float32))
        return float(action.item())

    def update(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray | None = None, done: bool = False) -> None:
        if next_state is None:
            next_state = state
        self.memory.append(Transition(state.copy(), float(action), float(reward), next_state.copy(), bool(done)))
        if len(self.memory) < self.min_replay_size:
            return
        batch_size = min(64, len(self.memory))
        idx = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        batch = [list(self.memory)[int(i)] for i in idx]
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32)
        actions = torch.tensor([[b.action] for b in batch], dtype=torch.float32)
        rewards = torch.tensor([[b.reward] for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32)
        dones = torch.tensor([[b.done] for b in batch], dtype=torch.float32)

        with torch.no_grad():
            next_actions, next_logp = self.actor.sample(next_states)
            target_q = torch.min(self.target_q1(next_states, next_actions), self.target_q2(next_states, next_actions)) - self.alpha * next_logp
            y = rewards + self.gamma * (1.0 - dones) * target_q
        q1_loss = F.mse_loss(self.q1(states, actions), y)
        q2_loss = F.mse_loss(self.q2(states, actions), y)
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        sampled_actions, logp = self.actor.sample(states)
        actor_loss = (self.alpha * logp - torch.min(self.q1(states, sampled_actions), self.q2(states, sampled_actions))).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        self.last_loss = float((q1_loss + q2_loss + actor_loss).item())
        self.gradient_updates += 1

        with torch.no_grad():
            for target, source in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
                for target_param, source_param in zip(target.parameters(), source.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * source_param.data)
