"""Adaptive policy engines for recommender parameters."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


class LinUCBPolicy:
    """Contextual bandit over a discrete set of Layer-A parameters."""

    def __init__(self, context_dim: int, actions: Sequence[float], alpha: float = 0.8):
        self.actions = list(actions)
        self.alpha = alpha
        self.A = [np.eye(context_dim) for _ in self.actions]
        self.b = [np.zeros(context_dim) for _ in self.actions]
        self.last_action_idx = 0

    def select(self, context: np.ndarray) -> float:
        scores = []
        for A, b in zip(self.A, self.b):
            inv = np.linalg.inv(A)
            theta = inv @ b
            scores.append(float(theta @ context + self.alpha * np.sqrt(context @ inv @ context)))
        self.last_action_idx = int(np.argmax(scores))
        return self.actions[self.last_action_idx]

    def update(self, context: np.ndarray, reward: float) -> None:
        a = self.last_action_idx
        self.A[a] += np.outer(context, context)
        self.b[a] += reward * context


class SACPolicy:
    """Lightweight SAC-shaped continuous actor for stateful adaptation."""

    def __init__(self, state_dim: int, seed: int):
        torch.manual_seed(seed)
        self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.memory: List[Tuple[np.ndarray, float, float]] = []

    def select(self, state: np.ndarray) -> float:
        with torch.no_grad():
            base = float(self.actor(torch.tensor(state, dtype=torch.float32)).item())
        return float(np.clip(np.random.normal(base, 0.08), 0.05, 0.95))

    def update(self, state: np.ndarray, action: float, reward: float) -> None:
        self.memory.append((state.copy(), action, reward))
        batch = self.memory[-64:]
        states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([[b[1]] for b in batch], dtype=torch.float32)
        rewards = torch.tensor([[b[2]] for b in batch], dtype=torch.float32)
        pred = self.actor(states)
        loss = ((pred - actions).pow(2) * (1.2 - rewards.clamp(0, 1))).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
