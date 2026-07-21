"""Taste and pleasure prediction module.

Run directly to train the pleasure model on synthetic Food.com-like interactions
and print a small prediction preview:

    python -m foodrecom.taste --seed 42 --epochs 2 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Recipe, UserProfile, build_synthetic_foodcom
from .utils import set_global_seed


class NCF(nn.Module):
    """Small PyTorch neural collaborative filtering model for pleasure scores."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 16):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.user_embedding(users), self.item_embedding(items)], dim=-1)).squeeze(-1)


class TasteModule:
    """Trainable pleasure predictor returning normalized ``P_{u,i} in [0, 1]``."""

    def __init__(self, users: Sequence[UserProfile], recipes: Sequence[Recipe], seed: int, embedding_dim: int = 16):
        self.users: Dict[str, int] = {u.user_id: idx for idx, u in enumerate(users)}
        self.items: Dict[str, int] = {r.recipe_id: idx for idx, r in enumerate(recipes)}
        self.user_profiles = {u.user_id: u for u in users}
        self.recipe_map = {r.recipe_id: r for r in recipes}
        torch.manual_seed(seed)
        self.model = NCF(len(users), len(recipes), embedding_dim=embedding_dim)
        self.trained = False
        self.training_losses: List[float] = []

    def train_from_interactions(self, interactions: List[Tuple[str, str, float]], epochs: int = 4, batch_size: int = 512) -> List[float]:
        """Train NCF on ``(user_id, recipe_id, rating_1_to_5)`` records."""
        if not interactions:
            return []
        users = torch.tensor([self.users[u] for u, _, _ in interactions], dtype=torch.long)
        items = torch.tensor([self.items[i] for _, i, _ in interactions], dtype=torch.long)
        ratings = torch.tensor([(r - 1.0) / 4.0 for _, _, r in interactions], dtype=torch.float32)
        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        self.model.train()
        self.training_losses = []
        for _ in range(epochs):
            epoch_losses: List[float] = []
            perm = torch.randperm(len(ratings))
            for start in range(0, len(ratings), batch_size):
                idx = perm[start : start + batch_size]
                loss = F.mse_loss(self.model(users[idx], items[idx]), ratings[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.item()))
            self.training_losses.append(float(np.mean(epoch_losses)))
        self.trained = True
        return self.training_losses

    def predict(self, user_id: str, recipe_id: str) -> float:
        """Predict normalized pleasure for a user/recipe pair."""
        if self.trained and user_id in self.users and recipe_id in self.items:
            self.model.eval()
            with torch.no_grad():
                return float(self.model(torch.tensor([self.users[user_id]]), torch.tensor([self.items[recipe_id]])).item())
        return self._latent_fallback(user_id, recipe_id)

    def predict_many(self, user_id: str, recipe_ids: Sequence[str]) -> List[Dict[str, float | str]]:
        """Return a testable table of pleasure predictions for multiple recipes."""
        return [{"user_id": user_id, "recipe_id": rid, "pleasure_score": self.predict(user_id, rid)} for rid in recipe_ids]

    def top_k_for_user(self, user_id: str, recipes: Sequence[Recipe], k: int = 5) -> List[Dict[str, float | str]]:
        """Rank recipes by predicted pleasure for quick inspection."""
        rows = [
            {"recipe_id": r.recipe_id, "name": r.name, "pleasure_score": self.predict(user_id, r.recipe_id)}
            for r in recipes
        ]
        return sorted(rows, key=lambda row: float(row["pleasure_score"]), reverse=True)[:k]

    def _latent_fallback(self, user_id: str, recipe_id: str) -> float:
        user = self.user_profiles[user_id]
        recipe = self.recipe_map[recipe_id]
        dot = float(np.dot(user.latent, recipe.latent) / (np.linalg.norm(user.latent) * np.linalg.norm(recipe.latent) + 1e-8))
        tag_bonus = 0.08 * len(set(user.preferred_tags).intersection(recipe.tags))
        return float(np.clip(1.0 / (1.0 + math.exp(-2.3 * dot)) + tag_bonus, 0.0, 1.0))


def train_and_preview_taste_model(seed: int = 42, epochs: int = 2, target_user_id: str = "food_com_user_10842", top_k: int = 5) -> Dict[str, object]:
    """Convenience function for testing the pleasure model independently."""
    set_global_seed(seed)
    users, recipes, interactions = build_synthetic_foodcom(seed)
    if target_user_id not in {u.user_id for u in users}:
        target_user_id = users[0].user_id
    module = TasteModule(users, recipes, seed)
    losses = module.train_from_interactions(interactions, epochs=epochs)
    preview = module.top_k_for_user(target_user_id, recipes, k=top_k)
    return {"target_user_id": target_user_id, "training_losses": losses, "top_predictions": preview}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and preview the taste/pleasure model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--target-user-id", default="food_com_user_10842")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(train_and_preview_taste_model(args.seed, args.epochs, args.target_user_id, args.top_k), indent=2))


if __name__ == "__main__":
    main()
