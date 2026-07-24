"""Taste and pleasure prediction module.

Run directly to train the pleasure model on synthetic Food.com-like interactions
and print a small prediction preview:

    python -m foodrecom.taste --seed 42 --epochs 15 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import Recipe, UserProfile, build_synthetic_foodcom, load_real_foodcom
from .utils import set_global_seed


class NCF(nn.Module):
    """
    Neural Collaborative Filtering model using a dual-pathway Neural Matrix Factorization
    (NeuMF) architecture combining GMF (linear interactions) and MLP (non-linear interactions).
    """

    def __init__(
        self, 
        n_users: int, 
        n_items: int, 
        embedding_dim: int = 16, 
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        # 1. GMF Branch Embeddings (Linear Interaction)
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        
        # 2. MLP Branch Embeddings (Deep Non-linear Interaction)
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)
        
        # Deep Neural Network for non-linear pattern extraction
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Final output projection (GMF output dimension + MLP output dimension -> 1)
        self.output_layer = nn.Linear(32 + embedding_dim, 1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # GMF Pathway: Element-wise Hadamard product
        user_gmf = self.user_embedding_gmf(users)
        item_gmf = self.item_embedding_gmf(items)
        gmf_vector = user_gmf * item_gmf

        # MLP Pathway: Concatenation through dense layers
        user_mlp = self.user_embedding_mlp(users)
        item_mlp = self.item_embedding_mlp(items)
        mlp_vector = self.mlp(torch.cat([user_mlp, item_mlp], dim=-1))

        # Concatenate linear and non-linear representations
        combined = torch.cat([gmf_vector, mlp_vector], dim=-1)
        
        # Bounded probability prediction in [0, 1]
        return torch.sigmoid(self.output_layer(combined)).squeeze(-1)


class TasteModule:
    """Trainable pleasure predictor returning normalized ``P_{u,i} in [0, 1]``."""

    def __init__(
        self, 
        users: Sequence[UserProfile], 
        recipes: Sequence[Recipe], 
        seed: int, 
        embedding_dim: int = 16,
        dropout_rate: float = 0.2
    ):
        self.users: Dict[str, int] = {u.user_id: idx for idx, u in enumerate(users)}
        self.items: Dict[str, int] = {r.recipe_id: idx for idx, r in enumerate(recipes)}
        self.user_profiles = {u.user_id: u for u in users}
        self.recipe_map = {r.recipe_id: r for r in recipes}
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        torch.manual_seed(seed)
        self.model = NCF(
            len(users), 
            len(recipes), 
            embedding_dim=embedding_dim, 
            dropout_rate=dropout_rate
        )
        self.trained = False
        self.training_losses: List[float] = []
        self.validation_history: List[Dict[str, float]] = []
        self.test_metrics: Dict[str, float] = {}

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _evaluate(self, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        """Compute average loss and Mean Absolute Error (MAE)."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_users, batch_items, batch_ratings in dataloader:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)
                batch_ratings = batch_ratings.to(device)

                preds = self.model(batch_users, batch_items)
                loss = criterion(preds, batch_ratings)

                total_loss += float(loss.item()) * len(batch_ratings)
                total_mae += float(torch.sum(torch.abs(preds - batch_ratings)).item())
                total_samples += len(batch_ratings)

        avg_loss = total_loss / max(1, total_samples)
        avg_mae = total_mae / max(1, total_samples)
        return avg_loss, avg_mae

    def train_from_interactions(
        self, 
        interactions: List[Tuple[str, str, float]], 
        epochs: int = 15, 
        batch_size: int = 512, 
        lr: float = 1e-3,
        val_split: float = 0.1,
        test_split: float = 0.1,
        eval_interval: int = 5
    ) -> List[float]:
        """Train NCF on ``(user_id, recipe_id, rating_1_to_5)`` records with validation and testing."""
        if not interactions:
            return []

        # 1. Dataset Split (Train / Validation / Test)
        shuffled_interactions = list(interactions)
        np.random.shuffle(shuffled_interactions)
        
        n_total = len(shuffled_interactions)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val - n_test

        train_data = shuffled_interactions[:n_train]
        val_data = shuffled_interactions[n_train : n_train + n_val]
        test_data = shuffled_interactions[n_train + n_val :]

        def create_dataloader(data: List[Tuple[str, str, float]], shuffle: bool = True) -> DataLoader:
            u_tens = torch.tensor([self.users[u] for u, _, _ in data], dtype=torch.long)
            i_tens = torch.tensor([self.items[i] for _, i, _ in data], dtype=torch.long)
            r_tens = torch.tensor([(r - 1.0) / 4.0 for _, _, r in data], dtype=torch.float32)
            dataset = TensorDataset(u_tens, i_tens, r_tens)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        train_loader = create_dataloader(train_data, shuffle=True)
        val_loader = create_dataloader(val_data, shuffle=False) if val_data else None
        test_loader = create_dataloader(test_data, shuffle=False) if test_data else None

        # 2. Optimization and hardware setup
        device = self._get_device()
        self.model.to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        self.training_losses = []
        self.validation_history = []

        print(f"\n--- Training NCF on Device: {device.type.upper()} ---")
        print(f"Dataset Split: Train={n_train} | Val={n_val} | Test={n_test}\n")

        # 3. Training Loop
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses: List[float] = []

            for batch_users, batch_items, batch_ratings in train_loader:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)
                batch_ratings = batch_ratings.to(device)

                preds = self.model(batch_users, batch_items)
                loss = criterion(preds, batch_ratings)

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_losses.append(float(loss.item()))

            avg_train_loss = float(np.mean(epoch_losses))
            self.training_losses.append(avg_train_loss)

            # Evaluate on Validation set every `eval_interval` (5) epochs
            if val_loader and (epoch % eval_interval == 0 or epoch == epochs):
                val_loss, val_mae = self._evaluate(val_loader, criterion, device)
                self.validation_history.append({"epoch": epoch, "loss": val_loss, "mae": val_mae})
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

        # 4. Final Evaluation on Test set
        if test_loader:
            test_loss, test_mae = self._evaluate(test_loader, criterion, device)
            self.test_metrics = {"test_loss": test_loss, "test_mae": test_mae}
            print(f"\n--- Final Test Set Results ---")
            print(f"Test BCE Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}\n")

        self.trained = True
        self.model.to("cpu")
        return self.training_losses

    def save_model(self, filepath: str) -> None:
        """Save model checkpoint, parameters, and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "users": self.users,
            "items": self.items,
            "embedding_dim": self.embedding_dim,
            "dropout_rate": self.dropout_rate,
            "trained": self.trained,
            "training_losses": self.training_losses,
            "validation_history": self.validation_history,
            "test_metrics": self.test_metrics,
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved successfully to '{filepath}'")

    def load_model(self, filepath: str) -> None:
        """Load model checkpoint and restore state from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found at '{filepath}'")

        checkpoint = torch.load(filepath, map_location="cpu")
        self.users = checkpoint["users"]
        self.items = checkpoint["items"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.dropout_rate = checkpoint["dropout_rate"]

        self.model = NCF(
            len(self.users), 
            len(self.items), 
            embedding_dim=self.embedding_dim, 
            dropout_rate=self.dropout_rate
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.trained = checkpoint["trained"]
        self.training_losses = checkpoint.get("training_losses", [])
        self.validation_history = checkpoint.get("validation_history", [])
        self.test_metrics = checkpoint.get("test_metrics", {})
        print(f"Model loaded successfully from '{filepath}'")

    def predict(self, user_id: str, recipe_id: str) -> float:
        """Predict normalized pleasure for a user/recipe pair."""
        if self.trained and user_id in self.users and recipe_id in self.items:
            self.model.eval()
            with torch.no_grad():
                u_idx = torch.tensor([self.users[user_id]], dtype=torch.long)
                r_idx = torch.tensor([self.items[recipe_id]], dtype=torch.long)
                return float(self.model(u_idx, r_idx).item())
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


def train_and_preview_taste_model(
    seed: int = 42, 
    epochs: int = 15, 
    target_user_id: str = "food_com_user_10842", 
    top_k: int = 5,
    save_path: Optional[str] = None,
    data_dir: str ='./',
    n_users: int =10**10,
    n_recipes: int=10**10
) -> Dict[str, object]:
    """Convenience function for testing the pleasure model independently."""
    set_global_seed(seed)
    users, recipes, interactions = load_real_foodcom(seed=seed,
                                                     recipes_path=data_dir,
                                                     ratings_path=data_dir,
                                                     n_users=n_users,
                                                     n_recipes=n_recipes)
    
    if target_user_id not in {u.user_id for u in users}:
        target_user_id = users[0].user_id

    module = TasteModule(users, recipes, seed)
    losses = module.train_from_interactions(interactions, epochs=epochs, eval_interval=5)
    
    if save_path:
        module.save_model(save_path)

    preview = module.top_k_for_user(target_user_id, recipes, k=top_k)
    return {
        "target_user_id": target_user_id,
        "training_losses": losses,
        "validation_history": module.validation_history,
        "test_metrics": module.test_metrics,
        "top_predictions": preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and preview the taste/pleasure model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--target-user-id", default="food_com_user_10842")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-path", type=str, default="checkpoints/taste_model.pt")
    args = parser.parse_args()

    results = train_and_preview_taste_model(
        seed=args.seed, 
        epochs=args.epochs, 
        target_user_id=args.target_user_id, 
        top_k=args.top_k,
        save_path=args.save_path
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()