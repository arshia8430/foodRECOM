"""Taste and pleasure prediction module.

Run directly to train the pleasure model on synthetic Food.com-like interactions
and print a small prediction preview:

    python -m foodrecom.taste --seed 42 --epochs 2 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Recipe, UserProfile, build_synthetic_foodcom
from .utils import set_global_seed


class NCF(nn.Module):
    """Hybrid neural collaborative filtering model for pleasure scores."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_tags: int,
        embedding_dim: int = 16,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            n_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            n_items,
            embedding_dim,
        )

        self.tag_embedding = nn.Embedding(
            max(n_tags, 1),
            embedding_dim,
        )

        self.recipe_encoder = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                embedding_dim,
            ),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        item_tags: torch.Tensor,
    ) -> torch.Tensor:
        user_vector = self.user_embedding(users)

        item_vector = self.item_embedding(items)

        tag_vectors = self.tag_embedding(
            item_tags.clamp(min=0)
        )

        valid_tags = (
            item_tags >= 0
        ).unsqueeze(-1)

        tag_vectors = (
            tag_vectors
            * valid_tags
        )

        tag_counts = (
            valid_tags
            .sum(dim=1)
            .clamp(min=1)
        )

        tag_vector = (
            tag_vectors.sum(dim=1)
            / tag_counts
        )

        recipe_vector = self.recipe_encoder(
            torch.cat(
                [
                    item_vector,
                    tag_vector,
                ],
                dim=-1,
            )
        )

        combined = torch.cat(
            [
                user_vector,
                recipe_vector,
            ],
            dim=-1,
        )

        return self.mlp(
            combined
        ).squeeze(-1)


class TasteModule:
    """Trainable pleasure predictor returning normalized P_{u,i} in [0, 1]."""

    def __init__(
        self,
        users: Sequence[UserProfile],
        recipes: Sequence[Recipe],
        seed: int,
        embedding_dim: int = 16,
    ):
        self.users: Dict[str, int] = {
            u.user_id: idx
            for idx, u in enumerate(users)
        }

        self.items: Dict[str, int] = {
            r.recipe_id: idx
            for idx, r in enumerate(recipes)
        }

        self.user_profiles = {
            u.user_id: u
            for u in users
        }

        self.recipe_map = {
            r.recipe_id: r
            for r in recipes
        }

        all_tags = sorted(
            {
                tag
                for recipe in recipes
                for tag in recipe.tags
            }
        )

        self.tags: Dict[str, int] = {
            tag: idx
            for idx, tag in enumerate(all_tags)
        }

        self.recipe_tags: Dict[str, List[int]] = {
            recipe.recipe_id: [
                self.tags[tag]
                for tag in recipe.tags
                if tag in self.tags
            ]
            for recipe in recipes
        }

        torch.manual_seed(seed)

        self.model = NCF(
            n_users=len(users),
            n_items=len(recipes),
            n_tags=len(self.tags),
            embedding_dim=embedding_dim,
        )

        self.trained = False

        self.training_losses: List[float] = []

        self.validation_losses: List[float] = []

        self.test_loss: float | None = None

        self.best_val_loss = float("inf")

        self.best_epoch = -1

        self.train_interactions: List[
            Tuple[str, str, float]
        ] = []

        self.validation_interactions: List[
            Tuple[str, str, float]
        ] = []

        self.test_interactions: List[
            Tuple[str, str, float]
        ] = []

    def _build_tensors(
        self,
        interactions: List[
            Tuple[str, str, float]
        ],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        users = torch.tensor(
            [
                self.users[user_id]
                for user_id, _, _ in interactions
            ],
            dtype=torch.long,
        )

        items = torch.tensor(
            [
                self.items[recipe_id]
                for _, recipe_id, _ in interactions
            ],
            dtype=torch.long,
        )

        ratings = torch.tensor(
            [
                (rating - 1.0) / 4.0
                for _, _, rating in interactions
            ],
            dtype=torch.float32,
        )

        max_tags = max(
            (
                len(
                    self.recipe_tags.get(
                        recipe_id,
                        [],
                    )
                )
                for _, recipe_id, _ in interactions
            ),
            default=1,
        )

        item_tags = torch.full(
            (
                len(interactions),
                max_tags,
            ),
            -1,
            dtype=torch.long,
        )

        for row, (
            _,
            recipe_id,
            _,
        ) in enumerate(interactions):
            tags = self.recipe_tags.get(
                recipe_id,
                [],
            )

            if tags:
                item_tags[
                    row,
                    :len(tags),
                ] = torch.tensor(
                    tags,
                    dtype=torch.long,
                )

        return (
            users,
            items,
            ratings,
            item_tags,
        )

    def _evaluate(
        self,
        interactions: List[
            Tuple[str, str, float]
        ],
    ) -> float:
        if not interactions:
            return float("nan")

        (
            users,
            items,
            ratings,
            item_tags,
        ) = self._build_tensors(
            interactions
        )

        self.model.eval()

        with torch.no_grad():
            predictions = self.model(
                users,
                items,
                item_tags,
            )

            loss = F.mse_loss(
                predictions,
                ratings,
            )

        return float(
            loss.item()
        )

    def train_from_interactions(
        self,
        interactions: List[
            Tuple[str, str, float]
        ],
        epochs: int = 4,
        batch_size: int = 512,
        lr: float = 1e-4,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        patience: int = 5,
    ) -> List[float]:
        """Train on train split, validate during training, and evaluate on test."""

        if not interactions:
            return []

        if (
            validation_split < 0.0
            or validation_split >= 1.0
        ):
            raise ValueError(
                "validation_split must be in [0, 1)."
            )

        if (
            test_split < 0.0
            or test_split >= 1.0
        ):
            raise ValueError(
                "test_split must be in [0, 1)."
            )

        if (
            validation_split
            + test_split
            >= 1.0
        ):
            raise ValueError(
                "validation_split + test_split must be less than 1."
            )

        if patience < 1:
            raise ValueError(
                "patience must be at least 1."
            )

        if len(interactions) < 3:
            raise ValueError(
                "At least 3 interactions are required for train/validation/test split."
            )

        generator = torch.Generator()

        generator.manual_seed(
            torch.initial_seed()
        )

        permutation = torch.randperm(
            len(interactions),
            generator=generator,
        ).tolist()

        shuffled = [
            interactions[index]
            for index in permutation
        ]

        test_size = max(
            1,
            int(
                len(shuffled)
                * test_split
            ),
        )

        validation_size = max(
            1,
            int(
                len(shuffled)
                * validation_split
            ),
        )

        if (
            test_size
            + validation_size
            >= len(shuffled)
        ):
            raise ValueError(
                "Dataset is too small for the requested train/validation/test split."
            )

        self.test_interactions = shuffled[
            :test_size
        ]

        self.validation_interactions = shuffled[
            test_size:
            test_size + validation_size
        ]

        self.train_interactions = shuffled[
            test_size + validation_size:
        ]

        (
            train_users,
            train_items,
            train_ratings,
            train_item_tags,
        ) = self._build_tensors(
            self.train_interactions
        )

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        self.training_losses = []

        self.validation_losses = []

        self.test_loss = None

        self.best_val_loss = float("inf")

        self.best_epoch = -1

        best_model_state = None

        epochs_without_improvement = 0

        for epoch in range(epochs):

            self.model.train()

            epoch_losses = []

            permutation = torch.randperm(
                len(train_ratings)
            )

            for start in range(
                0,
                len(train_ratings),
                batch_size,
            ):
                indices = permutation[
                    start:
                    start + batch_size
                ]

                predictions = self.model(
                    train_users[indices],
                    train_items[indices],
                    train_item_tags[indices],
                )

                loss = F.mse_loss(
                    predictions,
                    train_ratings[indices],
                )

                opt.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                opt.step()

                epoch_losses.append(
                    float(
                        loss.item()
                    )
                )

            train_loss = float(
                np.mean(
                    epoch_losses
                )
            )

            self.training_losses.append(
                train_loss
            )

            val_loss = self._evaluate(
                self.validation_interactions
            )

            self.validation_losses.append(
                val_loss
            )

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- train_loss: {train_loss:.6f} "
                f"- val_loss: {val_loss:.6f}"
            )

            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

                self.best_epoch = epoch + 1

                best_model_state = deepcopy(
                    self.model.state_dict()
                )

                epochs_without_improvement = 0

            else:

                epochs_without_improvement += 1

                if (
                    epochs_without_improvement
                    >= patience
                ):
                    print(
                        f"Early stopping at epoch {epoch + 1}"
                    )
                    break

        if best_model_state is not None:

            self.model.load_state_dict(
                best_model_state
            )

        self.model.eval()

        self.test_loss = self._evaluate(
            self.test_interactions
        )

        self.trained = True

        print(
            f"Best epoch: {self.best_epoch}"
        )

        print(
            f"Best validation loss: "
            f"{self.best_val_loss:.6f}"
        )

        print(
            f"Test loss: "
            f"{self.test_loss:.6f}"
        )

        return self.training_losses

    def save_model(
        self,
        path: str,
    ) -> None:
        """Save the trained model and all metadata required for inference."""

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "users": self.users,
            "items": self.items,
            "tags": self.tags,
            "recipe_tags": self.recipe_tags,
            "trained": self.trained,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "test_loss": self.test_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }

        torch.save(
            checkpoint,
            path,
        )

    def load_model(
        self,
        path: str,
    ) -> None:
        """Load a previously saved model checkpoint."""

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.trained = checkpoint.get(
            "trained",
            True,
        )

        self.training_losses = checkpoint.get(
            "training_losses",
            [],
        )

        self.validation_losses = checkpoint.get(
            "validation_losses",
            [],
        )

        self.test_loss = checkpoint.get(
            "test_loss",
            None,
        )

        self.best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf"),
        )

        self.best_epoch = checkpoint.get(
            "best_epoch",
            -1,
        )

        self.model.eval()

    def predict(
        self,
        user_id: str,
        recipe_id: str,
    ) -> float:
        """Predict normalized pleasure for a user/recipe pair."""

        if (
            self.trained
            and user_id in self.users
            and recipe_id in self.items
        ):
            self.model.eval()

            user_tensor = torch.tensor(
                [
                    self.users[
                        user_id
                    ]
                ],
                dtype=torch.long,
            )

            item_tensor = torch.tensor(
                [
                    self.items[
                        recipe_id
                    ]
                ],
                dtype=torch.long,
            )

            tags = self.recipe_tags.get(
                recipe_id,
                [],
            )

            if tags:

                item_tags = torch.tensor(
                    [tags],
                    dtype=torch.long,
                )

            else:

                item_tags = torch.full(
                    (1, 1),
                    -1,
                    dtype=torch.long,
                )

            with torch.no_grad():

                return float(
                    self.model(
                        user_tensor,
                        item_tensor,
                        item_tags,
                    ).item()
                )

        return self._latent_fallback(
            user_id,
            recipe_id,
        )

    def predict_many(
        self,
        user_id: str,
        recipe_ids: Sequence[str],
    ) -> List[
        Dict[str, float | str]
    ]:
        """Return a testable table of pleasure predictions for multiple recipes."""

        return [
            {
                "user_id": user_id,
                "recipe_id": recipe_id,
                "pleasure_score": self.predict(
                    user_id,
                    recipe_id,
                ),
            }
            for recipe_id in recipe_ids
        ]

    def top_k_for_user(
        self,
        user_id: str,
        recipes: Sequence[Recipe],
        k: int = 5,
    ) -> List[
        Dict[str, float | str]
    ]:
        """Rank recipes by predicted pleasure for quick inspection."""

        rows = [
            {
                "recipe_id": recipe.recipe_id,
                "name": recipe.name,
                "pleasure_score": self.predict(
                    user_id,
                    recipe.recipe_id,
                ),
            }
            for recipe in recipes
        ]

        return sorted(
            rows,
            key=lambda row: float(
                row[
                    "pleasure_score"
                ]
            ),
            reverse=True,
        )[:k]

    def _latent_fallback(
        self,
        user_id: str,
        recipe_id: str,
    ) -> float:
        user = self.user_profiles[
            user_id
        ]

        recipe = self.recipe_map[
            recipe_id
        ]

        dot = float(
            np.dot(
                user.latent,
                recipe.latent,
            )
            / (
                np.linalg.norm(
                    user.latent
                )
                * np.linalg.norm(
                    recipe.latent
                )
                + 1e-8
            )
        )

        tag_bonus = (
            0.08
            * len(
                set(
                    user.preferred_tags
                ).intersection(
                    recipe.tags
                )
            )
        )

        return float(
            np.clip(
                1.0
                / (
                    1.0
                    + math.exp(
                        -2.3 * dot
                    )
                )
                + tag_bonus,
                0.0,
                1.0,
            )
        )


def train_and_preview_taste_model(
    seed: int = 42,
    epochs: int = 2,
    target_user_id: str = "food_com_user_10842",
    top_k: int = 5,
) -> Dict[str, object]:
    """Convenience function for testing the pleasure model independently."""

    set_global_seed(
        seed
    )

    users, recipes, interactions = (
        build_synthetic_foodcom(
            seed
        )
    )

    if target_user_id not in {
        user.user_id
        for user in users
    }:
        target_user_id = users[
            0
        ].user_id

    module = TasteModule(
        users,
        recipes,
        seed,
    )

    losses = module.train_from_interactions(
        interactions,
        epochs=epochs,
    )

    preview = module.top_k_for_user(
        target_user_id,
        recipes,
        k=top_k,
    )

    return {
        "target_user_id": target_user_id,
        "training_losses": losses,
        "validation_losses": (
            module.validation_losses
        ),
        "best_epoch": (
            module.best_epoch
        ),
        "best_validation_loss": (
            module.best_val_loss
        ),
        "test_loss": (
            module.test_loss
        ),
        "top_predictions": preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and preview "
            "the taste/pleasure model."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--target-user-id",
        default=(
            "food_com_user_10842"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    result = (
        train_and_preview_taste_model(
            args.seed,
            args.epochs,
            args.target_user_id,
            args.top_k,
        )
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()