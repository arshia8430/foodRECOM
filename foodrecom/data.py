"""Data structures and synthetic Food.com-style data generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Recipe:
    recipe_id: str
    name: str
    calories: float
    saturated_fat_g: float
    sugar_g: float
    sodium_mg: float
    fiber_g: float
    protein_g: float
    tags: List[str]
    latent: np.ndarray
    health_score: float = 0.0


@dataclass
class UserProfile:
    user_id: str
    big_five: Dict[str, float]
    preferred_tags: List[str]
    latent: np.ndarray
    historical_mean_rating: float


def build_synthetic_foodcom(seed: int, n_users: int = 80, n_recipes: int = 350) -> Tuple[List[UserProfile], List[Recipe], List[Tuple[str, str, float]]]:
    """Build a deterministic Food.com-like catalogue, users, and rating table."""
    from .health import HealthModule

    rng = np.random.default_rng(seed)
    tags = ["comfort", "quick", "vegetarian", "spicy", "sweet", "high-protein", "low-sodium", "fresh", "pasta", "soup"]
    recipes: List[Recipe] = []
    for i in range(n_recipes):
        recipe_tags = list(rng.choice(tags, size=int(rng.integers(2, 5)), replace=False))
        recipe = Recipe(
            recipe_id=f"rec_{i:04d}",
            name=f"Food.com Inspired Recipe {i:04d}",
            calories=float(rng.uniform(180, 950)),
            saturated_fat_g=float(rng.uniform(1, 24)),
            sugar_g=float(rng.uniform(1, 65)),
            sodium_mg=float(rng.uniform(80, 2600)),
            fiber_g=float(rng.uniform(0, 14)),
            protein_g=float(rng.uniform(3, 48)),
            tags=recipe_tags,
            latent=rng.normal(size=12).astype(np.float32),
        )
        recipe.health_score = HealthModule.score(recipe)
        recipes.append(recipe)

    users: List[UserProfile] = []
    for u in range(n_users):
        users.append(
            UserProfile(
                user_id=f"food_com_user_{10800 + u}",
                latent=rng.normal(size=12).astype(np.float32),
                preferred_tags=list(rng.choice(tags, size=3, replace=False)),
                historical_mean_rating=float(rng.uniform(3.2, 4.7)),
                big_five={
                    k: float(rng.uniform(0.15, 0.95))
                    for k in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
                },
            )
        )

    interactions: List[Tuple[str, str, float]] = []
    for user in users:
        for recipe in rng.choice(recipes, size=min(45, len(recipes)), replace=False):
            affinity = np.dot(user.latent, recipe.latent) / (np.linalg.norm(user.latent) * np.linalg.norm(recipe.latent) + 1e-8)
            rating = float(np.clip(3.2 + affinity + rng.normal(0, 0.65), 1.0, 5.0))
            interactions.append((user.user_id, recipe.recipe_id, rating))
    return users, recipes, interactions
