"""Data structures and synthetic/real Food.com-style data generation."""

from __future__ import annotations

import csv
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .health import HealthModule

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


# URL for a clean, public sample of the Food.com dataset
#FOODCOM_RECIPES_URL = "https://raw.githubusercontent.com/majid-sari/FoodCom-Dataset-Sample/main/recipes_sample.csv"
#FOODCOM_RATINGS_URL = "https://raw.githubusercontent.com/majid-sari/FoodCom-Dataset-Sample/main/interactions_sample.csv"


def ensure_data_directory(data_dir: str = "data") -> Tuple[str, str]:
    """Check if the dataset exists locally; download it if it's missing."""
    os.makedirs(data_dir, exist_ok=True)
    recipes_path = os.path.join(data_dir, "RAW_recipes.csv")
    ratings_path = os.path.join(data_dir, "RAW_interactions.csv")

    if not os.path.exists(recipes_path):
        print(f"[Data Pipeline] Recipes data not found in '{data_dir}/'. ")
        raise FileNotFoundError(
        f"Recipes data not found: '{recipes_path}'"
        )
    if not os.path.exists(ratings_path):
        print(f"[Data Pipeline] Interactions data not found in '{data_dir}/'.")
        raise FileNotFoundError(
                f"interactions data not found: '{ratings_path}'"
        )
    return recipes_path, ratings_path


def load_real_foodcom(
    recipes_path: str, 
    ratings_path: str, 
    seed: int, 
    n_users: int, 
    n_recipes: int
) -> Tuple[List[UserProfile], List[Recipe], List[Tuple[str, str, float]]] | None:
    """Attempt to parse local Food.com CSV files into structured objects."""
    if not (os.path.exists(recipes_path) and os.path.exists(ratings_path)):
        return None

    try:
        rng = np.random.default_rng(seed)

        # 1. Parse Recipes CSV
        recipes: List[Recipe] = []
        recipe_lookup: Dict[str, Recipe] = {}

        with open(recipes_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= n_recipes:
                    break

                rec_id = str(row.get("recipe_id", row.get("id", f"rec_{idx:04d}")))
                name = row.get("name", f"Food.com Recipe {idx:04d}")

                # Extract or generate tags
                raw_tags = row.get("tags", "comfort,quick").replace("'", "").replace("[", "").replace("]", "")
                tags_list = [t.strip() for t in raw_tags.split(",") if t.strip()][:5]
                if not tags_list:
                    tags_list = ["comfort", "quick"]

                recipe = Recipe(
                    recipe_id=rec_id,
                    name=name,
                    calories=float(row.get("calories", rng.uniform(180, 950))),
                    saturated_fat_g=float(row.get("saturated_fat_g", rng.uniform(1, 24))),
                    sugar_g=float(row.get("sugar_g", rng.uniform(1, 65))),
                    sodium_mg=float(row.get("sodium_mg", rng.uniform(80, 2600))),
                    fiber_g=float(row.get("fiber_g", rng.uniform(0, 14))),
                    protein_g=float(row.get("protein_g", rng.uniform(3, 48))),
                    tags=tags_list,
                    latent=rng.normal(size=12).astype(np.float32),
                )
                recipe.health_score = HealthModule.score(recipe)
                recipes.append(recipe)
                recipe_lookup[rec_id] = recipe

        if not recipes:
            return None

        # 2. Parse Interactions CSV
        raw_interactions: List[Tuple[str, str, float]] = []
        user_ratings_map: Dict[str, List[float]] = {}

        with open(ratings_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u_id = str(row.get("user_id", ""))
                r_id = str(row.get("recipe_id", ""))
                try:
                    rating = float(row.get("rating", 5.0))
                except ValueError:
                    rating = 5.0

                if r_id in recipe_lookup:
                    raw_interactions.append((u_id, r_id, rating))
                    user_ratings_map.setdefault(u_id, []).append(rating)

        # Filter top N active users
        sorted_users = sorted(user_ratings_map.keys(), key=lambda u: len(user_ratings_map[u]), reverse=True)[:n_users]
        active_user_set = set(sorted_users)

        # 3. Build User Profiles
        users: List[UserProfile] = []
        tags = ["comfort", "quick", "vegetarian", "spicy", "sweet", "high-protein", "low-sodium", "fresh", "pasta", "soup"]

        for u_id in sorted_users:
            mean_r = float(np.mean(user_ratings_map[u_id])) if u_id in user_ratings_map else 4.0
            users.append(
                UserProfile(
                    user_id=f"food_com_user_{u_id}",
                    latent=rng.normal(size=12).astype(np.float32),
                    preferred_tags=list(rng.choice(tags, size=3, replace=False)),
                    historical_mean_rating=mean_r,
                    big_five={
                        k: float(rng.uniform(0.15, 0.95))
                        for k in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
                    },
                )
            )

        # Filter interactions matching selected users
        interactions = [
            (f"food_com_user_{u}", r, float(rating))
            for u, r, rating in raw_interactions
            if u in active_user_set
        ]

        if not users or not interactions:
            return None

        print(f"[Data Pipeline] Loaded real dataset: {len(users)} users, {len(recipes)} recipes, {len(interactions)} interactions.")
        return users, recipes, interactions

    except Exception as e:
        print(f"[Data Pipeline] Note: Falling back to synthetic generation due to parsing issue ({e}).")
        return None


def build_synthetic_foodcom(
    seed: int, 
    n_users: int = 80, 
    n_recipes: int = 350
) -> Tuple[List[UserProfile], List[Recipe], List[Tuple[str, str, float]]]:
    """Build a deterministic Food.com-like catalogue, users, and rating table."""
    
    # Step 1: Check directory and attempt downloading/loading real data
    recipes_path, ratings_path = ensure_data_directory("data")
    real_data = load_real_foodcom(recipes_path, ratings_path, seed, n_users, n_recipes)
    
    if real_data is not None:
        return real_data

    # Step 2: Synthetic Generator Fallback (Preserved original shell)

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