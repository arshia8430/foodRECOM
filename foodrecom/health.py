"""Objective recipe health scoring."""

from __future__ import annotations

import numpy as np

from .data import Recipe


class HealthModule:
    """Deterministic Nutri-Score-inspired health scorer normalized to [0, 1]."""

    @staticmethod
    def score(recipe: Recipe) -> float:
        negative = (
            min(recipe.calories / 800.0, 1.5) * 0.30
            + min(recipe.saturated_fat_g / 20.0, 1.5) * 0.20
            + min(recipe.sugar_g / 50.0, 1.5) * 0.20
            + min(recipe.sodium_mg / 2300.0, 1.5) * 0.30
        )
        positive = min(recipe.fiber_g / 12.0, 1.0) * 0.45 + min(recipe.protein_g / 40.0, 1.0) * 0.55
        raw = 0.55 + 0.45 * positive - 0.55 * negative
        return float(np.clip(raw, 0.0, 1.0))
