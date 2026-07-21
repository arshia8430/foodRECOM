"""Layer-A recommendation decision logic."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from .data import Recipe, UserProfile
from .taste import TasteModule


def rank_candidates(
    recipes: Sequence[Recipe],
    taste: TasteModule,
    user_id: str,
    strategy: str,
    param: float,
    k: int = 5,
    user_profile: UserProfile | None = None,
) -> List[Dict[str, Any]]:
    """Rank recipes using one of the required Layer-A strategies.

    ``param`` is interpreted as a taste weight for weighted-sum, a minimum
    health threshold for constrained optimization, and an angular preference
    over the Pareto frontier for Pareto ranking.
    """
    preferred_tags = set(user_profile.preferred_tags if user_profile else [])
    rows = [
        {
            "recipe_id": r.recipe_id,
            "name": r.name,
            "taste": float(taste.predict(user_id, r.recipe_id)),
            "health": float(r.health_score),
            "tags": r.tags,
            "persona_tag_overlap": sorted(preferred_tags.intersection(r.tags)),
            "nutrition": {
                "calories": r.calories,
                "saturated_fat_g": r.saturated_fat_g,
                "sugar_g": r.sugar_g,
                "sodium_mg": r.sodium_mg,
                "fiber_g": r.fiber_g,
                "protein_g": r.protein_g,
            },
        }
        for r in recipes
    ]
    strategy_upper = strategy.upper()
    if "CONSTRAINED" in strategy_upper:
        filtered = [x for x in rows if x["health"] >= param] or sorted(rows, key=lambda x: x["health"], reverse=True)[:50]
        ranked = sorted(filtered, key=lambda x: (x["taste"], x["health"]), reverse=True)
    elif "LEXICOGRAPHIC" in strategy_upper:
        ranked = sorted(rows, key=lambda x: (x["health"], x["taste"]), reverse=True)
    elif "PARETO" in strategy_upper:
        frontier = [
            x
            for x in rows
            if not any(
                y["taste"] >= x["taste"] and y["health"] >= x["health"] and (y["taste"] > x["taste"] or y["health"] > x["health"])
                for y in rows
            )
        ]
        target_angle = float(param) * (math.pi / 2.0)
        ranked = sorted(frontier, key=lambda x: (abs(math.atan2(x["health"], x["taste"]) - target_angle), -(x["health"] + x["taste"])))
    else:
        taste_weight = float(min(max(param, 0.0), 1.0))
        ranked = sorted(rows, key=lambda x: taste_weight * x["taste"] + (1.0 - taste_weight) * x["health"], reverse=True)
    return ranked[:k]
