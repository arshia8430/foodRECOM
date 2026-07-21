"""Layer-A recommendation decision logic."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from .data import Recipe
from .taste import TasteModule


def rank_candidates(recipes: Sequence[Recipe], taste: TasteModule, user_id: str, strategy: str, param: float, k: int = 5) -> List[Dict[str, Any]]:
    rows = [
        {"recipe_id": r.recipe_id, "name": r.name, "taste": taste.predict(user_id, r.recipe_id), "health": r.health_score, "tags": r.tags}
        for r in recipes
    ]
    if "CONSTRAINED" in strategy:
        filtered = [x for x in rows if x["health"] >= param] or sorted(rows, key=lambda x: x["health"], reverse=True)[:50]
        ranked = sorted(filtered, key=lambda x: x["taste"], reverse=True)
    elif "LEXICOGRAPHIC" in strategy:
        ranked = sorted(rows, key=lambda x: (x["health"], x["taste"]), reverse=True)
    elif "PARETO" in strategy:
        frontier = [
            x
            for x in rows
            if not any(y["taste"] >= x["taste"] and y["health"] >= x["health"] and y != x for y in rows)
        ]
        ranked = sorted(frontier, key=lambda x: abs(math.atan2(x["health"], x["taste"]) - param))
    else:
        ranked = sorted(rows, key=lambda x: param * x["taste"] + (1 - param) * x["health"], reverse=True)
    return ranked[:k]
