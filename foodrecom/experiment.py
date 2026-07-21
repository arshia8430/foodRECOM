"""Single-entrypoint simulation driver."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from .data import build_synthetic_foodcom
from .environment import sample_dynamic_state, update_long_state
from .persona import LLMPersonaClient
from .policies import LinUCBPolicy, SACPolicy
from .recommender import rank_candidates
from .taste import TasteModule
from .utils import set_global_seed


def run_simulation_experiment(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model_name: str = "gpt-4o-mini",
    num_days: int = 30,
    meals_per_day: int = 3,
    strategy_type: str = "LINUCB_WEIGHTED",
    target_user_id: str = "food_com_user_10842",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute the adaptive LLM-persona food recommendation loop."""
    set_global_seed(seed)
    users, recipes, interactions = build_synthetic_foodcom(seed)
    if target_user_id not in {u.user_id for u in users}:
        target_user_id = users[0].user_id
    user = next(u for u in users if u.user_id == target_user_id)

    taste = TasteModule(users, recipes, seed)
    taste.train_from_interactions(interactions)
    persona = LLMPersonaClient(api_key, base_url, model_name)
    linucb = LinUCBPolicy(context_dim=6, actions=[0.2, 0.35, 0.5, 0.65, 0.8])
    sac = SACPolicy(state_dim=6, seed=seed)

    willpower, fatigue, sleep_quality = 100.0, 0.0, 7.0
    trajectory: List[Dict[str, Any]] = []
    diet_failure_step: Optional[int] = None
    total_steps = int(num_days * meals_per_day)

    for step in range(total_steps):
        day, meal = step // meals_per_day + 1, step % meals_per_day + 1
        if meal == 1:
            sleep_quality = float(np.clip(np.random.normal(7.0, 1.5), 1, 10))
        state = sample_dynamic_state(day, meal, meals_per_day, willpower, fatigue, sleep_quality)
        context = state.as_vector()

        if strategy_type.startswith("STATIC"):
            param = 0.5
        elif strategy_type.startswith("SAC"):
            param = sac.select(context)
        elif strategy_type.startswith("LINUCB"):
            param = linucb.select(context)
        else:
            param = 0.5

        candidates = rank_candidates(recipes, taste, target_user_id, strategy_type, param)
        choice = persona.choose(user, state, candidates)
        accepted = choice.get("action") == "ACCEPTED" and choice.get("selected_recipe_id") is not None
        selected = next((c for c in candidates if c["recipe_id"] == choice.get("selected_recipe_id")), None)
        health = float(selected["health"] if selected else 0.0)
        pleasure = float(selected["taste"] if selected else 0.0)
        reward = (0.45 * health + 0.35 * pleasure + 0.20 * float(accepted)) if accepted else -0.15

        if strategy_type.startswith("LINUCB"):
            linucb.update(context, reward)
        if strategy_type.startswith("SAC"):
            sac.update(context, param, reward)
        willpower, fatigue = update_long_state(willpower, fatigue, accepted, pleasure, health)
        if diet_failure_step is None and (not accepted or willpower <= 5 or fatigue >= 9.5):
            diet_failure_step = step + 1

        trajectory.append(
            {
                "step": step + 1,
                "state": state.__dict__,
                "policy_param": param,
                "candidates": candidates,
                "llm_choice": choice,
                "accepted": accepted,
                "health": health,
                "taste": pleasure,
                "reward": reward,
            }
        )

    accepted_rows = [x for x in trajectory if x["accepted"]]
    metrics = {
        "adherence_rate": float(np.mean([x["accepted"] for x in trajectory])),
        "avg_health": float(np.mean([x["health"] for x in accepted_rows])) if accepted_rows else 0.0,
        "avg_taste": float(np.mean([x["taste"] for x in accepted_rows])) if accepted_rows else 0.0,
        "time_to_diet_failure": diet_failure_step,
        "total_steps": total_steps,
    }
    return {
        "config": {
            "num_days": num_days,
            "meals_per_day": meals_per_day,
            "strategy_type": strategy_type,
            "target_user_id": target_user_id,
            "seed": seed,
        },
        "metrics_summary": metrics,
        "trajectory_history": trajectory,
        "execution_logs": {
            "trained_taste_model": taste.trained,
            "taste_training_losses": taste.training_losses,
            "fallback_llm_enabled": True,
            "generated_at_unix": time.time(),
        },
    }
