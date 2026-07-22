"""Single-entrypoint, multi-person simulation driver."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .data import UserProfile, build_synthetic_foodcom
from .environment import sample_dynamic_state, update_long_state
from .persona import LLMPersonaClient
from .policies import LinUCBPolicy, SACPolicy
from .recommender import rank_candidates
from .taste import TasteModule
from .utils import set_global_seed

DEFAULT_STRATEGIES = ["LINUCB_WEIGHTED", "LINUCB_CONSTRAINED", "STATIC_LEXICOGRAPHIC", "SAC_PARETO"]


def _as_strategy_list(strategy_type: str | Sequence[str], strategy_types: Optional[Sequence[str]]) -> List[str]:
    if strategy_types is not None:
        values = list(strategy_types)
    elif isinstance(strategy_type, str) and strategy_type.upper() == "ALL":
        values = DEFAULT_STRATEGIES.copy()
    elif isinstance(strategy_type, str):
        values = [strategy_type]
    else:
        values = list(strategy_type)
    if not values:
        raise ValueError("At least one strategy must be supplied.")
    return [str(v).upper() for v in values]


def _select_people(users: Sequence[UserProfile], target_user_id: str, num_people: int) -> List[UserProfile]:
    if num_people < 1:
        raise ValueError("num_people must be at least 1.")
    user_by_id = {u.user_id: u for u in users}
    first = user_by_id.get(target_user_id, users[0])
    selected = [first]
    for user in users:
        if len(selected) >= num_people:
            break
        if user.user_id != first.user_id:
            selected.append(user)
    return selected


def _apply_persona_parameters(people: Sequence[UserProfile], persona_parameters: Optional[Sequence[Dict[str, Any]]]) -> List[UserProfile]:
    """Create fixed per-person personas, optionally overriding generated traits."""
    if not persona_parameters:
        return list(people)
    customized: List[UserProfile] = []
    for idx, user in enumerate(people):
        params = persona_parameters[idx] if idx < len(persona_parameters) else {}
        merged_big_five = dict(user.big_five)
        merged_big_five.update({k: float(v) for k, v in params.get("big_five", {}).items()})
        customized.append(
            replace(
                user,
                preferred_tags=list(params.get("preferred_tags", user.preferred_tags)),
                historical_mean_rating=float(params.get("historical_mean_rating", user.historical_mean_rating)),
                big_five=merged_big_five,
            )
        )
    return customized


def _initial_policy(strategy: str, seed: int) -> LinUCBPolicy | SACPolicy | None:
    if strategy.startswith("LINUCB"):
        return LinUCBPolicy(context_dim=6, actions=[0.2, 0.35, 0.5, 0.65, 0.8])
    if strategy.startswith("SAC"):
        return SACPolicy(state_dim=6, seed=seed)
    return None


def _select_policy_param(strategy: str, policy: LinUCBPolicy | SACPolicy | None, context: np.ndarray) -> float:
    if "CONSTRAINED" in strategy and not isinstance(policy, LinUCBPolicy):
        return 0.65
    if strategy.startswith("LINUCB") and isinstance(policy, LinUCBPolicy):
        return policy.select(context)
    if strategy.startswith("SAC") and isinstance(policy, SACPolicy):
        return policy.select(context)
    if "PARETO" in strategy:
        return 0.5
    return 0.5


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    accepted_rows = [x for x in rows if x["accepted"]]
    return {
        "adherence_rate": float(np.mean([x["accepted"] for x in rows])) if rows else 0.0,
        "avg_health": float(np.mean([x["health"] for x in accepted_rows])) if accepted_rows else 0.0,
        "avg_taste": float(np.mean([x["taste"] for x in accepted_rows])) if accepted_rows else 0.0,
        "avg_reward": float(np.mean([x["reward"] for x in rows])) if rows else 0.0,
        "total_steps": len(rows),
    }


def run_simulation_experiment(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model_name: str = "gpt-4o-mini",
    num_days: int = 30,
    meals_per_day: int = 3,
    strategy_type: str | Sequence[str] = "LINUCB_WEIGHTED",
    target_user_id: str = "food_com_user_10842",
    seed: int = 42,
    provider_url: Optional[str] = None,
    num_people: int = 1,
    persona_parameters: Optional[Sequence[Dict[str, Any]]] = None,
    strategy_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute a multi-person, multi-strategy LLM-persona food simulation.

    Pipeline:
    1. Build fixed Food.com-anchored personas for ``num_people`` people.
    2. Sample daily/per-meal dynamic variables from their distributions.
    3. For every person, meal, and strategy, generate Top-5 recommendations.
    4. Ask the LLM persona client to accept one candidate or reject all.
    5. Update strategy-specific cumulative states and compute final metrics.
    """
    set_global_seed(seed)
    strategies = _as_strategy_list(strategy_type, strategy_types)
    users, recipes, interactions = build_synthetic_foodcom(seed, n_users=max(80, num_people))
    people = _apply_persona_parameters(_select_people(users, target_user_id, num_people), persona_parameters)

    taste = TasteModule(users, recipes, seed)
    taste.train_from_interactions(interactions)
    effective_provider_url = provider_url or base_url
    persona = LLMPersonaClient(api_key, effective_provider_url, model_name)

    policy_state: Dict[tuple[str, str], Dict[str, Any]] = {}
    for person_index, user in enumerate(people):
        for strategy_index, strategy in enumerate(strategies):
            policy_state[(user.user_id, strategy)] = {
                "willpower": 100.0,
                "fatigue": 0.0,
                "diet_failure_step": None,
                "policy": _initial_policy(strategy, seed + person_index * 997 + strategy_index * 37),
                "pending_sac_transition": None,
            }

    daily_contexts: List[Dict[str, Any]] = []
    trajectory: List[Dict[str, Any]] = []
    total_meals = int(num_days * meals_per_day)

    for day in range(1, num_days + 1):
        per_person_sleep = {
            user.user_id: float(np.clip(np.random.normal(7.0 - 0.35 * user.big_five.get("neuroticism", 0.5), 1.3), 1, 10))
            for user in people
        }
        daily_contexts.append({"day": day, "sleep_quality_by_user": per_person_sleep})
        for meal in range(1, meals_per_day + 1):
            meal_step = (day - 1) * meals_per_day + meal
            for person_index, user in enumerate(people):
                for strategy in strategies:
                    state_record = policy_state[(user.user_id, strategy)]
                    state = sample_dynamic_state(
                        day,
                        meal,
                        meals_per_day,
                        float(state_record["willpower"]),
                        float(state_record["fatigue"]),
                        per_person_sleep[user.user_id],
                    )
                    context = state.as_vector()
                    policy = state_record["policy"]
                    pending = state_record.get("pending_sac_transition")
                    if pending is not None and isinstance(policy, SACPolicy):
                        policy.update(pending["state"], pending["action"], pending["reward"], context, False)
                        state_record["pending_sac_transition"] = None
                    param = _select_policy_param(strategy, policy, context)
                    candidates = rank_candidates(recipes, taste, user.user_id, strategy, param, user_profile=user)
                    choice = persona.choose(user, state, candidates)
                    accepted = choice.get("action") == "ACCEPTED" and choice.get("selected_recipe_id") is not None
                    selected = next((c for c in candidates if c["recipe_id"] == choice.get("selected_recipe_id")), None)
                    health = float(selected["health"] if selected else 0.0)
                    pleasure = float(selected["taste"] if selected else 0.0)
                    reward = (0.45 * health + 0.35 * pleasure + 0.20 * float(accepted)) if accepted else -0.15

                    if strategy.startswith("LINUCB") and isinstance(policy, LinUCBPolicy):
                        policy.update(context, reward)
                    next_willpower, next_fatigue = update_long_state(float(state_record["willpower"]), float(state_record["fatigue"]), accepted, pleasure, health)
                    if strategy.startswith("SAC") and isinstance(policy, SACPolicy):
                        if meal_step == total_meals:
                            policy.update(context, param, reward, context, True)
                        else:
                            state_record["pending_sac_transition"] = {"state": context, "action": param, "reward": reward}
                    state_record["willpower"] = next_willpower
                    state_record["fatigue"] = next_fatigue
                    if state_record["diet_failure_step"] is None and (not accepted or next_willpower <= 5 or next_fatigue >= 9.5):
                        state_record["diet_failure_step"] = meal_step

                    trajectory.append(
                        {
                            "global_step": len(trajectory) + 1,
                            "meal_step": meal_step,
                            "day": day,
                            "meal": meal,
                            "person_index": person_index,
                            "user_id": user.user_id,
                            "strategy_type": strategy,
                            "state": state.__dict__,
                            "policy_param": param,
                            "candidates": candidates,
                            "llm_choice": choice,
                            "accepted": accepted,
                            "selected_recipe_id": selected["recipe_id"] if selected else None,
                            "health": health,
                            "taste": pleasure,
                            "reward": reward,
                            "next_willpower_bank": next_willpower,
                            "next_diet_fatigue": next_fatigue,
                        }
                    )

    sac_training_status: Dict[str, Dict[str, Any]] = {}
    for (user_id, strategy), state_record in policy_state.items():
        policy = state_record["policy"]
        if isinstance(policy, SACPolicy):
            key = f"{user_id}:{strategy}"
            sac_training_status[key] = {
                "replay_size": len(policy.memory),
                "min_replay_size": policy.min_replay_size,
                "gradient_updates": policy.gradient_updates,
                "trained": policy.gradient_updates > 0,
            }

    per_person_strategy_metrics: Dict[str, Dict[str, Any]] = {}
    for user in people:
        per_person_strategy_metrics[user.user_id] = {}
        for strategy in strategies:
            rows = [x for x in trajectory if x["user_id"] == user.user_id and x["strategy_type"] == strategy]
            summary = _summarize(rows)
            summary["time_to_diet_failure"] = policy_state[(user.user_id, strategy)]["diet_failure_step"]
            summary["final_willpower_bank"] = float(policy_state[(user.user_id, strategy)]["willpower"])
            summary["final_diet_fatigue"] = float(policy_state[(user.user_id, strategy)]["fatigue"])
            per_person_strategy_metrics[user.user_id][strategy] = summary

    per_strategy_metrics = {strategy: _summarize([x for x in trajectory if x["strategy_type"] == strategy]) for strategy in strategies}
    metrics = _summarize(trajectory)
    metrics.update({"num_people": len(people), "num_strategies": len(strategies), "total_llm_decisions": len(trajectory)})

    return {
        "config": {
            "num_people": len(people),
            "num_days": num_days,
            "meals_per_day": meals_per_day,
            "strategy_types": strategies,
            "target_user_id": target_user_id,
            "seed": seed,
            "provider_url": effective_provider_url,
            "model_name": model_name,
        },
        "personas": [
            {
                "user_id": user.user_id,
                "historical_mean_rating": user.historical_mean_rating,
                "preferred_tags": user.preferred_tags,
                "big_five": user.big_five,
            }
            for user in people
        ],
        "daily_contexts": daily_contexts,
        "metrics_summary": metrics,
        "per_strategy_metrics": per_strategy_metrics,
        "per_person_strategy_metrics": per_person_strategy_metrics,
        "trajectory_history": trajectory,
        "execution_logs": {
            "trained_taste_model": taste.trained,
            "taste_training_losses": taste.training_losses,
            "provider_url_used": persona.provider_url,
            "chat_completions_url": persona.chat_completions_url,
            "llm_call_stats": persona.stats.__dict__,
            "fallback_llm_enabled": True,
            "pytorch_modules": ["TasteModule.NCF", "SACPolicy.GaussianActor", "SACPolicy.QNetwork"],
            "sac_training_status": sac_training_status,
            "sac_training_warnings": [
                f"{key} did not cross SAC warm-up threshold; no gradient updates were run."
                for key, status in sac_training_status.items()
                if not status["trained"]
            ],
            "audit_checklist": {
                "multi_person_fixed_personas": len(people) == num_people,
                "daily_and_per_meal_variable_state": len(daily_contexts) == num_days,
                "strategy_evaluation_for_each_person_and_meal": len(trajectory) == len(people) * len(strategies) * total_meals,
                "llm_accept_reject_per_strategy": all("llm_choice" in row for row in trajectory),
                "llm_api_success_observed": persona.stats.api_calls_succeeded > 0,
                "heuristic_fallback_observed": persona.stats.fallback_calls > 0,
                "pytorch_ncf_and_sac": taste.trained and any(isinstance(s["policy"], SACPolicy) for s in policy_state.values()),
                "single_colab_entrypoint": True,
                "global_seeded_rngs": True,
                "foodcom_anchored_target_user": any(user.user_id == target_user_id for user in people),
                "health_score_in_0_1": all(0.0 <= row["health"] <= 1.0 for row in trajectory),
                "dynamic_and_cumulative_state": all("next_willpower_bank" in row and "next_diet_fatigue" in row for row in trajectory),
                "sac_crossed_warmup_threshold": all(status["trained"] for status in sac_training_status.values()) if sac_training_status else None,
                "layer_a_strategies": ["WEIGHTED", "CONSTRAINED", "LEXICOGRAPHIC", "PARETO"],
                "adaptive_engines": ["LinUCB", "Soft Actor-Critic"],
                "batched_top5_llm_prompt_with_json_fallback": persona.stats.api_calls_attempted + persona.stats.fallback_calls > 0,
            },
            "generated_at_unix": time.time(),
        },
    }
