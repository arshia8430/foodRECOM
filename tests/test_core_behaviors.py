import numpy as np
import torch

from foodrecom.data import Recipe, UserProfile
from foodrecom.environment import DynamicState, update_long_state
from foodrecom.experiment import _select_policy_param, run_simulation_experiment
from foodrecom.health import HealthModule
from foodrecom.persona import LLMPersonaClient
from foodrecom.policies import LinUCBPolicy, SACPolicy


def test_lincub_constrained_static_fallback_uses_health_threshold():
    assert _select_policy_param("STATIC_CONSTRAINED", None, np.zeros(6, dtype=np.float32)) == 0.65


def test_lincub_policy_branch_still_selects_action():
    policy = LinUCBPolicy(context_dim=6, actions=[0.2, 0.8])
    assert _select_policy_param("LINUCB_CONSTRAINED", policy, np.ones(6, dtype=np.float32)) in {0.2, 0.8}


def test_rejecting_costs_at_least_as_much_willpower_as_strict_compliance():
    rejected_willpower, _ = update_long_state(50.0, 0.0, False, 0.0, 0.0)
    strict_willpower, _ = update_long_state(50.0, 0.0, True, 0.2, 0.9)
    assert rejected_willpower <= strict_willpower


def test_sac_updates_weights_after_short_run_threshold():
    policy = SACPolicy(state_dim=6, seed=123, min_replay_size=2)
    before = [p.detach().clone() for p in policy.actor.parameters()]
    state = np.zeros(6, dtype=np.float32)
    next_state = np.ones(6, dtype=np.float32) * 0.1
    policy.update(state, 0.5, 0.1, next_state, False)
    policy.update(next_state, 0.4, 0.2, state, True)
    assert policy.gradient_updates > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.actor.parameters()))


def test_health_score_bounds():
    module = HealthModule()
    recipe = Recipe("r", "test", 2000, 30, 80, 4000, 1, 40, [], np.zeros(12, dtype=np.float32))
    score = module.score(recipe)
    assert 0.0 <= score <= 1.0


def test_persona_validation_rejects_malformed_choice_and_fallback_schema():
    user = UserProfile("u1", {"neuroticism": 0.5}, ["quick"], np.zeros(12, dtype=np.float32), 4.0)
    state = DynamicState(1, 1, 5, 22, 6, 7, 100, 0)
    client = LLMPersonaClient(api_key="", provider_url="https://example.test/v1", model_name="test")
    choice = client.choose(user, state, [{"recipe_id": "r1", "taste": 0.8, "health": 0.8}])
    assert choice["action"] in {"ACCEPTED", "REJECTED_ALL"}
    assert choice["source"] == "heuristic_fallback"
    assert client.stats.fallback_calls == 1


def test_short_sac_experiment_records_real_training_status():
    result = run_simulation_experiment(api_key="", num_days=2, meals_per_day=2, strategy_type="SAC_PARETO", num_people=1, seed=7)
    statuses = result["execution_logs"]["sac_training_status"]
    assert statuses
    assert all(status["replay_size"] == 4 for status in statuses.values())
    assert all(status["trained"] for status in statuses.values())
