# foodRECOM

A modular Python framework for simulating an LLM-persona food recommendation loop with adaptive contextual-bandit and PyTorch Soft Actor-Critic policies.

## Module layout

- `foodrecom/data.py` defines Food.com-like `Recipe` and `UserProfile` records plus deterministic synthetic data generation.
- `foodrecom/health.py` defines the deterministic nutrition/health model.
- `foodrecom/taste.py` defines the taste and pleasure model (`NCF`, `TasteModule`) and can be run independently for testing.
- `foodrecom/environment.py` defines short-term dynamic state sampling and long-term willpower/fatigue updates.
- `foodrecom/policies.py` defines LinUCB plus a compact PyTorch Soft Actor-Critic actor-critic engine with twin Q networks and replay updates.
- `foodrecom/recommender.py` defines Layer-A ranking strategies.
- `foodrecom/persona.py` defines the LLM persona client and heuristic fallback.
- `foodrecom/experiment.py` defines the main `run_simulation_experiment(...)` entrypoint.

## Quick start: full simulation

```python
from foodrecom import run_simulation_experiment

result = run_simulation_experiment(
    api_key="",  # empty string uses deterministic heuristic fallback
    provider_url="https://api.openai.com/v1",  # OpenAI-compatible provider URL
    model_name="gpt-4o-mini",
    num_days=30,
    meals_per_day=3,
    strategy_type="LINUCB_WEIGHTED",
    seed=42,
)
print(result["metrics_summary"])
```

The main entrypoint accepts an explicit OpenAI-compatible `provider_url` (or the backward-compatible `base_url`) and returns a structured dictionary containing configuration, execution logs, metrics summary, and full trajectory history. The module generates a deterministic Food.com-like dataset when a real export is unavailable, making it suitable for Google Colab smoke tests without external data downloads. The returned `execution_logs.audit_checklist` records the implemented experimental-design requirements.

## Test only the taste / pleasure model

Use the dedicated module script to train the Neural Collaborative Filtering pleasure model and inspect predicted top recipes for a target user:

```bash
python -m foodrecom.taste --seed 42 --epochs 2 --target-user-id food_com_user_10842 --top-k 5
```

Or call it from Python:

```python
from foodrecom.taste import train_and_preview_taste_model

preview = train_and_preview_taste_model(seed=42, epochs=2, target_user_id="food_com_user_10842", top_k=5)
print(preview["training_losses"])
print(preview["top_predictions"])
```

## Dependencies

Install the Colab-compatible runtime dependencies with:

```bash
pip install -r requirements.txt
```
