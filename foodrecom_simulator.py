"""Backward-compatible entrypoint for the modular foodrecom package."""

from __future__ import annotations

import json

from foodrecom import HealthModule, NCF, TasteModule, run_simulation_experiment, train_and_preview_taste_model

__all__ = ["HealthModule", "NCF", "TasteModule", "run_simulation_experiment", "train_and_preview_taste_model"]


if __name__ == "__main__":
    result = run_simulation_experiment(api_key="", num_days=1, meals_per_day=3)
    print(json.dumps(result["metrics_summary"], indent=2))
