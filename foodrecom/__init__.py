"""Modular food recommendation simulator package."""

from .experiment import run_simulation_experiment
from .taste import NCF, TasteModule, train_and_preview_taste_model
from .health import HealthModule

__all__ = ["run_simulation_experiment", "NCF", "TasteModule", "HealthModule", "train_and_preview_taste_model"]
