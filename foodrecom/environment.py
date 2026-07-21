"""Dynamic psychophysiological state engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DynamicState:
    day: int
    meal: int
    stress_level: float
    ambient_temp_c: float
    hunger_level: float
    sleep_quality: float
    willpower_bank: float
    diet_fatigue: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.stress_level / 10.0,
                (self.ambient_temp_c - 10.0) / 32.0,
                self.hunger_level / 10.0,
                self.sleep_quality / 10.0,
                self.willpower_bank / 100.0,
                self.diet_fatigue / 10.0,
            ],
            dtype=np.float32,
        )


def sample_dynamic_state(day: int, meal: int, meals_per_day: int, willpower: float, fatigue: float, sleep_quality: float) -> DynamicState:
    hunger = float(np.clip(5.5 + 2.2 * math.sin((meal / meals_per_day) * math.pi) + np.random.normal(0, 1), 1, 10))
    return DynamicState(
        day=day,
        meal=meal,
        stress_level=float(np.clip(np.random.normal(5, 2), 1, 10)),
        ambient_temp_c=float(np.clip(np.random.normal(24, 6), 10, 42)),
        hunger_level=hunger,
        sleep_quality=sleep_quality,
        willpower_bank=willpower,
        diet_fatigue=fatigue,
    )


def update_long_state(willpower: float, fatigue: float, accepted: bool, taste_score: float, health_score: float) -> Tuple[float, float]:
    if not accepted:
        return max(0.0, willpower - 5.0), min(10.0, fatigue + 0.8)
    strict = health_score > 0.72 and taste_score < 0.55
    indulgent = taste_score > 0.75
    willpower += -8.0 if strict else 4.0 if indulgent else 0.5
    fatigue += 0.7 if strict else -0.5 if indulgent else -0.1
    return float(np.clip(willpower, 0, 100)), float(np.clip(fatigue, 0, 10))
