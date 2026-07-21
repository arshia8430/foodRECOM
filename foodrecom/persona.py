"""LLM persona client with robust fallback behavior."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List

import numpy as np
import requests

from .data import UserProfile
from .environment import DynamicState


class LLMPersonaClient:
    def __init__(self, api_key: str, base_url: str, model_name: str, timeout_s: int = 30):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_s = timeout_s

    def choose(self, user: UserProfile, state: DynamicState, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.api_key:
            return self._heuristic(state, candidates, "No API key supplied.")
        prompt = {
            "persona": user.__dict__ | {"latent": "omitted"},
            "state": state.__dict__,
            "candidates": candidates,
            "schema": {
                "selected_recipe_id": "string or null",
                "action": "ACCEPTED or REJECTED_ALL",
                "perceived_satisfaction": "0..1",
                "rationale": "short",
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a Food.com user persona. Return only valid JSON."},
                        {"role": "user", "content": json.dumps(prompt)},
                    ],
                    "temperature": 0.2,
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content[content.find("{") : content.rfind("}") + 1])
            if parsed.get("action") in {"ACCEPTED", "REJECTED_ALL"}:
                return parsed
        except Exception as exc:
            return self._heuristic(state, candidates, f"LLM fallback: {exc}")
        return self._heuristic(state, candidates, "LLM returned invalid schema.")

    @staticmethod
    def _heuristic(state: DynamicState, candidates: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
        comfort_need = (state.stress_level + state.hunger_level + state.diet_fatigue) / 30.0
        scored = [(c["taste"] * (0.55 + comfort_need) + c["health"] * (0.75 - 0.35 * comfort_need), c) for c in candidates]
        best_score, best = max(scored, key=lambda x: x[0])
        reject_risk = 0.18 + 0.35 * (state.diet_fatigue / 10.0) + 0.25 * max(0, 0.45 - best["taste"])
        if random.random() < reject_risk:
            return {"selected_recipe_id": None, "action": "REJECTED_ALL", "perceived_satisfaction": 0.0, "rationale": reason}
        return {
            "selected_recipe_id": best["recipe_id"],
            "action": "ACCEPTED",
            "perceived_satisfaction": float(np.clip(best_score, 0, 1)),
            "rationale": reason,
        }
