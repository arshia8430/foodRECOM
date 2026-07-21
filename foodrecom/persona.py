"""LLM persona client with stateless batched candidate evaluation.

The client is intentionally OpenAI-compatible rather than OpenAI-specific: the
caller supplies the provider URL, API key, and model name, and the simulator
posts one Top-5 candidate batch per meal to ``/chat/completions``. If the
provider is unavailable or returns malformed data, a deterministic persona
heuristic supplies the same structured schema so long simulations can continue.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import requests

from .data import UserProfile
from .environment import DynamicState


VALID_ACTIONS = {"ACCEPTED", "REJECTED_ALL"}


@dataclass
class PersonaCallStats:
    """Counters exposed in experiment logs for LLM auditability."""

    api_calls_attempted: int = 0
    api_calls_succeeded: int = 0
    fallback_calls: int = 0
    malformed_responses: int = 0


class LLMPersonaClient:
    """OpenAI-compatible stateless LLM persona evaluator."""

    def __init__(self, api_key: str, provider_url: str, model_name: str, timeout_s: int = 30):
        self.api_key = api_key
        self.provider_url = self._normalize_provider_url(provider_url)
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.stats = PersonaCallStats()

    @staticmethod
    def _normalize_provider_url(provider_url: str) -> str:
        """Accept either a provider root or a versioned API base URL."""
        cleaned = (provider_url or "https://api.openai.com/v1").strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned[: -len("/chat/completions")]
        return cleaned

    @property
    def chat_completions_url(self) -> str:
        return f"{self.provider_url}/chat/completions"

    def choose(self, user: UserProfile, state: DynamicState, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all Top-5 candidates in one stateless prompt and return JSON-like output."""
        if not candidates:
            return self._fallback(state, candidates, "No candidates supplied.")
        if not self.api_key:
            return self._fallback(state, candidates, "No API key supplied.")

        self.stats.api_calls_attempted += 1
        request_json = self._build_request(user, state, candidates)
        try:
            response = requests.post(
                self.chat_completions_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=request_json,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            content = self._extract_content(response.json())
            parsed = self._parse_json_object(content)
            validated = self._validate_choice(parsed, candidates)
            self.stats.api_calls_succeeded += 1
            return validated
        except Exception as exc:
            self.stats.malformed_responses += 1
            return self._fallback(state, candidates, f"LLM fallback: {exc}")

    def _build_request(self, user: UserProfile, state: DynamicState, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = {
            "instruction": (
                "Act as the specified Food.com user for this single meal only. "
                "Use only the static persona, current state, and Top-5 candidates below. "
                "Do not rely on prior conversation history. Choose one recipe or reject all."
            ),
            "static_persona": {
                "user_id": user.user_id,
                "historical_mean_rating": round(user.historical_mean_rating, 3),
                "preferred_tags": user.preferred_tags,
                "big_five": user.big_five,
            },
            "current_step_state": state.__dict__,
            "top_5_candidates": candidates[:5],
            "required_json_schema": {
                "selected_recipe_id": "one candidate recipe_id if ACCEPTED, otherwise null",
                "action": "ACCEPTED or REJECTED_ALL",
                "perceived_satisfaction": "number in [0, 1]",
                "rationale": "brief explanation grounded in persona/state/candidates",
            },
        }
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You simulate a Food.com user persona for a diet recommendation experiment. "
                        "Return exactly one valid JSON object and no surrounding prose."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("provider response has no choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            content = "".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("provider response has empty message content")
        return content

    @staticmethod
    def _parse_json_object(content: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end < start:
                raise
            parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("provider JSON is not an object")
        return parsed

    @staticmethod
    def _validate_choice(parsed: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        candidate_ids = {str(c["recipe_id"]) for c in candidates}
        action = str(parsed.get("action", "")).upper()
        selected = parsed.get("selected_recipe_id")
        if selected is not None:
            selected = str(selected)
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid action: {action}")
        if action == "ACCEPTED" and selected not in candidate_ids:
            raise ValueError("accepted recipe is not in candidate batch")
        if action == "REJECTED_ALL":
            selected = None
        satisfaction = float(np.clip(float(parsed.get("perceived_satisfaction", 0.0)), 0.0, 1.0))
        return {
            "selected_recipe_id": selected,
            "action": action,
            "perceived_satisfaction": satisfaction,
            "rationale": str(parsed.get("rationale", ""))[:500],
            "source": "llm",
        }

    def _fallback(self, state: DynamicState, candidates: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
        self.stats.fallback_calls += 1
        return self._heuristic(state, candidates, reason)

    @staticmethod
    def _heuristic(state: DynamicState, candidates: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
        if not candidates:
            return {
                "selected_recipe_id": None,
                "action": "REJECTED_ALL",
                "perceived_satisfaction": 0.0,
                "rationale": reason,
                "source": "heuristic_fallback",
            }
        comfort_need = (state.stress_level + state.hunger_level + state.diet_fatigue + (10.0 - state.sleep_quality)) / 40.0
        willpower_factor = state.willpower_bank / 100.0
        scored = []
        for candidate in candidates:
            taste = float(candidate["taste"])
            health = float(candidate["health"])
            comfort_score = taste * (0.60 + comfort_need) + health * (0.65 + 0.35 * willpower_factor - 0.30 * comfort_need)
            tag_match = 0.03 * len(candidate.get("persona_tag_overlap", []))
            scored.append((comfort_score + tag_match, candidate))
        best_score, best = max(scored, key=lambda x: x[0])
        reject_risk = 0.12 + 0.38 * (state.diet_fatigue / 10.0) + 0.20 * max(0.0, 0.50 - float(best["taste"]))
        reject_risk += 0.12 * max(0.0, 0.30 - state.willpower_bank / 100.0)
        if random.random() < float(np.clip(reject_risk, 0.02, 0.85)):
            return {
                "selected_recipe_id": None,
                "action": "REJECTED_ALL",
                "perceived_satisfaction": 0.0,
                "rationale": reason,
                "source": "heuristic_fallback",
            }
        return {
            "selected_recipe_id": best["recipe_id"],
            "action": "ACCEPTED",
            "perceived_satisfaction": float(np.clip(best_score, 0.0, 1.0)),
            "rationale": reason,
            "source": "heuristic_fallback",
        }
