"""LLM-assisted parameter optimizer."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .pydantic_shim import ValidationError

from .registry import StrategyRegistry
from .schemas import OptimizeResponse

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class LLMOptimizer:
    """High-level helper for invoking the language model safely."""

    def __init__(
        self,
        registry: StrategyRegistry,
        data_service: DataService,
        prompt_dir: Path,
        client: Optional[Any] = None,
        model: str = "gpt-5",
        temperature: Optional[float] = 0.1,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> None:
        self.registry = registry
        self.data_service = data_service
        self.prompt_dir = prompt_dir
        self.client = client or self._build_default_client()
        self.model = model.strip()
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        if self.client is None:
            LOGGER.warning(
                "OpenAI client unavailable; provide a client instance or set OPENAI_API_KEY when running live optimizations."
            )

    def optimize(
        self,
        strategy_id: str,
        timeframe: str,
        data_summary: Dict[str, Any],
        prior_bests: Optional[Iterable[Dict[str, Any]]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        n: int = 8,
    ) -> OptimizeResponse:
        prior_bests = list(prior_bests or [])
        prompt = self._render_prompt(strategy_id, timeframe, data_summary, prior_bests, constraints, n)
        LOGGER.debug("Rendered optimization prompt: %s", prompt[:2000])

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._call_llm(prompt)
                response = OptimizeResponse.parse_raw(raw)
                LOGGER.info("Received %s proposals from LLM", len(response.proposals))
                clamped = [
                    self.registry.clamp(strategy_id, dict(proposal.params))
                    for proposal in response.proposals
                ]
                for proposal, new_params in zip(response.proposals, clamped):
                    proposal.params = new_params
                return response
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                LOGGER.warning("Validation failure on attempt %s/%s: %s", attempt, self.max_retries, exc)
        raise RuntimeError("LLM failed to produce valid response") from last_error

    def _render_prompt(
        self,
        strategy_id: str,
        timeframe: str,
        data_summary: Dict[str, Any],
        prior_bests: Iterable[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]],
        n: int,
    ) -> str:
        template_path = self.prompt_dir / "optimize_template.md"
        template = template_path.read_text()
        schema_json = self.registry.schema_json(strategy_id)
        payload = template
        payload = payload.replace("{{strategy_id}}", strategy_id)
        payload = payload.replace("{{optimize_schema_json}}", OptimizeResponse.schema_json(indent=2))
        payload = payload.replace("{{data_summary_json}}", json.dumps(data_summary, indent=2))
        payload = payload.replace("{{param_schema_json}}", schema_json)
        payload = payload.replace("{{prior_bests_json}}", json.dumps(list(prior_bests), indent=2))
        return payload

    def _call_llm(self, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client unavailable")
        request: Dict[str, Any] = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": prompt}],
        }
        if self._is_gpt5_model():
            if self.reasoning_effort:
                request["reasoning"] = {"effort": self.reasoning_effort}
            if self.verbosity:
                request["text"] = {"verbosity": self.verbosity}
            if self.max_output_tokens:
                request["max_output_tokens"] = int(self.max_output_tokens)
        elif self.temperature is not None:
            request["temperature"] = float(self.temperature)
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content  # type: ignore[attr-defined]
        if not content:
            raise ValueError("Empty response from LLM")
        return content

    def _is_gpt5_model(self) -> bool:
        return self.model.lower().startswith("gpt-5")

    def _build_default_client(self) -> Any | None:
        if OpenAI is None:
            return None
        try:
            return OpenAI()
        except Exception as exc:  # pragma: no cover - exercised via unit tests
            LOGGER.warning("Failed to initialise OpenAI client: %s", exc)
            return None
