from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from strategy_builder.datasvc import DataService
from strategy_builder.llm_optimizer import LLMOptimizer
import strategy_builder.llm_optimizer as llm_optimizer
from strategy_builder.registry import StrategyRegistry
from strategy_builder.schemas import OptimizeResponse


def sample_summary() -> dict:
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=10, freq="H", tz="UTC"),
            "Open": range(10),
            "High": range(1, 11),
            "Low": range(10),
            "Close": range(1, 11),
            "Volume": [1] * 10,
        }
    )
    service = DataService(cache_dir=Path("."))
    return service.summarize(df)


def test_optimizer_retry_and_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = StrategyRegistry()
    prompts = Path("strategy_builder/prompts")
    optimizer = LLMOptimizer(registry, DataService(cache_dir=tmp_path), prompts, client=object())

    calls = {"count": 0}

    def fake_call(prompt: str) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            return "not-json"
        payload = {
            "proposals": [
                {
                    "params": {"ema_fast": 200, "ema_slow": 300, "donchian": 20, "atr_len": 14, "stop_atr": 2.0, "trail_atr": 3.0},
                    "predicted_metrics": {"cagr": 0.5, "maxdd": 0.05, "sharpe": 1.4},
                    "rationale": "test",
                    "confidence": 0.5,
                }
            ]
        }
        return json.dumps(payload)

    optimizer._call_llm = fake_call  # type: ignore[assignment]
    response = optimizer.optimize("trend_atr", "1h", sample_summary(), n=1)
    assert isinstance(response, OptimizeResponse)
    # Clamped to bounds (ema_fast <= 80)
    assert response.proposals[0].params["ema_fast"] <= 80
    assert calls["count"] == 2


def test_optimizer_failure_after_retries(tmp_path: Path) -> None:
    registry = StrategyRegistry()
    prompts = Path("strategy_builder/prompts")
    optimizer = LLMOptimizer(registry, DataService(cache_dir=tmp_path), prompts, client=object(), max_retries=2)

    optimizer._call_llm = lambda prompt: "{"  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        optimizer.optimize("trend_atr", "1h", sample_summary(), n=1)


def test_optimizer_without_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry = StrategyRegistry()
    data_service = DataService(cache_dir=tmp_path)

    class RaisingClient:
        def __init__(self) -> None:
            raise RuntimeError("missing api key")

    monkeypatch.setattr(llm_optimizer, "OpenAI", RaisingClient)
    optimizer = LLMOptimizer(registry, data_service, tmp_path)
    assert optimizer.client is None
