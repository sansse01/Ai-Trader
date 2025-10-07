from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from strategy_builder import cli
from strategy_builder.schemas import OptimizeResponse, ParamProposal


@pytest.fixture()
def sample_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=48, freq="H", tz="UTC"),
            "Open": [100 + i for i in range(48)],
            "High": [101 + i for i in range(48)],
            "Low": [99 + i for i in range(48)],
            "Close": [100 + i for i in range(48)],
            "Volume": [10] * 48,
        }
    )
    df.to_csv(data_dir / "BTC_EUR_1h.csv", index=False)
    return data_dir


def test_cli_optimize_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_data: Path) -> None:
    proposals = OptimizeResponse(
        proposals=[
            ParamProposal(
                params={"ema_fast": 10, "ema_slow": 50, "donchian": 20, "atr_len": 14, "stop_atr": 2.0, "trail_atr": 3.0},
                predicted_metrics={"cagr": 0.3, "maxdd": 0.08, "sharpe": 1.5},
                rationale="baseline",
                confidence=0.8,
            )
        ]
    )

    monkeypatch.setattr(cli.LLMOptimizer, "optimize", lambda self, **kwargs: proposals)

    out = tmp_path / "card.json"
    args = [
        "optimize",
        "--strategy",
        "trend_atr",
        "--tf",
        "1h",
        "--symbol",
        "BTC/EUR",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-03",
        "--data",
        str(sample_data),
        "--out",
        str(out),
    ]
    assert cli.main(args) == 0
    card = json.loads(out.read_text())
    assert card["strategy"] == "trend_atr"
    assert card["proposals"][0]["accepted"] in (True, False)
    assert "champion" in card
