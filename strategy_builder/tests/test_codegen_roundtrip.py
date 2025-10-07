from __future__ import annotations

import pandas as pd

from strategy_builder.codegen import GraphExecutor, compile_and_load
from strategy_builder.schemas import Node, NodeType, StrategyGraph


def test_codegen_executes_on_synthetic() -> None:
    graph = StrategyGraph(
        nodes=[
            Node(id="ema_fast", type=NodeType.INDICATOR, op="EMA", params={"period": 5}),
            Node(id="ema_slow", type=NodeType.INDICATOR, op="EMA", params={"period": 15}),
            Node(id="signal", type=NodeType.SIGNAL, op="Greater", inputs=["ema_fast", "ema_slow"]),
            Node(id="position", type=NodeType.POSITION, op="PositionLong", inputs=["signal"]),
        ],
        outputs=["position"],
    )
    module, cls = compile_and_load(graph, class_name="MyStrategy")
    strategy = cls(params={}, graph=graph)
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=50, freq="H", tz="UTC"),
            "Open": range(50),
            "High": range(1, 51),
            "Low": range(50),
            "Close": range(1, 51),
            "Volume": [1] * 50,
        }
    )
    output = strategy.run(df)
    assert "position" in output
    assert len(output["position"]) == len(df)

    executor = GraphExecutor(graph=graph, params={})
    signals = executor.run(df)
    assert signals["position"].sum() >= 0
