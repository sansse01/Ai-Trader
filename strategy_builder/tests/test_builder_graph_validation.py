from __future__ import annotations

import json
from pathlib import Path

import pytest

from strategy_builder.builder_graph import GraphValidator, OperatorCatalog, load_graph_from_json
from strategy_builder.schemas import Node, NodeType, StrategyGraph


def catalog() -> OperatorCatalog:
    return OperatorCatalog(Path("strategy_builder/configs/operators_catalog.yaml"))


def valid_graph() -> StrategyGraph:
    nodes = [
        Node(id="ema_fast", type=NodeType.INDICATOR, op="EMA", params={"period": 20}),
        Node(id="ema_slow", type=NodeType.INDICATOR, op="EMA", params={"period": 50}),
        Node(id="signal", type=NodeType.SIGNAL, op="Greater", inputs=["ema_fast", "ema_slow"]),
        Node(id="position", type=NodeType.POSITION, op="PositionLong", inputs=["signal"]),
    ]
    return StrategyGraph(nodes=nodes, outputs=["position"])


def test_valid_graph_passes() -> None:
    validator = GraphValidator(catalog())
    validator.validate(valid_graph())


def test_cycle_detection() -> None:
    graph = valid_graph()
    graph.nodes[0].inputs = ["position"]
    validator = GraphValidator(catalog())
    with pytest.raises(ValueError):
        validator.validate(graph)


def test_param_bounds() -> None:
    graph = valid_graph()
    graph.nodes[0].params["period"] = 1000
    validator = GraphValidator(catalog())
    with pytest.raises(ValueError):
        validator.validate(graph)


def test_load_graph_from_json() -> None:
    payload = {
        "nodes": [
            {"id": "a", "type": "indicator", "op": "EMA", "params": {"period": 10}},
            {"id": "b", "type": "indicator", "op": "EMA", "params": {"period": 20}},
            {"id": "c", "type": "signal", "op": "Greater", "inputs": ["a", "b"]},
        ],
        "outputs": ["c"],
    }
    graph = load_graph_from_json(json.dumps(payload))
    assert isinstance(graph, StrategyGraph)
