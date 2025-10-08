"""Strategy graph code generation and execution helpers."""
from __future__ import annotations

import logging
import sys
import textwrap
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from .pydantic_shim import BaseModel, create_model

from .schemas import StrategyGraph

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphExecutor:
    """Simple interpreter to execute a StrategyGraph over OHLCV data."""

    graph: StrategyGraph
    params: Dict[str, Any]

    def run(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        order = self._topological_order()
        registry = _get_operator_registry()
        values: Dict[str, pd.Series] = {}
        for node_id in order:
            node = next(n for n in self.graph.nodes if n.id == node_id)
            inputs = [values[i] for i in node.inputs]
            op = registry[node.op]
            params = {k: self._resolve_param(v) for k, v in node.params.items()}
            LOGGER.debug("Executing node %s (%s)", node.id, node.op)
            values[node.id] = op(data, inputs, params)
        return {out: values[out] for out in self.graph.outputs}

    def _topological_order(self) -> List[str]:
        indegree: Dict[str, int] = {node.id: 0 for node in self.graph.nodes}
        edges: Dict[str, List[str]] = {node.id: [] for node in self.graph.nodes}
        for node in self.graph.nodes:
            for child in node.inputs:
                indegree[node.id] += 1
                edges.setdefault(child, []).append(node.id)
        queue = [node_id for node_id, deg in indegree.items() if deg == 0]
        order: List[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for nxt in edges.get(current, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return order

    def _resolve_param(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            key = value[2:-1]
            return self.params.get(key, value)
        return value


def _get_operator_registry() -> Dict[str, Any]:
    return {
        "EMA": _op_ema,
        "SMA": _op_sma,
        "RSI": _op_rsi,
        "Donchian": _op_donchian,
        "ATR": _op_atr,
        "BB": _op_bb,
        "ADX": _op_adx,
        "CrossOver": _op_crossover,
        "Greater": _op_greater,
        "And": _op_and,
        "Or": _op_or,
        "PositionLong": _op_position_long,
        "PositionFlat": _op_position_flat,
        "StopATR": _op_stop_atr,
        "TrailATR": _op_trail_atr,
        "RiskCaps": _op_risk_caps,
    }


def render_strategy_code(graph: StrategyGraph, class_name: str = "GeneratedStrategy") -> str:
    """Return Python source representing the strategy."""

    code = f'''\
import logging
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from strategy_builder.codegen import GraphExecutor

LOGGER = logging.getLogger(__name__)


@dataclass
class {class_name}:
    """Auto-generated strategy produced from a StrategyGraph."""

    params: Dict[str, Any]
    graph: Dict[str, Any] = None

    def run(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        executor = GraphExecutor(graph=self.graph, params=self.params)
        return executor.run(data)
'''
    return textwrap.dedent(code)


def build_param_schema(graph: StrategyGraph, name: str = "GeneratedParams") -> type[BaseModel]:
    fields: Dict[str, Tuple[type, Any]] = {}
    for node in graph.nodes:
        for key, value in node.params.items():
            if isinstance(value, (int, float)):
                fields.setdefault(key, (float, value))
            elif isinstance(value, str) and value.startswith("${"):
                param_name = value[2:-1]
                fields.setdefault(param_name, (float, ...))
    if not fields:
        fields["dummy"] = (float, 0.0)
    model = create_model(name, **fields)  # type: ignore[arg-type]
    return model


def compile_and_load(
    graph: StrategyGraph,
    module_name: str = "generated_strategy_module",
    class_name: str = "GeneratedStrategy",
) -> Tuple[ModuleType, type]:
    source = render_strategy_code(graph, class_name=class_name)
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    module.__dict__.update({"pd": pd})
    module.__dict__.update({"GraphExecutor": GraphExecutor})
    exec(source, module.__dict__)
    return module, module.__dict__[class_name]


# --- Operator implementations -------------------------------------------------

def _series(data: pd.DataFrame, column: str = "Close") -> pd.Series:
    return data[column].astype(float)


def _op_ema(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 20))
    return _series(data).ewm(span=period, adjust=False).mean()


def _op_sma(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 20))
    return _series(data).rolling(period, min_periods=1).mean()


def _op_rsi(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 14))
    delta = _series(data).diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _op_donchian(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 20))
    high = data["High"].rolling(period, min_periods=1).max()
    low = data["Low"].rolling(period, min_periods=1).min()
    return (high + low) / 2


def _op_atr(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 14))
    high = data["High"]
    low = data["Low"]
    close = data["Close"].shift(1)
    tr = pd.concat([(high - low), (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _op_bb(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 20))
    std_mult = float(params.get("std_mult", 2.0))
    mean = _series(data).rolling(period, min_periods=1).mean()
    std = _series(data).rolling(period, min_periods=1).std(ddof=0)
    return mean + std_mult * std


def _op_adx(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    period = int(params.get("period", 14))
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    tr = pd.concat([(high - low), (high - close.shift()), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(period, min_periods=1).mean()


def _op_crossover(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    if len(inputs) != 2:
        raise ValueError("CrossOver requires two inputs")
    return (inputs[0] > inputs[1]).astype(int)


def _op_greater(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    if len(inputs) != 2:
        raise ValueError("Greater requires two inputs")
    return (inputs[0] > inputs[1]).astype(int)


def _op_and(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    result = pd.Series(1, index=data.index)
    for series in inputs:
        result = result & series.astype(int)
    return result.astype(int)


def _op_or(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    result = pd.Series(0, index=data.index)
    for series in inputs:
        result = result | series.astype(int)
    return result.astype(int)


def _op_position_long(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    if not inputs:
        raise ValueError("PositionLong requires a signal input")
    return inputs[0].astype(int)


def _op_position_flat(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    return pd.Series(0, index=data.index)


def _op_stop_atr(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    atr = inputs[0] if inputs else _op_atr(data, [], params)
    mult = float(params.get("multiple", 2.0))
    close = data["Close"]
    return close - atr * mult


def _op_trail_atr(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    atr = inputs[0] if inputs else _op_atr(data, [], params)
    mult = float(params.get("multiple", 2.0))
    close = data["Close"]
    stop = close - atr * mult
    return stop.cummax()


def _op_risk_caps(data: pd.DataFrame, inputs: List[pd.Series], params: Dict[str, Any]) -> pd.Series:
    cap = float(params.get("max_dd", 0.1))
    return pd.Series(cap, index=data.index)
