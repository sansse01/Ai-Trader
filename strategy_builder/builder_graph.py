"""Validation and utilities around StrategyGraph specifications."""
from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Set

import yaml

from .schemas import Node, StrategyGraph

LOGGER = logging.getLogger(__name__)


class OperatorCatalog:
    """Represents the allowed operators and their metadata."""

    def __init__(self, catalog_path: Path) -> None:
        self.path = catalog_path
        self.raw = yaml.safe_load(catalog_path.read_text())
        self.ops: Dict[str, Dict[str, object]] = self.raw.get("operators", {})
        if not self.ops:
            raise ValueError("Operator catalog cannot be empty")

    def get(self, op_name: str) -> Dict[str, object]:
        try:
            return self.ops[op_name]
        except KeyError as exc:
            raise ValueError(f"Unknown operator '{op_name}'") from exc


class GraphValidator:
    """Validate StrategyGraph objects against the operator catalog."""

    def __init__(self, catalog: OperatorCatalog) -> None:
        self.catalog = catalog

    def validate(self, graph: StrategyGraph) -> StrategyGraph:
        LOGGER.info("Validating strategy graph containing %s nodes", len(graph.nodes))
        self._check_duplicate_ids(graph.nodes)
        self._check_acyclic(graph)
        self._validate_nodes(graph.nodes)
        self._validate_outputs(graph)
        return graph

    def _check_duplicate_ids(self, nodes: Iterable[Node]) -> None:
        seen: Set[str] = set()
        for node in nodes:
            if node.id in seen:
                raise ValueError(f"Duplicate node id '{node.id}'")
            seen.add(node.id)

    def _check_acyclic(self, graph: StrategyGraph) -> None:
        indegree: Dict[str, int] = defaultdict(int)
        edges: Dict[str, List[str]] = defaultdict(list)
        for node in graph.nodes:
            for child in node.inputs:
                edges[child].append(node.id)
                indegree[node.id] += 1
                indegree.setdefault(child, 0)

        queue = deque([node.id for node in graph.nodes if indegree[node.id] == 0])
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for nxt in edges.get(current, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if visited != len(graph.nodes):
            raise ValueError("Graph contains cycles")

    def _validate_nodes(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            meta = self.catalog.get(node.op)
            expected_type = meta.get("type")
            if expected_type and node.type.value != expected_type:
                raise ValueError(f"Node {node.id} type mismatch: {node.type} vs {expected_type}")
            inputs = node.inputs or []
            arity = meta.get("inputs")
            if arity is not None and len(inputs) not in self._expand_arity(arity):
                raise ValueError(f"Node {node.id} expects {arity} inputs, got {len(inputs)}")
            params_meta: Mapping[str, Mapping[str, float]] = meta.get("params", {})  # type: ignore[assignment]
            for key, value in node.params.items():
                if key not in params_meta:
                    raise ValueError(f"Node {node.id} has unsupported param '{key}'")
                bounds = params_meta[key]
                lo, hi = bounds.get("min", float("-inf")), bounds.get("max", float("inf"))
                if isinstance(value, (int, float)) and not (lo <= value <= hi):
                    raise ValueError(f"Node {node.id} param '{key}' out of range {lo}-{hi}")

    def _expand_arity(self, arity: object) -> Set[int]:
        if isinstance(arity, int):
            return {arity}
        if isinstance(arity, list):
            return {int(v) for v in arity}
        if isinstance(arity, str) and "-" in arity:
            start, end = arity.split("-", 1)
            return set(range(int(start), int(end) + 1))
        return set(range(0, 5))

    def _validate_outputs(self, graph: StrategyGraph) -> None:
        node_ids = {node.id for node in graph.nodes}
        for output in graph.outputs:
            if output not in node_ids:
                raise ValueError(f"Unknown output node '{output}'")


def load_graph_from_json(data: str | Dict[str, object]) -> StrategyGraph:
    if isinstance(data, str):
        payload = json.loads(data)
    else:
        payload = data
    return StrategyGraph.parse_obj(payload)
