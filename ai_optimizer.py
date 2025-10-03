"""Optimization helpers for adaptive parameter search."""
from __future__ import annotations

import random
from collections import deque
from typing import Callable, Iterable

import numpy as np
import pandas as pd


def _to_native(value):
    """Convert numpy/pandas scalars into native Python values for JSON/Streamlit."""
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value
    return value


def run_ai_optimizer(
    *,
    base_params: dict,
    range_values: dict,
    evaluate_fn: Callable[[dict], dict],
    objective_metric: str,
    maximize: bool,
    max_evaluations: int,
    progress_callback: Callable[[int, int, dict, dict, float | None], None] | None = None,
    seed: int | None = None,
) -> dict:
    """Evolutionary-inspired optimizer that adapts proposals from prior runs."""

    combos_keys = list(range_values.keys())
    value_lists: list[Iterable] = [list(range_values[k]) for k in combos_keys]
    if not value_lists:
        return {"results": [], "best_params": None, "best_row": None, "evaluations": 0, "limit": 0}

    rng = random.Random(seed)
    total_space = 1
    for values in value_lists:
        if not values:
            return {"results": [], "best_params": None, "best_row": None, "evaluations": 0, "limit": 0}
        total_space *= len(values)

    limit = int(max(1, min(max_evaluations, total_space)))

    seen: set[tuple[int, ...]] = set()
    candidate_queue: deque[tuple[int, ...]] = deque()
    radices = [len(values) for values in value_lists]
    dimension = len(value_lists)
    seq_cursor = 0

    def sequential_candidate():
        nonlocal seq_cursor
        while seq_cursor < total_space:
            rem = seq_cursor
            idx_components = []
            for base in radices:
                idx_components.append(rem % base)
                rem //= base
            seq_cursor += 1
            candidate = tuple(idx_components)
            if candidate not in seen and candidate not in candidate_queue:
                return candidate
        return None

    def sample_random():
        attempts = 0
        while attempts < 1000:
            idx_tuple = tuple(rng.randrange(base) for base in radices)
            if idx_tuple not in seen and idx_tuple not in candidate_queue:
                return idx_tuple
            attempts += 1
        return sequential_candidate()

    def register_candidate(candidate):
        if candidate in seen or candidate in candidate_queue:
            return
        for dim, base in enumerate(radices):
            if candidate[dim] < 0 or candidate[dim] >= base:
                return
        candidate_queue.append(candidate)

    def build_params(idx_tuple):
        params = base_params.copy()
        for key, values, idx in zip(combos_keys, value_lists, idx_tuple):
            params[key] = values[idx]
        return params

    results_rows = []
    best_state = None
    evaluations = 0

    while evaluations < limit:
        if candidate_queue:
            idx_tuple = candidate_queue.popleft()
            if idx_tuple in seen:
                continue
        else:
            idx_tuple = sample_random()
            if idx_tuple is None:
                break

        seen.add(idx_tuple)
        combo_params = build_params(idx_tuple)
        evaluation = evaluate_fn(combo_params)
        metrics = {}
        if isinstance(evaluation, dict):
            metrics = evaluation.get("metrics", {}) or {}
        metrics = {k: _to_native(v) for k, v in metrics.items()}

        objective_raw = metrics.get(objective_metric)
        if objective_raw is None or pd.isna(objective_raw) or (
            isinstance(objective_raw, (float, int, np.floating, np.integer)) and not np.isfinite(objective_raw)
        ):
            objective_score = float("-inf") if maximize else float("inf")
        else:
            objective_score = float(objective_raw)

        row = {k: combo_params[k] for k in combos_keys}
        row.update(metrics)
        row["Objective"] = objective_raw if objective_raw is not None else objective_score
        results_rows.append(row)
        evaluations += 1

        if progress_callback is not None:
            progress_callback(evaluations, limit, combo_params, metrics, objective_raw)

        is_better = False
        if best_state is None:
            is_better = True
        else:
            if maximize:
                is_better = objective_score > best_state["score"]
            else:
                is_better = objective_score < best_state["score"]

        if is_better:
            best_state = {
                "score": objective_score,
                "params": combo_params.copy(),
                "row": row.copy(),
            }

            idx_list = list(idx_tuple)
            neighbour_seeds = [idx_list]
            for _ in range(2):
                mutated = []
                for dim, base in enumerate(radices):
                    current_idx = idx_tuple[dim]
                    if rng.random() < 0.3:
                        mutated.append(rng.randrange(base))
                    else:
                        mutated.append(current_idx)
                neighbour_seeds.append(mutated)

            for seed_indices in neighbour_seeds:
                for dim in range(dimension):
                    for offset in (-1, 1):
                        neighbour = seed_indices.copy()
                        neighbour[dim] += offset
                        register_candidate(tuple(neighbour))

        if not candidate_queue and evaluations < limit:
            next_candidate = sequential_candidate()
            if next_candidate is not None:
                register_candidate(next_candidate)

    return {
        "results": results_rows,
        "best_params": best_state["params"] if best_state else None,
        "best_row": best_state["row"] if best_state else None,
        "evaluations": evaluations,
        "limit": limit,
    }


__all__ = ["run_ai_optimizer", "_to_native"]
