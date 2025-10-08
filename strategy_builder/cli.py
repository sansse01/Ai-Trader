"""Command line interface for the strategy builder."""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from . import __version__
from .backtest_svc import BacktestService
from .builder_graph import GraphValidator, OperatorCatalog, load_graph_from_json
from .codegen import GraphExecutor, compile_and_load, render_strategy_code
from .datasvc import DataService
from .evaluator import accept as evaluator_accept
from .evaluator import choose_champion
from .llm_optimizer import LLMOptimizer
from .registry import StrategyRegistry
from .schemas import StrategyGraph

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")


def command_optimize(args: argparse.Namespace) -> int:
    registry = StrategyRegistry()
    data_service = DataService(cache_dir=Path(args.data))
    df = data_service.get_history(args.symbol, args.tf, args.start, args.end)
    summary = data_service.summarize(df)
    max_tokens = args.max_output_tokens if args.max_output_tokens and args.max_output_tokens > 0 else None
    optimizer = LLMOptimizer(
        registry=registry,
        data_service=data_service,
        prompt_dir=Path(args.prompts),
        model=args.model,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        max_output_tokens=max_tokens,
    )
    if getattr(args, "mock", None):
        optimizer._call_llm = args.mock  # type: ignore[attr-defined]
    response = optimizer.optimize(
        strategy_id=args.strategy,
        timeframe=args.tf,
        data_summary=summary,
        prior_bests=None,
        constraints=None,
        n=args.n,
    )

    backtester = BacktestService()
    key_metrics: Dict[str, Any] = {}

    def _run(proposal: Dict[str, Any]) -> tuple[str, Any]:
        params = dict(proposal)
        metrics = backtester.run(args.strategy, params, df)
        return json.dumps(params, sort_keys=True), metrics

    with ThreadPoolExecutor(max_workers=min(args.n, 4)) as executor:
        futures = [executor.submit(_run, proposal.params) for proposal in response.proposals]
        for future in futures:
            key, metrics = future.result()
            key_metrics[key] = metrics

    card = {
        "version": __version__,
        "generated_at": datetime.utcnow().isoformat(),
        "prompt_hash": DataService.summary_hash(summary),
        "strategy": args.strategy,
        "timeframe": args.tf,
        "summary": summary,
        "proposals": [],
    }

    evaluations = []
    for proposal in response.proposals:
        key = json.dumps(proposal.params, sort_keys=True)
        metrics = key_metrics[key]
        ok, reason = evaluator_accept(proposal.predicted_metrics, metrics)
        evaluations.append((proposal, metrics, ok, reason))
        card["proposals"].append(
            {
                "proposal": proposal.dict(),
                "measured": metrics.dict(),
                "accepted": ok,
                "reason": reason,
            }
        )

    try:
        champion_params, champion_metrics, report = choose_champion([p for p, *_ in evaluations], key_metrics)
        accepted, reason = evaluator_accept({}, champion_metrics)
        card["champion"] = {
            "params": champion_params,
            "metrics": champion_metrics.dict(),
            "report": report,
            "accepted": accepted,
            "reason": reason,
        }
    except ValueError as exc:
        card["champion_error"] = str(exc)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(card, indent=2, default=_json_default))
    LOGGER.info("Strategy card written to %s", args.out)
    return 0


def command_compose(args: argparse.Namespace) -> int:
    registry = StrategyRegistry()
    max_tokens = args.max_output_tokens if args.max_output_tokens and args.max_output_tokens > 0 else None
    optimizer = LLMOptimizer(
        registry=registry,
        data_service=DataService(cache_dir=Path(args.data)),
        prompt_dir=Path(args.prompts),
        model=args.model,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        max_output_tokens=max_tokens,
    )
    spec_text = Path(args.spec).read_text()
    template = Path(args.prompts) / "compose_template.md"
    prompt = template.read_text()
    prompt = prompt.replace("{{graph_schema_json}}", StrategyGraph.schema_json(indent=2))
    prompt = prompt.replace("{{nl_spec}}", spec_text)
    raw = optimizer._call_llm(prompt)
    graph = load_graph_from_json(raw)
    catalog = OperatorCatalog(Path(args.catalog))
    validator = GraphValidator(catalog)
    validator.validate(graph)

    module, cls = compile_and_load(graph)
    code = render_strategy_code(graph)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(code)

    df = _load_small_dataframe(Path(args.data), args.symbol, args.tf, args.start, args.end)
    executor = GraphExecutor(graph=graph, params={})
    executor.run(df)
    LOGGER.info("Generated strategy module %s and validated via dry run", args.out)
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    data_service = DataService(cache_dir=Path(args.data))
    df = data_service.get_history(args.symbol, args.tf, args.start, args.end)
    params = json.loads(Path(args.params).read_text())
    backtester = BacktestService()
    metrics = backtester.run(args.strategy, params, df)
    print(json.dumps(metrics.dict(), indent=2, default=_json_default))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    data_service = DataService(cache_dir=Path(args.data))
    df = data_service.get_history(args.symbol, args.tf, args.start, args.end)
    backtester = BacktestService()
    params = json.loads(Path(args.params).read_text())
    strat_metrics = backtester.run(args.strategy, params, df)
    bh_metrics = backtester.buy_and_hold(df)
    rows = [
        ("Candidate", strat_metrics.cagr, strat_metrics.max_drawdown, strat_metrics.sharpe),
        ("BuyHold", bh_metrics.cagr, bh_metrics.max_drawdown, bh_metrics.sharpe),
    ]
    print("Label    CAGR    MaxDD   Sharpe")
    for label, cagr, maxdd, sharpe in rows:
        print(f"{label:<8} {cagr:>6.2%} {maxdd:>6.2%} {sharpe:>6.2f}")
    return 0


def command_promote(args: argparse.Namespace) -> int:
    card = json.loads(Path(args.card).read_text())
    champion = card.get("champion", {})
    if not champion or not champion.get("accepted"):
        print("Champion not accepted. Promotion aborted.")
        return 1
    version = card.get("version", __version__)
    print(f"Promoting strategy with version {version} using params {champion['params']}")
    return 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _load_small_dataframe(data_dir: Path, symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
    service = DataService(cache_dir=data_dir)
    return service.get_history(symbol, tf, start, end)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strategy_builder", description="AI Strategy Builder CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser("optimize", help="Optimize strategy parameters")
    optimize.add_argument("--strategy", required=True)
    optimize.add_argument("--tf", required=True)
    optimize.add_argument("--symbol", required=True)
    optimize.add_argument("--start", required=True)
    optimize.add_argument("--end", required=True)
    optimize.add_argument("--n", type=int, default=8)
    optimize.add_argument("--out", required=True)
    optimize.add_argument("--data", default="data")
    optimize.add_argument("--prompts", default="strategy_builder/prompts")
    optimize.add_argument("--model", default="gpt-5")
    optimize.add_argument("--temperature", type=float, default=None)
    optimize.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default="medium")
    optimize.add_argument("--verbosity", choices=["low", "medium", "high"], default="medium")
    optimize.add_argument("--max-output-tokens", type=int, default=2048)
    optimize.set_defaults(func=command_optimize)

    compose = subparsers.add_parser("compose", help="Compose a strategy from NL spec")
    compose.add_argument("--spec", required=True)
    compose.add_argument("--tf", required=True)
    compose.add_argument("--symbol", default="BTC/EUR")
    compose.add_argument("--start", required=True)
    compose.add_argument("--end", required=True)
    compose.add_argument("--out", required=True)
    compose.add_argument("--data", default="data")
    compose.add_argument("--prompts", default="strategy_builder/prompts")
    compose.add_argument("--catalog", default="strategy_builder/configs/operators_catalog.yaml")
    compose.add_argument("--model", default="gpt-5")
    compose.add_argument("--temperature", type=float, default=None)
    compose.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default="medium")
    compose.add_argument("--verbosity", choices=["low", "medium", "high"], default="medium")
    compose.add_argument("--max-output-tokens", type=int, default=2048)
    compose.set_defaults(func=command_compose)

    backtest = subparsers.add_parser("backtest", help="Run backtest for params")
    backtest.add_argument("--strategy", required=True)
    backtest.add_argument("--params", required=True)
    backtest.add_argument("--tf", required=True)
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--start", required=True)
    backtest.add_argument("--end", required=True)
    backtest.add_argument("--data", default="data")
    backtest.set_defaults(func=command_backtest)

    compare = subparsers.add_parser("compare", help="Compare strategies against benchmark")
    compare.add_argument("--strategy", required=True)
    compare.add_argument("--params", required=True)
    compare.add_argument("--tf", required=True)
    compare.add_argument("--symbol", required=True)
    compare.add_argument("--start", required=True)
    compare.add_argument("--end", required=True)
    compare.add_argument("--data", default="data")
    compare.set_defaults(func=command_compare)

    promote = subparsers.add_parser("promote", help="Promote accepted strategy")
    promote.add_argument("--card", required=True)
    promote.set_defaults(func=command_promote)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
