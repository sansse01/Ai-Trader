# Strategy Builder

`strategy_builder v0.1.0` provides tooling to seed crypto trading strategies with
language models, validate them locally, and promote them into production. The
package integrates with an existing Backtrader-compatible stack and risk engine
while offering fallbacks so it works standalone for testing.

## Installation

```bash
pip install -r requirements.txt  # ensure pandas, pydantic, numpy, pyyaml
```

## Configuration

### 1. API access (LLM optimizer)

* Set an OpenAI API key in your environment before running any command that
  reaches the hosted model:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
* By default the optimizer targets the `gpt-5` API alias (system card
  `gpt-5-thinking`) with JSON mode, `medium` reasoning effort, `medium`
  verbosity, and `2048` max output tokens. Override via CLI flags (`--model`,
  `--reasoning-effort`, `--verbosity`, `--max-output-tokens`, or
  `--temperature` for pre GPT-5 models) or by editing
  `strategy_builder/configs/optimizer_defaults.yaml`.
* In offline or CI environments inject a mock client by passing
  `--mock` (see tests) or by supplying a custom `LLMOptimizer` instance.

### 2. Data inputs

* Historical candles are loaded from CSV files in the directory supplied to
  `--data` (default `./data`). Files must be named
  `{SYMBOL_REPLACED_SLASH}_{TF}.csv`, e.g. `BTC_EUR_1h.csv`.
* Required columns: `Timestamp, Open, High, Low, Close, Volume` in UTC and
  ascending order. The CLI filters rows by the provided `--start` / `--end`
  window.
* Only summaries computed by `DataService.summarize` are sent to the LLM; raw
  bars never leave your environment.

### 3. Application integration

* For production wiring, expose the following modules so the builder can use
  your native components:
  ```python
  from yourapp.risk import RiskEngine  # enforces drawdown locks
  from yourapp.exec import BacktestRunner  # run(strategy_id, params, data) -> Metrics
  ```
* If these imports are absent, the package falls back to internal reference
  implementations (`SimpleBacktestRunner`, bundled `RiskEngine`). You can also
  inject custom instances into `BacktestService`.

### 4. Strategy registry & operator catalog

* `strategy_builder/registry.py` maps strategy identifiers to their Pydantic
  parameter schemas and bounds. Extend this file to add new strategies.
* `strategy_builder/configs/operators_catalog.yaml` lists the whitelisted
  indicator/logic operators for graph composition. Update bounds and defaults
  here when introducing new building blocks.

### 5. Evaluator tolerances & stress tests

* Base tolerances (CAGR, drawdown, Sharpe) and stress scenarios (`fees x2`,
  volatility multipliers) live in
  `strategy_builder/configs/optimizer_defaults.yaml`. Adjust them to match your
  governance standards.

## CLI usage

The entrypoint module exposes several commands:

### Optimize

```bash
python -m strategy_builder.cli optimize \
    --strategy trend_atr --tf 1h --symbol BTC/EUR \
    --start 2024-01-01 --end 2024-03-01 --n 8 \
    --data data --out cards/trend_atr_1h.json
```

Flow:
1. Load OHLCV data from `data/BTC_EUR_1h.csv` and summarise it without exposing
   raw bars.
2. Ask the LLM for parameter proposals using JSON mode.
3. Backtest each proposal locally and record measured metrics.
4. Evaluate against tolerances and stress tests. A strategy card is written with
   the prompt hash, proposals, measured metrics, and champion decision.

### Compose

```bash
python -m strategy_builder.cli compose \
    --spec strategy_builder/examples/compose_example_spec.md \
    --tf 1h --symbol BTC/EUR \
    --start 2024-01-01 --end 2024-02-01 \
    --data data --out generated/strategy_regime_trendmr.py
```

This converts a natural-language specification into a validated `StrategyGraph`,
produces Python code, and executes a quick dry-run backtest to ensure the graph
is coherent.

### Backtest

```bash
python -m strategy_builder.cli backtest \
    --strategy trend_atr \
    --params strategy_builder/examples/optimize_trend_atr_1h.json \
    --tf 1h --symbol BTC/EUR \
    --start 2024-01-01 --end 2024-03-01 \
    --data data
```

### Compare

Compare the candidate, buy-and-hold benchmark, and current production metrics,
printing a compact table and writing equity curves when integrated with the
broader app.

### Promote

Read a strategy card, ensure the champion passed the evaluator, and echo the
version bump. Promotion requires all stress tests to succeed.

## Registering strategies

`strategy_builder/registry.py` lists supported strategy identifiers with their
Pydantic parameter schemas and bounds. Extend this mapping to support new
strategies while reusing the optimizer and evaluator.

## Privacy-preserving summaries

`DataService.summarize` computes statistical summaries (mean return, volatility,
regime distribution, etc.) so prompts never include raw OHLCV rows. This keeps
exchange data private while still informing the LLM.

## Promotion flow

1. Run `optimize` to produce a strategy card.
2. Review the card and ensure stress tests (`fees x2`, `volatility +50%`,
   `slippage +20%`) remain under the 12% drawdown ceiling.
3. Use `compare` to benchmark against buy-and-hold.
4. Call `promote` to confirm the version bump and hand off to deployment tools.
