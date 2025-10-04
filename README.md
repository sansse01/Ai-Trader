# EMA9 Trend Trader — v3.8 (Contract Size for Simple Engine)

**Option 2 implemented** for the Simple (`backtesting.py`) engine:

- New **Contract size (BTC per unit)** input. If you set it to **0.001**, then **1 unit = 0.001 BTC**.
- The app scales prices *inside the Simple engine only* and sizes trades in **integer contracts** against the **scaled price**.
- Charts and signals still use the **original** price series, so visuals remain familiar.
- You can still switch to **Fraction of equity (0–1)** if you prefer.

Backtrader (TRUE STOP) still supports **fractional units** via a `CashSizer` (`retint=False`).

## Getting started

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: . .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Strategy selector overview

The sidebar strategy selector is backed by `strategies.STRATEGY_REGISTRY`. Each entry is a
`StrategyDefinition` that declares:

- Display metadata (`key`, `name`, and a human-readable `description`).
- Control metadata used to render Streamlit widgets (`controls` and `range_controls`).
- Parameter defaults (`default_params`) that seed the UI and downstream calculations.
- Callbacks for data preparation, signal generation, order construction, and backtest builders.
- Optional validation hooks to ensure the chosen context (symbols, datasets, etc.) is compatible.

Selecting a strategy automatically loads its default parameters and wires the correct handlers
into both the simple and TRUE STOP backtesting flows. To register a strategy, import the new
`StrategyDefinition` in `strategies/__init__.py` and add it to `STRATEGY_REGISTRY` with a unique key.

## Risk engine toggles

The Streamlit UI exposes a set of risk controls that feed into `risk.engine.RiskEngineConfig`:

- **Enable risk engine** toggles the entire subsystem on or off.
- **Drawdown guardrails** (drawdown limit, cooldown duration, and circuit breaker thresholds)
  modulate sizing or flatten positions when losses breach configured levels.
- **Volatility targeting** uses recent returns to scale exposure toward a target annualized volatility.
- **Exposure caps** limit gross leverage and single-position notional relative to current equity.
- **Correlation cuts** prevent adding size to positions that are already highly correlated with
  existing exposure.

These settings map one-to-one with `RiskEngineConfig` fields, so adjustments in the UI immediately
affect order sizing in both backtest engines and live signal previews.

## Adding a new strategy

1. Implement any helper indicators in `strategies/indicators.py` if they are broadly useful.
2. Create a new module in `strategies/` that exposes a `StrategyDefinition` describing how the
   strategy prepares data, emits signals, and (optionally) builds backtest classes.
3. Populate `data_requirements` with chart overlays, signal columns, and preview fields so the UI
   can render a consistent experience without additional wiring.
4. Register the strategy in `strategies/__init__.py` so it appears in the selector.
5. Add lightweight smoke coverage that exercises `prepare_data` and `generate_signals` using the
   deterministic dataset under `tests/data/sample_ohlcv.csv` to guard against regressions.

## Tests and smoke checks

The repository bundles a reproducible OHLCV sample (`tests/data/sample_ohlcv.csv`) that includes a
synthetic funding-rate column. The automated test suite uses this dataset to assert that every
registered strategy generates the expected number of long/short signals and that key risk engine
state transitions behave as designed.

Run the suite locally with:

```bash
pytest
```

For quick manual verification or experimentation, use the updated smoke script:

```bash
# Validate against the bundled dataset
python signals_smoke_test.py

# Or fetch fresh data (requires ccxt credentials/network access)
python signals_smoke_test.py --use-live --symbol BTC/EUR --timeframe 1h
```

GitHub Actions executes the same `pytest` suite on every push so that strategy or risk-engine
changes cannot silently regress signal generation.
