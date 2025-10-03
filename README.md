# AI Trader (EMA9 Trend Trader)

AI Trader is a Streamlit application that backtests and paper-trades an EMA9 trend-following strategy on Kraken markets. It supports two execution engines, contract-size scaling, and an adaptive parameter optimizer that searches for promising configurations using feedback from previous evaluations.

## Features

- **Streamlit UI** for loading Kraken OHLCV data, configuring strategy parameters, and visualizing results.
- **Two backtest engines**:
  - `Simple (backtesting.py)` uses the Backtesting.py library with contract-size scaling.
  - `TRUE STOP (backtrader)` runs the same signal logic inside Backtrader with native stop and trailing orders.
- **Flexible position sizing** with whole-contract or fraction-of-equity modes.
- **Adaptive optimizer** that proposes new parameter combinations based on metric feedback instead of brute-force grid search.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## Launching the App

```bash
streamlit run app.py
```

The Streamlit interface opens in your browser. Use the sidebar to select the backtest mode, engine, data source, and strategy parameters.

## Data & Strategy Inputs

1. **Data settings**: Choose a Kraken symbol (e.g., `BTC/EUR`), timeframe (`1h`, `4h`, or `1d`), and start date in ISO8601 format. The app fetches data with ccxt and caches results during the session.
2. **Strategy configuration**: Adjust EMA lookback, slope filters, stop/trailing percentages, fee/slippage assumptions, leverage caps, risk sizing, and contract size. Each control can run as a single value or, in optimization mode, as a min/max/step range.
3. **Execution options**: Toggle shorting, ignore the trend filter for debugging, and pick a sizing mode (`Whole units (int)` or `Fraction of equity (0-1)`).

## Running a Single Backtest

1. Select **Run mode → Single Run**.
2. Configure the desired parameters in the sidebar.
3. Click **Run Backtest**.
4. Review the summary metrics table, trade log (if available), and annotated price chart. Metrics differ slightly by engine but always include final equity and return information.

## Optimization Mode

When you want to explore parameter combinations:

1. Select **Run mode → Optimization**.
2. For each tunable parameter, provide a min/max/step range. Leave a control in single-value mode to keep it fixed during the search.
3. Pick an **Objective metric** (e.g., `Return [%]`, `Sharpe Ratio`, `Win Rate [%]`, etc.) and whether to maximize or minimize it.
4. Set **Max evaluations** to cap how many combinations the optimizer may test (it never exceeds the total size of the parameter grid).
5. Click **Run Optimization**.

### How the Optimizer Works

The app delegates parameter search to `run_ai_optimizer` in `ai_optimizer.py`. The helper treats each parameter grid as a discrete search space and adaptively explores it using an evolutionary-inspired strategy:

- Starts with random proposals and tracks evaluated combinations.
- Scores each run using the selected objective metric (handling missing or invalid values safely).
- Whenever it finds a new best score, it seeds nearby combinations by perturbing indices and queuing neighbours for future evaluation.
- Falls back to sequential coverage when the queue empties, ensuring the search respects the `max_evaluations` budget and never repeats combos.

The optimizer returns:

- A full table of evaluated parameters with metrics and the objective value.
- The best-performing parameter set and its metrics (`best_run`).
- Counts of evaluations performed versus the configured limit.

The UI displays the results table sorted by objective, updates progress/status indicators during the run, and enables **Apply top parameters to controls** so you can immediately rerun a single backtest or further tune the winning configuration.

## Tips

- Use smaller ranges or fewer parameters when experimenting to keep evaluation times manageable.
- The `Simple` engine is faster for quick iteration; switch to `TRUE STOP` for more detailed stop-handling once you have promising settings.
- Keep `Max evaluations` below the total combination count for adaptive search; setting it equal to the total grid reproduces exhaustive coverage.

## Testing

To ensure the application scripts compile, run:

```bash
python -m compileall app.py ai_optimizer.py
```
