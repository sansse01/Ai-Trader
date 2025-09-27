
# EMA9 Trend Trader – Web App (Streamlit) v2

Now with a **mode switch** for:
- **Backtest** (using backtesting.py)
- **Dry-Run (paper)** with stateful equity/position, trailing/SL, and CSV export of trades

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: . .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Works with **Kraken** via **ccxt** (symbols like `BTC/EUR`, `BTC/USD`).
- **Fees** default to **0.26%** (Kraken spot taker). **Slippage** is % per fill.
- **Max leverage** is a **cap for exposure** (simulated).
- Strategy mirrors your Pine idea with configurable thresholds in the sidebar.
- Dry-run updates when you click **Step 1 bar** or enable **Auto-refresh**.
