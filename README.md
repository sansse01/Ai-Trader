
# EMA9 Trend Trader — v3.6 (Trade Analysis)

Adds a **Trade Analysis** section for both backtest engines:
- **Simple (backtesting.py)**: uses `stats._trades`, augments with Direction, Duration, ReturnPct_est, CSV export.
- **TRUE STOP (backtrader)**: uses a custom Analyzer (`TradeLogger`) to build a trades table with Direction, Entry/Exit, Size, PnL, Duration, ReturnPct_est.

Other features stay: Plotly charting, Kraken paging, fee+slippage, signal diagnostics, leverage cap.

Run:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: . .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
