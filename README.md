
# EMA9 Trend Trader – Web App (Streamlit) v3.4

- Backtest trailing stop implemented **without** Position.set_sl / Position.sl
- Initial SL still set via order param; trailing exits close position when breach is detected
- Plotly chart, paging fix, equity-fraction sizing, commission=fee+slippage, diagnostics, paper mode

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: . .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
