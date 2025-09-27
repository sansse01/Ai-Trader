
# EMA9 Trend Trader – Web App (Streamlit) v2 (Bokeh-free)

This build removes the Bokeh dependency from backtesting.py's plot and uses Plotly for charting,
fixing the `DatetimeTickFormatter ... expected str got list` error.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: . .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
