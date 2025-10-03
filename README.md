
# EMA9 Trend Trader — v3.8 (Contract Size for Simple Engine)

**Option 2 implemented** for the Simple (backtesting.py) engine:
- New **Contract size (BTC per unit)** input. If you set it to **0.001**, then **1 unit = 0.001 BTC**.
- The app scales prices *inside the Simple engine only* and sizes trades in **integer contracts** against the **scaled price**.
- Charts and signals still use the **original** price series, so visuals remain familiar.
- You can still switch to **Fraction of equity (0–1)** if you prefer.

Backtrader (TRUE STOP) still supports **fractional units** via a CashSizer (retint=False).

Run:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: . .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
