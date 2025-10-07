# AI Trader Strategies

Strategy bundle providing a daily core trend booster and an hourly overlay for BTC/EUR.

## Components

- `DailyTrendBooster`: boosts exposure to 1.3× during confirmed breakouts and reverts to 0.3× otherwise while honouring hard risk locks.
- `HourlyTrendOverlay`: adds up to 0.2× extra notional when intra-day momentum confirms the daily breakout and the overall leverage stays below 1.5×.
- `RiskEngineAdapter`: synchronises the strategies with the platform risk engine enforcing drawdown and daily loss locks.
- `rebalance_to_notional`: helper that targets a desired notional accounting for trading fees.
- `MultiTimeframeOrchestrator`: command line entry point to wire feeds, broker and strategies together.

## Usage

```bash
python -m ai_trader_strats.orchestrator \
  --config-daily ai_trader_strats/configs/btc_eur_daily_trend_booster.json \
  --data-daily /path/to/BTC_EUR_1d.csv \
  --config-hourly ai_trader_strats/configs/btc_eur_hourly_overlay.json \
  --data-hourly /path/to/BTC_EUR_1h.csv
```

Feeds must contain the columns `Timestamp, Open, High, Low, Close, Volume` (case-insensitive) sorted in ascending order. Timestamps are parsed as UTC and signals execute on the next bar.

## Testing

Run the suite with:

```bash
pytest ai_trader_strats/tests
```
