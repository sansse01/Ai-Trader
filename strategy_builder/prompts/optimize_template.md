You are optimizing parameters for strategy: {{strategy_id}} on BTC/EUR.
Reply ONLY with a JSON object that conforms to this schema:

{{optimize_schema_json}}

Objective:
- Maximize CAGR and Sharpe with MaxDD ≤ 0.10.
- Prefer smoother equity and reasonable turnover.

Data summary (no raw bars):
{{data_summary_json}}

Parameter schema and allowed bounds (inclusive):
{{param_schema_json}}

Previous good params (optional):
{{prior_bests_json}}

Return JSON:
{
  "proposals": [
    {
      "params": { ... },
      "predicted_metrics": {"cagr": ..., "maxdd": ..., "sharpe": ..., "trades": ...},
      "rationale": "why these ranges fit the regime distribution",
      "confidence": 0.0-1.0
    }
  ],
  "notes": "assumptions/risks"
}
Do not include extra fields.
