You are designing a trading strategy as a DAG called StrategyGraph.
Return ONLY JSON that conforms to this schema:
{{graph_schema_json}}

Constraints:
- Beat Buy&Hold CAGR if possible with MaxDD ≤ 0.10.
- Use only allowed ops in operators catalog.
- Target trades per year 10–80; avoid excessive complexity.

Given natural-language spec:
{{nl_spec}}

Return:
{
 "nodes":[...],
 "outputs":["position_stream"]
}
No prose.
