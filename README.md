MCP-master

connects to CRM mcp and Controlroom MCP.

Lars Cowé & Abdellah El Morabit

## Observability — Prometheus metrics

`--server-mode` exposes `GET /metrics` as a Prometheus text exposition (no auth,
no rate-limit — scrape from inside the network only). Exposed metrics:

| Metric | Type | Labels |
|---|---|---|
| `http_requests_total` | counter | method, route, status |
| `http_request_duration_seconds` | summary (incl. p95) | route |
| `chat_requests_total` | counter | mode, outcome |
| `llm_tokens_total` | counter | kind (input / output / cache_creation / cache_read) |
| `llm_request_cost` | summary | currency, mode |
| `mcp_tool_calls_total` | counter | tool, server, ok |
| `mcp_tool_call_duration_seconds` | summary | tool, server |
| `incidents_total` | counter | event, detail |
| `incident_pipeline_duration_seconds` | summary | — |

`llm_request_cost` is derived from token counts × per-million-token prices.
Defaults are Claude Sonnet 4.x USD list prices; override per deployment:

| Env-var | Default |
|---|---|
| `LLM_PRICE_INPUT_PER_MTOK` | `3.0` |
| `LLM_PRICE_OUTPUT_PER_MTOK` | `15.0` |
| `LLM_PRICE_CACHE_WRITE_PER_MTOK` | `3.75` |
| `LLM_PRICE_CACHE_READ_PER_MTOK` | `0.30` |
| `LLM_PRICE_CURRENCY` | `usd` |
