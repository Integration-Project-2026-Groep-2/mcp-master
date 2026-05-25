# MCP-master

> Rust master agent for the Integration Project AI team — an MCP **client** that
> lets Claude orchestrate tool-calls across the team's MCP servers.

`mcp-master` — the agent end users meet as **Jarvis** — is the client side of
[Model Context Protocol](https://modelcontextprotocol.io) for Integratieproject
2025/2026, Groep 2 (Desideriushogeschool / ShiftFestival).
It connects to N team-MCP servers (CRM in Python, Controlroom in Go, …), exposes
their tools to Claude Sonnet 4.6, and runs the tool-calling loop. The team-MCP
**servers** live in their own repositories.

## Features

- **Multi-server tool-calling** — connects to several MCP servers over Streamable
  HTTP and lets Claude route tool-calls across them by name.
- **Read-only by default** — write-tools are blocked unless the caller presents a
  `read+act` JWT scope *and* a human approves the proposed action.
- **Chat API** — `POST /chat` and SSE `POST /chat/stream`, with multi-turn history,
  tool-trace, and token/cost accounting. The agent is stateless; the client carries
  history.
- **Incident response** — consumes `heartbeat_failed` events from AMQP and runs a
  dual-LLM diagnose pipeline (evidence-gathering + reasoning), publishing a
  root-cause hypothesis; auto-resolves on recovery and is cost-bounded by a
  per-hour circuit-breaker.
- **Autonomous fix-flow** — `POST /fix-flow` proposes a GitHub PR fix for an
  incident through the same approval flow.
- **Live service status** — taps the `heartbeat.direct` stream to derive up/down
  per service (`GET /status`), and emits its own liveness heartbeat.
- **Scheduled briefings** — posts a summary to Teams at 00:00 / 08:30 / 12:30 /
  16:30 UTC (server-mode).
- **Optional memory** (`MEMORY_ENABLED`) — Qdrant-backed RAG (needs a Qdrant +
  embeddings endpoint) plus a SQLite response cache.
- **Observability** — Prometheus `/metrics`, an `ai.events` AMQP audit feed, and
  structured `tracing` logs.

## Architecture

A single-crate Rust binary; one CLI flag selects the run mode:

| Flag | Mode |
|---|---|
| `--list-tools` | Print the aggregated tool list from all MCP servers, then exit |
| `--terminal-mode` | One prompt (argv/stdin) → run agent → publish answer to Teams |
| `--server-mode` | axum HTTP API on `:8080` + scheduled trigger + AMQP consumers (**production**) |
| `--debug-client` | Interactive CLI client against a running server (`BACKEND_URL`) |

Code is organised by concern under `src/`: `agent/` (LLM client, orchestrator,
modes, prompts), `gateway/` (auth, approval state-machine, audit), `incident/`
(the R3 pipeline), `memory/`, `mcp.rs` (the multi-server pool), `http_api.rs`,
`rabbitmq/`, and `teams.rs`. Per-request caps: up to 10 tool-loop iterations
(20 in actionable mode) and `MAX_TOKENS=8192`.

## Getting started

Prerequisites: Rust 2024 (1.91+), an Anthropic API key, and at least one reachable
MCP server.

```sh
cp src/.env.example .env          # then set ANTHROPIC_API_KEY + MCP_SERVERS, e.g.
                                  #   MCP_SERVERS=crm@http://localhost:7001/mcp,controlroom@http://localhost:5555/mcp
cargo run -- --list-tools         # verify MCP connectivity (no Anthropic credits used)
cargo run -- --server-mode        # start the HTTP API on :8080
cargo run -- --terminal-mode "..."   # one-shot to Teams (needs TEAMS_ID/CHANNEL_ID/TEAMS_TOKEN)
```

## Configuration

Loaded from `.env` at startup (`dotenvy`). The essentials:

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude Messages API key | — (**required**) |
| `MCP_SERVERS` | Endpoints as `label@url,label@url` | — (**required**) |
| `RABBITMQ_URL` | AMQP broker for `ai.events` / incidents / heartbeat; absent → runs without (WARN) | — |
| `CHAT_JWT_SECRET` | HS256 secret enabling `read+act` scope (write approvals) | — |
| `CHAT_BEARER_TOKEN` | Optional bearer auth on `/chat`; unset → no-auth (WARN) | — |
| `RUST_LOG` | `tracing` filter | `info,mcp_master=debug` |

CORS is permissive until you set `CHAT_ALLOWED_ORIGINS` (enforce with
`CHAT_CORS_STRICT=true`). See `src/.env.example` for a starting template; further
per-area variables exist: `CHAT_*` (auth, CORS, approval TTL, streaming),
`INCIDENT_*` (debounce, rate-limit), `HEARTBEAT_INTERVAL_MS`, `SERVICE_REPO_MAP`,
`MEMORY_*`, `LLM_PRICE_*`, and `TEAMS_ID` / `CHANNEL_ID` / `TEAMS_TOKEN`
(terminal-mode).

## HTTP API (`--server-mode`, `:8080`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Chat — single-shot (`{prompt}`) or multi-turn (`{messages}`) |
| `POST` | `/chat/stream` | Same, as Server-Sent Events |
| `POST` | `/chat/approve`, `/chat/reject` | Resolve a pending write action |
| `POST` | `/fix-flow` | Start an async incident-fix job |
| `DELETE` | `/memory/user/{user_id}` | Erase a user's stored memory |
| `GET` | `/status` | Per-service heartbeat up/down |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Prometheus exposition |

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

## Development

```sh
cargo test
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

CI runs the same three gates; on merge to `main` the pipeline builds a container
image to GHCR and triggers the `k8s-manifests` GitOps repo.

## Authors

Lars Cowé & Abdellah El Morabit
