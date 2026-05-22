//! Prometheus metrics: install the global recorder once at startup and hold the
//! render handle that the `/metrics` endpoint serves. Call-sites elsewhere
//! record through the `record_*` helpers here (or the `metrics::*!` macros), so
//! metric names live in one place and only this module touches the exporter.

use std::sync::OnceLock;
use std::time::Instant;

use axum::{
    extract::{MatchedPath, Request},
    middleware::Next,
    response::Response,
};
use metrics_exporter_prometheus::PrometheusBuilder;

use crate::agent::llm::TokenUsage;

/// Render handle for the `/metrics` exposition. Cheap to clone (Arc inside).
pub type Handle = metrics_exporter_prometheus::PrometheusHandle;

/// Install the process-wide recorder. Call exactly once (from `serve()`);
/// metric macros are no-ops until it runs.
pub fn install() -> anyhow::Result<Handle> {
    PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| anyhow::anyhow!("install prometheus recorder: {e}"))
}

/// Matched route pattern (`/chat`, `/memory/user/{user_id}`) rather than the raw
/// URI, so path params don't blow up label cardinality. Falls back to the path.
fn route_label(req: &Request) -> String {
    req.extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| req.uri().path().to_owned())
}

/// Count and time every HTTP request. Apply via `route_layer` — `MatchedPath`
/// is only populated after routing, so a plain `layer` would see it as unset.
pub async fn track_http(req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_owned();
    let route = route_label(&req);
    let start = Instant::now();
    let response = next.run(req).await;
    let status = response.status().as_u16().to_string();
    metrics::counter!(
        "http_requests_total",
        "method" => method,
        "route" => route.clone(),
        "status" => status,
    )
    .increment(1);
    metrics::histogram!("http_request_duration_seconds", "route" => route)
        .record(start.elapsed().as_secs_f64());
    response
}

/// Record one MCP tool dispatch — called from `McpPool::call` next to the
/// `tool_called` event so metrics work even without a broker.
pub fn record_tool_call(tool: &str, server: &str, ok: bool, duration_ms: u64) {
    metrics::counter!(
        "mcp_tool_calls_total",
        "tool" => tool.to_string(),
        "server" => server.to_string(),
        "ok" => ok.to_string(),
    )
    .increment(1);
    metrics::histogram!(
        "mcp_tool_call_duration_seconds",
        "tool" => tool.to_string(),
        "server" => server.to_string(),
    )
    .record(duration_ms as f64 / 1000.0);
}

/// Per-million-token prices for cost metering. Defaults are Claude Sonnet 4.x
/// USD list prices; override per-deployment via `LLM_PRICE_*_PER_MTOK` and
/// `LLM_PRICE_CURRENCY` (e.g. set EUR-adjusted values + `eur`).
#[derive(Debug, Clone)]
struct Prices {
    input: f64,
    output: f64,
    cache_write: f64,
    cache_read: f64,
    currency: String,
}

impl Prices {
    fn from_env() -> Self {
        let num = |key: &str, default: f64| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .unwrap_or(default)
        };
        Self {
            input: num("LLM_PRICE_INPUT_PER_MTOK", 3.0),
            output: num("LLM_PRICE_OUTPUT_PER_MTOK", 15.0),
            cache_write: num("LLM_PRICE_CACHE_WRITE_PER_MTOK", 3.75),
            cache_read: num("LLM_PRICE_CACHE_READ_PER_MTOK", 0.30),
            currency: std::env::var("LLM_PRICE_CURRENCY")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "usd".to_string()),
        }
    }
}

fn prices() -> &'static Prices {
    static P: OnceLock<Prices> = OnceLock::new();
    P.get_or_init(Prices::from_env)
}

/// Cost of one request from token counts and per-MTok prices.
fn compute_cost(p: &Prices, t: &TokenUsage) -> f64 {
    let part = |n: u32, price: f64| (n as f64) / 1_000_000.0 * price;
    part(t.input, p.input)
        + part(t.output, p.output)
        + part(t.cache_creation_input.unwrap_or(0), p.cache_write)
        + part(t.cache_read_input.unwrap_or(0), p.cache_read)
}

fn record_tokens(t: &TokenUsage) {
    if t.input > 0 {
        metrics::counter!("llm_tokens_total", "kind" => "input").increment(t.input as u64);
    }
    if t.output > 0 {
        metrics::counter!("llm_tokens_total", "kind" => "output").increment(t.output as u64);
    }
    if let Some(c) = t.cache_creation_input
        && c > 0
    {
        metrics::counter!("llm_tokens_total", "kind" => "cache_creation").increment(c as u64);
    }
    if let Some(c) = t.cache_read_input
        && c > 0
    {
        metrics::counter!("llm_tokens_total", "kind" => "cache_read").increment(c as u64);
    }
}

/// Record one chat request: outcome counter, per-class token counters, and the
/// per-request cost (histogram → `_sum`/`_count` give total + cost/request).
pub fn record_chat(mode: &str, outcome: &str, tokens: &TokenUsage) {
    metrics::counter!(
        "chat_requests_total",
        "mode" => mode.to_string(),
        "outcome" => outcome.to_string(),
    )
    .increment(1);
    record_tokens(tokens);
    let p = prices();
    metrics::histogram!(
        "llm_request_cost",
        "currency" => p.currency.clone(),
        "mode" => mode.to_string(),
    )
    .record(compute_cost(p, tokens));
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::Request as HttpRequest, routing::get};
    use metrics::with_local_recorder;
    use tower::ServiceExt;

    fn test_prices() -> Prices {
        Prices {
            input: 3.0,
            output: 15.0,
            cache_write: 3.75,
            cache_read: 0.30,
            currency: "usd".to_string(),
        }
    }

    #[test]
    fn handle_renders_recorded_metric() {
        // build_recorder() stays off the global slot so the test is isolated.
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        with_local_recorder(&recorder, || {
            metrics::counter!("mcp_master_test_total").increment(1);
        });
        assert!(handle.render().contains("mcp_master_test_total"));
    }

    #[test]
    fn route_label_falls_back_to_uri_path_without_matched_path() {
        let req = HttpRequest::builder()
            .uri("/raw/path")
            .body(Body::empty())
            .unwrap();
        assert_eq!(route_label(&req), "/raw/path");
    }

    #[test]
    fn track_http_records_request_total() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let app: Router = Router::new()
            .route("/ping", get(|| async { "ok" }))
            .route_layer(axum::middleware::from_fn(track_http));
        // current-thread runtime so the with_local_recorder thread-local is the
        // recorder the middleware records into (it runs on the calling thread).
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        with_local_recorder(&recorder, || {
            let req = HttpRequest::builder()
                .uri("/ping")
                .body(Body::empty())
                .unwrap();
            let resp = rt.block_on(async { app.oneshot(req).await.unwrap() });
            assert_eq!(resp.status(), 200);
        });
        let out = handle.render();
        assert!(out.contains("http_requests_total"), "render:\n{out}");
        assert!(out.contains("route=\"/ping\""), "render:\n{out}");
    }

    #[test]
    fn record_tool_call_emits_counter_and_histogram() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        with_local_recorder(&recorder, || {
            record_tool_call("count_contacts", "crm", true, 412);
        });
        let out = handle.render();
        assert!(out.contains("mcp_tool_calls_total"), "render:\n{out}");
        assert!(out.contains("tool=\"count_contacts\""), "render:\n{out}");
        assert!(
            out.contains("mcp_tool_call_duration_seconds"),
            "render:\n{out}"
        );
    }

    #[test]
    fn compute_cost_sums_token_classes() {
        let p = test_prices();
        let one_m = TokenUsage {
            input: 1_000_000,
            output: 1_000_000,
            cache_creation_input: None,
            cache_read_input: None,
        };
        assert!((compute_cost(&p, &one_m) - 18.0).abs() < 1e-9);

        let cached = TokenUsage {
            input: 0,
            output: 0,
            cache_creation_input: Some(1_000_000),
            cache_read_input: Some(1_000_000),
        };
        assert!((compute_cost(&p, &cached) - (3.75 + 0.30)).abs() < 1e-9);
    }

    #[test]
    fn record_chat_emits_requests_tokens_and_cost() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let t = TokenUsage {
            input: 1000,
            output: 500,
            cache_creation_input: None,
            cache_read_input: None,
        };
        with_local_recorder(&recorder, || record_chat("sync", "ok", &t));
        let out = handle.render();
        assert!(out.contains("chat_requests_total"), "render:\n{out}");
        assert!(out.contains("llm_tokens_total"), "render:\n{out}");
        assert!(out.contains("kind=\"input\""), "render:\n{out}");
        assert!(out.contains("llm_request_cost"), "render:\n{out}");
    }
}
