//! Prometheus metrics: install the global recorder once at startup and hold the
//! render handle that the `/metrics` endpoint serves. Call-sites elsewhere
//! record through the `metrics::{counter,histogram,gauge}!` macros, so only this
//! module and the endpoint touch the exporter directly.

use std::time::Instant;

use axum::{
    extract::{MatchedPath, Request},
    middleware::Next,
    response::Response,
};
use metrics_exporter_prometheus::PrometheusBuilder;

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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::Request as HttpRequest, routing::get};
    use metrics::with_local_recorder;
    use tower::ServiceExt;

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
}
