//! Prometheus metrics: install the global recorder once at startup and hold the
//! render handle that the `/metrics` endpoint serves. Call-sites elsewhere
//! record through the `metrics::{counter,histogram,gauge}!` macros, so only this
//! module and the endpoint touch the exporter directly.

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

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::with_local_recorder;

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
}
