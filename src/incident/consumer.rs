use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use lapin::{
    Connection, ConnectionProperties, ExchangeKind,
    options::{
        BasicAckOptions, BasicConsumeOptions, BasicNackOptions, ExchangeDeclareOptions,
        QueueBindOptions, QueueDeclareOptions,
    },
    types::FieldTable,
};
use tokio::sync::watch;
use uuid::Uuid;

use super::budget::{Budget, BudgetOutcome};
use super::debounce::Debouncer;
use super::diagnose::DiagnosePipeline;
use super::schema::{IncidentDiagnosis, IncidentEvent};
use crate::rabbitmq::config::RabbitMqConfig;
use crate::rabbitmq::publisher::Publisher;
use crate::retry::backoff_with_jitter;

const QUEUE_NAME: &str = "mcp-master.incidents";
const ROUTING_KEY: &str = "event.heartbeat_failed";
const CONSUMER_TAG: &str = "mcp-master-incident";
const SKIP_EVENT_NAME: &str = "incident_skipped";
const DIAGNOSED_EVENT_NAME: &str = "incident_diagnosed";
const CIRCUIT_OPEN_EVENT_NAME: &str = "incident_circuit_open";

/// Consume `event.heartbeat_failed` until shutdown, reconnecting through
/// transient broker outages. Mirrors `rabbitmq::consumer::run` but with a
/// durable named queue + manual ack so an mcp-master restart during an
/// incident-piek doesn't drop deliveries — the broker buffers until we
/// reconnect.
///
/// `publisher` is optional: when absent (skip-warn path), skip-events are
/// only logged. The Debouncer lives here (not per-session) so it persists
/// across reconnects — a service flapping during a broker-blip shouldn't
/// reset its debounce slot.
///
/// `pipeline` is optional too: when absent, accepted events are logged but
/// no Step A+B run. Tests pass `None` to skip the (heavy to mock) pipeline;
/// production wires `Some(DefaultDiagnosePipeline::new(state))`.
pub async fn run(
    config: RabbitMqConfig,
    publisher: Option<Arc<Publisher>>,
    pipeline: Option<Arc<dyn DiagnosePipeline>>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let debouncer = Arc::new(Debouncer::from_env());
    let budget = Arc::new(Budget::from_env());
    tracing::info!(
        window_s = debouncer.window().as_secs(),
        max_per_hour = budget.max_per_hour(),
        "incident consumer initialised"
    );

    let mut attempt: u32 = 0;
    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }

        match consume_session(
            &config,
            &debouncer,
            &budget,
            publisher.as_ref(),
            pipeline.as_ref(),
            &mut shutdown_rx,
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempt = attempt.saturating_add(1);
                let delay = backoff_with_jitter(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "incident consumer connection lost: {e:#} — reconnecting after backoff"
                );
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            tracing::info!("incident consumer shutting down during backoff");
                            return Ok(());
                        }
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
    }
}

async fn consume_session(
    config: &RabbitMqConfig,
    debouncer: &Debouncer,
    budget: &Budget,
    publisher: Option<&Arc<Publisher>>,
    pipeline: Option<&Arc<dyn DiagnosePipeline>>,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> Result<()> {
    let conn = Connection::connect(&config.url, ConnectionProperties::default())
        .await
        .context("AMQP connection")?;
    let channel = conn.create_channel().await.context("AMQP channel")?;

    channel
        .exchange_declare(
            &config.exchange,
            ExchangeKind::Topic,
            ExchangeDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP exchange_declare")?;

    channel
        .queue_declare(
            QUEUE_NAME,
            QueueDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP queue_declare (durable named)")?;

    channel
        .queue_bind(
            QUEUE_NAME,
            &config.exchange,
            ROUTING_KEY,
            QueueBindOptions::default(),
            FieldTable::default(),
        )
        .await
        .context("AMQP queue_bind")?;

    let mut consumer = channel
        .basic_consume(
            QUEUE_NAME,
            CONSUMER_TAG,
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .context("AMQP basic_consume")?;

    tracing::info!(
        exchange = %config.exchange,
        queue = QUEUE_NAME,
        routing_key = ROUTING_KEY,
        "incident consumer started"
    );

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("incident consumer shutting down");
                    return Ok(());
                }
            }
            delivery = consumer.next() => match delivery {
                Some(Ok(msg)) => match handle_delivery(&msg.data, debouncer, budget, publisher, pipeline).await {
                    Ok(()) => {
                        if let Err(e) = msg.ack(BasicAckOptions::default()).await {
                            tracing::warn!("incident ack failed: {e:#}");
                        }
                    }
                    Err(e) => {
                        tracing::error!("incident handler failed: {e:#}");
                        if let Err(nack_err) = msg
                            .nack(BasicNackOptions {
                                requeue: false,
                                multiple: false,
                            })
                            .await
                        {
                            tracing::warn!("incident nack failed: {nack_err:#}");
                        }
                    }
                },
                Some(Err(e)) => {
                    return Err(anyhow::Error::from(e).context("incident consumer delivery error"));
                }
                None => {
                    anyhow::bail!("incident consumer stream ended unexpectedly");
                }
            }
        }
    }
}

async fn handle_delivery(
    body: &[u8],
    debouncer: &Debouncer,
    budget: &Budget,
    publisher: Option<&Arc<Publisher>>,
    pipeline: Option<&Arc<dyn DiagnosePipeline>>,
) -> Result<()> {
    let received_at = Instant::now();
    let correlation_id = Uuid::new_v4().to_string();

    let evt: IncidentEvent =
        serde_json::from_slice(body).context("decoding IncidentEvent envelope")?;

    if !evt.payload.severity.is_actionable() {
        tracing::info!(
            correlation_id = %correlation_id,
            service = %evt.payload.component,
            severity = ?evt.payload.severity,
            "incident skipped (severity_too_low)"
        );
        publish_skip(publisher, &evt, &correlation_id, "severity_too_low", None).await;
        return Ok(());
    }

    if let Err(elapsed) = debouncer.check(&evt.payload.component) {
        tracing::info!(
            correlation_id = %correlation_id,
            service = %evt.payload.component,
            elapsed_s = elapsed.as_secs(),
            "incident skipped (debounced)"
        );
        publish_skip(publisher, &evt, &correlation_id, "debounced", Some(elapsed)).await;
        return Ok(());
    }

    if let BudgetOutcome::CircuitOpen { reset_at } = budget.try_consume() {
        let reset_in_s = reset_at.saturating_duration_since(Instant::now()).as_secs();
        tracing::warn!(
            correlation_id = %correlation_id,
            service = %evt.payload.component,
            reset_in_s,
            "incident circuit open — skipping diagnosis (budget exhausted)"
        );
        publish_circuit_open(publisher, &evt, &correlation_id, reset_in_s).await;
        return Ok(());
    }

    tracing::info!(
        correlation_id = %correlation_id,
        service = %evt.payload.component,
        severity = ?evt.payload.severity,
        timestamp = %evt.timestamp,
        "incident accepted for diagnosis"
    );

    let Some(pl) = pipeline else {
        tracing::warn!(
            correlation_id = %correlation_id,
            service = %evt.payload.component,
            "no DiagnosePipeline configured — Step A+B skipped"
        );
        return Ok(());
    };

    let pipeline_start = Instant::now();
    match pl.diagnose(&evt).await {
        Ok(diagnosis) => {
            let pipeline_ms = pipeline_start.elapsed().as_millis() as u64;
            let total_ms = received_at.elapsed().as_millis() as u64;
            tracing::info!(
                correlation_id = %correlation_id,
                service = %evt.payload.component,
                confidence = ?diagnosis.confidence,
                pipeline_ms,
                total_ms,
                "incident diagnosed"
            );
            publish_diagnosis(publisher, &evt, &correlation_id, &diagnosis).await;
        }
        Err(e) => {
            let pipeline_ms = pipeline_start.elapsed().as_millis() as u64;
            tracing::error!(
                correlation_id = %correlation_id,
                service = %evt.payload.component,
                pipeline_ms,
                "diagnose pipeline failed: {e:#}"
            );
        }
    }

    Ok(())
}

async fn publish_skip(
    publisher: Option<&Arc<Publisher>>,
    evt: &IncidentEvent,
    correlation_id: &str,
    reason: &str,
    elapsed: Option<Duration>,
) {
    let Some(p) = publisher else {
        return;
    };
    let mut payload = serde_json::json!({
        "service": evt.payload.component,
        "correlation_id": correlation_id,
        "reason": reason,
        "severity": evt.payload.severity,
        "original_summary": evt.payload.summary,
        "original_timestamp": evt.timestamp.to_rfc3339(),
    });
    if let Some(e) = elapsed {
        payload["since_last_diagnosis_seconds"] = serde_json::json!(e.as_secs());
    }
    if let Err(e) = p.publish_event(SKIP_EVENT_NAME, payload).await {
        tracing::warn!("publish_event({SKIP_EVENT_NAME}) failed: {e:#}");
    }
}

async fn publish_diagnosis(
    publisher: Option<&Arc<Publisher>>,
    event: &IncidentEvent,
    correlation_id: &str,
    diagnosis: &IncidentDiagnosis,
) {
    let Some(p) = publisher else {
        return;
    };
    let payload = serde_json::json!({
        "service": event.payload.component,
        "correlation_id": correlation_id,
        "severity": event.payload.severity,
        "diagnosis": diagnosis,
        "original_summary": event.payload.summary,
        "original_timestamp": event.timestamp.to_rfc3339(),
    });
    if let Err(e) = p.publish_event(DIAGNOSED_EVENT_NAME, payload).await {
        tracing::warn!("publish_event({DIAGNOSED_EVENT_NAME}) failed: {e:#}");
    }
}

async fn publish_circuit_open(
    publisher: Option<&Arc<Publisher>>,
    event: &IncidentEvent,
    correlation_id: &str,
    reset_in_seconds: u64,
) {
    let Some(p) = publisher else {
        return;
    };
    let payload = serde_json::json!({
        "service": event.payload.component,
        "correlation_id": correlation_id,
        "reset_in_seconds": reset_in_seconds,
        "original_summary": event.payload.summary,
        "original_timestamp": event.timestamp.to_rfc3339(),
    });
    if let Err(e) = p.publish_event(CIRCUIT_OPEN_EVENT_NAME, payload).await {
        tracing::warn!("publish_event({CIRCUIT_OPEN_EVENT_NAME}) failed: {e:#}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use tokio::sync::watch;

    #[tokio::test]
    async fn run_returns_immediately_on_pre_set_shutdown() {
        let (tx, rx) = watch::channel(false);
        tx.send(true).unwrap();

        let cfg = RabbitMqConfig {
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let started = Instant::now();
        let result = run(cfg, None, None, rx).await;
        let elapsed = started.elapsed();

        assert!(result.is_ok(), "shutdown-first should succeed: {result:?}");
        assert!(
            elapsed.as_millis() < 100,
            "should return fast (<100ms), took {}ms",
            elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn run_honors_shutdown_during_backoff() {
        let (tx, rx) = watch::channel(false);

        let cfg = RabbitMqConfig {
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let handle = tokio::spawn(run(cfg, None, None, rx));

        tokio::time::sleep(Duration::from_millis(150)).await;
        tx.send(true).unwrap();

        let started = Instant::now();
        let result = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("run should not hang past 2s after shutdown signal");

        assert!(
            started.elapsed().as_millis() < 2000,
            "shutdown should wake the backoff sleep early"
        );
        assert!(
            result.is_ok(),
            "join should succeed: {:?}",
            result.unwrap_err()
        );
        let inner = result.unwrap();
        assert!(inner.is_ok(), "run should return Ok on shutdown: {inner:?}");
    }

    fn body(severity: &str, component: &str) -> Vec<u8> {
        format!(
            r#"{{
                "event": "heartbeat_failed",
                "source": "controlroom-watchdog",
                "timestamp": "2026-05-10T14:23:17Z",
                "payload": {{
                    "summary": "{component} down",
                    "severity": "{severity}",
                    "component": "{component}"
                }}
            }}"#
        )
        .into_bytes()
    }

    #[tokio::test]
    async fn handle_delivery_accepts_critical_first_time() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn handle_delivery_skips_warning_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(&body("warning", "kassa"), &d, &b, None, None).await;
        assert!(r.is_ok(), "skip is not an error path");
    }

    #[tokio::test]
    async fn handle_delivery_skips_info_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(&body("info", "kassa"), &d, &b, None, None).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn handle_delivery_debounces_rapid_repeats() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let first = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        let second = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        assert!(first.is_ok());
        assert!(second.is_ok(), "second is skipped, not failed");
        assert!(d.check("kassa").is_err(), "first allow consumed the slot");
    }

    #[tokio::test]
    async fn handle_delivery_severity_skip_does_not_consume_debounce_slot() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let warn_result = handle_delivery(&body("warning", "kassa"), &d, &b, None, None).await;
        assert!(warn_result.is_ok());
        // Critical should now be allowed — warning didn't burn the slot.
        let crit_result = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        assert!(crit_result.is_ok());
        assert!(d.check("kassa").is_err());
    }

    #[tokio::test]
    async fn handle_delivery_rejects_malformed_json() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(b"not json", &d, &b, None, None).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn handle_delivery_rejects_unknown_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let body = br#"{
            "event": "heartbeat_failed",
            "source": "controlroom-watchdog",
            "timestamp": "2026-05-10T14:23:17Z",
            "payload": {
                "summary": "x",
                "severity": "meltdown",
                "component": "x"
            }
        }"#;
        let r = handle_delivery(body, &d, &b, None, None).await;
        assert!(r.is_err());
    }

    use crate::incident::schema::{Confidence, IncidentDiagnosis};
    use async_trait::async_trait;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Mutex;

    struct MockPipeline {
        queued: Mutex<VecDeque<Result<IncidentDiagnosis, String>>>,
        call_count: AtomicUsize,
    }

    impl MockPipeline {
        fn new() -> Self {
            Self {
                queued: Mutex::new(VecDeque::new()),
                call_count: AtomicUsize::new(0),
            }
        }

        async fn with_ok(self, d: IncidentDiagnosis) -> Self {
            self.queued.lock().await.push_back(Ok(d));
            self
        }

        async fn with_err(self, msg: &str) -> Self {
            self.queued.lock().await.push_back(Err(msg.into()));
            self
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl DiagnosePipeline for MockPipeline {
        async fn diagnose(&self, _event: &IncidentEvent) -> Result<IncidentDiagnosis> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            match self.queued.lock().await.pop_front() {
                Some(Ok(d)) => Ok(d),
                Some(Err(e)) => Err(anyhow::anyhow!("{e}")),
                None => Err(anyhow::anyhow!("MockPipeline: no canned response")),
            }
        }
    }

    fn diag(confidence: Confidence) -> IncidentDiagnosis {
        IncidentDiagnosis {
            root_cause: "test cause".into(),
            critical_failure: "test failure".into(),
            impact: "test impact".into(),
            confidence,
            suggested_action: None,
            evidence_summary: "test evidence".into(),
        }
    }

    #[tokio::test]
    async fn handle_delivery_calls_pipeline_when_configured_and_accepted() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let mock = Arc::new(MockPipeline::new().with_ok(diag(Confidence::High)).await);
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(r.is_ok());
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn handle_delivery_skips_pipeline_when_severity_filtered() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let mock = Arc::new(MockPipeline::new());
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();
        let r = handle_delivery(&body("warning", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(r.is_ok());
        assert_eq!(mock.call_count(), 0);
    }

    #[tokio::test]
    async fn handle_delivery_skips_pipeline_when_debounced() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let mock = Arc::new(MockPipeline::new().with_ok(diag(Confidence::High)).await);
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();

        let first =
            handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        let second =
            handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(first.is_ok());
        assert!(second.is_ok());
        assert_eq!(mock.call_count(), 1, "second event was debounced");
    }

    #[tokio::test]
    async fn handle_delivery_swallows_pipeline_errors_so_message_is_acked() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let mock = Arc::new(MockPipeline::new().with_err("LLM timeout").await);
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(
            r.is_ok(),
            "pipeline failure must not propagate (would trigger DLQ)"
        );
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn handle_delivery_accepts_when_no_pipeline_configured() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(u32::MAX);
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, None).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn handle_delivery_skips_pipeline_when_budget_exhausted() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(0);
        let mock = Arc::new(MockPipeline::new());
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        assert!(r.is_ok());
        assert_eq!(
            mock.call_count(),
            0,
            "pipeline must not run when circuit is open"
        );
    }

    #[tokio::test]
    async fn handle_delivery_budget_caps_total_pipeline_calls() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(2);
        let mock = Arc::new(
            MockPipeline::new()
                .with_ok(diag(Confidence::High))
                .await
                .with_ok(diag(Confidence::Medium))
                .await
                .with_ok(diag(Confidence::Low))
                .await,
        );
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();

        let r1 = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;
        let r2 = handle_delivery(&body("critical", "crm"), &d, &b, None, Some(&pipeline)).await;
        let r3 = handle_delivery(
            &body("critical", "controlroom"),
            &d,
            &b,
            None,
            Some(&pipeline),
        )
        .await;

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(r3.is_ok());
        assert_eq!(mock.call_count(), 2, "third event must hit circuit-open");
    }

    #[tokio::test]
    async fn handle_delivery_severity_skip_does_not_consume_budget() {
        let d = Debouncer::new(Duration::from_secs(60));
        let b = Budget::new(1);
        let mock = Arc::new(MockPipeline::new().with_ok(diag(Confidence::High)).await);
        let pipeline: Arc<dyn DiagnosePipeline> = mock.clone();

        let _ = handle_delivery(&body("warning", "kassa"), &d, &b, None, Some(&pipeline)).await;
        let r = handle_delivery(&body("critical", "kassa"), &d, &b, None, Some(&pipeline)).await;

        assert!(r.is_ok());
        assert_eq!(
            mock.call_count(),
            1,
            "warning skip didn't burn the budget slot"
        );
    }
}
