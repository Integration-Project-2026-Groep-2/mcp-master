use std::sync::Arc;
use std::time::Duration;

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

use super::debounce::Debouncer;
use super::schema::IncidentEvent;
use crate::rabbitmq::config::RabbitMqConfig;
use crate::rabbitmq::publisher::Publisher;
use crate::retry::backoff_with_jitter;

const QUEUE_NAME: &str = "mcp-master.incidents";
const ROUTING_KEY: &str = "event.heartbeat_failed";
const CONSUMER_TAG: &str = "mcp-master-incident";
const SKIP_EVENT_NAME: &str = "incident_skipped";

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
pub async fn run(
    config: RabbitMqConfig,
    publisher: Option<Arc<Publisher>>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let debouncer = Arc::new(Debouncer::from_env());
    tracing::info!(
        window_s = debouncer.window().as_secs(),
        "incident debouncer initialised"
    );

    let mut attempt: u32 = 0;
    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }

        match consume_session(&config, &debouncer, publisher.as_ref(), &mut shutdown_rx).await {
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
    publisher: Option<&Arc<Publisher>>,
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
                Some(Ok(msg)) => match handle_delivery(&msg.data, debouncer, publisher).await {
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
    publisher: Option<&Arc<Publisher>>,
) -> Result<()> {
    let evt: IncidentEvent =
        serde_json::from_slice(body).context("decoding IncidentEvent envelope")?;

    if !evt.payload.severity.is_actionable() {
        tracing::info!(
            service = %evt.payload.component,
            severity = ?evt.payload.severity,
            "incident skipped (severity_too_low)"
        );
        publish_skip(publisher, &evt, "severity_too_low", None).await;
        return Ok(());
    }

    if let Err(elapsed) = debouncer.check(&evt.payload.component) {
        tracing::info!(
            service = %evt.payload.component,
            elapsed_s = elapsed.as_secs(),
            "incident skipped (debounced)"
        );
        publish_skip(publisher, &evt, "debounced", Some(elapsed)).await;
        return Ok(());
    }

    tracing::info!(
        service = %evt.payload.component,
        severity = ?evt.payload.severity,
        timestamp = %evt.timestamp,
        summary = %evt.payload.summary,
        "incident accepted for diagnosis"
    );
    Ok(())
}

async fn publish_skip(
    publisher: Option<&Arc<Publisher>>,
    evt: &IncidentEvent,
    reason: &str,
    elapsed: Option<Duration>,
) {
    let Some(p) = publisher else {
        return;
    };
    let mut payload = serde_json::json!({
        "service": evt.payload.component,
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
        let result = run(cfg, None, rx).await;
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

        let handle = tokio::spawn(run(cfg, None, rx));

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
        let r = handle_delivery(&body("critical", "kassa"), &d, None).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn handle_delivery_skips_warning_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
        let r = handle_delivery(&body("warning", "kassa"), &d, None).await;
        assert!(r.is_ok(), "skip is not an error path");
    }

    #[tokio::test]
    async fn handle_delivery_skips_info_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
        let r = handle_delivery(&body("info", "kassa"), &d, None).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn handle_delivery_debounces_rapid_repeats() {
        let d = Debouncer::new(Duration::from_secs(60));
        let first = handle_delivery(&body("critical", "kassa"), &d, None).await;
        let second = handle_delivery(&body("critical", "kassa"), &d, None).await;
        assert!(first.is_ok());
        assert!(second.is_ok(), "second is skipped, not failed");
        assert!(d.check("kassa").is_err(), "first allow consumed the slot");
    }

    #[tokio::test]
    async fn handle_delivery_severity_skip_does_not_consume_debounce_slot() {
        let d = Debouncer::new(Duration::from_secs(60));
        let warn_result = handle_delivery(&body("warning", "kassa"), &d, None).await;
        assert!(warn_result.is_ok());
        // Critical should now be allowed — warning didn't burn the slot.
        let crit_result = handle_delivery(&body("critical", "kassa"), &d, None).await;
        assert!(crit_result.is_ok());
        assert!(d.check("kassa").is_err());
    }

    #[tokio::test]
    async fn handle_delivery_rejects_malformed_json() {
        let d = Debouncer::new(Duration::from_secs(60));
        let r = handle_delivery(b"not json", &d, None).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn handle_delivery_rejects_unknown_severity() {
        let d = Debouncer::new(Duration::from_secs(60));
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
        let r = handle_delivery(body, &d, None).await;
        assert!(r.is_err());
    }
}
