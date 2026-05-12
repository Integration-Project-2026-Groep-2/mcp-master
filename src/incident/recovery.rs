//! Stateless forwarder: consume `event.heartbeat_succeeded` from Controlroom's
//! watchdog and re-publish as `event.incident_resolved` so the Drupal
//! `/ai/incidents` dashboard can close the open incident for that service.
//!
//! Mirrors `incident::consumer`'s reconnect-loop shape but drops the LLM
//! pipeline, debouncer, and budget — recovery is a parse + scrub + publish,
//! ~ms latency. Separate durable queue isolates head-of-line blocking from
//! the failure-side dual-LLM cascade.
//!
//! Three controlroom body bugs (filed upstream) are sanitised here so the
//! operator-facing UI doesn't display them:
//!   - `severity: "critical"` → forced to `"info"`
//!   - `class: "heartbeat-loss"` → dropped from outbound payload
//!   - `summary` trailing `%!(EXTRA <type>=<value>)` (Go fmt arg-mismatch) → stripped

use std::sync::Arc;

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
use serde_json::json;
use tokio::sync::watch;
use uuid::Uuid;

use super::schema::IncidentEvent;
use crate::rabbitmq::config::RabbitMqConfig;
use crate::rabbitmq::publisher::Publisher;
use crate::retry::backoff_with_jitter;

const QUEUE_NAME: &str = "mcp-master.recoveries";
const ROUTING_KEY: &str = "event.heartbeat_succeeded";
const CONSUMER_TAG: &str = "mcp-master-recovery";
const RESOLVED_EVENT_NAME: &str = "incident_resolved";
const EXPECTED_BODY_EVENT: &str = "heartbeat_online";

pub async fn run(
    config: RabbitMqConfig,
    publisher: Option<Arc<Publisher>>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    tracing::info!("recovery consumer initialised");

    let mut attempt: u32 = 0;
    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }

        match consume_session(&config, publisher.as_ref(), &mut shutdown_rx).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempt = attempt.saturating_add(1);
                let delay = backoff_with_jitter(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "recovery consumer connection lost: {e:#} — reconnecting after backoff"
                );
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            tracing::info!("recovery consumer shutting down during backoff");
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
        "recovery consumer started"
    );

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("recovery consumer shutting down");
                    return Ok(());
                }
            }
            delivery = consumer.next() => match delivery {
                Some(Ok(msg)) => match handle_recovery(&msg.data, publisher).await {
                    Ok(()) => {
                        if let Err(e) = msg.ack(BasicAckOptions::default()).await {
                            tracing::warn!("recovery ack failed: {e:#}");
                        }
                    }
                    Err(e) => {
                        tracing::error!("recovery handler failed: {e:#}");
                        if let Err(nack_err) = msg
                            .nack(BasicNackOptions {
                                requeue: false,
                                multiple: false,
                            })
                            .await
                        {
                            tracing::warn!("recovery nack failed: {nack_err:#}");
                        }
                    }
                },
                Some(Err(e)) => {
                    return Err(anyhow::Error::from(e).context("recovery consumer delivery error"));
                }
                None => {
                    anyhow::bail!("recovery consumer stream ended unexpectedly");
                }
            }
        }
    }
}

async fn handle_recovery(body: &[u8], publisher: Option<&Arc<Publisher>>) -> Result<()> {
    let correlation_id = Uuid::new_v4().to_string();

    let evt: IncidentEvent =
        serde_json::from_slice(body).context("decoding recovery IncidentEvent envelope")?;

    if evt.event != EXPECTED_BODY_EVENT {
        tracing::warn!(
            correlation_id = %correlation_id,
            body_event = %evt.event,
            expected = EXPECTED_BODY_EVENT,
            "recovery body.event differs from expected — schema drift?"
        );
    }

    let service = evt.payload.component.clone();
    let original_summary = scrub_summary(&evt.payload.summary);

    tracing::info!(
        correlation_id = %correlation_id,
        service = %service,
        original_timestamp = %evt.timestamp,
        "publishing incident_resolved"
    );

    let payload = json!({
        "correlation_id": correlation_id,
        "service": service,
        "severity": "info",
        "original_summary": original_summary,
        "original_timestamp": evt.timestamp.to_rfc3339(),
        "source": "controlroom-watchdog",
    });

    match publisher {
        Some(p) => {
            if let Err(e) = p.publish_event(RESOLVED_EVENT_NAME, payload).await {
                tracing::warn!("publish_event({RESOLVED_EVENT_NAME}) failed: {e:#}");
            }
        }
        None => {
            tracing::warn!(
                correlation_id = %correlation_id,
                "publisher unavailable — skip-warn for incident_resolved"
            );
        }
    }

    Ok(())
}

fn scrub_summary(s: &str) -> String {
    match s.find("%!(EXTRA") {
        Some(idx) => s[..idx].trim_end().to_string(),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const RECOVERY_BODY: &str = r#"{
        "event": "heartbeat_online",
        "source": "controlroom-watchdog",
        "timestamp": "2026-05-12T16:30:00Z",
        "payload": {
            "summary": "KASSA heartbeat is back online%!(EXTRA float64=47)",
            "severity": "critical",
            "component": "kassa",
            "group": "festival-services",
            "class": "heartbeat-loss",
            "custom_details": {
                "heartbeat_count_last_60s": 47,
                "threshold": 30,
                "last_check_at": "2026-05-12T16:30:00Z"
            }
        }
    }"#;

    #[test]
    fn parses_heartbeat_online_body() {
        let evt: IncidentEvent = serde_json::from_str(RECOVERY_BODY).expect("parses");
        assert_eq!(evt.event, "heartbeat_online");
        assert_eq!(evt.payload.component, "kassa");
        assert!(evt.payload.summary.contains("%!(EXTRA"));
    }

    #[test]
    fn scrub_summary_strips_extra_tail() {
        let scrubbed = scrub_summary("KASSA heartbeat is back online%!(EXTRA float64=47)");
        assert_eq!(scrubbed, "KASSA heartbeat is back online");
    }

    #[test]
    fn scrub_summary_no_marker_unchanged() {
        let clean = "CRM heartbeat is back online";
        assert_eq!(scrub_summary(clean), clean);
    }

    #[test]
    fn scrub_summary_handles_empty_and_marker_only() {
        assert_eq!(scrub_summary(""), "");
        assert_eq!(scrub_summary("%!(EXTRA int=0)"), "");
    }

    #[tokio::test]
    async fn handle_recovery_accepts_valid_body_without_publisher() {
        let r = handle_recovery(RECOVERY_BODY.as_bytes(), None).await;
        assert!(r.is_ok(), "skip-warn path must not fail: {r:?}");
    }

    #[tokio::test]
    async fn handle_recovery_rejects_malformed_json() {
        let r = handle_recovery(b"not json", None).await;
        assert!(r.is_err(), "malformed JSON must surface as Err for DLQ");
    }

    #[tokio::test]
    async fn handle_recovery_rejects_unknown_severity() {
        let bad = RECOVERY_BODY.replace("\"critical\"", "\"meltdown\"");
        let r = handle_recovery(bad.as_bytes(), None).await;
        assert!(r.is_err());
    }

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

        assert!(started.elapsed().as_millis() < 2000);
        let inner = result.expect("join should succeed");
        assert!(inner.is_ok(), "run should return Ok on shutdown: {inner:?}");
    }
}
