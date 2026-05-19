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
use quick_xml::de::from_str as xml_from_str;
use serde::Deserialize;

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
                Some(Ok(msg)) => {
                    let content_type = msg.properties.content_type().as_ref().map(|s| s.as_str());
                    match handle_recovery_with_content(&msg.data, publisher, content_type).await {
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
                    }
                }
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

async fn handle_recovery_with_content(
    body: &[u8],
    publisher: Option<&Arc<Publisher>>,
    content_type: Option<&str>,
) -> Result<()> {
    let correlation_id = Uuid::new_v4().to_string();

    let evt: IncidentEvent = if matches!(content_type, Some(ct) if ct.contains("json")) {
        serde_json::from_slice(body).context("decoding recovery IncidentEvent JSON envelope")?
    } else if matches!(content_type, Some(ct) if ct.contains("xml")) || (!body.is_empty() && body[0] == b'<') {
        #[derive(Debug, Deserialize)]
        struct XmlHeartbeatCustomDetails {
            #[serde(rename = "HeartbeatCountLast60s")]
            heartbeat_count_last_60s: f64,
            #[serde(rename = "Threshold")]
            threshold: Option<i64>,
            #[serde(rename = "LastCheckAt")]
            last_check_at: Option<String>,
        }

        #[derive(Debug, Deserialize)]
        struct XmlHeartbeatPayload {
            #[serde(rename = "Summary")]
            summary: String,
            #[serde(rename = "Severity")]
            severity: String,
            #[serde(rename = "Component")]
            component: String,
            #[serde(rename = "Group")]
            group: Option<String>,
            #[serde(rename = "Class")]
            class: Option<String>,
            #[serde(rename = "CustomDetails")]
            custom_details: XmlHeartbeatCustomDetails,
        }

        #[derive(Debug, Deserialize)]
        struct XmlHeartbeatEvent {
            #[serde(rename = "Event")]
            event: String,
            #[serde(rename = "Source")]
            source: String,
            #[serde(rename = "Timestamp")]
            timestamp: chrono::DateTime<chrono::Utc>,
            #[serde(rename = "Payload")]
            payload: XmlHeartbeatPayload,
        }

        let s = std::str::from_utf8(body).context("utf8 from xml body")?;
        let xml_evt: XmlHeartbeatEvent = xml_from_str(s).context("decoding Controlroom XML heartbeat")?;

        // Map to IncidentEvent
        let payload = crate::incident::schema::IncidentPayload {
            summary: xml_evt.payload.summary,
            severity: match xml_evt.payload.severity.to_lowercase().as_str() {
                "critical" => crate::incident::schema::Severity::Critical,
                "error" => crate::incident::schema::Severity::Error,
                "warning" => crate::incident::schema::Severity::Warning,
                "info" => crate::incident::schema::Severity::Info,
                other => anyhow::bail!("unknown severity: {other}"),
            },
            component: xml_evt.payload.component,
            group: xml_evt.payload.group,
            class: xml_evt.payload.class,
            custom_details: serde_json::json!({
                "heartbeat_count_last_60s": xml_evt.payload.custom_details.heartbeat_count_last_60s,
                "threshold": xml_evt.payload.custom_details.threshold,
                "last_check_at": xml_evt.payload.custom_details.last_check_at,
            }),
        };

        IncidentEvent {
            event: xml_evt.event,
            source: xml_evt.source,
            timestamp: xml_evt.timestamp,
            payload,
        }
    } else {
        serde_json::from_slice(body).context("decoding recovery IncidentEvent envelope")?
    };

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

// Backwards-compatible wrapper for callers that don't provide content_type
async fn handle_recovery(body: &[u8], publisher: Option<&Arc<Publisher>>) -> Result<()> {
    handle_recovery_with_content(body, publisher, None).await
}

fn scrub_summary(s: &str) -> String {
    match s.find("%!(EXTRA") {
        Some(idx) => s[..idx].trim_end().to_string(),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests;
