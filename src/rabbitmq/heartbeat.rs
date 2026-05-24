//! Liveness heartbeat publisher — emits a `<Heartbeat>` XML document on the
//! `heartbeat.direct` exchange so Controlroom's watchdog monitors this service
//! the same way it monitors the other teams (Facturatie, Kassa, ...).
//!
//! Wire-contract mirrors Facturatie's `RabbitMQService::publishHeartbeat`:
//! a `direct` exchange, routing-key `routing.heartbeat`, `application/xml` body
//! `<Heartbeat><serviceId>…</serviceId><timestamp>…</timestamp></Heartbeat>`,
//! persistent delivery, one publish per `interval` (default 1s). The producer
//! only declares the exchange; the `heartbeat_queue` bind is the watchdog side.

use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use lapin::{
    BasicProperties, ExchangeKind,
    options::{BasicPublishOptions, ExchangeDeclareOptions},
    types::FieldTable,
};
use tokio::sync::watch;

use crate::retry::backoff_with_jitter;

const DEFAULT_EXCHANGE: &str = "heartbeat.direct";
const DEFAULT_ROUTING_KEY: &str = "routing.heartbeat";
const DEFAULT_SERVICE_ID: &str = "mcp-master";
const DEFAULT_INTERVAL_MS: u64 = 1000;

// Floor matches Facturatie's clamp: a too-small interval would hammer the
// broker without making the watchdog any happier.
const MIN_INTERVAL_MS: u64 = 100;

// Ceiling guards the watchdog contract: it raises a heartbeat-loss incident
// below ~30 beats/60s (one per 2s), so the ceiling sits below that with margin
// — a config typo can't push cadence onto the boundary and self-trigger.
const MAX_INTERVAL_MS: u64 = 1500;

// Cap reconnect backoff well under the watchdog window: the shared schedule
// climbs to 32s, but 32s of silence after a broker blip trips the watchdog by
// itself. Facturatie retries at ~1s; this keeps post-recovery cadence that fast.
const MAX_RECONNECT_BACKOFF: Duration = Duration::from_secs(2);

const DELIVERY_MODE_PERSISTENT: u8 = 2;

#[derive(Clone, Debug)]
pub struct HeartbeatConfig {
    pub exchange: String,
    pub routing_key: String,
    pub service_id: String,
    pub interval: Duration,
}

impl HeartbeatConfig {
    pub fn from_env() -> Self {
        let requested_ms = std::env::var("HEARTBEAT_INTERVAL_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_INTERVAL_MS);
        let interval_ms = requested_ms.clamp(MIN_INTERVAL_MS, MAX_INTERVAL_MS);
        if interval_ms != requested_ms {
            tracing::warn!(
                requested_ms,
                clamped_ms = interval_ms,
                "HEARTBEAT_INTERVAL_MS outside [{MIN_INTERVAL_MS}, {MAX_INTERVAL_MS}] — clamped to keep cadence within the watchdog window"
            );
        }

        Self {
            exchange: env_nonempty("HEARTBEAT_EXCHANGE")
                .unwrap_or_else(|| DEFAULT_EXCHANGE.to_string()),
            routing_key: env_nonempty("HEARTBEAT_ROUTING_KEY")
                .unwrap_or_else(|| DEFAULT_ROUTING_KEY.to_string()),
            service_id: env_nonempty("HEARTBEAT_SERVICE_ID")
                .or_else(|| env_nonempty("SERVICE_ID"))
                .unwrap_or_else(|| DEFAULT_SERVICE_ID.to_string()),
            interval: Duration::from_millis(interval_ms),
        }
    }
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Byte-matches Facturatie's `DOMDocument::saveXML` output: declaration line
/// then a single-line element, `serviceId` XML-escaped (saveXML escapes too).
/// Timestamp is `xs:dateTime` at second precision with a numeric offset
/// (`+00:00`, not `Z`) — chrono's `SecondsFormat::Secs` + `use_z=false` equals
/// PHP's `Y-m-d\TH:i:sP`.
pub fn build_heartbeat_xml(service_id: &str, timestamp: &DateTime<Utc>) -> String {
    let service_id = quick_xml::escape::partial_escape(service_id);
    let ts = timestamp.to_rfc3339_opts(SecondsFormat::Secs, false);
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <Heartbeat><serviceId>{service_id}</serviceId><timestamp>{ts}</timestamp></Heartbeat>\n"
    )
}

/// Publish a heartbeat every `cfg.interval` until shutdown, reconnecting
/// through transient broker outages with the shared backoff schedule. Mirrors
/// `rabbitmq::consumer::run`: returns `Ok(())` only on the shutdown signal —
/// any publish/connection failure bubbles up so the outer loop reconnects.
pub async fn run(
    url: String,
    cfg: HeartbeatConfig,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let mut attempt: u32 = 0;
    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }

        match publish_session(&url, &cfg, &mut shutdown_rx).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempt = attempt.saturating_add(1);
                let delay = backoff_with_jitter(attempt).min(MAX_RECONNECT_BACKOFF);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "heartbeat publisher connection lost: {e:#} — reconnecting after backoff"
                );
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            tracing::info!("heartbeat publisher shutting down during backoff");
                            return Ok(());
                        }
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
    }
}

/// One connect-and-publish lifecycle. Returns `Ok(())` only when the shutdown
/// signal fires; a publish error is an `Err` so the outer loop reconnects.
async fn publish_session(
    url: &str,
    cfg: &HeartbeatConfig,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> Result<()> {
    let conn = crate::rabbitmq::connect_with_timeout(url).await?;
    let channel = conn.create_channel().await.context("AMQP channel")?;

    channel
        .exchange_declare(
            &cfg.exchange,
            ExchangeKind::Direct,
            ExchangeDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP exchange_declare")?;

    let props = BasicProperties::default()
        .with_content_type("application/xml".into())
        .with_delivery_mode(DELIVERY_MODE_PERSISTENT);

    tracing::info!(
        exchange = %cfg.exchange,
        routing_key = %cfg.routing_key,
        service_id = %cfg.service_id,
        interval_ms = cfg.interval.as_millis() as u64,
        "heartbeat publisher started"
    );

    let mut ticker = tokio::time::interval(cfg.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        // Fixed-rate cadence: the period stays `interval` regardless of publish
        // latency (not interval+publish), and the first tick fires immediately.
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("heartbeat publisher shutting down");
                    return Ok(());
                }
            }
            _ = ticker.tick() => {}
        }

        // A server-closed channel lets `basic_publish` resolve Ok into the void
        // — the loop would "publish" forever while the watchdog hears nothing.
        // Bail so the outer loop reconnects (same pre-flight guard as publisher.rs).
        if !channel.status().connected() {
            anyhow::bail!("heartbeat channel disconnected");
        }

        let xml = build_heartbeat_xml(&cfg.service_id, &Utc::now());

        // Race the publish against shutdown so a wedged broker can't pin the
        // task past the graceful-shutdown drain budget.
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("heartbeat publisher shutting down");
                    return Ok(());
                }
            }
            result = channel.basic_publish(
                &cfg.exchange,
                &cfg.routing_key,
                BasicPublishOptions::default(),
                xml.as_bytes(),
                props.clone(),
            ) => {
                result.context("AMQP basic_publish heartbeat")?;
            }
        }
    }
}

#[cfg(test)]
mod tests;
