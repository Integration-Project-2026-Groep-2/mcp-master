//! Live service-status from the raw heartbeat stream.
//!
//! Teams publish ~1 Hz heartbeats (XML `<heartbeat>`) to the `heartbeat.direct`
//! exchange (key `routing.heartbeat`); Controlroom indexes them. We tap the same
//! exchange with our own queue (a direct exchange copies to every bound queue,
//! so this doesn't disturb Controlroom) and keep a last-seen-per-service map.
//! `GET /status` derives up/down from each service's age. Read-only, additive.

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures_util::StreamExt;
use lapin::{
    ExchangeKind,
    options::{BasicConsumeOptions, ExchangeDeclareOptions, QueueBindOptions, QueueDeclareOptions},
    types::FieldTable,
};
use quick_xml::de::from_str as xml_from_str;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::rabbitmq::config::RabbitMqConfig;
use crate::retry::backoff_with_jitter;

const EXCHANGE: &str = "heartbeat.direct";
const ROUTING_KEY: &str = "routing.heartbeat";
const CONSUMER_TAG: &str = "mcp-master-heartbeat";

/// A service is "up" if its last heartbeat is within this window. Lenient —
/// heartbeats are ~1 Hz, so this only flags a genuinely silent service, not a
/// missed beat or jitter.
pub const STALE_SECS: i64 = 90;

/// service (lowercased) -> last-seen heartbeat timestamp. Written by the AMQP
/// consumer, read by the `/status` handler.
pub type HeartbeatState = DashMap<String, DateTime<Utc>>;

/// Controlroom's `<heartbeat>` wire shape (`pkg/gen/heartbeat.go`): `serviceId`
/// + `timestamp`. `indexed` is set by Controlroom on ingest, absent inbound.
#[derive(Debug, Deserialize)]
struct RawHeartbeat {
    #[serde(rename = "serviceId")]
    service_id: String,
    timestamp: Option<DateTime<Utc>>,
}

/// Parse a heartbeat XML body into `(service, last_seen)`. `last_seen` falls
/// back to `now` when the `<timestamp>` is absent/unparseable.
fn parse_heartbeat(body: &[u8], now: DateTime<Utc>) -> Result<(String, DateTime<Utc>)> {
    let s = std::str::from_utf8(body).context("heartbeat body utf8")?;
    let hb: RawHeartbeat = xml_from_str(s).context("decoding heartbeat XML")?;
    let service = hb.service_id.trim().to_lowercase();
    if service.is_empty() {
        anyhow::bail!("heartbeat missing serviceId");
    }
    Ok((service, hb.timestamp.unwrap_or(now)))
}

/// Per-service status as served by `GET /status`. `status` is `up`/`down`;
/// services never seen are simply absent (the Frontend renders them "unknown").
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ServiceStatus {
    pub name: String,
    pub status: &'static str,
    pub last_seen: String,
    pub age_seconds: i64,
}

/// Snapshot the heartbeat map into a name-sorted per-service status list.
pub fn snapshot(state: &HeartbeatState, now: DateTime<Utc>) -> Vec<ServiceStatus> {
    let mut out: Vec<ServiceStatus> = state
        .iter()
        .map(|e| {
            let last = *e.value();
            let age = (now - last).num_seconds();
            ServiceStatus {
                name: e.key().clone(),
                status: if age <= STALE_SECS { "up" } else { "down" },
                last_seen: last.to_rfc3339(),
                age_seconds: age,
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Consume raw heartbeats into `state` until shutdown, reconnecting through
/// transient broker outages. Mirrors `incident::consumer::run` but taps a
/// different exchange with a throwaway queue.
pub async fn run(
    config: RabbitMqConfig,
    state: Arc<HeartbeatState>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let mut attempt: u32 = 0;
    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }
        match consume_session(&config, &state, &mut shutdown_rx).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempt = attempt.saturating_add(1);
                let delay = backoff_with_jitter(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "heartbeat consumer connection lost: {e:#} — reconnecting after backoff"
                );
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
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
    state: &HeartbeatState,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> Result<()> {
    let conn = crate::rabbitmq::connect_with_timeout(&config.url).await?;
    let channel = conn.create_channel().await.context("AMQP channel")?;

    // Same params Controlroom declares (direct, durable) — idempotent no-op.
    channel
        .exchange_declare(
            EXCHANGE,
            ExchangeKind::Direct,
            ExchangeDeclareOptions {
                durable: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP exchange_declare heartbeat.direct")?;

    // Anonymous exclusive auto-delete: a transient tap. The map refills within
    // ~1s after a reconnect, so no durability/replay is needed, and the queue
    // vanishes on disconnect (no broker-side residue).
    let queue = channel
        .queue_declare(
            "",
            QueueDeclareOptions {
                exclusive: true,
                auto_delete: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP queue_declare (anonymous)")?;
    let queue_name = queue.name().as_str().to_owned();

    channel
        .queue_bind(
            &queue_name,
            EXCHANGE,
            ROUTING_KEY,
            QueueBindOptions::default(),
            FieldTable::default(),
        )
        .await
        .context("AMQP queue_bind")?;

    let mut consumer = channel
        .basic_consume(
            &queue_name,
            CONSUMER_TAG,
            BasicConsumeOptions {
                no_ack: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP basic_consume")?;

    tracing::info!(
        exchange = EXCHANGE,
        routing_key = ROUTING_KEY,
        "heartbeat consumer started"
    );

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("heartbeat consumer shutting down");
                    return Ok(());
                }
            }
            delivery = consumer.next() => match delivery {
                Some(Ok(msg)) => match parse_heartbeat(&msg.data, Utc::now()) {
                    Ok((service, ts)) => { state.insert(service, ts); }
                    Err(e) => tracing::debug!("skipping unparseable heartbeat: {e:#}"),
                },
                Some(Err(e)) => {
                    return Err(anyhow::Error::from(e).context("heartbeat delivery error"));
                }
                None => anyhow::bail!("heartbeat consumer stream ended unexpectedly"),
            }
        }
    }
}

#[cfg(test)]
mod tests;
