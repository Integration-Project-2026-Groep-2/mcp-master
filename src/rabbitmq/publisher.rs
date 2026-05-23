use anyhow::{Context, Result};
use lapin::{
    BasicProperties, Channel, Connection, ExchangeKind,
    options::{BasicPublishOptions, ExchangeDeclareOptions},
    types::FieldTable,
};
use serde_json::Value;
use tokio::sync::Mutex;

use super::config::RabbitMqConfig;

/// AMQP publisher with single-flight reconnect on transient disconnects.
///
/// Holds the connection + channel behind an async `Mutex` so we can swap
/// them out when lapin reports the channel/connection has gone away. Lock
/// contention is fine here: publishes happen on the order of a few per
/// `/chat` request (a small handful per second under normal load), and
/// each call holds the mutex only for the brief publish round-trip.
pub struct Publisher {
    inner: Mutex<PublisherInner>,
    config: RabbitMqConfig,
    exchange: String,
}

struct PublisherInner {
    // Connection must outlive the Channel — dropping it kills the IO loop
    // which silently breaks publishes. See lapin docs.
    _conn: Connection,
    channel: Channel,
}

impl Publisher {
    pub async fn connect(config: &RabbitMqConfig) -> Result<Self> {
        let inner = Self::open_inner(config).await?;
        Ok(Self {
            inner: Mutex::new(inner),
            config: config.clone(),
            exchange: config.exchange.clone(),
        })
    }

    /// Connect once, declare the exchange, return the (Conn, Channel) pair.
    /// Used both by `connect` and by the reconnect path inside `publish_event`.
    async fn open_inner(config: &RabbitMqConfig) -> Result<PublisherInner> {
        let conn = super::connect_with_timeout(&config.url).await?;
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
        Ok(PublisherInner {
            _conn: conn,
            channel,
        })
    }

    pub async fn publish_event(&self, event: &str, payload: Value) -> Result<()> {
        let envelope = serde_json::json!({
            "event": event,
            "source": "mcp-master",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "payload": payload,
        });
        let body = serde_json::to_vec(&envelope)?;
        let routing_key = format!("event.{event}");

        let mut guard = self.inner.lock().await;

        // Pre-flight check: if lapin says the channel/connection is dead,
        // skip the doomed first attempt and reconnect immediately. This
        // matters because `basic_publish` against a dead channel can hang
        // for the full lapin internal timeout instead of erroring fast.
        if !guard.channel.status().connected() {
            tracing::warn!("AMQP publisher channel disconnected — reconnecting");
            *guard = Self::open_inner(&self.config)
                .await
                .context("AMQP publisher reconnect (pre-flight)")?;
        }

        match Self::do_publish(&guard.channel, &self.exchange, &routing_key, &body).await {
            Ok(()) => Ok(()),
            Err(e) if is_likely_disconnect(&e) || !guard.channel.status().connected() => {
                tracing::warn!(
                    "AMQP publish failed (likely disconnect): {e:#} — reconnecting and retrying"
                );
                *guard = Self::open_inner(&self.config)
                    .await
                    .context("AMQP publisher reconnect")?;
                Self::do_publish(&guard.channel, &self.exchange, &routing_key, &body)
                    .await
                    .context("AMQP publisher retry after reconnect")
            }
            Err(e) => Err(e),
        }
    }

    async fn do_publish(
        channel: &Channel,
        exchange: &str,
        routing_key: &str,
        body: &[u8],
    ) -> Result<()> {
        channel
            .basic_publish(
                exchange,
                routing_key,
                BasicPublishOptions::default(),
                body,
                BasicProperties::default().with_content_type("application/json".into()),
            )
            .await
            .context("AMQP basic_publish")?;
        Ok(())
    }
}

/// Heuristic that catches IO-level failures (connection closed, protocol
/// error, channel closed) without forcibly retrying deterministic errors
/// like exchange-not-found or auth-failure. Conservative — false-negatives
/// just propagate the error to the caller; false-positives waste one
/// reconnect attempt that immediately re-fails on the second publish.
pub(crate) fn is_likely_disconnect(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}").to_ascii_lowercase();
    [
        "connectionreset",
        "connection reset",
        "connection closed",
        "channel closed",
        "broken pipe",
        "io error",
        "protocol error",
        "transport",
    ]
    .iter()
    .any(|needle| msg.contains(needle))
}

#[cfg(test)]
mod tests;
