use anyhow::{Context, Result};
use futures_util::StreamExt;
use lapin::{
    Connection, ConnectionProperties, ExchangeKind,
    options::{BasicConsumeOptions, ExchangeDeclareOptions, QueueBindOptions, QueueDeclareOptions},
    types::FieldTable,
};
use tokio::sync::watch;

use super::config::RabbitMqConfig;
use crate::retry::backoff_with_jitter;

/// Consume `ai.events` until shutdown, reconnecting through transient
/// broker outages. Returns `Ok(())` only on the explicit shutdown signal —
/// every other exit path (stream end, IO error, broker close) re-enters
/// the connect loop with exponential backoff.
pub async fn run(config: RabbitMqConfig, mut shutdown_rx: watch::Receiver<bool>) -> Result<()> {
    let mut attempt: u32 = 0;
    loop {
        if *shutdown_rx.borrow() {
            return Ok(());
        }

        match consume_session(&config, &mut shutdown_rx).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                attempt = attempt.saturating_add(1);
                let delay = backoff_with_jitter(attempt);
                tracing::warn!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "AMQP consumer connection lost: {e:#} — reconnecting after backoff"
                );
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            tracing::info!("AMQP consumer shutting down during backoff");
                            return Ok(());
                        }
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
            }
        }
    }
}

/// One connect-and-consume lifecycle. Returns `Ok(())` only when the
/// shutdown signal fires — anything else (stream end, IO error) is an
/// `Err` so the outer loop reconnects.
async fn consume_session(
    config: &RabbitMqConfig,
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

    // Anonymous, exclusive, auto-delete queue: server picks the name; queue
    // dies with this connection. Matches broadcast semantics — every
    // master-agent instance gets its own ephemeral queue.
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
        .context("AMQP queue_declare")?;

    let queue_name = queue.name().as_str().to_owned();

    channel
        .queue_bind(
            &queue_name,
            &config.exchange,
            "event.#",
            QueueBindOptions::default(),
            FieldTable::default(),
        )
        .await
        .context("AMQP queue_bind")?;

    // no_ack=true: at-most-once delivery — fits v1 logging-only behaviour.
    let mut consumer = channel
        .basic_consume(
            &queue_name,
            "mcp-master-consumer",
            BasicConsumeOptions {
                no_ack: true,
                ..Default::default()
            },
            FieldTable::default(),
        )
        .await
        .context("AMQP basic_consume")?;

    tracing::info!(exchange = %config.exchange, queue = %queue_name, "AMQP consumer started");

    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!("AMQP consumer shutting down");
                    return Ok(());
                }
            }
            delivery = consumer.next() => match delivery {
                Some(Ok(msg)) => {
                    tracing::info!(
                        routing_key = %msg.routing_key.as_str(),
                        body = %String::from_utf8_lossy(&msg.data),
                        "AMQP event received"
                    );
                }
                Some(Err(e)) => {
                    // Surface the error to the outer loop so we reconnect
                    // instead of silently swallowing the IO break and
                    // continuing on a dead stream.
                    return Err(anyhow::Error::from(e).context("AMQP consumer delivery error"));
                }
                None => {
                    // Stream ended unexpectedly — broker closed channel,
                    // network blip, etc. Bubble up as an error so the outer
                    // loop reconnects.
                    anyhow::bail!("AMQP consumer stream ended unexpectedly");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
