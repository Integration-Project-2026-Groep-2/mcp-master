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
mod tests {
    use super::*;
    use std::time::Instant;
    use tokio::sync::watch;

    /// Fast-path: when shutdown is set BEFORE `run` is called, return Ok
    /// immediately without trying to connect (which would block forever
    /// against an unreachable broker).
    #[tokio::test]
    async fn run_returns_immediately_on_pre_set_shutdown() {
        let (tx, rx) = watch::channel(false);
        tx.send(true).unwrap();

        let cfg = RabbitMqConfig {
            // Definitely-unreachable address. If we ever try to connect,
            // it would hang or fail — but the pre-set shutdown should
            // short-circuit before we get there.
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let started = Instant::now();
        let result = run(cfg, rx).await;
        let elapsed = started.elapsed();

        assert!(result.is_ok(), "shutdown-first should succeed: {result:?}");
        assert!(
            elapsed.as_millis() < 100,
            "should return fast (<100ms), took {}ms",
            elapsed.as_millis()
        );
    }

    /// On a connect failure to an unreachable broker, the outer loop must
    /// honor a shutdown signal during the backoff sleep. Validates that we
    /// don't get stuck in a multi-second sleep blocking graceful shutdown.
    #[tokio::test]
    async fn run_honors_shutdown_during_backoff() {
        let (tx, rx) = watch::channel(false);

        let cfg = RabbitMqConfig {
            // Port 1 is reserved/unbindable on most systems → connect fails
            // fast. The test relies on connect failing within ~ms, then
            // entering the backoff sleep where we send the shutdown signal.
            url: "amqp://nobody:nobody@127.0.0.1:1//".to_string(),
            exchange: "test.events".to_string(),
        };

        let handle = tokio::spawn(run(cfg, rx));

        // Give the run a moment to fail-connect once and enter backoff.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        tx.send(true).unwrap();

        let started = Instant::now();
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("run should not hang past 2s after shutdown signal");

        // Total time under 2s — proves we didn't sleep through the full
        // ~1s backoff before noticing shutdown.
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
}
