use anyhow::{Context, Result};
use futures_util::StreamExt;
use lapin::{
    Connection, ConnectionProperties, ExchangeKind,
    options::{BasicConsumeOptions, ExchangeDeclareOptions, QueueBindOptions, QueueDeclareOptions},
    types::FieldTable,
};
use tokio::sync::watch;

use super::config::RabbitMqConfig;

pub async fn run(config: RabbitMqConfig, mut shutdown_rx: watch::Receiver<bool>) -> Result<()> {
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
                Some(Err(e)) => tracing::error!("AMQP consumer error: {e:#}"),
                None => {
                    tracing::warn!("AMQP consumer stream ended");
                    return Ok(());
                }
            }
        }
    }
}
