use anyhow::{Context, Result};
use lapin::{
    BasicProperties, Channel, Connection, ConnectionProperties, ExchangeKind,
    options::{BasicPublishOptions, ExchangeDeclareOptions},
    types::FieldTable,
};
use serde_json::Value;

use super::config::RabbitMqConfig;

pub struct Publisher {
    // Keep Connection alive next to Channel — dropping the Connection closes
    // the underlying IO task, which would silently break publishes.
    _conn: Connection,
    channel: Channel,
    exchange: String,
}

impl Publisher {
    pub async fn connect(config: &RabbitMqConfig) -> Result<Self> {
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

        Ok(Self {
            _conn: conn,
            channel,
            exchange: config.exchange.clone(),
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

        self.channel
            .basic_publish(
                &self.exchange,
                &routing_key,
                BasicPublishOptions::default(),
                &body,
                BasicProperties::default().with_content_type("application/json".into()),
            )
            .await
            .context("AMQP basic_publish")?;
        Ok(())
    }
}
